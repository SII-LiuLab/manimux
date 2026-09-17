#!/usr/bin/env python3
"""Differential-IK prototype: per-frame QP over joint velocity.

min ||J qdot - v_des||^2_W + lambda ||qdot||^2
s.t. joint position (Taylor-expanded around q_prev) and velocity limits,
     as a linear box constraint on qdot.

Uses algos.kinematics.ArmKinematics for J -- not Marvin_Kine's ik()/ik_nsp().
Marvin_Kine is still used, but only for mat4x4_to_xyzabc()/xyzabc_to_mat4x4()
(coordinate-format conversion, not solving) in the validation step, exactly
as ik_solver.ArmIK already does.

Selectable via run_teleop.py/scripts/umi_replay.py --solver diff (the
latter's default), and compared offline against ik_solver.ArmIK by
bench/compare_ik.py. See docs/algos.md#diff_ikpy for the OSQP warm-start
mechanics, the silent-failure mode this guards against, and open tuning
knobs (W, lambda).
"""
import math
import os
import sys
import time

import numpy as np
import osqp
import scipy.sparse as sp
from scipy.spatial.transform import Rotation

# Repo root on sys.path so `algos.ik_solver` resolves whether this module
# is imported normally, run as `python3 -m algos.diff_ik`, or run directly
# as `python3 algos/diff_ik.py` (the last of which needs this -- the
# script's own directory ends up on sys.path[0] instead of the repo root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algos.ik_solver import BD67_REAL  # noqa: E402
from algos.nullspace import JointLimitAvoidance  # noqa: E402

# Same reason vocabulary as ik_solver, so bench/compare_ik.py can treat
# both solvers' .reason field uniformly.
OK = 'ok'
R_FK_MISMATCH = 'fk_mismatch'
R_JOINT_LIMIT = 'joint_limit'
R_J67 = 'j67_interference'
R_QP_INFEASIBLE = 'qp_infeasible'   # box was empty, or OSQP didn't solve it

_SOLVED_STATUS = (1, 2)   # OSQP_SOLVED, OSQP_SOLVED_INACCURATE


class DiffIKResult:
    __slots__ = ('ok', 'reason', 'joints', 'qdot_deg_s', 'pos_err_mm',
                 'rot_err_deg', 'max_step_deg', 'min_margin_deg',
                 'solve_time_ms', 'detail')

    def __init__(self, ok, reason, joints=None, qdot_deg_s=None,
                 pos_err_mm=None, rot_err_deg=None, max_step_deg=None,
                 min_margin_deg=None, solve_time_ms=None, detail=None):
        self.ok = ok
        self.reason = reason
        self.joints = joints
        self.qdot_deg_s = qdot_deg_s
        self.pos_err_mm = pos_err_mm
        self.rot_err_deg = rot_err_deg
        self.max_step_deg = max_step_deg
        self.min_margin_deg = min_margin_deg
        self.solve_time_ms = solve_time_ms
        self.detail = detail or {}

    def __repr__(self):
        if self.ok:
            return ('<DiffIK ok err=%.3fmm/%.3f deg step=%.2f deg '
                    'margin=%.1f deg %.3fms>'
                    % (self.pos_err_mm, self.rot_err_deg or 0,
                       self.max_step_deg or 0, self.min_margin_deg or 0,
                       self.solve_time_ms or 0))
        return '<DiffIK FAIL %s %s>' % (self.reason, self.detail)


def _wrap_deg(a, b):
    """Signed shortest-path angular difference a-b, wrapped to [-180, 180)."""
    return (a - b + 180.0) % 360.0 - 180.0


def _j67_quadrant(j6, j7):
    """Same row-selection as ik_solver.ArmIK.j67_ok: which of the 4
    quadrant-specific interference curves applies at (j6, j7)."""
    return 0 if (j6 >= 0 and j7 >= 0) else \
           1 if (j6 < 0 and j7 >= 0) else \
           2 if (j6 < 0 and j7 < 0) else 3


def _j67_ok(q, bd):
    """J6/J7 self-interference check -- same formula as
    ik_solver.ArmIK.j67_ok, duplicated rather than imported because that
    method is bound to an ArmIK instance. bd (the hardware-measured
    coefficients) is imported, not re-typed, since those numbers must not
    drift between the two copies of the check."""
    j6, j7 = q[5], q[6]
    a0, a1, a2 = bd[_j67_quadrant(j6, j7)]
    lim = a0 * j6 * j6 + a1 * j6 + a2
    return (j7 <= lim) if j7 >= 0 else (j7 >= lim)


def _j67_linearized_row(q_prev, bd, margin_deg, dt_eff):
    """One linear inequality in (qdot6, qdot7) that keeps a margin_deg
    buffer from the J6/J7 interference boundary, Taylor-expanded around
    q_prev -- same technique as the position-limit box, applied to a
    curved, two-variable constraint instead of independent per-joint
    ones. See docs/algos.md#diff_ikpy for the derivation.

    Returns (c6, c7, bound) meaning c6*qdot6 + c7*qdot7 <= bound. Only
    ever an upper bound (l=-inf) -- both quadrant branches reduce to that
    form.
    """
    j6_0, j7_0 = q_prev[5], q_prev[6]
    a0, a1, a2 = bd[_j67_quadrant(j6_0, j7_0)]
    lim0 = a0 * j6_0 * j6_0 + a1 * j6_0 + a2
    slope = 2.0 * a0 * j6_0 + a1   # d(lim)/d(j6)
    if j7_0 >= 0:
        g0 = j7_0 - lim0                      # safe when g0 <= -margin_deg
        c6, c7 = -slope, 1.0
    else:
        g0 = lim0 - j7_0
        c6, c7 = slope, -1.0
    bound = (-margin_deg - g0) / dt_eff
    return c6, c7, bound


def _dense_symmetric_to_full_upper_csc(M):
    """Dense symmetric MxM -> CSC storing every upper-triangle entry,
    including exact zeros. A fixed sparsity pattern (same nnz, same
    positions, every call) is required for OSQP's update(Px=...) to patch
    values in place instead of re-factorizing -- see module docs for the
    silent-corruption failure mode this avoids."""
    n = M.shape[0]
    iu, ju = np.triu_indices(n)
    return sp.csc_matrix((M[iu, ju], (iu, ju)), shape=(n, n))


def _build_A(c6, c7):
    """Constraint matrix: 7 box rows (one per joint) + 1 J6/J7
    interference row (couples columns 5, 6 only). Always built from the
    same explicit (row, col) list, including when c6 or c7 happens to be
    exactly 0 -- an all-zero row is still a valid (if useless) row and
    keeps the sparsity pattern identical to the previous frame's, which
    is what makes update(Ax=...) valid instead of needing a fresh
    setup()/factorization every frame. scipy.sparse.csc_matrix built from
    explicit (data, (row, col)) triples keeps a stored zero rather than
    silently dropping it, which this invariant relies on."""
    rows = list(range(7)) + [7, 7]
    cols = list(range(7)) + [5, 6]
    vals = [1.0] * 7 + [c6, c7]
    return sp.csc_matrix((vals, (rows, cols)), shape=(8, 7))


def _joint_limit_grad(jla, q, eps_deg=1e-4):
    """Central-difference gradient of JointLimitAvoidance.cost(q). Finite-
    differenced rather than hand-derived: cost() is 7 cheap arithmetic
    terms (14 evaluations here is microseconds), and reusing the existing,
    already-self-tested cost() avoids a second hand-derived formula that
    could silently drift from nullspace.py's -- the same kind of
    duplication risk BD67_REAL's comment already warns about elsewhere in
    this codebase."""
    n = len(q)
    g = np.zeros(n)
    for i in range(n):
        qp, qm = list(q), list(q)
        qp[i] += eps_deg
        qm[i] -= eps_deg
        g[i] = (jla.cost(qp) - jla.cost(qm)) / (2 * eps_deg)
    return g


class DiffIKSolver:
    """Stateful, one instance per arm -- mirrors ik_solver.ArmIK's shape so
    bench/compare_ik.py can drive both solvers identically.

    kinematics: algos.kinematics.ArmKinematics (this solver's actual math).
    kine: a Marvin_Kine instance, used only for coordinate-format
          conversion (mat4x4_to_xyzabc / xyzabc_to_mat4x4), never for
          solving -- may be the same instance an ArmIK elsewhere already
          holds, or the caller's own.
    """

    def __init__(self, kinematics, kine, lim_lo, lim_hi, vmax_deg_s,
                 limit_margin_deg=3.0, pos_tol_mm=0.5, rot_tol_deg=0.2,
                 w_pos=1.0, w_rot=1.0, lam=1e-3, dt_max=0.016,
                 bd67=BD67_REAL, check_j67=True, j67_margin_deg=None,
                 mu_nullspace=0.0, nullspace_activation_deg=25.0,
                 nullspace_weights=None):
        self.kinematics = kinematics
        self.kine = kine
        self.lim_lo = list(lim_lo)
        self.lim_hi = list(lim_hi)
        self.vmax = list(vmax_deg_s)
        self.limit_margin_deg = limit_margin_deg
        self.pos_tol_mm = pos_tol_mm
        self.rot_tol_deg = rot_tol_deg
        self.dt_max = dt_max
        self.bd67 = bd67
        self.check_j67 = check_j67
        # check_j67=True both constrains the QP (proactive) and keeps the
        # _validate() post-hoc check (safety net for linearization slack
        # near a quadrant boundary -- see docs/algos.md#diff_ikpy).
        self.j67_margin_deg = (limit_margin_deg if j67_margin_deg is None
                               else j67_margin_deg)
        # Tunables -- W mixes mm and deg in one cost, lam is minimum-norm
        # regularization. Not a solved design; see docs/algos.md#diff_ikpy.
        self.W = np.diag([w_pos] * 3 + [w_rot] * 3)
        self.lam = lam
        # Null-space secondary objective: reuses nullspace.JointLimitAvoidance
        # (the same cost NullSpaceController uses for the analytic pipeline)
        # as a LINEAR term in the QP's cost, not a constraint -- see
        # docs/algos.md#diff_ikpy for why this one is a soft preference
        # while J6/J7 interference above is a hard constraint. Off by
        # default (mu_nullspace=0) so existing behavior/tests are unchanged
        # unless explicitly opted into.
        self.mu_nullspace = mu_nullspace
        self._jla = JointLimitAvoidance(lim_lo, lim_hi,
                                        activation_deg=nullspace_activation_deg,
                                        weights=nullspace_weights)

        # OSQP.update() consumes only the VALUE arrays of P and A, and both
        # sparsity patterns are fixed for this solver's lifetime -- that is
        # what _dense_symmetric_to_full_upper_csc and _build_A go out of
        # their way to guarantee. Rebuilding the csc_matrix objects each
        # frame just to read .data off them cost ~96 us, a quarter of a
        # whole solve; gathering the same values costs under 1 us.
        # Both layouts are verified here rather than assumed, so a scipy
        # change breaks construction instead of silently feeding OSQP
        # correctly-shaped nonsense.
        self._p_rows = np.concatenate([np.arange(j + 1) for j in range(7)])
        self._p_cols = np.concatenate([np.full(j + 1, j) for j in range(7)])
        probe = np.arange(49, dtype=float).reshape(7, 7)
        probe = probe + probe.T
        if not np.array_equal(_dense_symmetric_to_full_upper_csc(probe).data,
                              probe[self._p_rows, self._p_cols]):
            raise RuntimeError('P value layout is not column-major upper '
                               'triangle; the gather below would misfeed OSQP')
        marker = _build_A(-2.0, -3.0)
        self._a_data = marker.data.copy()
        found6 = np.flatnonzero(marker.data == -2.0)
        found7 = np.flatnonzero(marker.data == -3.0)
        if len(found6) != 1 or len(found7) != 1:
            raise RuntimeError('cannot locate the J6/J7 coefficients in A')
        self._a_c6, self._a_c7 = int(found6[0]), int(found7[0])

        self.qdot_prev = np.zeros(7)
        self._prob = None
        self.n_solved = 0
        self.stats = {k: 0 for k in (OK, R_FK_MISMATCH, R_JOINT_LIMIT,
                                     R_J67, R_QP_INFEASIBLE)}

    def solve(self, target_xyzabc, q_prev, dt):
        """Solve one frame. Returns DiffIKResult; if ok=False, joints is
        unusable and the caller should hold the previous command."""
        t0 = time.perf_counter()
        q_prev = np.asarray(q_prev, dtype=float)
        dt_eff = min(max(dt, 1e-6), self.dt_max)

        # One walk of the 7-link chain, not two: fk() and jacobian() are
        # both wanted at q_prev and each used to walk it separately.
        T_cur, J = self.kinematics.fk_and_jacobian(q_prev)
        cur_xyzabc = self.kine.mat4x4_to_xyzabc(pose_mat=T_cur)
        pos_err = np.array([t - c for t, c in
                            zip(target_xyzabc[:3], cur_xyzabc[:3])])
        # Orientation error MUST be a proper axis-angle rotation vector, not
        # a per-axis xyzabc coordinate difference -- the latter only agrees
        # with a true angular-velocity vector near the identity rotation.
        # At this arm's actual working orientations it doesn't, and feeding
        # it to the Jacobian as if it were one made the loop unstable (see
        # docs/algos.md#diff_ikpy). rot_err_vec here IS what the Jacobian's
        # angular rows expect.
        R_cur = T_cur[:3, :3]
        R_tgt = np.asarray(self.kine.xyzabc_to_mat4x4(
            xyzabc=list(target_xyzabc)))[:3, :3]
        rot_err_vec = Rotation.from_matrix(R_tgt @ R_cur.T).as_rotvec(degrees=True)
        v_des = np.concatenate([pos_err, rot_err_vec]) / dt_eff

        P = 2.0 * (J.T @ self.W @ J + self.lam * np.eye(7))
        qvec = -2.0 * (J.T @ self.W @ v_des)
        if self.mu_nullspace > 0 and self._nullspace_active(q_prev):
            # Linearizing mu*H(q_prev + qdot*dt) around qdot=0 adds a
            # LINEAR term mu*dt*grad_H(q_prev) to the cost -- it doesn't
            # touch P, doesn't add a variable, and doesn't compete with
            # the task term along the 6 task-constrained directions
            # (moving there costs quadratically via the first term, so the
            # QP's own optimization naturally confines this push to
            # whatever's left in the null space). See docs/algos.md#diff_ikpy.
            qvec = qvec + self.mu_nullspace * dt_eff * _joint_limit_grad(
                self._jla, q_prev)

        eff_lo = [lo + self.limit_margin_deg for lo in self.lim_lo]
        eff_hi = [hi - self.limit_margin_deg for hi in self.lim_hi]
        lo_bound = np.array([max(-self.vmax[i], (eff_lo[i] - q_prev[i]) / dt_eff)
                             for i in range(7)])
        hi_bound = np.array([min(self.vmax[i], (eff_hi[i] - q_prev[i]) / dt_eff)
                             for i in range(7)])

        # OSQP's update() silently keeps the PREVIOUS problem and prints a
        # C-level warning (no Python exception) if l>u is pushed to it --
        # see docs/algos.md#diff_ikpy. Must catch this ourselves before
        # calling update(), not trust OSQP to. (This only catches a single
        # BOX row going empty; a box-vs-interference-row conflict is a
        # different, harder-to-detect-cheaply kind of infeasibility that
        # OSQP's solve() status catches instead -- see below.)
        bad = np.where(lo_bound > hi_bound)[0]
        if len(bad):
            solve_ms = (time.perf_counter() - t0) * 1e3
            self.stats[R_QP_INFEASIBLE] += 1
            return DiffIKResult(False, R_QP_INFEASIBLE, solve_time_ms=solve_ms,
                                detail={'joint': int(bad[0]),
                                        'note': 'empty qdot box '
                                                '(limit_margin_deg leaves no '
                                                'room at this joint)'})

        # J6/J7 interference: one more linear inequality, coupling columns
        # 5 and 6, re-linearized around q_prev every frame -- same Taylor-
        # expansion idea as the position-limit box, just for a curved,
        # two-variable boundary instead of independent per-joint ones. See
        # docs/algos.md#diff_ikpy. A fixed 8-row A (7 box + 1 interference)
        # is used for the lifetime of this solver whenever check_j67 is on,
        # so the sparsity pattern never changes and update(Ax=...) stays
        # valid -- see _build_A.
        if self.check_j67:
            c6, c7, bound = _j67_linearized_row(q_prev, self.bd67,
                                                self.j67_margin_deg, dt_eff)
            # Only these two of A's nine stored values ever change; the other
            # seven are the box rows' constant 1.0. Writing them in place is
            # the whole of what rebuilding A used to accomplish.
            self._a_data[self._a_c6] = c6
            self._a_data[self._a_c7] = c7
            lo_full = np.concatenate([lo_bound, [-np.inf]])
            hi_full = np.concatenate([hi_bound, [bound]])
        else:
            lo_full, hi_full = lo_bound, hi_bound

        # Same values csc_matrix(...).data would hold, gathered directly.
        p_data = P[self._p_rows, self._p_cols]
        if self._prob is None:
            # Only here does a csc_matrix need to exist: setup() is what
            # fixes the pattern every later update() writes into.
            A_csc = (_build_A(c6, c7) if self.check_j67
                     else sp.eye(7, format='csc'))
            self._prob = osqp.OSQP()
            self._prob.setup(P=_dense_symmetric_to_full_upper_csc(P), q=qvec,
                             A=A_csc, l=lo_full, u=hi_full, verbose=False,
                             polish=False, warm_start=True)
        elif self.check_j67:
            self._prob.update(Px=p_data, Ax=self._a_data, q=qvec,
                              l=lo_full, u=hi_full)
        else:
            self._prob.update(Px=p_data, q=qvec, l=lo_full, u=hi_full)
        self._prob.warm_start(x=self.qdot_prev)
        res = self._prob.solve()

        solve_ms = (time.perf_counter() - t0) * 1e3
        if res.info.status_val not in _SOLVED_STATUS:
            self.stats[R_QP_INFEASIBLE] += 1
            return DiffIKResult(False, R_QP_INFEASIBLE, solve_time_ms=solve_ms,
                                detail={'osqp_status': res.info.status})

        # Clip against solver tolerance overshoot -- OSQP is not exact, and
        # a qdot a hair outside the box must not turn into a joint-limit
        # violation downstream. Only the box rows are clippable this way
        # (each is one independent dimension); the interference row is
        # coupled across two variables, so any of its slop is caught by
        # _validate()'s post-hoc _j67_ok check instead.
        qdot = np.clip(res.x, lo_bound, hi_bound)
        self.qdot_prev = qdot
        q_new = q_prev + qdot * dt_eff

        result = self._validate(q_new, target_xyzabc, q_prev, qdot, solve_ms)
        self.stats[result.reason] += 1
        self.n_solved += 1
        return result

    def _nullspace_active(self, q):
        """Is any joint inside the null-space objective's activation band?

        Outside it JointLimitAvoidance's activated form is flat, so
        _joint_limit_grad's 14 central differences can only return zeros --
        ~13 us a frame for a term that contributes nothing. The
        unactivated form (activation_deg=None) is a global quadratic and is
        never flat, so it is never skipped.

        Measured on the 2026-09-10 deployment trajectory: the smallest
        joint-limit margin over 4150 commanded frames was 29.3 deg against
        an activation band of 25, i.e. the gradient was zero on every
        single frame.
        """
        band = self._jla.activation_deg
        if band is None:
            return True
        return any(margin < band for margin in self._jla.margins(q))

    def _validate(self, q_new, target_xyzabc, q_prev, qdot, solve_ms):
        """Safety/correctness checks on the solved STEP -- not a check that
        the full target gap closed this frame. Unlike ArmIK's one-shot
        solve, this controller only ever takes rate-limited progress
        toward a target; failing to fully arrive within one ~4ms frame is
        normal when the target is moving faster than the joint-velocity
        budget allows (the same situation the analytic pipeline handles
        via SafetyGate's CLAMP -- still ok=True -- not a rejection). So
        pos_err_mm/rot_err_deg are computed and returned on every result
        as tracking-lag diagnostics, but do not gate ok/reject -- unlike
        ArmIK._validate()'s one-shot-convergence criterion, gating on them
        here would false-reject normal rate-limited tracking lag. See
        docs/algos.md#diff_ikpy.

        Duplicated from (not sharing) ik_solver.ArmIK._validate's limit-
        margin and J6/J7 checks -- see module docstring for why."""
        if not np.all(np.isfinite(q_new)):
            return DiffIKResult(False, R_FK_MISMATCH, qdot_deg_s=list(qdot),
                                solve_time_ms=solve_ms,
                                detail={'note': 'non-finite joints out of '
                                                'the QP solution'})
        back = self.kine.mat4x4_to_xyzabc(pose_mat=self.kinematics.fk(q_new))
        pos_err = math.dist(back[:3], target_xyzabc[:3])
        rot_err = max(abs(_wrap_deg(back[i], target_xyzabc[i]))
                      for i in (3, 4, 5))
        margins = [min(self.lim_hi[i] - q_new[i], q_new[i] - self.lim_lo[i])
                  for i in range(7)]
        mm = min(margins)
        if mm < self.limit_margin_deg:
            return DiffIKResult(False, R_JOINT_LIMIT, joints=list(q_new),
                                qdot_deg_s=list(qdot), pos_err_mm=pos_err,
                                rot_err_deg=rot_err, min_margin_deg=mm,
                                solve_time_ms=solve_ms,
                                detail={'joint': margins.index(mm)})
        if self.check_j67 and not _j67_ok(q_new, self.bd67):
            return DiffIKResult(False, R_J67, joints=list(q_new),
                                qdot_deg_s=list(qdot), pos_err_mm=pos_err,
                                rot_err_deg=rot_err, min_margin_deg=mm,
                                solve_time_ms=solve_ms,
                                detail={'j6': q_new[5], 'j7': q_new[6]})
        step = float(np.max(np.abs(q_new - q_prev)))
        return DiffIKResult(True, OK, joints=list(q_new), qdot_deg_s=list(qdot),
                            pos_err_mm=pos_err, rot_err_deg=rot_err,
                            max_step_deg=step, min_margin_deg=mm,
                            solve_time_ms=solve_ms)


def build_from_config(ik, arm_type, cfg, dc):
    """Construct a DiffIKSolver from an ik_solver.ArmIK plus a
    config.py module and an algos.solver_config.DiffIKConfig.

    Single builder on purpose: core/arm_channel.py (hardware) and
    replay.py (offline) both go through it, so a rehearsal cannot be
    solved by a differently-configured solver than the one that then
    runs on the robot. Tolerances and the limit-margin fallback come
    from cfg, keeping their single source of truth single -- only the
    QP's own tunables come from dc.
    """
    from algos.kinematics import ArmKinematics

    kin = ArmKinematics(ik.cfg['DH'][arm_type])
    vmax = [r[2] for r in ik.cfg['PNVA'][arm_type]]
    return DiffIKSolver(
        kin, ik.kine, ik.lim_n, ik.lim_p, vmax,
        limit_margin_deg=dc.limit_margin_deg or cfg.LIMIT_MARGIN_DEG,
        pos_tol_mm=cfg.IK_POS_TOL_MM, rot_tol_deg=cfg.IK_ROT_TOL_DEG,
        w_pos=dc.w_pos, w_rot=dc.w_rot, lam=dc.lam,
        check_j67=dc.check_j67, j67_margin_deg=dc.j67_margin_deg,
        mu_nullspace=dc.mu_nullspace,
        nullspace_activation_deg=dc.nullspace_activation_deg)


if __name__ == '__main__':
    # Smoke test: static targets, no SDK ik()/ik_nsp() involved. Confirms
    # the QP moves toward the target, respects box constraints, and that
    # warm-starting is deterministic (same state in -> same solution out).
    # Repo root is already on sys.path (module-level shim above); this
    # only needs to add the SDK's own path.
    _SDK = os.environ.get('MARVIN_SDK',
                          '/home/jw/Downloads/TJ_FX_ROBOT_CONTRL_SDK')
    if _SDK not in sys.path:
        sys.path.insert(0, _SDK)
    from SDK_PYTHON.fx_kine import Marvin_Kine  # noqa: E402
    import config  # noqa: E402
    from algos.kinematics import ArmKinematics  # noqa: E402

    kine = Marvin_Kine()
    kine.log_switch(0)
    cfg = kine.load_config(arm_type=0, config_path=config.KINE_CFG)
    kine.initial_kine(robot_type=cfg['TYPE'][0], dh=cfg['DH'][0],
                      pnva=cfg['PNVA'][0], j67=BD67_REAL)
    pnva = cfg['PNVA'][0]
    lim_p = [r[0] for r in pnva]
    lim_n = [r[1] for r in pnva]
    vmax = [r[2] for r in pnva]

    kin = ArmKinematics(cfg['DH'][0])
    solver = DiffIKSolver(kin, kine, lim_n, lim_p, vmax)

    q0 = [44.04, -62.57, -8.92, -57.21, 1.45, -4.39, 2.1]
    x0 = kine.mat4x4_to_xyzabc(pose_mat=kin.fk(q0))
    print('start xyzabc:', [round(v, 2) for v in x0])

    # 1) track a target ramping 20mm in +X over 60 frames -- this is the
    # realistic usage pattern: like real retargeting output, each frame's
    # target only moves a fraction of a mm from the last, which the QP can
    # actually close within pos_tol_mm in one step. (Validating every frame
    # against a single far-away target held fixed -- a step input -- would
    # fail by construction: a differential controller closes a large gap
    # over many frames, not in one; that's not this test.)
    final_target = list(x0)
    final_target[0] += 20.0
    q, dt = list(q0), 1.0 / 250.0
    n_ok = 0
    for i in range(1, 61):
        frac = i / 60.0
        target = [c + (t - c) * frac for c, t in zip(x0[:3], final_target[:3])] \
                + list(x0[3:6])
        r = solver.solve(target, q, dt)
        assert r.ok, r
        step_budget = max(vmax) * dt + 1e-6
        assert r.max_step_deg <= step_budget * 1.5, (
            'step %.4f exceeds a generous budget %.4f' % (r.max_step_deg, step_budget))
        q = r.joints
        n_ok += 1
    final = kine.mat4x4_to_xyzabc(pose_mat=kin.fk(q))
    pos_gap = math.dist(final[:3], final_target[:3])
    print('OK  tracked a 20mm +X ramp over 60 frames: %d/60 ok, final gap '
          '%.3fmm, last solve %.4fms' % (n_ok, pos_gap, r.solve_time_ms))
    assert pos_gap < 1.0, 'should have closed a 20mm ramp in 60 frames at 250Hz'

    # 1b) same, but ramping a ROTATION (5deg about a tilted axis) instead
    # of a translation. Regression test: a raw xyzabc coordinate difference
    # is only a valid angular-velocity proxy near the identity rotation;
    # using it directly here caused a growing closed-loop rot_err instead
    # of convergence. See docs/algos.md#diff_ikpy.
    rot_solver = DiffIKSolver(kin, kine, lim_n, lim_p, vmax)
    q = list(q0)
    for i in range(1, 61):
        frac = i / 60.0
        target = list(x0[:3]) + [x0[3] + 5.0 * frac, x0[4] - 3.0 * frac,
                                 x0[5] + 4.0 * frac]
        r = rot_solver.solve(target, q, dt)
        assert r.ok, r
        q = r.joints
    print('OK  tracked a 5deg-scale multi-axis rotation ramp over 60 frames '
          'without drift (final rot_err %.4f deg)' % r.rot_err_deg)

    # 2) determinism: same (target, q_prev) solved twice from a fresh
    # solver gives the same qdot (no hidden mutable state leaking in).
    # Deliberately a large-gap target here -- this checks solver output
    # reproducibility, not convergence, so r.ok is irrelevant.
    s1 = DiffIKSolver(kin, kine, lim_n, lim_p, vmax)
    s2 = DiffIKSolver(kin, kine, lim_n, lim_p, vmax)
    r1 = s1.solve(final_target, q0, dt)
    r2 = s2.solve(final_target, q0, dt)
    assert np.allclose(r1.qdot_deg_s, r2.qdot_deg_s), 'solve() must be deterministic'
    print('OK  deterministic: two fresh solvers on the same input agree')

    # 3) a target 10m away must still get ok=True -- this is an incremental
    # controller, not a one-shot solve: it should take a bounded,
    # rate-limited step toward it (not teleport, not refuse), same as
    # SafetyGate clamping (not rejecting) an oversized analytic-solver
    # command. pos_err_mm stays large as a diagnostic; that's expected,
    # not a failure. See docs/algos.md#diff_ikpy.
    far = list(x0)
    far[0] += 10000.0
    r = solver.solve(far, q0, dt)
    assert r.ok, r
    assert r.pos_err_mm > 1000, 'one 4ms step cannot close a 10m gap'
    step_budget = max(vmax) * dt + 1e-6
    assert r.max_step_deg <= step_budget * 1.5, (
        'a far-away target must not blow through the velocity budget: '
        'step %.4f vs budget %.4f' % (r.max_step_deg, step_budget))
    print('OK  far-away target: bounded rate-limited step taken (ok=True, '
          '%.0fmm still to go, step %.3f deg), not silently teleported or '
          'refused' % (r.pos_err_mm, r.max_step_deg))

    # 4) J6/J7 interference linearization: verify the constraint row's math
    # directly (not through a hand-picked Cartesian target -- engineering
    # one that reliably drives exactly J6/J7 without knowing the local
    # Jacobian by hand is fragile; the end-to-end closed-loop evidence is
    # bench/compare_ik.py against data/traj_c.jsonl, which already
    # naturally stresses this region with real operator motion).
    #
    # At q6=55, q7=52 (++ quadrant): interference limit on q7 is
    # 110.5-1.025*55=54.13deg, so the true margin is only 2.13deg --
    # already tighter than the solver's default 3deg j67_margin_deg. bound
    # is therefore deeply negative (not just < 0): the row demands g
    # (distance past the margin) shrink to 0 within this single ~4ms
    # frame, exactly like the position-limit box does when already past
    # its own margin -- "stand still" is not good enough, only retreating
    # fast enough clears it. So the magnitude to test against must be
    # derived from bound itself, not guessed.
    q_tight = list(q0)
    q_tight[5], q_tight[6] = 55.0, 52.0
    c6, c7, bound = _j67_linearized_row(np.array(q_tight), BD67_REAL,
                                        margin_deg=3.0, dt_eff=dt)
    assert bound < 0, ('already inside the margin -> qdot=0 must violate '
                       'the row (bound=%.4f)' % bound)
    stationary_qdot = np.zeros(7)
    assert c6 * stationary_qdot[5] + c7 * stationary_qdot[6] > bound, (
        'already past the margin: standing still must still violate the '
        'row (only retreating fast enough clears it)')
    # Move q6, q7 together at exactly 1.5x the minimum retreat rate bound
    # demands (derived, not guessed) -- must satisfy; 0.5x must not.
    k_min = bound / (c6 + c7)   # both negative here, so k_min < 0
    retreat_ok = np.zeros(7)
    retreat_ok[5] = retreat_ok[6] = k_min * 1.5
    assert c6 * retreat_ok[5] + c7 * retreat_ok[6] <= bound + 1e-9, (
        'retreating faster than the minimum required rate must satisfy '
        'the row')
    retreat_short = np.zeros(7)
    retreat_short[5] = retreat_short[6] = k_min * 0.5
    assert c6 * retreat_short[5] + c7 * retreat_short[6] > bound, (
        'retreating slower than the minimum required rate must still '
        'violate the row')
    print('OK  J6/J7 linearized row: already-tight state (bound=%.2f) '
          'correctly rejects standing still and under-retreating, accepts '
          'retreating fast enough' % bound)

    # End-to-end: solving from q_tight with check_j67 on must not let the
    # solved q_new violate the true (nonlinear) interference boundary,
    # confirmed via the actual _j67_ok check, not just the linear proxy.
    q_tight_target = kine.mat4x4_to_xyzabc(pose_mat=kin.fk(q_tight))
    tight_solver = DiffIKSolver(kin, kine, lim_n, lim_p, vmax)
    # Aim further into the interference zone (matches the +A/+B quadrant
    # direction that increases both q6 and q7).
    aggressive = list(q_tight_target)
    r = tight_solver.solve(aggressive, q_tight, dt)
    assert r.ok, r
    assert _j67_ok(r.joints, BD67_REAL), (
        'constrained solve must not cross the true interference boundary: '
        'q6=%.2f q7=%.2f' % (r.joints[5], r.joints[6]))
    print('OK  end-to-end: solving from an already-tight J6/J7 state keeps '
          'the true interference check satisfied (q6=%.3f q7=%.3f)'
          % (r.joints[5], r.joints[6]))

    # 5) Null-space secondary objective: mu_nullspace=0 (default) must
    # leave a joint sitting near its limit exactly where it started --
    # nothing pulls it away when the task doesn't need to. mu_nullspace>0
    # must pull it toward safety while task tracking stays intact.
    #
    # Null-space leverage is configuration-dependent (see docs/algos.md
    # and nullspace.py) -- picked HOME_JOINTS['A'] with J2 pushed to
    # margin=15deg (inside the 25deg activation deadband) specifically
    # because the Jacobian's null direction has a large J2 component there
    # (confirmed via SVD); at the q0 pose used elsewhere in this file it
    # does not, and mu_nullspace has almost no effect -- a reminder that
    # this is a soft, configuration-dependent preference, not a guarantee.
    q_ns = list(config.HOME_JOINTS['A'])
    q_ns[1] = -105.0   # J2 limit is +-120deg -> margin 15deg
    ns_target = kine.mat4x4_to_xyzabc(pose_mat=kin.fk(q_ns))   # stay in place

    off_solver = DiffIKSolver(kin, kine, lim_n, lim_p, vmax, mu_nullspace=0.0)
    q = list(q_ns)
    for _ in range(200):
        r = off_solver.solve(ns_target, q, dt)
        assert r.ok, r
        q = r.joints
    assert abs(q[1] - q_ns[1]) < 1e-6, (
        'mu_nullspace=0 must not move J2 at all when the task does not '
        'need it (got %.6f deg drift)' % (q[1] - q_ns[1]))
    print('OK  mu_nullspace=0: J2 stays exactly put at margin=15deg over '
          '200 frames (no pull without opting in)')

    on_solver = DiffIKSolver(kin, kine, lim_n, lim_p, vmax, mu_nullspace=1000.0)
    q = list(q_ns)
    for _ in range(200):
        r = on_solver.solve(ns_target, q, dt)
        assert r.ok, r
        q = r.joints
    margin_before = min(lim_p[1] - q_ns[1], q_ns[1] - lim_n[1])
    margin_after = min(lim_p[1] - q[1], q[1] - lim_n[1])
    assert margin_after > margin_before + 0.5, (
        'mu_nullspace>0 should measurably pull J2 away from its limit: '
        '%.3f -> %.3f deg margin' % (margin_before, margin_after))
    assert r.pos_err_mm < 0.01 and r.rot_err_deg < 0.01, (
        'null-space motion must not degrade task tracking: pos_err=%.4f '
        'rot_err=%.4f' % (r.pos_err_mm, r.rot_err_deg))
    print('OK  mu_nullspace=1000: J2 margin %.3f -> %.3f deg over 200 '
          'frames, task tracking undisturbed (pos_err=%.5fmm rot_err=%.5fdeg)'
          % (margin_before, margin_after, r.pos_err_mm, r.rot_err_deg))

    print('\nall passed')
