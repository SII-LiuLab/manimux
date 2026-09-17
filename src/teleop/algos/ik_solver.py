#!/usr/bin/env python3
"""IK wrapper — makes Marvin_Kine safe to call inside the teleop control loop.

Adds FK back-substitution validation, branch-jump detection, and
backoff/projection retries on top of the raw ik()/ik_nsp() calls.

See docs/algos.md for design rationale, empirical data, and known pitfalls.
"""
import contextlib
import math
import os
import sys

_SDK = os.environ.get('MARVIN_SDK',
                      '/home/jw/Downloads/TJ_FX_ROBOT_CONTRL_SDK')
if _SDK not in sys.path:
    sys.path.insert(0, _SDK)
from SDK_PYTHON.fx_kine import (Marvin_Kine, FX_InvKineSolvePara,  # noqa: E402
                                convert_to_8x8_matrix)
from algos.safety import merge_limit_override  # noqa: E402  (safety has no SDK dep)

DEFAULT_CFG = os.path.join(_SDK, 'CommonConfig/ccs_m6_40.MvKDCfg')

# J6/J7 self-interference params, measured on hardware. These agree exactly
# with ccs_m6_40.MvKDCfg's BD table and not at all with ccs_m6_31's -- which is
# how we know this is a 4.0 machine (see config.KINE_CFG). Kept as an explicit
# constant rather than read from the file: it is the hardware's own table, and
# it stays correct if someone points KINE_CFG somewhere else.
# See docs/algos.md#bd67_real.
BD67_REAL = [[0.0, -1.025, 110.5],
             [0.0, 1.025, 110.5],
             [0.0, -1.025, -110.5],
             [0.0, 1.025, -110.5]]

# Rejection reasons
OK = 'ok'
R_IK_FAIL = 'ik_failed'          # ik() itself failed (mostly out-of-reach)
R_NSP_FAIL = 'ik_nsp_failed'
R_FK_MISMATCH = 'fk_mismatch'    # "solved" but FK back-substitution misses target
R_JOINT_LIMIT = 'joint_limit'
R_J67 = 'j67_interference'
R_JUMP = 'branch_jump'           # solution-branch jump / overspeed
R_SINGULAR = 'singular'

# Recoverable via backoff/projection — see docs/algos.md#retryable.
RETRYABLE = (R_JUMP, R_IK_FAIL, R_NSP_FAIL, R_FK_MISMATCH,
             R_JOINT_LIMIT, R_J67)


class IKResult:
    __slots__ = ('ok', 'reason', 'joints', 'pos_err_mm', 'rot_err_deg',
                 'max_step_deg', 'min_margin_deg', 'detail')

    def __init__(self, ok, reason, joints=None, pos_err_mm=None,
                 rot_err_deg=None, max_step_deg=None, min_margin_deg=None,
                 detail=None):
        self.ok = ok
        self.reason = reason
        self.joints = joints
        self.pos_err_mm = pos_err_mm
        self.rot_err_deg = rot_err_deg
        self.max_step_deg = max_step_deg
        self.min_margin_deg = min_margin_deg
        self.detail = detail or {}

    def __repr__(self):
        if self.ok:
            return ('<IK ok err=%.3fmm/%.3f° step=%.2f° margin=%.1f°>'
                    % (self.pos_err_mm, self.rot_err_deg or 0,
                       self.max_step_deg or 0, self.min_margin_deg or 0))
        return '<IK FAIL %s %s>' % (self.reason, self.detail)


class ArmIK:
    """Single-arm IK solver. Left arm_type=0, right arm_type=1 — one instance per arm."""

    def __init__(self, arm_type, config_path=DEFAULT_CFG,
                 pos_tol_mm=0.5, rot_tol_deg=0.2,
                 max_step_deg=1.0, limit_margin_deg=3.0,
                 check_j67=True, verbose=False, limit_override=None):
        """
        :param pos_tol_mm:      FK back-substitution position tolerance
        :param rot_tol_deg:     FK back-substitution orientation tolerance
        :param max_step_deg:    max single-joint jump from ref config per
                                frame (250Hz/180°·s⁻¹ limit ≈ 0.72°; 1.0 is looser)
        :param limit_margin_deg: minimum margin to the soft limits
        """
        if arm_type not in (0, 1):
            raise ValueError('arm_type must be 0 (left) or 1 (right)')
        self.arm_type = arm_type
        self.pos_tol_mm = pos_tol_mm
        self.rot_tol_deg = rot_tol_deg
        self.max_step_deg = max_step_deg
        self.limit_margin_deg = limit_margin_deg
        self.check_j67 = check_j67

        self.kine = Marvin_Kine()
        self.kine.log_switch(1 if verbose else 0)
        cfg = self.kine.load_config(arm_type=arm_type,
                                    config_path=config_path)
        if not cfg:
            raise RuntimeError('load_config failed: %s' % config_path)
        if not self.kine.initial_kine(robot_type=cfg['TYPE'][arm_type],
                                      dh=cfg['DH'][arm_type],
                                      pnva=cfg['PNVA'][arm_type],
                                      j67=BD67_REAL):
            raise RuntimeError('initial_kine failed')
        self.cfg = cfg
        pnva = cfg['PNVA'][arm_type]
        # Single source of truth for limits — safety gate and nullspace
        # controller both read lim_n/lim_p directly. See docs/algos.md#limit-override.
        self.lim_n, self.lim_p = merge_limit_override(
            [r[1] for r in pnva], [r[0] for r in pnva], limit_override)
        self.vmax = [r[2] for r in pnva]
        # BD rows are ordered ++, -+, --, +- by (j6, j7) sign quadrant
        self.bd = BD67_REAL
        self._sp = FX_InvKineSolvePara()
        self.n_branch_switch = 0
        self.n_backoff = 0
        self.n_projected = 0
        self._count = True
        self.stats = {k: 0 for k in (OK, R_IK_FAIL, R_NSP_FAIL, R_FK_MISMATCH,
                                     R_JOINT_LIMIT, R_J67, R_JUMP)}

    # ---------- stats ----------

    def _bump(self, reason):
        if self._count:
            self.stats[reason] = self.stats.get(reason, 0) + 1

    @contextlib.contextmanager
    def probing(self):
        """Exclude solves in this block from stats/n_backoff/n_projected.

        Used by the nullspace controller's per-frame probe solves.
        See docs/algos.md#probing.
        """
        prev = self._count
        self._count = False
        try:
            yield
        finally:
            self._count = prev

    # ---------- basic wrappers ----------

    def fk(self, joints):
        return self.kine.fk(joints=list(joints))

    def fk_xyzabc(self, joints):
        return self.kine.mat4x4_to_xyzabc(pose_mat=self.fk(joints))

    def nsp_dir(self, joints):
        """Arm-angle-plane X vector at a config, used as the nullspace
        reference direction (zsp_type=1). Compute once at clutch-engage."""
        _, nsp = self.kine.fk_nsp(joints=list(joints))
        return [nsp[0][0], nsp[1][0], nsp[2][0], 0, 0, 0]

    # ---------- validation ----------

    def j67_ok(self, q):
        """J6/J7 self-interference check. See docs/algos.md#bd67_real."""
        j6, j7 = q[5], q[6]
        row = 0 if (j6 >= 0 and j7 >= 0) else \
              1 if (j6 < 0 and j7 >= 0) else \
              2 if (j6 < 0 and j7 < 0) else 3
        a0, a1, a2 = self.bd[row]
        lim = a0 * j6 * j6 + a1 * j6 + a2
        return (j7 <= lim) if j7 >= 0 else (j7 >= lim)

    def _validate(self, q, target_mat, ref_joints):
        # 1) FK back-substitution — the critical check
        back = self.kine.mat4x4_to_xyzabc(pose_mat=self.fk(q))
        tgt = self.kine.mat4x4_to_xyzabc(pose_mat=target_mat)
        pos_err = math.dist(back[:3], tgt[:3])
        rot_err = max(abs((back[i] - tgt[i] + 180) % 360 - 180)
                      for i in (3, 4, 5))
        if pos_err > self.pos_tol_mm or rot_err > self.rot_tol_deg:
            return IKResult(False, R_FK_MISMATCH, joints=q,
                            pos_err_mm=pos_err, rot_err_deg=rot_err,
                            detail={'note': 'solved but unreachable, rejected'})
        # 2) limit margin
        margins = [min(self.lim_p[i] - q[i], q[i] - self.lim_n[i])
                   for i in range(7)]
        mm = min(margins)
        if mm < self.limit_margin_deg:
            return IKResult(False, R_JOINT_LIMIT, joints=q,
                            pos_err_mm=pos_err, rot_err_deg=rot_err,
                            min_margin_deg=mm,
                            detail={'joint': margins.index(mm)})
        # 3) J6/J7 self-interference
        if self.check_j67 and not self.j67_ok(q):
            return IKResult(False, R_J67, joints=q, pos_err_mm=pos_err,
                            rot_err_deg=rot_err, min_margin_deg=mm,
                            detail={'j6': q[5], 'j7': q[6]})
        # 4) branch jump / overspeed
        step = max(abs(a - b) for a, b in zip(q, ref_joints))
        if step > self.max_step_deg:
            return IKResult(False, R_JUMP, joints=q, pos_err_mm=pos_err,
                            rot_err_deg=rot_err, max_step_deg=step,
                            min_margin_deg=mm,
                            detail={'note': 'suspected branch jump'})
        return IKResult(True, OK, joints=q, pos_err_mm=pos_err,
                        rot_err_deg=rot_err, max_step_deg=step,
                        min_margin_deg=mm)

    # ---------- main entry points ----------

    def solve(self, target_mat, ref_joints, zsp_dir=None, arm_angle=0.0):
        """Solve one frame.

        :param target_mat: 4x4 target pose (flange or TCP, per set_tool_kine)
        :param ref_joints: reference config — pass the PREVIOUS solution, not
                           the measured angle, or servo tracking error feeds
                           back in and causes jitter
        :param zsp_dir:    nullspace reference direction, computed once at
                           clutch-engage via nsp_dir()
        :param arm_angle:  arm angle (deg), operator-controlled, for active
                           singularity/obstacle avoidance
        :return: IKResult. If ok=False, caller should hold the previous
                 command and not use .joints
        """
        ref = list(ref_joints)
        sp = self._sp
        sp.set_input_ik_target_tcp(self.kine.mat4x4_to_mat1x16(target_mat))
        sp.set_input_ik_ref_joint(ref)
        if zsp_dir is not None:
            sp.set_input_ik_zsp_type(1)
            sp.set_input_ik_zsp_para(zsp_dir)
        else:
            sp.set_input_ik_zsp_type(0)

        # Order matters: with zsp_type=1, ik_nsp depends on state set by ik()
        if not self.kine.ik(structure_data=sp):
            self._bump(R_IK_FAIL)
            return IKResult(False, R_IK_FAIL, detail={
                'out_of_range': bool(sp.m_Output_IsOutRange),
                'singular_joints': [i for i, v
                                    in enumerate(sp.m_Output_IsDeg[:]) if v],
                'joint_exceed': [i for i, v
                                 in enumerate(sp.m_Output_JntExdTags[:]) if v]})

        sp.set_input_zsp_angle(float(arm_angle))
        sp.set_dgr1(0.05)
        sp.set_dgr2(0.05)
        if not self.kine.ik_nsp(sturcture_data=sp):
            self._bump(R_NSP_FAIL)
            return IKResult(False, R_NSP_FAIL, detail={
                'out_of_range': bool(sp.m_Output_IsOutRange),
                'singular_joints': [i for i, v
                                    in enumerate(sp.m_Output_IsDeg[:]) if v]})

        # Try RetJoint (the SDK's own pick) first
        q = list(sp.m_Output_RetJoint.to_list())
        res = self._validate(q, target_mat, ref)

        # RetJoint rejected for branch_jump: search AllJoint for a more
        # continuous solution. See docs/algos.md#branch-jump.
        if not res.ok and res.reason == R_JUMP:
            best, best_step = None, float('inf')
            for cand in self._all_solutions(sp):
                step = max(abs(a - b) for a, b in zip(cand, ref))
                if step >= best_step:
                    continue
                c_res = self._validate(cand, target_mat, ref)
                if c_res.ok:
                    best, best_step = c_res, step
            if best is not None:
                best.detail['note'] = 'switched to a more continuous AllJoint solution'
                res = best
                if self._count:
                    self.n_branch_switch += 1

        self._bump(res.reason)
        return res

    def _all_solutions(self, sp):
        """Unpack all candidate solutions from m_OutPut_AllJoint (7 joints/row)."""
        n = int(sp.m_OutPut_Result_Num)
        if n <= 0:
            return []
        flat = sp.m_OutPut_AllJoint.to_list()
        rows = convert_to_8x8_matrix(flat)
        return [list(rows[i][:7]) for i in range(min(n, len(rows)))]

    def solve_xyzabc(self, xyzabc, ref_joints, zsp_dir=None, arm_angle=0.0):
        mat = self.kine.xyzabc_to_mat4x4(xyzabc=list(xyzabc))
        return self.solve(mat, ref_joints, zsp_dir, arm_angle)

    def solve_with_backoff(self, xyzabc, ref_joints, zsp_dir=None,
                           arm_angle=0.0, tries=(1.0, 0.5, 0.25, 0.1)):
        """Back the target off toward the current pose instead of giving up.

        See docs/algos.md#backoff for why unrecovered rejections cascade.
        Isotropic backoff alone can't rescue every case; falls through to
        axis projection (_solve_axis_proj) when it doesn't.
        """
        cur = self.fk_xyzabc(ref_joints)
        res = self._solve_scaled(xyzabc, cur, ref_joints, zsp_dir, arm_angle,
                                 tries)
        if res.ok or res.reason not in RETRYABLE:
            return res
        return self._solve_axis_proj(xyzabc, cur, ref_joints, zsp_dir,
                                     arm_angle, tries) or res

    def _solve_scaled(self, xyzabc, cur, ref_joints, zsp_dir, arm_angle,
                      tries):
        """Scale the target toward cur per `tries`; return the first success."""
        last = None
        for f in tries:
            if f >= 1.0:
                tgt = list(xyzabc)
            else:
                tgt = [c + (t - c) * f for c, t in zip(cur[:3], xyzabc[:3])]
                # orientation interpolated the same way (fine for small
                # angles; large ones are bounded by the Cartesian speed cap)
                tgt += [c + ((t - c + 180) % 360 - 180) * f
                        for c, t in zip(cur[3:6], xyzabc[3:6])]
            res = self.solve_xyzabc(tgt, ref_joints, zsp_dir, arm_angle)
            if res.ok:
                if f < 1.0:
                    res.detail['backoff'] = f
                    if self._count:
                        self.n_backoff += 1
                return res
            last = res
            if res.reason not in RETRYABLE:
                break
        return last

    def _solve_axis_proj(self, xyzabc, cur, ref_joints, zsp_dir, arm_angle,
                         tries):
        """Drop one base-frame axis's displacement component and retry.

        See docs/algos.md#axis-projection for why isotropic backoff alone
        isn't enough. Only translation is projected, not orientation — see
        same doc section. Returns None if every axis also fails.
        """
        best, best_keep = None, -1.0
        for axis in range(3):
            proj = list(xyzabc)
            proj[axis] = cur[axis]              # drop this axis's displacement
            keep = math.dist(proj[:3], cur[:3])
            if keep <= best_keep:               # can't retain more, skip
                continue
            res = self._solve_scaled(proj, cur, ref_joints, zsp_dir,
                                     arm_angle, tries)
            if res is not None and res.ok:
                res.detail['projected'] = 'XYZ'[axis]
                best, best_keep = res, keep
        if best is not None:
            if self._count:
                self.n_projected += 1
        return best


if __name__ == '__main__':
    # Sanity check: move along -X (this config has ~80mm of +X margin) and
    # confirm the validation layer rejects out-of-reach targets.
    import time
    ik = ArmIK(arm_type=0)
    q0 = [44.04, -62.57, -8.92, -57.21, 1.45, -4.39, 2.1]
    x0 = ik.fk_xyzabc(q0)
    zsp = ik.nsp_dir(q0)
    print('start:', [round(v, 2) for v in x0])
    ref, t0 = list(q0), time.perf_counter()
    n_ok = 0
    for i in range(300):
        p = list(x0)
        p[0] -= 0.4 * i                       # 0.4mm back per frame
        p[2] += 0.2 * i
        r = ik.solve_xyzabc(p, ref, zsp_dir=zsp)
        if r.ok:
            ref = r.joints
            n_ok += 1
        elif i % 50 == 0:
            print('  frame %3d rejected: %s %s' % (i, r.reason, r.detail))
    dt = (time.perf_counter() - t0) / 300
    print('success %d/300, per-frame %.3f ms' % (n_ok, dt * 1e3))
    print('stats:', {k: v for k, v in ik.stats.items() if v})
