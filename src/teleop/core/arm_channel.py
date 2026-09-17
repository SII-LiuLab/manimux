"""ArmChannel: one arm's pipeline, controller pose -> retarget -> IK -> gate -> driver.

See docs/core.md for design rationale and empirical data.
"""
from algos.diff_ik import build_from_config
from algos.ik_solver import ArmIK
from algos.nullspace import JointLimitAvoidance, NullSpaceController
from algos.retarget import Retargeter
from algos.safety import LowPass, SafetyGate
from drivers.arm_driver import ArmDriver


class ArmChannel:
    """One arm's pipeline: controller pose -> retarget -> IK -> safety gate -> driver.

    solver='analytic' (default, hardware-validated) uses ik_solver.ArmIK.
    solver='diff' uses the diff_ik.py prototype instead -- see
    docs/algos.md#diff_ikpy. diff_ik_config (an
    algos.solver_config.DiffIKConfig) is required when solver='diff';
    run_teleop.py is responsible for loading/validating it from YAML
    before constructing this class, so this module doesn't need to know
    about YAML/pydantic at all.

    impedance_config (an algos.solver_config.ImpedanceConfig) is
    orthogonal to solver -- it governs how the driver executes whatever
    q the solver computes (config.ARM_STATE position vs. torque), not
    how q is computed. Forwarded straight to ArmDriver; see
    drivers/arm_driver.py for what it's used for.

    axis_map selects the source frame the retargeter maps from -- None
    means config.AXIS_MAP (the PICO controller). A different input device
    passes its own table rather than mutating config. It applies only to
    delta_frame='base'; a body-frame delta has no world->base mapping to
    do and ignores it.

    tool is an optional algos.tool_frame.ToolFrame: the flange -> fingertip
    offset to retarget at. None (default) retargets the flange, which is
    what live PICO teleop wants -- the operator watches the tool and closes
    the loop by hand. A recorded UMI pose is a fingertip midpoint with
    nobody in the loop, so that path passes one; see
    docs/algos.md#tool_framepy.

    delta_frame selects which frame the relative motion is taken in --
    'base' (default) is live PICO teleop's world-referenced delta,
    'ee' is the body-frame delta the policy pipeline trains on and
    deployment executes, which is what recorded UMI replay uses. It
    changes what a tool frame does: under 'base' only ToolFrame.t
    matters, under 'ee' the commanded flange is T_f(0).X.dT.X^-1 so the
    rotation matters too. See algos/retarget.py's module docstring.

    quiet suppresses the constructor's own advisory prints (currently the
    solver='diff' nullspace notice). For throwaway channels built only to
    simulate -- scripts/umi_replay.py's pre-flight preview constructs one
    set per arm on top of the real ones, and printing the same warning
    twice per arm teaches people to skim past it.
    """

    def __init__(self, arm, hand, conn, cfg, solver='analytic',
                 diff_ik_config=None, impedance_config=None, axis_map=None,
                 tool=None, quiet=False, delta_frame='base'):
        self.arm = arm
        self.hand = hand
        self.cfg = cfg
        self.solver = solver
        self.tool = tool
        self.driver = ArmDriver(conn, arm, cfg,
                                impedance_config=impedance_config)
        arm_type = 0 if arm == 'A' else 1
        # ArmIK is constructed regardless of solver: fk_xyzabc()/nsp_dir()/
        # lim_n/lim_p feed the retargeter and safety gate either way, and
        # diff_ik.py's own module docstring is explicit that this coordinate-
        # format usage (not solving) is fine to depend on.
        self.ik = ArmIK(arm_type=arm_type,
                        config_path=cfg.KINE_CFG,
                        pos_tol_mm=cfg.IK_POS_TOL_MM,
                        rot_tol_deg=cfg.IK_ROT_TOL_DEG,
                        max_step_deg=cfg.IK_MAX_STEP_DEG,
                        limit_margin_deg=cfg.LIMIT_MARGIN_DEG,
                        limit_override=cfg.JOINT_LIMIT_OVERRIDE.get(arm))
        # Single source of truth for joint limits; gate, nullspace, and
        # (if enabled) diff_ik all read it.
        lo, hi = self.ik.lim_n, self.ik.lim_p
        self.gate = SafetyGate(lo, hi, cfg.MAX_JOINT_RATE_DEG_S,
                               cfg.LIMIT_MARGIN_DEG,
                               cfg.MAX_TRACKING_ERR_DEG,
                               cfg.MAX_CONSEC_REJECT, cfg.XR_STALE_S,
                               dt_max=cfg.MAX_STEP_DT_S,
                               max_tracking_err_s=cfg.MAX_TRACKING_ERR_S)

        self.diff_solver = None
        self.ns = None
        if solver == 'diff':
            if diff_ik_config is None:
                raise ValueError("solver='diff' requires diff_ik_config "
                                 '(an algos.solver_config.DiffIKConfig)')
            if cfg.NULLSPACE_ENABLED and not quiet:
                # DiffIKSolver doesn't yet support NullSpaceController's
                # reentrant probe solves -- see docs/algos.md#diff_ikpy.
                # Its own redundancy handling (mu_nullspace, a QP cost
                # term) is independent and set via diff_ik_config instead.
                print('⚠️  臂%s: solver=diff 不支持 NullSpaceController，'
                      '已关闭（零空间规避请配置 diff_ik_config.mu_nullspace）'
                      % arm)
            # Shared builder, not an inline construction: replay.py's
            # offline rehearsal builds its solver the same way, so the
            # two cannot drift apart in tolerances or margins.
            self.diff_solver = build_from_config(self.ik, arm_type, cfg,
                                                 diff_ik_config)
        elif cfg.NULLSPACE_ENABLED:
            self.ns = NullSpaceController(
                [JointLimitAvoidance(
                    lo, hi, activation_deg=cfg.NULLSPACE_ACTIVATION_DEG)],
                rate_deg_s=cfg.NULLSPACE_RATE_DEG_S,
                limit_deg=cfg.ARM_ANGLE_LIMIT,
                probe_deg=cfg.NULLSPACE_PROBE_DEG)
        self.rt = Retargeter(cfg, arm, lowpass=LowPass(cfg.LOWPASS_HZ),
                             nullspace=self.ns, axis_map=axis_map, tool=tool,
                             delta_frame=delta_frame)
        self.q_cmd = None
        self.blocked_reason = None
        self.n_sent = 0
        # Command-vs-measured drift; see docs/core.md (P0-2 health metric).
        self.track_err = 0.0
        self.track_err_max = 0.0
        # solver='diff' only: worst-case per-frame solve time this run, so
        # a live dry-run can be compared directly against bench/compare_ik.py's
        # offline solve-time percentiles (see docs/algos.md#diff_ikpy).
        self.diff_solve_ms_max = 0.0

    def tip(self, q):
        """Fingertip position for a configuration, base frame mm -- the
        flange itself when no tool frame is set. What the keep-out box and
        the pre-flight envelope should be measuring: the tool is what
        reaches the robot's body first."""
        x = self.ik.fk_xyzabc(q)
        return list((self.tool.tip_from_flange(x) if self.tool else x)[:3])

    def prepare(self):
        self.driver.prepare()
        self.q_cmd = self.driver.joints()
        print('臂%s 就绪，当前构型 %s'
              % (self.arm, [round(v, 2) for v in self.q_cmd]))

    def _solve_joints(self, target, q_now, zsp_dir, arm_angle):
        """Nullspace probe callback: solve for a candidate arm angle, or None."""
        with self.ik.probing():
            r = self.ik.solve_with_backoff(target, q_now, zsp_dir=zsp_dir,
                                           arm_angle=arm_angle)
        return r.joints if r.ok else None

    def step(self, frame, age_s, dt, q_meas):
        """Return this frame's joint command, or None if nothing should be sent."""
        if q_meas is not None and self.q_cmd is not None:
            self.track_err = max(abs(a - b)
                                 for a, b in zip(self.q_cmd, q_meas))
            self.track_err_max = max(self.track_err_max, self.track_err)
        if frame is None:
            self.blocked_reason = 'no_xr_frame'
            return None
        if not self.rt.update(frame, self.hand, self.q_cmd,
                              self.ik.fk_xyzabc, self.ik.nsp_dir, dt=dt,
                              solve=self._solve_joints):
            self.blocked_reason = 'clutch_released'
            return None

        if self.solver == 'diff':
            r = self.diff_solver.solve(self.rt.target_xyzabc, self.q_cmd, dt)
            if r.solve_time_ms is not None:
                self.diff_solve_ms_max = max(self.diff_solve_ms_max,
                                             r.solve_time_ms)
        else:
            r = self.ik.solve_with_backoff(self.rt.target_xyzabc, self.q_cmd,
                                           zsp_dir=self.rt.zsp_dir,
                                           arm_angle=self.rt.arm_angle)
        if not r.ok:
            self.blocked_reason = 'ik:' + r.reason
            if not self.gate.rejects.hit(r.reason):
                print('⚠️  臂%s 连续拒解 %d 次（%s），退出跟随（松开 clutch 后可重新进入）'
                      % (self.arm, self.gate.rejects.consec, r.reason))
                self.rt.disengage(fault=True)
                self.gate.rejects.clear()
            return None
        self.gate.rejects.clear()

        v = self.gate.check(r.joints, self.q_cmd, q_meas, age_s, dt)
        if not v.ok:
            self.blocked_reason = 'gate:' + v.reason
            # Instantaneous tracking-error overshoot = backpressure, not a fault.
            fault = (v.detail.get('sustained', True)
                     if v.reason == 'tracking_error'
                     else v.reason == 'joint_limit')
            if fault:
                print('⚠️  臂%s 安全闸拦截：%s %s（松开 clutch 后可重新进入）'
                      % (self.arm, v.reason, v.detail))
                self.rt.disengage(fault=True)
            return None

        self.blocked_reason = None
        self.q_cmd = v.joints
        self.n_sent += 1
        return v.joints
