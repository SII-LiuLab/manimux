"""Measured joint-arrival gate for position-controlled CAN arms.

Device adapters own freshness and fault checks. This monitor never sends commands
or interprets a transport timestamp as proof of fresh device feedback.
"""

from __future__ import annotations

import numpy as np

from manimux.embodiments.arm.base import ArmController, ArmState


class JointHomeMonitor:
    def __init__(self, target, *, tolerance_rad, speed_rad_s, settle_s):
        self.target = np.array(target, dtype=float, copy=True)
        limits = np.asarray([tolerance_rad, speed_rad_s, settle_s], dtype=float)
        if (self.target.ndim != 1 or not self.target.size or not np.isfinite(self.target).all()
                or not np.isfinite(limits).all() or np.any(limits <= 0)):
            raise ValueError("Home needs finite joint angles and positive arrival limits")
        self.tolerance_rad, self.speed_rad_s, self.settle_s = limits
        self.supported_release = False
        self.invalidate()

    def invalidate(self):
        self._previous = None
        self._stable_since = None
        self._settled = False

    def observe(self, state: ArmState) -> bool:
        previous = self._previous
        dofs = self.target.size
        error = float(np.max(np.abs(state.joints[:dofs] - self.target)))
        self.detail = f"Home position error={error:.6f} rad"
        if previous is not None and state.monotonic_ns < previous.monotonic_ns:
            self.invalidate()
            raise RuntimeError("Home feedback timestamp moved backwards")
        if previous is not None and state.monotonic_ns == previous.monotonic_ns:
            # Re-reading a cached sample must not advance the stability window.
            return self._settled and error <= self.tolerance_rad
        self._previous = state
        if previous is None:
            return False
        dt = (state.monotonic_ns - previous.monotonic_ns) / 1e9
        speed = float(np.max(np.abs(state.joints[:dofs] - previous.joints[:dofs])) / dt)
        self.detail += f", sampled joint speed={speed:.6f} rad/s"
        if error > self.tolerance_rad or speed > self.speed_rad_s:
            self._stable_since = None
            self._settled = False
        else:
            if self._stable_since is None:
                self._stable_since = state.monotonic_ns
            self._settled = (state.monotonic_ns - self._stable_since) / 1e9 >= self.settle_s
        return self._settled

    def validate(self, state: ArmState):
        if not self.supported_release and not self.observe(state):
            raise RuntimeError(f"{self.detail}; retaining control until stationary at Home")


class CalibratedHomeController(ArmController):
    """Shared release contract for single-channel six-joint CAN controllers.

    Subclasses own readiness checks, SDK resources and the point at which their
    connection can supply actuator support. No SDK calls belong in this class.
    """

    def validate_commands(self, targets):
        self.calibration.validate(targets)

    def command_bounds(self):
        # Six calibrated joint ranges plus the normalized integrated gripper.
        return {self.channel: np.vstack((self.calibration.bounds, [0.0, 1.0]))}

    def configure_home(self, target, **arrival_limits):
        if self._control_owned:
            raise RuntimeError("cannot change Home while motor control is owned")
        q = np.asarray(target, dtype=float)
        if q.shape != (6,):
            raise ValueError("Home requires six arm joints, excluding the gripper")
        self.calibration.validate({self.channel: np.r_[q, 0.5]})
        self._home_monitor = JointHomeMonitor(q, **arrival_limits)

    def require_home_configuration(self):
        if self.execute and self._home_monitor is None:
            raise ValueError(f"{self.channel}: configure Home before connecting motor control")

    def validate_release(self):
        if not self.execute or not self._control_owned:
            return
        self.require_home_configuration()
        if not self._home_monitor.supported_release:
            self.validate_motion_ready()
            self._home_monitor.validate(self.get_states()[self.channel])

    def confirm_supported_release(self):
        if self._home_monitor is not None:
            self._home_monitor.supported_release = True

    def await_release_permission(self):
        from manimux.embodiments.robot.release import ReleaseDecision, require_supported_release

        while True:
            try:
                self.validate_release()
                return
            except Exception as error:
                if require_supported_release(f"{self.channel}: {error}") is ReleaseDecision.SUPPORTED_RELEASE:
                    self.confirm_supported_release()
                    return
