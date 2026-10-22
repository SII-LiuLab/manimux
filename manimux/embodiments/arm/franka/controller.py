"""Franka joint-position sessions; no agent, model server or RLinf dependency."""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager

import numpy as np

from manimux.clock import SystemClock
from manimux.embodiments.arm.base import ArmController, ArmState

from .connection import FCIConnectionLease


def _current_thread_scheduler():
    if not hasattr(os, "sched_getscheduler"):
        return None
    return os.sched_getscheduler(0), os.sched_getparam(0)


def _optional_joint_limit(value, name):
    if value is None:
        return None
    limit = np.asarray(value, dtype=float)
    if limit.shape not in ((), (7,)) or not np.isfinite(limit).all() or np.any(limit <= 0):
        raise ValueError(f"{name} requires a positive finite scalar, seven values, or null")
    return limit.tolist()


def _check_joint_limit(values, limit, message, *, error=ValueError, tolerance=0.0):
    if limit is None:
        return
    limits = np.broadcast_to(limit, (7,))
    exceeded = np.flatnonzero(np.abs(values) > limits + tolerance)
    if exceeded.size:
        joint = int(exceeded[0])
        raise error(
            f"{message}: joint {joint + 1}, value={abs(values[joint]):.6g}, "
            f"limit={limits[joint]:.6g}"
        )


@contextmanager
def _scoped_thread_scheduler(target=None):
    """Contain SDK priority changes to this call, never the whole runtime loop."""
    previous = _current_thread_scheduler()
    try:
        if target is not None and target != previous:
            os.sched_setscheduler(0, *target)
        yield
    finally:
        if previous is not None and _current_thread_scheduler() != previous:
            os.sched_setscheduler(0, *previous)


class FrankaController(ArmController):
    """One FCI owner, with absolute seven-joint targets in radians.

    Franky runs its native real-time impedance controller; ManiMux remains the
    only target writer. Targets are asynchronous and returning is not arrival.
    Read-only sessions use pylibfranka because Franky's constructor writes
    collision thresholds. Neither path homes, unlocks, or recovers errors.
    """

    def __init__(
        self,
        *,
        ip,
        joint_limits,
        clock=None,
        execute=False,
        max_target_delta_rad=None,
        max_reference_step_rad=None,
        measured_velocity_limit_rad_s=None,
        joint_stiffness=(600, 600, 600, 600, 250, 150, 50),
        joint_damping=(50, 50, 50, 50, 30, 25, 15),
        max_delta_tau=1.0,
        torque_threshold=20.0,
        force_threshold=30.0,
        stop_timeout_s=5.0,
        feedback_timeout_s=0.2,
    ):
        if not isinstance(ip, str) or not ip.strip():
            raise ValueError("Franka requires a controller IP/hostname")
        bounds = np.array(joint_limits, dtype=float, copy=True)
        if (
            bounds.shape != (7, 2)
            or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] >= bounds[:, 1])
        ):
            raise ValueError("Franka requires seven finite ordered joint bounds")
        values = [
            max_delta_tau,
            torque_threshold,
            force_threshold,
            stop_timeout_s,
            feedback_timeout_s,
        ]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Franka dynamics, thresholds and timeouts must be positive")
        gains = np.array([joint_stiffness, joint_damping], dtype=float)
        if gains.shape != (2, 7) or not np.isfinite(gains).all() or np.any(gains <= 0):
            raise ValueError("Franka stiffness/damping require seven positive finite values")
        self.channel = ip
        self.bounds = bounds
        self.clock = clock or SystemClock()
        self.execute = execute is True
        self.max_target_delta_rad = _optional_joint_limit(
            max_target_delta_rad, "max_target_delta_rad"
        )
        self.max_reference_step = _optional_joint_limit(
            max_reference_step_rad, "max_reference_step_rad"
        )
        self.measured_velocity_limit = _optional_joint_limit(
            measured_velocity_limit_rad_s, "measured_velocity_limit_rad_s"
        )
        self.gains = gains
        self.max_delta_tau = max_delta_tau
        self.thresholds = (torque_threshold, force_threshold)
        self.stop_timeout_s = stop_timeout_s
        self.feedback_timeout_s = feedback_timeout_s
        self._robot = self._sdk = None
        self._state = self._raw = None
        self._device_ms = None
        self._target = None
        self._reference = None
        self._pending = False
        self._control_scheduler = None
        self._sequence = 0
        self._lock = threading.RLock()
        self._connection_lease = FCIConnectionLease(self.channel)

    def connect(self):
        with self._lock:
            if self._robot is not None:
                return
            self._connection_lease.acquire(execute=self.execute)
            try:
                # libfranka 0.19 promotes its constructor's calling thread.
                # Franky's native control thread inherits priority when move()
                # spawns it; ordinary Python work must retain its prior policy.
                with _scoped_thread_scheduler():
                    if self.execute:
                        import franky

                        if not all(
                            hasattr(franky, name)
                            for name in ("JointReferenceHandle", "JointImpedanceTrackingMotion")
                        ):
                            raise RuntimeError(
                                "Franka requires a Franky build with native joint tracking"
                            )
                        self._sdk = franky
                        self._robot = franky.Robot(
                            self.channel,
                            realtime_config=franky.RealtimeConfig.Enforce,
                            default_torque_threshold=self.thresholds[0],
                            default_force_threshold=self.thresholds[1],
                        )
                    else:
                        import pylibfranka

                        self._sdk = pylibfranka
                        self._robot = pylibfranka.Robot(self.channel)
                    self._control_scheduler = _current_thread_scheduler()
                self.get_states()
                if self.execute:
                    self._require_motion_ready()
                    if self._raw.robot_mode != self._sdk.RobotMode.Idle:
                        raise RuntimeError("Franka must be Idle before acquiring control")
            except BaseException:
                self._robot = self._sdk = None
                self._state = self._raw = None
                self._device_ms = None
                self._connection_lease.release()
                raise

    def _require_motion_ready(self):
        if self._raw is None:
            raise RuntimeError("Franka has no measured feedback")
        if self._raw.robot_mode not in (self._sdk.RobotMode.Idle, self._sdk.RobotMode.Move):
            raise RuntimeError(f"Franka is not ready for motion: {self._raw.robot_mode}")

    def get_states(self):
        with self._lock:
            if self._robot is None:
                raise RuntimeError("Franka is not connected")
            if self._pending and self._robot.poll_motion():
                raise RuntimeError("Franka tracking ended unexpectedly; explicit stop required")
            raw = self._robot.state if self.execute else self._robot.read_once()
            device_ms = raw.time.to_msec() if self.execute else round(raw.time.to_sec() * 1000)
            if self._device_ms is not None and device_ms < self._device_ms:
                raise RuntimeError("Franka device time moved backwards; reconnect required")
            if self._state is None or device_ms != self._device_ms:
                self._sequence += 1
                self._state = ArmState(raw.q, self.clock.now_ns(), self._sequence)
                self._device_ms = device_ms
            self._raw = raw
            if (
                not 0
                <= self.clock.now_ns() - self._state.monotonic_ns
                <= (self.feedback_timeout_s * 1e9)
            ):
                raise RuntimeError("Franka feedback is stale")
            # Franky 1.x Errors has no __bool__; even an empty object is truthy.
            # Its native has_errors property performs libfranka's actual check.
            has_errors = self._robot.has_errors if self.execute else bool(raw.current_errors)
            if has_errors:
                raise RuntimeError(
                    f"Franka fault; explicit recovery required: {raw.current_errors}"
                )
            if self._pending:
                _check_joint_limit(
                    raw.dq, self.measured_velocity_limit,
                    "Franka measured joint speed exceeds configured limit",
                    error=RuntimeError,
                )
            return {self.channel: self._state}

    def validate_commands(self, targets):
        if set(targets) != {self.channel}:
            raise ValueError("Franka command must address its single configured channel")
        q = np.asarray(targets[self.channel], dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError("Franka command requires seven finite radians")
        if np.any(q < self.bounds[:, 0]) or np.any(q > self.bounds[:, 1]):
            raise ValueError("Franka target exceeds model joint bounds")

    def command_bounds(self):
        return {self.channel: self.bounds.copy()}

    def send_commands(self, targets):
        self.validate_commands(targets)
        if not self.execute:
            raise PermissionError("Franka session is read-only")
        with self._lock:
            measured = self.get_states()[self.channel].joints
            self._require_motion_ready()
            q = np.array(targets[self.channel], dtype=float, copy=True)
            _check_joint_limit(
                q - measured, self.max_target_delta_rad,
                "Franka target exceeds maximum measured-to-target joint distance",
            )
            previous = measured if self._target is None else self._target
            _check_joint_limit(
                q - previous, self.max_reference_step,
                "Franka reference step exceeds max_reference_step_rad",
                tolerance=1e-10,
            )
            if self._reference is None:
                # The first accepted target has the same meaning as every later
                # target. Optional guards reject before changing the reference.
                self._reference = self._sdk.JointReferenceHandle()
                self._reference.set(q)
                motion = self._sdk.JointImpedanceTrackingMotion(
                    reference_handle=self._reference,
                    stiffness=self.gains[0],
                    damping=self.gains[1],
                    lower_joint_limits=self.bounds[:, 0],
                    upper_joint_limits=self.bounds[:, 1],
                    compensate_coriolis=True,
                    max_delta_tau=self.max_delta_tau,
                )
                with _scoped_thread_scheduler(self._control_scheduler):
                    self._robot.move(motion, asynchronous=True)
                    self._pending = True
            else:
                self._reference.set(q)
            self._target = q

    def stop(self):
        with self._lock:
            if not self.execute or self._robot is None:
                return
            requested_stop = self._robot.is_in_control
            if requested_stop:
                self._robot.stop()
            # join propagates errors from the native control thread, including a
            # motion that failed before stop was requested.
            try:
                if not self._robot.join_motion(timeout=self.stop_timeout_s):
                    raise TimeoutError(
                        "Franka control thread did not stop; cleanup remains retryable"
                    )
            except self._sdk.ControlException as exc:
                # Franky's tracking wrapper uses the same libfranka stop protocol.
                # Only our own preemption is normal; faults remain visible.
                if (
                    not requested_stop
                    or "Move command preempted!" not in str(exc)
                    or self._robot.has_errors
                    or self._robot.is_in_control
                ):
                    raise
            self._pending = False
            self._target = None
            self._reference = None

    def close(self):
        with self._lock:
            self.stop()
            self._robot = self._sdk = None  # SDK RAII releases the FCI connection.
            self._state = self._raw = None
            self._device_ms = None
            self._target = None
            # A failed stop keeps the SDK and lease owned for retryable cleanup.
            self._connection_lease.release()

    def runtime_metadata(self):
        return {
            "sdk": "franky" if self.execute else "pylibfranka",
            "connected": self._robot is not None,
            "execute": self.execute,
            "timestamp_semantics": (
                "first host observation of advancing device time; cached samples retain it"
            ),
            "dispatch": (
                "exact absolute joint references; native persistent impedance tracking; "
                "asynchronous"
            ),
            "max_reference_step_rad": self.max_reference_step,
            "max_target_delta_rad": self.max_target_delta_rad,
            "measured_velocity_limit_rad_s": self.measured_velocity_limit,
            "reference_limits": "optional explicit guards; no clipping or hidden interpolation",
            "command_evidence": "SDK reference submission, not torque output or measured arrival",
            "joint_stiffness": self.gains[0].tolist(),
            "joint_damping": self.gains[1].tolist(),
            "stop": "libfranka stop and finite join; no homing or fault recovery",
            "thread_scheduling": "SDK priority inherited by native control; caller policy restored",
            "collision_thresholds_on_execute_connect": self.thresholds,
        }
