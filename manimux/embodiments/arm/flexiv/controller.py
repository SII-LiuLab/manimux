"""Owned RDK session for one seven-axis Rizon arm, using NRT joint targets."""

from __future__ import annotations

import importlib
import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np

from manimux.clock import SystemClock
from manimux.embodiments.arm.base import ArmController, ArmState

from .safety import JointSafety

logger = logging.getLogger(__name__)


class FlexivController(ArmController):
    """Legacy flat and newer grouped RDK APIs behind the same arm contract.

    Connection never homes, clears faults, or switches motion modes. Enabling
    requires execute=True AND enable_on_connect=True. Read-only cleanup never
    calls Stop. RDK timestamps identify samples; host time records their first
    observation here, not a proven network receipt or device acquisition time.
    """

    def __init__(
        self,
        *,
        serial,
        backend="flexivpy",
        sdk_path=None,
        interfaces=(),
        group=None,
        execute=False,
        enable_on_connect=False,
        max_velocity=0.15,
        max_acceleration=0.3,
        control_mode="joint_position",
        joint_stiffness_scale=1.0,
        joint_damping_ratio=0.7,
        joint_margin=0.03490658503988659,
        joint_limit_policy="reject",
        joint_clip_margin=0.0,
        joint_limit_tolerance=0.0,
        joint_safety=None,
        feedback_timeout_s=0.2,
        enable_timeout_s=20.0,
        clock=None,
        sdk=None,
    ):
        self.channel = str(serial).replace(" ", "")
        if not self.channel or self.channel.startswith("REPLACE"):
            raise ValueError("supply the actual Flexiv serial in the local station")
        if backend not in {"flexivpy", "flexivrdk"}:
            raise ValueError("backend must be flexivpy or flexivrdk")
        if control_mode not in {"joint_position", "joint_impedance"}:
            raise ValueError("Flexiv control_mode must be joint_position or joint_impedance")
        if (not np.isfinite([joint_stiffness_scale, joint_damping_ratio]).all()
                or not 0 <= joint_stiffness_scale <= 1
                or not 0.3 <= joint_damping_ratio <= 0.8):
            raise ValueError("Flexiv joint_stiffness_scale must be in [0,1] and joint_damping_ratio in [0.3,0.8]")
        self.control_mode = control_mode
        self.joint_stiffness_scale = float(joint_stiffness_scale)
        self.joint_damping_ratio = float(joint_damping_ratio)
        self._policy_stiffness = None
        self._prepared_mode = None
        if joint_limit_policy not in {"reject", "clip"}:
            raise ValueError("Flexiv joint_limit_policy must be reject or clip")
        if not np.isfinite(joint_clip_margin) or joint_clip_margin < 0:
            raise ValueError("Flexiv joint_clip_margin must be finite and nonnegative")
        if not np.isfinite(joint_limit_tolerance) or joint_limit_tolerance < 0:
            raise ValueError("Flexiv joint_limit_tolerance must be finite and nonnegative")
        self.joint_limit_policy = joint_limit_policy
        self.joint_clip_margin = float(joint_clip_margin)
        self.joint_limit_tolerance = float(joint_limit_tolerance)
        self._beyond_tolerance_count = 0
        self._max_requested_excess = 0.0
        self._clip_severity = None
        self._joint_safety = None if joint_safety is None else JointSafety(**joint_safety)
        self._guard_count = 0
        self._guard_report_ns = None
        self._last_submitted = None
        self._effective_velocity = max_velocity
        self._clip_count = 0
        self._clip_active = False
        self._clip_report_ns = None
        self._clip_lower = self._clip_upper = None
        if interfaces is None:
            interfaces = ()
        if isinstance(interfaces, str) or not all(
            isinstance(item, str) and item for item in interfaces
        ):
            raise ValueError("Flexiv interfaces must be a list of non-empty host NIC addresses")
        if (
            not np.isfinite(
                [max_velocity, max_acceleration, joint_margin, feedback_timeout_s, enable_timeout_s]
            ).all()
            or min(max_velocity, max_acceleration, feedback_timeout_s, enable_timeout_s) <= 0
            or joint_margin < 0
        ):
            raise ValueError("Flexiv limits/timeouts must be finite and positive; margin >= 0")
        self.execute = execute is True
        self.enable_on_connect = enable_on_connect is True
        self.backend, self.sdk_path = backend, sdk_path
        self.interfaces, self.group_name = list(interfaces), group
        self.max_velocity, self.max_acceleration = max_velocity, max_acceleration
        self.joint_margin = joint_margin
        self.feedback_timeout_s, self.enable_timeout_s = feedback_timeout_s, enable_timeout_s
        self.clock = clock if clock is not None else SystemClock()
        self._sdk = sdk
        self._robot = None
        self._group = None
        self._grouped = False
        self._ready = False
        self._owned = False
        self._enable_pending = False
        self._prepared = False
        self._stamp = None
        self._state = None
        self._guiding = None
        self._lock = threading.RLock()

    def connect(self):
        with self._lock:
            if self._ready:
                return
            if self._robot is not None:
                raise RuntimeError("Flexiv cleanup required before reconnect")
            # A closed handle has no owned session; clear any stale flags before
            # constructing the new one so a fresh connection cannot inherit them.
            self._prepared = self._owned = self._enable_pending = False
            self._prepared_mode = self._policy_stiffness = None
            self._state = self._stamp = None
            self._clip_count = 0
            self._clip_active = False
            self._clip_report_ns = None
            self._beyond_tolerance_count = 0
            self._max_requested_excess = 0.0
            self._clip_severity = None
            self._clip_lower = self._clip_upper = None
            self._guard_count = 0
            self._guard_report_ns = self._last_submitted = None
            if self._joint_safety is not None:
                self._joint_safety.reset()
            if self._sdk is None:
                if self.sdk_path is not None:
                    path = str(Path(self.sdk_path).expanduser().resolve())
                    if path not in sys.path:
                        sys.path.insert(0, path)
                try:
                    self._sdk = importlib.import_module(self.backend)
                except ImportError as exc:
                    raise RuntimeError(
                        f"Cannot import {self.backend}; install a binding for this Python ABI "
                        "and the robot firmware (see Flexiv README)"
                    ) from exc
            sdk = self._sdk
            self._grouped = hasattr(sdk.RobotInfo, "single_arm_groups")
            if self._grouped and self.interfaces:
                raise ValueError("grouped RDK does not accept network interface selection")
            try:
                self._robot = (
                    sdk.Robot(self.channel)
                    if self._grouped
                    else sdk.Robot(self.channel, self.interfaces, True, False)
                )
                info = self._robot.info()
                if self._grouped:
                    groups = list(info.single_arm_groups)
                    if self.group_name is not None:
                        names = getattr(sdk, "kJointGroupNames", {})
                        groups = [
                            g
                            for g in groups
                            if self.group_name in (str(g), getattr(g, "name", ""), names.get(g, ""))
                        ]
                    if len(groups) != 1:
                        raise ValueError("Flexiv requires exactly one selected single-arm group")
                    self._group = groups[0]
                self._device_lower = self._vector(self._value(info.q_min))
                self._device_upper = self._vector(self._value(info.q_max))
                self._lower = self._device_lower + self.joint_margin
                self._upper = self._device_upper - self.joint_margin
                if np.any(self._lower >= self._upper):
                    raise ValueError("invalid Flexiv device limits or joint margin")
                # Clip clearance is a minimum inset from the device limits;
                # preserve a larger explicitly configured joint margin.
                extra = max(0.0, self.joint_clip_margin - self.joint_margin)
                self._clip_lower = self._lower + extra
                self._clip_upper = self._upper - extra
                if self.joint_limit_policy == "clip" and np.any(self._clip_lower >= self._clip_upper):
                    raise ValueError("invalid Flexiv device limits or clip margin")
                if self._joint_safety is not None:
                    velocity = self._vector(self._value(info.dq_max))
                    if np.any(velocity <= 0):
                        raise ValueError("Flexiv SDK joint velocity limits must be positive")
                    self._effective_velocity = min(self.max_velocity, float(np.min(velocity)))
                self.get_states()
                if self.execute and self.enable_on_connect:
                    self._health(require_operational=False)
                    if not self._robot.estop_released():
                        raise RuntimeError("Flexiv E-stop is not released")
                    if not self._robot.operational():
                        self._owned = self._enable_pending = True
                        if self._grouped:
                            self._robot.ServoOn()
                        else:
                            self._robot.Enable()
                        self._wait_operational()
                        self._enable_pending = False
                self._ready = True
            except Exception as error:
                try:
                    self.close()
                except Exception as cleanup_error:
                    raise ExceptionGroup(
                        "Flexiv connect and cleanup failed", [error, cleanup_error]
                    ) from None
                raise

    def _value(self, value):
        return value[self._group] if self._grouped else value

    @staticmethod
    def _vector(value):
        q = np.array(value, dtype=float, copy=True)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError("Flexiv requires seven finite joint values")
        return q

    def _health(self, *, require_operational=True):
        if self._robot is None or not self._robot.connected():
            raise RuntimeError("Flexiv is disconnected")
        if self._robot.fault():
            raise RuntimeError("Flexiv fault; recovery must be explicit")
        recovery = getattr(self._robot, "recovery", None)
        if callable(recovery) and recovery():
            raise RuntimeError(f"Flexiv {self.channel}: controller requires joint recovery")
        if require_operational and not self._robot.operational():
            raise RuntimeError(
                f"Flexiv {self.channel} is not operational; enable must be explicit; "
                f"status={self._operational_status()}"
            )

    def _operational_status(self):
        read = getattr(self._robot, "operational_status", None)
        return str(read()) if callable(read) else "unavailable"

    def _check_remote_mode(self):
        status = self._operational_status()
        if status.rsplit(".", 1)[-1] in {"IN_AUTO_MODE", "IN_MANUAL_MODE"}:
            raise RuntimeError(
                f"Flexiv {self.channel}: status={status}; select Auto (Remote) "
                "in Flexiv Elements before SDK control"
            )

    def _wait_operational(self, stop=None):
        deadline = time.monotonic() + self.enable_timeout_s
        while not self._robot.operational():
            self._health(require_operational=False)
            if stop is not None and stop.is_set():
                raise InterruptedError("Flexiv enable cancelled")
            if time.monotonic() >= deadline:
                reason = (
                    "enable did not complete; stop is not confirmed"
                    if stop is None else "enable timed out"
                )
                raise TimeoutError(
                    f"Flexiv {self.channel}: {reason}; "
                    f"status={self._operational_status()}"
                )
            if stop is None:
                time.sleep(0.02)
            else:
                stop.wait(0.02)

    def get_states(self):
        with self._lock:
            if self._robot is None or not self._robot.connected():
                raise RuntimeError("Flexiv is disconnected")
            raw = self._value(self._robot.states())
            stamp = tuple(raw.timestamp)
            if len(stamp) != 2:
                raise ValueError("Flexiv feedback requires the RDK timestamp pair")
            now = self.clock.now_ns()
            if self._state is None or stamp != self._stamp:
                if self._stamp is not None and stamp < self._stamp:
                    raise RuntimeError("Flexiv feedback timestamp regressed; reconnect required")
                self._state = ArmState(
                    self._vector(raw.q), now, 1 if self._state is None else self._state.sequence + 1
                )
                self._stamp = stamp
            if not 0 <= now - self._state.monotonic_ns <= self.feedback_timeout_s * 1e9:
                raise RuntimeError("Flexiv feedback timestamp stopped advancing")
            if self._joint_safety is not None:
                q = self._state.joints
                outside = np.abs(q - np.clip(q, self._device_lower, self._device_upper))
                if np.any(outside > self._joint_safety.tracking_limit):
                    raise RuntimeError(f"Flexiv {self.channel}: joint feedback outside tracking envelope")
            return {self.channel: self._state}

    def validate_commands(self, targets):
        self._command_target(targets, clip=self.joint_limit_policy == "clip")

    def validate_home_commands(self, targets):
        """Home must reach its configured target, never a saturated substitute."""
        self._command_target(targets, clip=False)

    def _command_target(self, targets, *, clip):
        if set(targets) != {self.channel}:
            raise ValueError("Flexiv command must name this controller's serial channel")
        q = self._vector(targets[self.channel])
        if self._robot is None:
            raise RuntimeError("Flexiv device limits are unavailable before connect")
        if clip:
            return np.clip(q, self._clip_lower, self._clip_upper)
        if np.any(q < self._lower) or np.any(q > self._upper):
            bad = np.flatnonzero((q < self._lower) | (q > self._upper))
            details = "; ".join(
                f"J{i + 1}={np.rad2deg(q[i]):.6f}deg allowed "
                f"[{np.rad2deg(self._lower[i]):.6f}, {np.rad2deg(self._upper[i]):.6f}]deg"
                for i in bad
            )
            raise ValueError(
                f"Flexiv {self.channel} target exceeds device joint limits with configured margin: {details}"
            )
        return q

    def send_commands(self, targets):
        self._send_commands(targets, control_mode=self.control_mode,
                            clip=self.joint_limit_policy == "clip", guard=True)

    def send_home_commands(self, targets):
        """Preserve position-mode Home independently of policy impedance settings."""
        self._send_commands(targets, control_mode="joint_position")

    def _send_commands(self, targets, *, control_mode, clip=False, guard=False):
        with self._lock:
            if not self.execute:
                raise PermissionError("Flexiv session is read-only")
            if not self._ready:
                raise RuntimeError("Flexiv is not connected")
            q = self._command_target(targets, clip=clip)
            measured = self.get_states()[self.channel].joints
            self._health()
            bounded = q.copy()
            now = self.clock.now_ns()
            if guard and self._joint_safety is not None:
                lower, upper = ((self._clip_lower, self._clip_upper) if clip
                                else (self._lower, self._upper))
                q = self._joint_safety.target(
                    q, measured, lower, upper, self._effective_velocity, now,
                )
                # Independent last line of defense after dynamic shaping.
                q = np.clip(q, lower, upper)
            mode = getattr(self._sdk.Mode, "NRT_" + control_mode.upper())
            if self._prepared and self._robot.mode() != self._prepared_mode:
                raise RuntimeError("Flexiv motion mode changed during execution")
            if not self._prepared or self._prepared_mode != mode:
                stiffness = None
                if control_mode == "joint_impedance":
                    nominal = self._vector(self._value(self._robot.info().K_q_nom))
                    if np.any(nominal <= 0):
                        raise ValueError("Flexiv nominal joint stiffness must be positive")
                    stiffness = (nominal * self.joint_stiffness_scale).tolist()
                    setter = self._robot.SetJointImpedance
                self._prepared = False
                self._owned = True
                self._robot.SwitchMode(mode)
                if self._robot.mode() != mode:
                    raise RuntimeError("Flexiv motion mode switch did not complete")
                if stiffness is not None:
                    damping = [self.joint_damping_ratio] * 7
                    if self._grouped:
                        setter(self._group, stiffness, damping)
                    else:
                        setter(stiffness, damping)
                    self._policy_stiffness = stiffness
                self._prepared_mode = mode
                self._prepared = True
            if self._robot.mode() != mode:
                raise RuntimeError("Flexiv motion mode changed during execution")
            args = (
                q.tolist(),
                [0.0] * 7,
                [self._effective_velocity] * 7,
                [self.max_acceleration] * 7,
            )
            if self._grouped:
                self._robot.SendJointPosition({self._group: self._sdk.NrtJointPositionCmd(*args)})
            else:
                self._robot.SendJointPosition(*args)
            self._last_submitted = q.copy()
            if self._joint_safety is not None:
                if guard:
                    self._joint_safety.commit(q, now)
                    if not np.array_equal(q, bounded):
                        self._guard_count += 1
                        if self._guard_report_ns is None or now - self._guard_report_ns >= 1_000_000_000:
                            logger.warning(
                                "joint_target_guarded channel=%s requested_deg=%s submitted_deg=%s "
                                "guarded_commands=%d", self.channel,
                                np.rad2deg(targets[self.channel]).tolist(),
                                np.rad2deg(q).tolist(), self._guard_count,
                            )
                            self._guard_report_ns = now
                else:
                    self._joint_safety.reset()
            if clip:
                self._report_joint_clip(self._vector(targets[self.channel]), bounded, submitted=q)

    def _report_joint_clip(self, requested, bounded, *, submitted):
        changed = np.flatnonzero(requested != bounded)
        if changed.size:
            self._clip_count += 1
            # Tolerance classifies the executor's requested target; it never
            # expands the command bounds or changes strict Home validation.
            excess = np.maximum(np.maximum(self._device_lower - requested,
                                           requested - self._device_upper), 0.0)
            beyond = ((requested < self._device_lower - self.joint_limit_tolerance)
                      | (requested > self._device_upper + self.joint_limit_tolerance))
            severity = ("beyond_tolerance" if np.any(beyond) else
                        "within_tolerance" if np.any(excess > 0) else "within_sdk_bounds")
            self._beyond_tolerance_count += int(np.any(beyond))
            self._max_requested_excess = max(self._max_requested_excess, float(np.max(excess)))
            now = self.clock.now_ns()
            if (not self._clip_active or severity != self._clip_severity
                    or self._clip_report_ns is None or now - self._clip_report_ns >= 1_000_000_000):
                logger.log(
                    logging.WARNING if np.any(beyond) else logging.INFO,
                    "joint_target_clipped channel=%s joints=%s requested_deg=%s "
                    "bounded_deg=%s submitted_deg=%s clipped_commands=%d "
                    "severity=%s excess_deg=%s tolerance_deg=%.6f beyond_tolerance_commands=%d",
                    self.channel, (changed + 1).tolist(),
                    np.rad2deg(requested[changed]).tolist(),
                    np.rad2deg(bounded[changed]).tolist(),
                    np.rad2deg(submitted[changed]).tolist(), self._clip_count,
                    severity, np.rad2deg(excess[changed]).tolist(),
                    np.rad2deg(self.joint_limit_tolerance), self._beyond_tolerance_count,
                )
                self._clip_report_ns = now
            self._clip_active = True
            self._clip_severity = severity
        elif self._clip_active:
            logger.info("joint_target_clip_recovered channel=%s clipped_commands=%d",
                        self.channel, self._clip_count)
            self._clip_active = False
            self._clip_severity = None

    def stop(self):
        with self._lock:
            if self._robot is None or not self.execute or not self._owned:
                return
            if self._enable_pending:
                self._wait_operational()
                self._enable_pending = False
            self._robot.Stop()  # Blocking stop and transition to IDLE, without ServoOff.
            self._prepared = self._owned = False
            self._prepared_mode = None
            if self._joint_safety is not None:
                self._joint_safety.reset()

    def enable(self, stop):
        """Explicit recovery enable; cancellation is checked between RDK calls."""
        with self._lock:
            if not self.execute or not self._ready:
                raise RuntimeError("recovery requires a connected executing controller")
            if stop.is_set():
                raise InterruptedError("Flexiv recovery cancelled")
            self._health(require_operational=False)
            self._check_remote_mode()
            if not self._robot.estop_released():
                raise RuntimeError("Flexiv E-stop is not released")
            if not self._robot.operational():
                self._owned = self._enable_pending = True
                (self._robot.ServoOn if self._grouped else self._robot.Enable)()
                self._wait_operational(stop)
                self._enable_pending = False

    def clear_errors(self):
        """Explicit fault reset in a temporary session, without enabling."""
        with self._lock:
            if not self.execute or self._robot is not None:
                raise RuntimeError("clear_errors requires a disconnected executing controller")
            previous = self.enable_on_connect
            self.enable_on_connect = False
            try:
                self.connect()
                if self._robot.fault() and not self._robot.ClearFault():
                    raise RuntimeError("Flexiv ClearFault failed")
                self._health(require_operational=False)
                self.get_states()
            finally:
                self.enable_on_connect = previous
                self.close()

    def prepare_hand_guiding(self, stiffness_scale, motion):
        """Validate NRT guiding before enabling any selected arm."""
        with self._lock:
            if not self.execute or not self._ready:
                raise RuntimeError("hand guiding requires a connected executing controller")
            if self._grouped:
                raise RuntimeError("hand guiding requires the flat flexivpy API")
            self.check_hand_guiding(require_mode=False)
            stiffness = self._vector(self._robot.info().K_q_nom) * stiffness_scale
            if np.any(stiffness < 0) or not 0 <= stiffness_scale <= 1:
                raise ValueError("invalid hand-guiding stiffness")
            self._guiding = (stiffness.tolist(), float(motion["damping_ratio"]),
                             float(np.deg2rad(motion["max_velocity_deg_s"])),
                             float(np.deg2rad(motion["max_acceleration_deg_s2"])))

    def begin_hand_guiding(self, stop):
        """Enter low-stiffness NRT impedance with the current measured position."""
        with self._lock:
            if self._guiding is None:
                raise RuntimeError("prepare_hand_guiding must precede guiding")
            self.enable(stop)
            if stop.is_set():
                raise InterruptedError("Flexiv hand guiding cancelled")
            self._owned = True
            self._prepared = False
            if self._joint_safety is not None:
                self._joint_safety.reset()
            self._robot.SwitchMode(self._sdk.Mode.NRT_JOINT_IMPEDANCE)
            if stop.is_set():
                raise InterruptedError("Flexiv hand guiding cancelled")
            stiffness, damping, _, _ = self._guiding
            self._robot.SetJointImpedance(stiffness, [damping] * 7)
            self.follow_hand_guiding(stop)

    def check_hand_guiding(self, *, require_mode=True):
        self._health(require_operational=require_mode)
        self._check_remote_mode()
        if require_mode and self._robot.mode() != self._sdk.Mode.NRT_JOINT_IMPEDANCE:
            raise RuntimeError("Flexiv hand-guiding mode changed")
        # The controller enforces its own limits in manual guiding. Additional
        # ManiMux policy margins remain exclusive to commanded trajectories.
        return self.get_states()[self.channel].joints

    def follow_hand_guiding(self, stop):
        """Refresh the impedance reference from feedback, without policy margins."""
        with self._lock:
            if not self.execute or not self._ready or self._guiding is None:
                raise RuntimeError("hand guiding has not been prepared")
            if stop.is_set():
                raise InterruptedError("Flexiv hand guiding cancelled")
            q = self.check_hand_guiding()
            if stop.is_set():
                raise InterruptedError("Flexiv hand guiding cancelled")
            _, _, velocity, acceleration = self._guiding
            self._robot.SendJointPosition(q.tolist(), [0.0] * 7,
                                         [velocity] * 7, [acceleration] * 7)

    def close(self):
        with self._lock:
            self._ready = False
            self.stop()  # Preserve the SDK handle if Stop fails, allowing cleanup retry.
            self._guiding = None
            self._robot = None  # RDK releases its connection through its destructor.
            self._state = self._stamp = None
            self._prepared = self._owned = self._enable_pending = False
            self._prepared_mode = self._policy_stiffness = None

    def runtime_metadata(self):
        return {
            "sdk": self.backend,
            "serial": self.channel,
            "execute": self.execute,
            "api": "grouped" if self._grouped else "flat",
            "timestamp_semantics": "host first observation of a new RDK timestamp",
            "max_velocity_rad_s": self.max_velocity,
            "max_acceleration_rad_s2": self.max_acceleration,
            "joint_margin_rad": self.joint_margin,
            "joint_limit_policy": self.joint_limit_policy,
            "joint_clip_margin_rad": self.joint_clip_margin,
            "joint_limit_tolerance_rad": self.joint_limit_tolerance,
            "joint_targets_beyond_tolerance": self._beyond_tolerance_count,
            "max_requested_joint_excess_rad": self._max_requested_excess,
            "joint_clipped_commands": self._clip_count,
            "joint_safety": self._joint_safety.metadata() if self._joint_safety else None,
            "joint_guarded_commands": self._guard_count,
            "effective_max_velocity_rad_s": self._effective_velocity,
            "last_sdk_submitted_rad": self._last_submitted.tolist() if self._last_submitted is not None else None,
            "joint_clip_lower_rad": self._clip_lower.tolist() if self._clip_lower is not None else None,
            "joint_clip_upper_rad": self._clip_upper.tolist() if self._clip_upper is not None else None,
            "control_mode": self.control_mode,
            "home_control_mode": "joint_position",
            "joint_stiffness_scale": self.joint_stiffness_scale,
            "joint_damping_ratio": self.joint_damping_ratio,
            "policy_joint_stiffness_nm_per_rad": self._policy_stiffness,
            "dispatch": "NRT internal interpolation; independent arms are not atomic",
            "stop": "blocking RDK Stop to IDLE; next send prepares NRT mode",
        }
