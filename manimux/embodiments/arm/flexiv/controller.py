"""Owned RDK session for one seven-axis Rizon arm, using NRT joint targets."""

from __future__ import annotations

import importlib
import sys
import threading
import time
from pathlib import Path

import numpy as np

from manimux.clock import SystemClock
from manimux.embodiments.arm.base import ArmController, ArmState


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
        joint_margin=0.03490658503988659,
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
            self._state = self._stamp = None
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
                self._lower = self._vector(self._value(info.q_min)) + self.joint_margin
                self._upper = self._vector(self._value(info.q_max)) - self.joint_margin
                if np.any(self._lower >= self._upper):
                    raise ValueError("invalid Flexiv device limits or joint margin")
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
        if require_operational and not self._robot.operational():
            raise RuntimeError("Flexiv is not operational; enable must be explicit")

    def _wait_operational(self):
        deadline = time.monotonic() + self.enable_timeout_s
        while not self._robot.operational():
            self._health(require_operational=False)
            if time.monotonic() >= deadline:
                raise TimeoutError("Flexiv enable did not complete; stop is not confirmed")
            time.sleep(0.02)

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
            return {self.channel: self._state}

    def validate_commands(self, targets):
        if set(targets) != {self.channel}:
            raise ValueError("Flexiv command must name this controller's serial channel")
        q = self._vector(targets[self.channel])
        if self._robot is None:
            raise RuntimeError("Flexiv device limits are unavailable before connect")
        if np.any(q < self._lower) or np.any(q > self._upper):
            raise ValueError("Flexiv target exceeds device joint limits with configured margin")

    def send_commands(self, targets):
        with self._lock:
            if not self.execute:
                raise PermissionError("Flexiv session is read-only")
            if not self._ready:
                raise RuntimeError("Flexiv is not connected")
            self.validate_commands(targets)
            self.get_states()
            self._health()
            mode = self._sdk.Mode.NRT_JOINT_POSITION
            if not self._prepared:
                self._owned = True
                self._robot.SwitchMode(mode)
                self._prepared = True
            if self._robot.mode() != mode:
                raise RuntimeError("Flexiv motion mode changed during execution")
            args = (
                self._vector(targets[self.channel]).tolist(),
                [0.0] * 7,
                [self.max_velocity] * 7,
                [self.max_acceleration] * 7,
            )
            if self._grouped:
                self._robot.SendJointPosition({self._group: self._sdk.NrtJointPositionCmd(*args)})
            else:
                self._robot.SendJointPosition(*args)

    def stop(self):
        with self._lock:
            if self._robot is None or not self.execute or not self._owned:
                return
            if self._enable_pending:
                self._wait_operational()
                self._enable_pending = False
            self._robot.Stop()  # Blocking stop and transition to IDLE, without ServoOff.
            self._prepared = self._owned = False

    def close(self):
        with self._lock:
            self._ready = False
            self.stop()  # Preserve the SDK handle if Stop fails, allowing cleanup retry.
            self._robot = None  # RDK releases its connection through its destructor.
            self._state = self._stamp = None
            self._prepared = self._owned = self._enable_pending = False

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
            "dispatch": "NRT internal interpolation; independent arms are not atomic",
            "stop": "blocking RDK Stop to IDLE; next send prepares NRT mode",
        }
