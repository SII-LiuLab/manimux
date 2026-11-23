"""Official X5 SDK controller; construction never opens CAN or starts a process."""

from __future__ import annotations

import multiprocessing
import threading
from collections.abc import Mapping, Sequence
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import ArrayLike

from manimux.clock import Clock, SystemClock
from manimux.embodiments.arm.base import ArmController, ArmState
from manimux.types import FloatArray

from .session import JointGripperTarget, SDKSample, SessionOperation, run_session


class _JointGripperCalibration:
    """Validate device commands and convert travel using measured endpoints.

    The hardware gripper units belong to each controller. Reversed endpoints
    are supported; out-of-range commands and measurements fail without clipping.
    Nominal URDF finger travel is independent of this station calibration.
    """

    def __init__(
        self,
        *,
        channel: str,
        joint_limits: ArrayLike,
        gripper_closed: float,
        gripper_open: float,
    ) -> None:
        if not isinstance(channel, str) or not channel.strip() or channel.startswith("REPLACE"):
            raise ValueError("channel must name the station's actual CAN interface")
        bounds = np.asarray(joint_limits, dtype=float)
        if (
            bounds.shape != (6, 2)
            or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] >= bounds[:, 1])
        ):
            raise ValueError("joint_limits must contain six finite lower/upper pairs")
        endpoints = np.asarray([gripper_closed, gripper_open], dtype=float)
        if not np.isfinite(endpoints).all() or endpoints[0] == endpoints[1]:
            raise ValueError("distinct finite closed/open gripper calibration is required")
        self.channel = channel
        self.bounds = bounds.copy()
        self.closed = float(gripper_closed)
        self.open = float(gripper_open)

    def validate(self, targets: Mapping[str, ArrayLike]) -> FloatArray:
        if set(targets) != {self.channel}:
            raise ValueError("command must select exactly this controller's CAN channel")
        q = np.asarray(targets[self.channel], dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError("command must contain six finite radians and one opening")
        outside = (q[:6] < self.bounds[:, 0]) | (q[:6] > self.bounds[:, 1])
        if np.any(outside):
            index = int(np.flatnonzero(outside)[0])
            raise ValueError(
                f"{self.channel}: joint {index + 1} target {q[index]:.9g} rad is outside "
                f"configured device bounds {self.bounds[index].tolist()} rad"
            )
        if not 0.0 <= q[6] <= 1.0:
            raise ValueError(
                f"{self.channel}: gripper target {q[6]:.9g} must be in [0, 1] normalized travel"
            )
        return q.copy()

    def opening(self, raw: float) -> float:
        value = (raw - self.closed) / (self.open - self.closed)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("measured gripper is outside its configured calibration")
        return float(value)

    def raw_opening(self, opening: float) -> float:
        return self.closed + opening * (self.open - self.closed)


class _JointHomeMonitor:
    """Measure joint arrival; the controller owns freshness and fault checks."""

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


class ArxX5Controller(ArmController):
    """Isolate a binding without close(); PROTECT stops owned position control.

    The vendor getter provides no device identity or receipt time. SDK polling
    must be selected explicitly and cannot prove hardware feedback freshness.
    Passive/read-only connection is unsupported by this pinned native SDK.
    """

    def __init__(
        self,
        *,
        channel: str,
        joint_limits: ArrayLike,
        gripper_closed: float,
        gripper_open: float,
        gripper_command_closed: float,
        gripper_command_open: float,
        execute: bool = False,
        feedback_policy: str | None = None,
        control_parameters: Sequence[float] = (35000.0, 6000.0, 10.0),
        poll_interval_s: float = 0.01,
        feedback_timeout_s: float = 0.2,
        rpc_timeout_s: float = 3.0,
        connect_timeout_s: float = 10.0,
        clock: Clock | None = None,
    ) -> None:
        self.calibration = _JointGripperCalibration(
            channel=channel,
            joint_limits=joint_limits,
            gripper_closed=gripper_closed,
            gripper_open=gripper_open,
        )
        commands = np.asarray([gripper_command_closed, gripper_command_open], dtype=float)
        if (
            not np.isfinite(commands).all()
            or np.any(commands < 0)
            or np.any(commands > 5)
            or commands[0] == commands[1]
        ):
            raise ValueError("X5 (2023) requires distinct set_catch endpoints in [0, 5]")
        parameters = np.asarray(control_parameters, dtype=float)
        times = np.asarray([poll_interval_s, feedback_timeout_s, rpc_timeout_s, connect_timeout_s])
        if parameters.shape != (3,) or not np.isfinite(parameters).all() or np.any(parameters <= 0):
            raise ValueError("control_parameters must contain three finite positive vendor values")
        if (
            not np.isfinite(times).all()
            or np.any(times <= 0)
            or poll_interval_s >= feedback_timeout_s
        ):
            raise ValueError(
                "positive timeouts and polling faster than the feedback timeout are required"
            )
        if feedback_policy not in {None, "sdk_poll"}:
            raise ValueError("official X5 SDK supports only explicit sdk_poll feedback")
        self.channel = channel
        self.execute = execute is True
        self.feedback_policy = feedback_policy
        self.clock = clock if clock is not None else SystemClock()
        self.command_closed, self.command_open = commands.tolist()
        self.control_parameters = parameters.tolist()
        self.poll_interval_s = poll_interval_s
        self.feedback_timeout_s = feedback_timeout_s
        self.rpc_timeout_s = rpc_timeout_s
        self.connect_timeout_s = connect_timeout_s
        self._process: BaseProcess | None = None
        self._connection: Connection | None = None
        # A timeout/startup failure requires cleanup; late replies cannot be reused.
        self._failed = False
        self._lock = threading.RLock()
        self._home_monitor = None
        self._control_owned = False

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
        self._home_monitor = _JointHomeMonitor(q, **arrival_limits)

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
                decision = require_supported_release(f"{self.channel}: {error}")
                if decision is ReleaseDecision.SUPPORTED_RELEASE:
                    self.confirm_supported_release()
                    return

    def connect(self) -> None:
        self.require_home_configuration()
        with self._lock:
            if self._process is not None:
                if self._process.is_alive() and not self._failed:
                    return
                raise RuntimeError("cleanup required before reconnecting X5")
            if not self.execute:
                raise NotImplementedError("official X5 SDK does not provide a read-only connection")
            if self.feedback_policy != "sdk_poll":
                raise NotImplementedError(
                    "X5 device freshness is unavailable; explicitly select sdk_poll"
                )
            context = multiprocessing.get_context("spawn")
            connection, child = context.Pipe()
            self._connection = connection
            self._process = context.Process(
                target=run_session,
                args=(
                    child,
                    self.channel,
                    Path(__file__).parent / "model.urdf",
                    self.control_parameters,
                    self.poll_interval_s,
                ),
                name=f"arx-x5-{self.channel}",
            )
            try:
                self._process.start()
                self._control_owned = True
                self._home_monitor.invalidate()
                self._home_monitor.supported_release = False
                child.close()
                self._receive(connection, self.connect_timeout_s)
                self.get_states()
            except Exception as error:
                child.close()
                self._failed = True
                try:
                    self.close()
                except Exception as cleanup_error:
                    raise ExceptionGroup(
                        "X5 connection and cleanup failed", [error, cleanup_error]
                    ) from None
                raise

    def _receive(self, connection: Connection, timeout: float) -> SDKSample | None:
        if not connection.poll(timeout):
            self._failed = True
            raise TimeoutError("X5 SDK process did not respond")
        reply = connection.recv()
        if not reply["ok"]:
            raise RuntimeError(reply["error"])
        return reply["value"]

    def _rpc(
        self, operation: SessionOperation, payload: JointGripperTarget | None = None
    ) -> SDKSample | None:
        if (
            self._process is None
            or not self._process.is_alive()
            or self._connection is None
            or self._failed
        ):
            raise RuntimeError("X5 SDK process is not ready")
        self._connection.send((operation, payload))
        return self._receive(self._connection, self.rpc_timeout_s)

    def get_states(self) -> dict[str, ArmState]:
        with self._lock:
            joints, poll_completed_ns, sequence = cast(SDKSample, self._rpc("state"))
            raw = np.asarray(joints, dtype=float)
            if raw.shape != (7,) or not np.isfinite(raw).all():
                raise RuntimeError("invalid X5 SDK feedback")
            q = np.r_[raw[:6], self.calibration.opening(raw[6])]
            if not 0 <= self.clock.now_ns() - poll_completed_ns <= self.feedback_timeout_s * 1e9:
                raise RuntimeError("X5 SDK polling is stale or its clock does not match")
            return {self.channel: ArmState(q, poll_completed_ns, sequence)}

    def send_commands(self, targets: Mapping[str, FloatArray]) -> None:
        q = self.calibration.validate(targets)
        if not self.execute:
            raise PermissionError("X5 commands require execute=True")
        with self._lock:
            self.get_states()
            self._home_monitor.invalidate()
            grip = self.command_closed + q[6] * (self.command_open - self.command_closed)
            self._rpc("send", (q[:6].tolist(), grip))

    def validate_motion_ready(self) -> None:
        # sdk_poll confirms the owned process responds; the vendor API exposes
        # no motor sample identity, so this is not a device-freshness guarantee.
        self.get_states()

    def stop(self) -> None:
        with self._lock:
            if self._process is not None:
                self._rpc("stop")

    def close(self) -> None:
        self.await_release_permission()
        with self._lock:
            if self._process is None:
                return
            process = self._process
            error: Exception | None = None
            if process.pid is not None and not process.is_alive() and not self._failed:
                error = RuntimeError(
                    "X5 SDK process exited unexpectedly; protective stop is unconfirmed"
                )
            if process.is_alive() and not self._failed:
                try:
                    self._rpc("close")
                except Exception as close_error:
                    error = close_error
            if process.pid is not None:
                process.join(timeout=1.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=1.0)
                    if error is None:
                        error = RuntimeError(
                            "X5 process required termination; protective stop is unconfirmed"
                        )
                if process.is_alive():
                    raise RuntimeError("X5 process still alive; cleanup remains retryable")
                process.close()
            if self._connection is not None:
                self._connection.close()
            self._process = None
            self._connection = None
            self._failed = False
            self._control_owned = False
            if error is not None:
                raise error

    def runtime_metadata(self) -> dict[str, object]:
        return {
            "sdk": "ARXroboticsX/X5",
            "model": "X5 (2023)",
            "channel": self.channel,
            "connected": self._process is not None and self._process.is_alive(),
            "execute": self.execute,
            "feedback_policy": self.feedback_policy,
            "timestamp_semantics": "SDK getter completion, not device receipt",
            "device_sample_identity_available": False,
            "read_only_connection": False,
            "gripper_encoder_endpoints": [self.calibration.closed, self.calibration.open],
            "gripper_command_endpoints": [self.command_closed, self.command_open],
            "dispatch": "targets prepared then POSITION_CONTROL; not atomic",
            "stop": "vendor PROTECT",
            "close": "PROTECT then release owned process resources",
        }
