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
from manimux.embodiments.arm._calibration import JointGripperCalibration
from manimux.embodiments.arm.base import ArmController, ArmState
from manimux.types import FloatArray

from .session import JointGripperTarget, SDKSample, SessionOperation, run_session


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
        self.calibration = JointGripperCalibration(
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

    def connect(self) -> None:
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

    def validate_commands(self, targets: Mapping[str, FloatArray]) -> None:
        self.calibration.validate(targets)

    def send_commands(self, targets: Mapping[str, FloatArray]) -> None:
        q = self.calibration.validate(targets)
        if not self.execute:
            raise PermissionError("X5 commands require execute=True")
        with self._lock:
            self.get_states()
            grip = self.command_closed + q[6] * (self.command_open - self.command_closed)
            self._rpc("send", (q[:6].tolist(), grip))

    def stop(self) -> None:
        with self._lock:
            if self._process is not None:
                self._rpc("stop")

    def close(self) -> None:
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
