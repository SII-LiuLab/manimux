"""Standard PiPER adapter with owned CAN receipt timestamps and strict cleanup."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from manimux.clock import Clock, SystemClock
from manimux.embodiments.arm.base import ArmController, ArmState
from manimux.types import FloatArray

if TYPE_CHECKING:
    import can

# Standard PiPER position packets and low-speed motor-status packets.
_JOINT_PACKET_IDS = frozenset({0x2A5, 0x2A6, 0x2A7})
_GRIPPER_PACKET_IDS = frozenset({0x2A8})
_POSITION_PACKET_IDS = _JOINT_PACKET_IDS | _GRIPPER_PACKET_IDS
_MOTOR_STATUS_PACKET_IDS = frozenset(range(0x261, 0x267))
_FEEDBACK_PACKET_IDS = _POSITION_PACKET_IDS | _MOTOR_STATUS_PACKET_IDS
_JOINT_FEEDBACK_FIELDS = (
    ("joint_12", ("joint_1", "joint_2")),
    ("joint_34", ("joint_3", "joint_4")),
    ("joint_56", ("joint_5", "joint_6")),
)


class _CanTransport:
    """pyAgxArm context transport; python-can failures must reach the caller.

    The pinned SDK communication wrapper suppresses some send/shutdown errors.
    This small bridge keeps its codecs and driver API while owning the bus.
    """

    def __init__(self, channel: str) -> None:
        self.channel = channel
        self.bus: can.BusABC | None = None

    def connect(self) -> None:
        import can

        self.bus = can.Bus(
            interface="socketcan",
            channel=self.channel,
            bitrate=1_000_000,
            receive_own_messages=False,
            local_loopback=False,
        )

    def is_connected(self) -> bool:
        return self.bus is not None

    def get_channel(self) -> str:
        return self.channel

    def _connected_bus(self) -> can.BusABC:
        if self.bus is None:
            raise RuntimeError("PiPER CAN transport is not connected")
        return self.bus

    def send(self, frame: can.Message) -> None:
        self._connected_bus().send(frame, timeout=0.1)

    def recv(self) -> can.Message | None:
        return self._connected_bus().recv(timeout=0.05)

    def close(self) -> None:
        if self.bus is not None:
            self.bus.shutdown()
            self.bus = None


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


class PiperController(ArmController):
    """One standard PiPER with its integrated width-controlled gripper.

    execute=False opens a receive-only session. Motors are enabled on connect
    only when execute and enable_on_connect are both explicitly True.
    Stop submits a measured hold, without reset, homing or torque release.
    """

    def __init__(
        self,
        *,
        channel: str,
        joint_limits: ArrayLike,
        gripper_closed: float,
        gripper_open: float,
        firmware: str = "default",
        execute: bool = False,
        enable_on_connect: bool = False,
        speed_percent: int = 20,
        gripper_force_n: float = 1.0,
        feedback_timeout_s: float = 0.2,
        connect_timeout_s: float = 3.0,
        clock: Clock | None = None,
    ) -> None:
        self.calibration = _JointGripperCalibration(
            channel=channel,
            joint_limits=joint_limits,
            gripper_closed=gripper_closed,
            gripper_open=gripper_open,
        )
        if firmware not in {"default", "v183", "v188", "v189"}:
            raise ValueError("unsupported standard PiPER firmware selector")
        if min(gripper_closed, gripper_open) < 0:
            raise ValueError("PiPER width calibration is in nonnegative metres")
        if not isinstance(speed_percent, int) or not 1 <= speed_percent <= 100:
            raise ValueError("speed_percent must be an integer in [1, 100]")
        if (
            not np.isfinite([gripper_force_n, feedback_timeout_s, connect_timeout_s]).all()
            or min(gripper_force_n, feedback_timeout_s, connect_timeout_s) <= 0
        ):
            raise ValueError("force and timeouts must be finite and positive")
        if enable_on_connect is True and execute is not True:
            raise ValueError("receive-only sessions cannot enable motors")
        self.channel = channel
        self.firmware = firmware
        self.clock = clock if clock is not None else SystemClock()
        self.execute = execute is True
        self.enable_on_connect = enable_on_connect is True
        self.speed_percent = speed_percent
        self.gripper_force_n = gripper_force_n
        self.feedback_timeout_s = feedback_timeout_s
        self.connect_timeout_s = connect_timeout_s
        # Vendor driver/effector types stay local to the lazily imported SDK.
        self._driver: Any = None
        self._gripper: Any = None
        self._transport: _CanTransport | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._stop_read = threading.Event()
        self._feedback_ready = threading.Event()
        self._receipts: dict[int, int] = {}
        self._status_receipts: dict[int, int] = {}
        self._state: ArmState | None = None
        self._read_error: Exception | None = None
        self._sequence = 0
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
        if self._driver is not None:
            if self._thread is not None and self._thread.is_alive() and self._read_error is None:
                return
            raise RuntimeError("cleanup required before reconnecting PiPER")
        from pyAgxArm import AgxArmFactory, create_agx_arm_config

        config = create_agx_arm_config(
            "piper",
            firmeware_version=self.firmware,
            channel=self.channel,
            auto_connect=False,
            enable_check_can=False,
            joint_limits={
                f"joint{i + 1}": row.tolist() for i, row in enumerate(self.calibration.bounds)
            },
        )
        self._driver = AgxArmFactory.create_arm(config)
        try:
            self._gripper = self._driver.init_effector("agx_gripper")
            transport = _CanTransport(self.channel)
            self._transport = transport
            transport.connect()
            self._control_owned = self.execute
            if self._home_monitor is not None:
                self._home_monitor.invalidate()
                self._home_monitor.supported_release = False
            context = self._driver.get_context()
            # Keep the pinned driver/parser contract; do not start SDK-owned threads.
            context.comm = transport
            context._comm_initialized = True
            self._driver.connect(start_read_thread=False)
            self._stop_read.clear()
            self._feedback_ready.clear()
            self._receipts.clear()
            self._status_receipts.clear()
            self._state = None
            self._read_error = None
            self._sequence = 0
            self._thread = threading.Thread(
                target=self._read_loop, args=(transport,), name=f"piper-{self.channel}"
            )
            self._thread.start()
            self._wait_for_feedback()
            if self.enable_on_connect:
                self._enable_motors()
        except Exception as error:
            try:
                self.close()
            except Exception as cleanup_error:
                raise ExceptionGroup(
                    "PiPER connection and cleanup failed", [error, cleanup_error]
                ) from None
            raise

    def _wait_for_feedback(self) -> None:
        if not self._feedback_ready.wait(self.connect_timeout_s):
            if self._read_error is not None:
                raise RuntimeError("PiPER receive failed during connection") from self._read_error
            raise TimeoutError("PiPER requires feedback from all six joints and the gripper")
        self.get_states()

    def _enable_motors(self) -> None:
        """Submit the explicitly requested enable once; do not recover faults."""
        self._driver.enable()
        deadline = time.monotonic() + self.connect_timeout_s
        while not all(self._driver.get_joints_enable_status_list()):
            self.get_states()
            if time.monotonic() >= deadline:
                raise TimeoutError("PiPER motors did not acknowledge enable")
            self._stop_read.wait(0.01)

    def _read_loop(self, transport: _CanTransport) -> None:
        try:
            while not self._stop_read.is_set():
                frame = transport.recv()
                if frame is not None:
                    self._accept_frame(frame, self.clock.now_ns())
        except Exception as error:
            with self._lock:
                self._read_error = error
            self._feedback_ready.set()

    def _accept_frame(self, frame: can.Message, received_ns: int) -> None:
        if frame.is_error_frame:
            raise RuntimeError("PiPER received a CAN error frame")
        packet_id = frame.arbitration_id
        if packet_id in _FEEDBACK_PACKET_IDS and (frame.is_extended_id or len(frame.data) != 8):
            raise RuntimeError("PiPER feedback must be an eight-byte standard CAN packet")
        with self._lock:
            self._driver.get_context()._run_parser_packet_funs(frame)
            if packet_id in _MOTOR_STATUS_PACKET_IDS:
                self._status_receipts[packet_id] = received_ns
            if packet_id not in _POSITION_PACKET_IDS:
                return
            self._receipts[packet_id] = received_ns
            if len(self._receipts) < len(_POSITION_PACKET_IDS):
                return
            self._state = self._measured_state()
            self._sequence = self._state.sequence
            self._feedback_ready.set()

    def _measured_state(self) -> ArmState:
        """Called under the receive lock after all position packets have arrived."""
        # The SDK aggregate getter timestamps only joint_56. Assemble all three
        # cached packets and use their oldest receipt, including the gripper.
        return ArmState(
            np.r_[self._measured_joints(), self._measured_opening()],
            min(self._receipts.values()),
            self._sequence + 1,
        )

    def _measured_joints(self) -> list[float]:
        joints = []
        for name, fields in _JOINT_FEEDBACK_FIELDS:
            part = getattr(self._driver._parser, name).msg
            joints.extend(getattr(part, field) for field in fields)
        return joints

    def _measured_opening(self) -> float:
        gripper = self._gripper.get_gripper_status().msg
        if gripper.mode != "width":
            raise RuntimeError("standard PiPER integration requires width-mode feedback")
        return self.calibration.opening(gripper.value)

    def _require_hold_feedback(self, packets: frozenset[int], component: str) -> None:
        """A hold needs fresh positions only for the component being stopped."""
        now_ns = self.clock.now_ns()
        if any(
            packet not in self._receipts
            or not 0 <= now_ns - self._receipts[packet] <= self.feedback_timeout_s * 1e9
            for packet in packets
        ):
            raise RuntimeError(f"PiPER {component} hold requires fresh measured feedback")

    def get_states(self) -> dict[str, ArmState]:
        with self._lock:
            if self._read_error is not None:
                raise RuntimeError("PiPER feedback receiver failed") from self._read_error
            if self._state is None:
                raise RuntimeError("PiPER has no complete measured feedback")
            age = self.clock.now_ns() - self._state.monotonic_ns
            if not 0 <= age <= self.feedback_timeout_s * 1e9:
                raise RuntimeError("PiPER feedback is stale or its clock does not match")
            return {self.channel: self._state}

    def _require_enabled_motors(self) -> None:
        if self._driver is None:
            raise RuntimeError("PiPER is not connected")
        if not all(self._driver.get_joints_enable_status_list()):
            raise RuntimeError("PiPER motors are not enabled; enable is an explicit station choice")
        now_ns = self.clock.now_ns()
        if len(self._status_receipts) != len(_MOTOR_STATUS_PACKET_IDS) or any(
            not 0 <= now_ns - stamp <= self.feedback_timeout_s * 1e9
            for stamp in self._status_receipts.values()
        ):
            raise RuntimeError("PiPER motor-enable feedback is incomplete or stale")
        for index in range(1, 7):
            status = self._driver.get_driver_states(index).msg.foc_status
            if any(
                (
                    status.voltage_too_low,
                    status.motor_overheating,
                    status.driver_overcurrent,
                    status.driver_overheating,
                    status.collision_status,
                    status.driver_error_status,
                    status.stall_status,
                )
            ):
                raise RuntimeError(f"PiPER motor {index} reports a fault; recovery is explicit")

    def _submit(self, q: FloatArray) -> None:
        """Caller owns the lock and has checked measured position freshness."""
        self._require_enabled_motors()
        self._driver.set_speed_percent(self.speed_percent)
        self._driver.move_j(q[:6].tolist())
        self._gripper.move_gripper_m(self.calibration.raw_opening(q[6]), force=self.gripper_force_n)

    def send_commands(self, targets: Mapping[str, FloatArray]) -> None:
        q = self.calibration.validate(targets)
        if not self.execute:
            raise PermissionError("PiPER session is receive-only")
        with self._lock:
            self.get_states()
            self._home_monitor.invalidate()
            self._submit(q)

    def validate_motion_ready(self) -> None:
        with self._lock:
            self.get_states()
            self._require_enabled_motors()

    def stop(self) -> None:
        """Attempt both measured holds without the normal motion readiness gate.

        Never use stale targets or an electronic emergency stop as a fallback:
        the vendor emergency stop permits a raised arm to descend under damping.
        Report each failed hold after attempting the other owned component.
        """
        if not self.execute or self._driver is None:
            return
        errors = []
        with self._lock:
            try:
                self._require_hold_feedback(_JOINT_PACKET_IDS, "arm")
                self._driver.set_speed_percent(self.speed_percent)
                self._driver.move_j(self._measured_joints())
            except Exception as error:
                errors.append(error)
            try:
                self._require_hold_feedback(_GRIPPER_PACKET_IDS, "gripper")
                self._gripper.move_gripper_m(
                    self.calibration.raw_opening(self._measured_opening()),
                    force=self.gripper_force_n,
                )
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("PiPER stop incomplete", errors)

    def close(self) -> None:
        self.await_release_permission()
        self._stop_read.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                raise RuntimeError("PiPER receiver did not stop; cleanup remains retryable")
        if self._transport is not None:
            self._transport.close()
        if self._driver is not None:
            context = self._driver.get_context()
            context.comm = None
            context._comm_initialized = False
            self._driver.disconnect()
        self._driver = None
        self._gripper = None
        self._transport = None
        self._thread = None
        self._state = None
        self._control_owned = False

    def runtime_metadata(self) -> dict[str, object]:
        return {
            "sdk": "pyAgxArm",
            "firmware": self.firmware,
            "connected": self._driver is not None,
            "channel": self.channel,
            "execute": self.execute,
            "timestamp_semantics": "oldest CAN packet host receipt",
            "gripper_closed_m": self.calibration.closed,
            "gripper_open_m": self.calibration.open,
            "dispatch": "arm frames then gripper; not atomic",
            "stop": "independent measured arm/gripper holds; failures reported",
        }
