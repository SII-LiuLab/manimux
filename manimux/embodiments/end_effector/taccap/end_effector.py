"""TacCap follower position and bounded-force position control."""

import importlib
import math
import threading
from dataclasses import dataclass
from typing import Literal

from manimux.clock import Clock, SystemClock
from manimux.embodiments.end_effector.base import EndEffectorModel
from manimux.embodiments.end_effector.gripper import GripperBase, GripperCommand, GripperState
from manimux.embodiments.end_effector.taccap.geometry import ASSET_DIRECTORY, TacCapGeometry
from manimux.kinematics.end_effector import load_end_effector

_PROTECTION_STATUS_MASK = 0x0FFE
_FORCE_ENDPOINT_EXCURSION = 0.05


@dataclass(frozen=True, slots=True)
class TacCapForcePositionState:
    """One read-only snapshot of TacCap's active force-position controller."""

    phase: str
    target_opening: float
    opening: float
    unclamped_opening: float
    grasp_torque_nm: float
    commanded_torque_nm: float
    measured_torque_nm: float
    motor_temperature_c: float
    holding: bool
    arrived: bool
    sequence: int
    age_ms: float
    timestamp: float
    fault_reason: str


class TacCapGripper(GripperBase):
    """Own one serial endpoint selected by exact firmware serial and side.

    connect opens serial with cameras disabled, validates calibration and reads
    feedback; it never enables, homes or clears faults. ``control_mode`` selects
    the existing direct impedance-position path or the SDK's supervised
    force-position controller. The latter keeps the same normalized position
    command and applies the configured grasp torque as a bounded budget.

    In force-position mode the first command starts the controller before enabling
    the motor, then submits the target. While it runs, feedback comes only from its
    status-stream snapshot; synchronous motor reads must not share that bus.

    get_state polls synchronously at most read_hz (<=100), caching measurements
    between polls. The timestamp is host receipt using the supplied Clock, not
    a firmware acquisition timestamp; firmware-internal cache age is unavailable
    through the synchronous SDK read. Controller snapshots preserve their frame
    identity and derive receipt time from ``age_ms``. Read failures and stale or
    invalid streamed observations are raised, never hidden by old state.

    stop disables motor torque (may release a grasp), not an emergency stop.
    A subsequent command re-enables. close disables an owned enabled motor before
    closing serial. Cleanup failure is reported and retained for retry.
    """

    @classmethod
    def load_model(
        cls, *, tcp_transform=None, base_frame="taccap_base", tcp_frame="taccap_tcp"
    ) -> EndEffectorModel:
        visual = load_end_effector(ASSET_DIRECTORY)
        geometry = TacCapGeometry(
            tcp_transform=visual.spec.tcp.matrix() if tcp_transform is None else tcp_transform,
            base_frame=base_frame,
            tcp_frame=tcp_frame,
        )
        return EndEffectorModel(geometry, visual)

    def __init__(
        self,
        *,
        serial: str | None = None,
        side: Literal["left", "right"],
        kp: float,
        kd: float,
        control_mode: Literal["impedance", "force_position"] = "impedance",
        grasp_torque_nm: float | None = None,
        close_speed_radps: float | None = None,
        clock: Clock | None = None,
        read_hz: float = 30.0,
        timeout_ms: int = 100,
    ) -> None:
        if not math.isfinite(read_hz) or not 0 < read_hz <= 100:
            raise ValueError("read_hz must be in (0, 100]")
        if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int) or timeout_ms <= 0:
            raise ValueError("timeout_ms must be a positive integer")
        if control_mode not in {"impedance", "force_position"}:
            raise ValueError("control_mode must be 'impedance' or 'force_position'")
        if control_mode == "force_position":
            if grasp_torque_nm is None or not math.isfinite(grasp_torque_nm):
                raise ValueError("force_position requires a finite grasp_torque_nm")
            if grasp_torque_nm <= 0:
                raise ValueError("grasp_torque_nm must be positive")
            if close_speed_radps is not None and (
                not math.isfinite(close_speed_radps) or close_speed_radps <= 0
            ):
                raise ValueError("close_speed_radps must be finite and positive")
        elif grasp_torque_nm is not None or close_speed_radps is not None:
            raise ValueError(
                "grasp_torque_nm and close_speed_radps require control_mode='force_position'"
            )
        self._serial, self._side = serial, side
        self._kp, self._kd = kp, kd
        self._control_mode = control_mode
        self._grasp_torque_nm = grasp_torque_nm
        self._close_speed_radps = close_speed_radps
        self._clock = clock if clock is not None else SystemClock()
        self._period = 1 / read_hz
        self._timeout_ms = timeout_ms
        self._device = None
        self._map = None
        self._force_config = None
        self._controller_factory = None
        self._controller = None
        self._controller_running = False
        self._controller_sequence = None
        self._controller_timestamp = None
        self._state = None
        self._enabled = False
        self._ready = False
        self._lock = threading.RLock()

    def connect(self) -> None:
        with self._lock:
            if self._ready:
                return
            if self._device is not None:
                raise RuntimeError("previous cleanup incomplete; call close before reconnecting")
            # Defer device binding to connect so offline robot assembly needs no serial.
            if not self._serial or self._side not in {"left", "right"}:
                raise ValueError("an explicit serial and left/right side are required")
            if (
                not math.isfinite(self._kp)
                or self._kp <= 0
                or not math.isfinite(self._kd)
                or self._kd < 0
            ):
                raise ValueError("kp must be positive and kd non-negative, both finite")
            sdk = importlib.import_module("xense.taccap")
            side = getattr(sdk.Side, self._side.capitalize())
            matches = [
                ep
                for ep in sdk.scan_grippers()
                if ep.firmware_sn == self._serial
                and ep.side == side
                and ep.role == sdk.Role.Follower
            ]
            if len(matches) != 1:
                raise ValueError("expected exactly one follower matching serial and side")
            try:
                self._device = sdk.FollowerGripper(
                    mcu_device=matches[0].mcu_device, open_cameras=False
                )
                mapping = self._device.position_map()
                if (
                    not mapping.valid
                    or not math.isfinite(mapping.min_open_rad)
                    or not math.isfinite(mapping.max_open_rad)
                    or mapping.max_open_rad - mapping.min_open_rad <= 1e-5
                ):
                    raise ValueError("TacCap calibration is invalid")
                self._map = mapping
                if self._control_mode == "force_position":
                    model = self._device.motor.get_model()
                    if not model.from_flash:
                        raise RuntimeError("TacCap motor model is not persisted in flash")
                    spec = self._device.motor.get_spec()
                    audit = self._device.audit_envelope(timeout_ms=self._timeout_ms)
                    if not audit.ok or audit.needs_write or audit.effective is None:
                        raise RuntimeError(f"TacCap safety envelope is not ready: {audit}")
                    limit = min(
                        float(spec.stall_cont_torque_nm),
                        float(audit.effective.cont_torque_nm),
                    )
                    if self._grasp_torque_nm > limit + 1e-6:
                        raise ValueError(
                            f"grasp_torque_nm={self._grasp_torque_nm:.3f} exceeds the "
                            f"effective continuous limit {limit:.3f} Nm"
                        )
                    config = sdk.ForcePositionConfig.for_spec(spec)
                    config.grasp_torque_nm = self._grasp_torque_nm
                    if self._close_speed_radps is not None:
                        config.close_speed_radps = self._close_speed_radps
                    self._force_config = config
                    # Construction has a destructor-side disable even if start()
                    # was never called. Delay it until the first authorized command
                    # so connect/close remains strictly read-only.
                    self._controller_factory = sdk.ForcePositionController
                self._sample()
                self._ready = True
            except Exception as error:
                try:
                    self.close()
                except Exception as cleanup_error:
                    raise ExceptionGroup(
                        "connect and cleanup failed", [error, cleanup_error]
                    ) from None
                raise

    def _opening_from_raw(
        self, raw: float, *, allow_endpoint_excursion: bool
    ) -> tuple[float, float]:
        if not math.isfinite(raw):
            raise ValueError("TacCap motor position is not finite")
        direction = -1 if self._map.reverse else 1
        unclamped = (raw * direction - self._map.min_open_rad) / (
            self._map.max_open_rad - self._map.min_open_rad
        )
        if not math.isfinite(unclamped):
            raise ValueError("TacCap normalized position is not finite")
        if allow_endpoint_excursion:
            if not -_FORCE_ENDPOINT_EXCURSION <= unclamped <= 1 + _FORCE_ENDPOINT_EXCURSION:
                raise ValueError(f"TacCap position is outside calibrated travel: {unclamped:.4f}")
            return min(1.0, max(0.0, unclamped)), unclamped
        if not 0 <= unclamped <= 1:
            raise ValueError(f"TacCap position is outside calibrated travel: {unclamped:.4f}")
        return unclamped, unclamped

    def _controller_snapshot(self, *, allow_fault: bool = False):
        snapshot = self._controller.snapshot()
        observation = snapshot.observation
        phase = snapshot.state.name
        if phase == "FAULT" and not allow_fault:
            raise RuntimeError(f"TacCap force-position fault: {snapshot.fault_reason}")
        if not observation.valid:
            raise RuntimeError("TacCap force-position observation is invalid")
        age_ms = float(observation.age_ms)
        if not math.isfinite(age_ms) or age_ms < 0:
            raise RuntimeError("TacCap force-position observation age is invalid")
        if age_ms > float(self._force_config.status_timeout_ms):
            raise TimeoutError(f"TacCap force-position observation is stale ({age_ms:.1f} ms)")
        if observation.status & _PROTECTION_STATUS_MASK:
            raise RuntimeError(f"TacCap motor protection status: 0x{observation.status:04x}")
        return snapshot, observation

    def _controller_sample_time(self, observation) -> float:
        sequence = int(observation.seq)
        if sequence != self._controller_sequence:
            self._controller_sequence = sequence
            self._controller_timestamp = (
                self._clock.now_ns() / 1e9 - float(observation.age_ms) / 1000.0
            )
        if self._controller_timestamp is None or not math.isfinite(self._controller_timestamp):
            raise RuntimeError("TacCap force-position timestamp is invalid")
        return self._controller_timestamp

    def _sample(self) -> GripperState:
        self._state = None
        if self._controller_running:
            _, observation = self._controller_snapshot()
            opening, _ = self._opening_from_raw(
                float(observation.raw_pos), allow_endpoint_excursion=True
            )
            self._state = GripperState(opening, self._controller_sample_time(observation))
            return self._state
        status = self._device.motor.read_status(timeout_ms=self._timeout_ms)
        # MotorStatusBit fault/protection bits in vendor protocol/payloads.hpp.
        if status.status & _PROTECTION_STATUS_MASK:
            raise RuntimeError(f"TacCap motor protection status: 0x{status.status:04x}")
        opening, _ = self._opening_from_raw(
            float(status.actual_pos), allow_endpoint_excursion=False
        )
        state = GripperState(opening, self._clock.now_ns() / 1e9)
        self._state = state
        return state

    def get_state(self) -> GripperState:
        with self._lock:
            self._require_ready()
            if self._controller_running:
                return self._sample()
            now = self._clock.now_ns() / 1e9
            if self._state is not None and 0 <= now - self._state.timestamp < self._period:
                return self._state
            return self._sample()

    def get_force_position_state(self) -> TacCapForcePositionState:
        """Return force-position telemetry without issuing any bus transaction."""

        with self._lock:
            self._require_ready()
            if self._control_mode != "force_position":
                raise RuntimeError("TacCap gripper is not configured for force_position")
            if not self._controller_running:
                raise RuntimeError("TacCap force-position controller is not running")
            snapshot, observation = self._controller_snapshot(allow_fault=True)
            opening, unclamped = self._opening_from_raw(
                float(observation.raw_pos), allow_endpoint_excursion=True
            )
            timestamp = self._controller_sample_time(observation)
            return TacCapForcePositionState(
                phase=snapshot.state.name,
                target_opening=float(snapshot.target_position),
                opening=opening,
                unclamped_opening=unclamped,
                grasp_torque_nm=float(snapshot.grasp_torque_nm),
                commanded_torque_nm=float(snapshot.commanded_torque_nm),
                measured_torque_nm=float(observation.torque),
                motor_temperature_c=float(observation.motor_temp_c),
                holding=bool(snapshot.holding),
                arrived=bool(snapshot.arrived),
                sequence=int(observation.seq),
                age_ms=float(observation.age_ms),
                timestamp=timestamp,
                fault_reason=str(snapshot.fault_reason),
            )

    def _require_ready(self) -> None:
        if not self._ready:
            raise RuntimeError("TacCap gripper is not connected")

    def send_command(self, command: GripperCommand) -> None:
        # 先读取指令字段，再进入硬件操作；开度范围由 GripperCommand 定义。
        opening = command.opening
        with self._lock:
            self._require_ready()
            try:
                self.get_state()  # reject failed/invalid feedback before commanding
                if self._control_mode == "force_position":
                    if not self._controller_running:
                        if self._controller is None:
                            self._controller = self._controller_factory(
                                self._device, self._force_config
                            )
                        # Mark cleanup responsibility before start: a failed start
                        # may still have acquired the stream or submit thread.
                        self._controller_running = True
                        self._controller.start()
                        self._enabled = True
                        self._device.motor.enable()
                    self._controller.set_target(opening)
                    return
                if not self._enabled:
                    self._enabled = True  # retain cleanup obligation even if enable raises
                    self._device.motor.enable()
                self._device.set_position(
                    opening,
                    kp_nm_per_rad=self._kp,
                    kd_nm_s_per_rad=self._kd,
                    feedforward_torque_nm=0.0,
                )
            except Exception as error:
                try:
                    self.stop()
                except Exception as stop_error:
                    raise ExceptionGroup(
                        "command and disable failed", [error, stop_error]
                    ) from None
                raise

    def stop(self) -> None:
        with self._lock:
            if self._device is None:
                return
            if self._controller is not None and self._controller_running:
                self._controller.stop()
                self._controller_running = False
                self._enabled = False
                self._controller_sequence = None
                self._controller_timestamp = None
                self._state = None
            elif self._enabled:
                self._device.motor.disable()
                self._enabled = False

    def close(self) -> None:
        with self._lock:
            self._ready = False
            self._state = None
            self.stop()  # on failure retain connection so disable can be retried
            if self._device is not None:
                self._device.transport.stop()
                self._controller = None
                self._force_config = None
                self._controller_factory = None
                self._device = None
            self._map = None
