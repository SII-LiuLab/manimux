"""Single TacCap follower position driver; SDK import is deferred to connect."""

import importlib
import logging
import math
import threading
from typing import Literal

from manimux.clock import Clock, SystemClock
from manimux.embodiments.end_effector.base import EndEffectorModel
from manimux.embodiments.end_effector.gripper_base import (
    GripperBase,
    GripperCommand,
    GripperState,
)
from manimux.embodiments.end_effector.taccap.geometry import ASSET_DIRECTORY, TacCapGeometry

logger = logging.getLogger(__name__)


class TacCapGripper(GripperBase):
    """Own one serial endpoint selected by exact firmware serial and side.

    connect opens serial with cameras disabled, validates calibration and reads
    feedback; it never enables, homes or clears faults. The first command enables
    the motor and submits the SDK's normalized impedance-position target (zero
    feedforward). kp/kd are explicit installation parameters in Nm/rad and
    Nm*s/rad. Submission has no ACK and does not imply target completion.

    get_state polls synchronously at most read_hz (<=100), caching measurements
    between polls. The timestamp is host receipt using the supplied Clock, not
    a firmware acquisition timestamp; firmware-internal cache age is unavailable
    through this SDK read. Read failures are raised, never hidden by old state.
    No application reader/control thread or SDK ControlLoop is started.

    stop disables motor torque (may release a grasp), not an emergency stop.
    A subsequent command re-enables. close disables an owned enabled motor before
    closing serial. Cleanup failure is reported and retained for retry.
    """

    @classmethod
    def load_model(
        cls, *, tcp_transform=None, base_frame="taccap_base", tcp_frame="taccap_tcp"
    ) -> EndEffectorModel:
        geometry = TacCapGeometry(
            tcp_transform=tcp_transform,
            base_frame=base_frame,
            tcp_frame=tcp_frame,
        )
        return EndEffectorModel(geometry, ASSET_DIRECTORY)

    def __init__(
        self,
        *,
        serial: str | None = None,
        side: Literal["left", "right"],
        kp: float,
        kd: float,
        clock: Clock | None = None,
        read_hz: float = 30.0,
        timeout_ms: int = 100,
    ) -> None:
        if not math.isfinite(read_hz) or not 0 < read_hz <= 100:
            raise ValueError("read_hz must be in (0, 100]")
        if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int) or timeout_ms <= 0:
            raise ValueError("timeout_ms must be a positive integer")
        self._serial, self._side = serial, side
        self._kp, self._kd = kp, kd
        self._clock = clock if clock is not None else SystemClock()
        self._period = 1 / read_hz
        self._timeout_ms = timeout_ms
        self._device = None
        self._map = None
        self._state = None
        self._enabled = False
        self._ready = False
        self._last_commanded_opening: float | None = None
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

    def _sample(self) -> GripperState:
        self._state = None
        status = self._device.motor.read_status(timeout_ms=self._timeout_ms)
        # MotorStatusBit fault/protection bits in vendor protocol/payloads.hpp.
        if status.status & 0x0FFE:
            raise RuntimeError(f"TacCap motor protection status: 0x{status.status:04x}")
        raw = float(status.actual_pos)
        # Avoid the SDK's silent [0, 1] clamp: invalid calibration/feedback must
        # not become a seemingly valid fully-open or fully-closed observation.
        direction = -1 if self._map.reverse else 1
        opening = (raw * direction - self._map.min_open_rad) / (
            self._map.max_open_rad - self._map.min_open_rad
        )
        state = GripperState(opening, self._clock.now_ns() / 1e9)
        self._state = state
        return state

    def get_state(self) -> GripperState:
        with self._lock:
            self._require_ready()
            now = self._clock.now_ns() / 1e9
            if self._state is not None and 0 <= now - self._state.timestamp < self._period:
                return self._state
            return self._sample()

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
                if not self._enabled:
                    self._enabled = True  # retain cleanup obligation even if enable raises
                    self._device.motor.enable()
                self._device.set_position(
                    opening,
                    kp_nm_per_rad=self._kp,
                    kd_nm_s_per_rad=self._kd,
                    feedforward_torque_nm=0.0,
                )
                if (
                    self._last_commanded_opening is None
                    or abs(opening - self._last_commanded_opening) >= 0.02
                ):
                    logger.info(
                        "taccap_set_position_ok serial=%s opening=%.4f enabled=%s",
                        self._serial,
                        opening,
                        self._enabled,
                    )
                    self._last_commanded_opening = opening
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
            if self._device is not None and self._enabled:
                self._device.motor.disable()
                self._enabled = False

    def close(self) -> None:
        with self._lock:
            self._ready = False
            self._state = None
            self.stop()  # on failure retain connection so disable can be retried
            if self._device is not None:
                self._device.transport.stop()
                self._device = None
            self._map = None
