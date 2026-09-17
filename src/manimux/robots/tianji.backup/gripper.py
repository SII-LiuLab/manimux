"""UMI follower grippers on arms A and B (TacCap SDK, one USB serial link each).

Two modes, both safe to query at any control rate:

* read-only: the serial link is opened and the calibration checked; a background
  thread reads ``position()`` at ``read_hz`` and :meth:`apertures` returns the
  latest value. The motor stays unpowered. Polling the firmware faster than
  ~100 Hz stalls its own status refresh, so the rate is fixed here rather than
  following the caller.
* control: the motor is enabled and the SDK's ``ControlLoop`` streams targets
  in its own thread (tianji-control ``UmiGrippers``). Targets are clamped to
  within ``grip_margin`` of the measured aperture, which bounds grip force.

Endpoints are resolved as in CalibWrist's ``TacCapCameraPair``: exactly one
follower per side, whose firmware SN must match the configured one. Apertures
are normalized: 0 closed, 1 open.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np

log = logging.getLogger("manimux.robots.tianji")

ARMS = ("A", "B")
SIDE_OF_ARM = {"A": "Left", "B": "Right"}
CALIBRATED = 0x0001


@dataclass(frozen=True, slots=True)
class GripperSettings:
    hz: int = 100
    kp: float = 8.0
    kd: float = 0.3
    grip_margin: float = 0.036
    max_torque_nm: float = 1.0
    stale_ms: float = 200.0
    read_hz: float = 30.0
    ready_timeout_s: float = 3.0


class TianjiGrippers:
    def __init__(
        self,
        taccap: ModuleType,
        serials: Mapping[str, str | None],
        *,
        control: bool,
        settings: GripperSettings | None = None,
    ) -> None:
        self._taccap = taccap
        self._serials = {arm: serials.get(arm) for arm in ARMS}
        self.control = control
        self._settings = settings or GripperSettings()
        self._readers: dict[str, Any] = {}
        self._controlled: dict[str, Any] = {}
        self._loops: dict[str, Any] = {}
        self._faults: dict[str, str] = {}
        self._lock = threading.Lock()
        self._latest: tuple[float, dict[str, float]] | None = None
        self._reader_error: BaseException | None = None
        self._reader_stop = threading.Event()
        self._reader_thread: threading.Thread | None = None

    def start(self) -> None:
        """Start or resume control, preserving any latched protection fault."""

        if self.control:
            fault = self.check()
            if fault:
                raise RuntimeError(f"gripper protection: {fault}")
            if all(arm in self._loops for arm in ARMS):
                return
            if self._loops:
                raise RuntimeError("gripper control is only partially started")
        endpoints = self._resolve(list(self._taccap.scan_grippers()))
        if self.control:
            self._start_control(endpoints)
        else:
            self._start_read_only(endpoints)

    def _resolve(self, found: list[Any]) -> dict[str, Any]:
        endpoints = {}
        for arm in ARMS:
            side = getattr(self._taccap.Side, SIDE_OF_ARM[arm])
            matches = [
                ep for ep in found if ep.side == side and ep.role == self._taccap.Role.Follower
            ]
            serial = self._serials[arm]
            if len(matches) != 1 or (serial is not None and matches[0].firmware_sn != serial):
                raise ValueError(
                    f"ambiguous/mismatched {SIDE_OF_ARM[arm]} follower mapping for arm {arm} "
                    f"(expected SN {serial}); connected: "
                    f"{[(ep.firmware_sn, str(ep.side), str(ep.role)) for ep in found]}"
                )
            endpoints[arm] = matches[0]
        return endpoints

    # ---------- read-only ----------

    def _start_read_only(self, endpoints: Mapping[str, Any]) -> None:
        try:
            for arm in ARMS:
                gripper = self._taccap.FollowerGripper(endpoints[arm].mcu_device)
                self._readers[arm] = gripper
                config = gripper.get_gripper_config()
                if (
                    not config.flags & CALIBRATED
                    or abs(config.max_open_rad - config.min_open_rad) < 1e-5
                ):
                    raise ValueError("gripper aperture calibration is invalid")
            self._sample_readers()  # a state is available as soon as start() returns
            self._reader_stop.clear()
            self._reader_thread = threading.Thread(
                target=self._read_loop, name="tianji-gripper-reader", daemon=True
            )
            self._reader_thread.start()
        except BaseException:
            self.close()
            raise

    def _sample_readers(self) -> None:
        values = {arm: float(gripper.position()) for arm, gripper in self._readers.items()}
        with self._lock:
            self._latest = (time.monotonic(), values)

    def _read_loop(self) -> None:
        period = 1.0 / self._settings.read_hz
        while not self._reader_stop.wait(period):
            try:
                self._sample_readers()
            except BaseException as exc:  # surfaced by apertures() on the caller's thread
                with self._lock:
                    self._reader_error = exc
                return

    # ---------- control ----------

    def _start_control(self, endpoints: Mapping[str, Any]) -> None:
        try:
            for arm in ARMS:
                gripper = self._taccap.FollowerGripper(mcu_device=endpoints[arm].mcu_device)
                self._controlled[arm] = gripper
                config = gripper.get_gripper_config()
                if not config.flags & CALIBRATED:
                    raise RuntimeError(
                        f"arm {arm} gripper is not calibrated (GripperConfig.flags="
                        f"0x{config.flags:04x}); normalized position is meaningless"
                    )
                gripper.motor.clear_fault()
                gripper.motor.enable()
                loop = self._taccap.ControlLoop(
                    gripper,
                    hz=int(self._settings.hz),
                    kp=float(self._settings.kp),
                    kd=float(self._settings.kd),
                )
                loop.start()
                self._loops[arm] = loop
            deadline = time.monotonic() + self._settings.ready_timeout_s
            while time.monotonic() < deadline:
                if all(self._loops[arm].observation().valid for arm in ARMS):
                    return
                time.sleep(0.01)
            missing = [arm for arm in ARMS if not self._loops[arm].observation().valid]
            raise RuntimeError(f"first gripper motor-status frame timed out: {missing}")
        except BaseException:
            self.stop_control()  # never leave a half-open set energized
            raise

    def apertures(self) -> dict[str, float]:
        """Latest normalized apertures; never touches the serial link."""

        if self._loops:
            values = []
            for arm in ARMS:
                observation = self._loops[arm].observation()
                if not observation.valid:
                    raise RuntimeError(f"arm {arm} gripper observation is invalid")
                if observation.age_ms > self._settings.stale_ms:
                    raise RuntimeError(f"arm {arm} gripper observation is stale")
                values.append(float(observation.position))
            result = dict(zip(ARMS, values, strict=True))
        else:
            with self._lock:
                error, latest = self._reader_error, self._latest
            if error is not None:
                raise RuntimeError("gripper aperture reader stopped") from error
            if latest is None:
                raise RuntimeError("grippers are not started")
            age_ms = (time.monotonic() - latest[0]) * 1e3
            if age_ms > self._settings.stale_ms:
                raise RuntimeError(f"gripper aperture is {age_ms:.0f} ms old")
            result = dict(latest[1])
        array = np.asarray(list(result.values()), dtype=np.float64)
        if not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 1):
            raise ValueError(f"invalid normalized aperture {array}")
        return result

    def set_target(self, arm: str, aperture: float) -> float:
        loop = self._loops.get(arm)
        if loop is None:
            raise RuntimeError(f"arm {arm} gripper is not under control")
        target = max(0.0, min(1.0, float(aperture)))
        observation = loop.observation()
        if observation.valid:
            margin = self._settings.grip_margin
            target = max(observation.position - margin, min(observation.position + margin, target))
        loop.set_target(target)
        return target

    def check(self) -> str | None:
        """Poll protections of controlled grippers; a tripped gripper stays tripped."""

        for arm in ARMS:
            if arm in self._faults:
                return self._faults[arm]
            loop = self._loops.get(arm)
            if loop is None:
                continue
            observation = loop.observation()
            if not observation.valid:
                self._faults[arm] = f"{arm} observation invalid (motor not enabled or link down)"
            elif observation.age_ms > self._settings.stale_ms:
                self._faults[arm] = (
                    f"{arm} observation {observation.age_ms:.0f} ms old "
                    f"(limit {self._settings.stale_ms:.0f}); link may be down"
                )
            elif abs(observation.torque) > self._settings.max_torque_nm:
                self._faults[arm] = (
                    f"{arm} torque {observation.torque:.3f} N*m over the "
                    f"{self._settings.max_torque_nm:.2f} limit"
                )
            if arm in self._faults:
                return self._faults[arm]
        return None

    def stop_control(self) -> None:
        """``UmiGrippers.stop``: stop each loop and de-energize, never raising."""

        for arm in ARMS:
            loop = self._loops.pop(arm, None)
            gripper = self._controlled.pop(arm, None)
            try:
                if loop is not None:
                    loop.stop()
            except Exception:  # noqa: BLE001
                log.exception("stopping arm %s gripper loop failed", arm)
            finally:
                if gripper is not None:
                    try:
                        gripper.motor.disable()
                    except Exception:  # noqa: BLE001
                        log.exception("disabling arm %s gripper motor failed", arm)

    def close(self) -> None:
        self.stop_control()
        self._reader_stop.set()
        thread, self._reader_thread = self._reader_thread, None
        if thread is not None:
            thread.join(timeout=1.0)
        for gripper in self._readers.values():
            gripper.transport.stop()
        self._readers.clear()
        self._latest = None
