"""UMI follower grippers on arms A and B (TacCap SDK, one USB serial link each).

Two modes per arm:

* read-only: the serial link is opened and the calibrated aperture is polled at
  most every ``read_period_s``; the motor stays unpowered. Status polling faster
  than ~100 Hz stalls the firmware's own refresh, hence the throttle.
* control: the motor is enabled and the SDK's C++ ``ControlLoop`` streams
  targets in phase with the motor status frames. Targets are clamped to within
  ``grip_margin`` of the measured aperture, which bounds grip force.

Apertures are normalized: 0 closed, 1 open.
"""

from __future__ import annotations

import time
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import Any

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
    read_period_s: float = 0.05
    ready_timeout_s: float = 3.0


class TianjiGrippers:
    def __init__(
        self,
        taccap: ModuleType,
        serials: Mapping[str, str | None],
        *,
        control_arms: Collection[str] = (),
        settings: GripperSettings | None = None,
    ) -> None:
        unknown = set(control_arms) - set(ARMS)
        if unknown:
            raise ValueError(f"unknown gripper arms: {sorted(unknown)}")
        self._taccap = taccap
        self._serials = {arm: serials.get(arm) for arm in ARMS}
        self._control_arms = frozenset(control_arms)
        self._settings = settings or GripperSettings()
        self._devices: dict[str, Any] = {}
        self._loops: dict[str, Any] = {}
        self._enabled: set[str] = set()
        self._cached: dict[str, tuple[float, float]] = {}
        self.fault: str | None = None

    def start(self) -> None:
        endpoints = self._resolve(list(self._taccap.scan_grippers()))
        try:
            for arm in ARMS:
                device = self._taccap.FollowerGripper(mcu_device=endpoints[arm].mcu_device)
                self._devices[arm] = device
                config = device.get_gripper_config()
                if (
                    not config.flags & CALIBRATED
                    or abs(config.max_open_rad - config.min_open_rad) < 1e-5
                ):
                    raise RuntimeError(
                        f"arm {arm} gripper (SN {endpoints[arm].firmware_sn}) is not calibrated "
                        f"(flags=0x{config.flags:04x}); its normalized aperture is meaningless"
                    )
                if arm in self._control_arms:
                    device.motor.clear_fault()
                    device.motor.enable()
                    self._enabled.add(arm)
                    loop = self._taccap.ControlLoop(
                        device,
                        hz=int(self._settings.hz),
                        kp=float(self._settings.kp),
                        kd=float(self._settings.kd),
                    )
                    loop.start()
                    self._loops[arm] = loop
            deadline = time.monotonic() + self._settings.ready_timeout_s
            while any(not loop.observation().valid for loop in self._loops.values()):
                if time.monotonic() > deadline:
                    missing = [
                        arm for arm, loop in self._loops.items() if not loop.observation().valid
                    ]
                    raise RuntimeError(f"gripper motor status did not arrive for arms {missing}")
                time.sleep(0.01)
        except BaseException:
            self.close()
            raise

    def _resolve(self, found: list[Any]) -> dict[str, Any]:
        if not found:
            raise RuntimeError(
                "no TacCap gripper found: check the USB cables (/dev/ttyACM*), that this user "
                "is in the dialout group, and that the grippers are powered"
            )
        listing = [(ep.firmware_sn, str(ep.side), str(ep.role)) for ep in found]
        endpoints = {}
        for arm in ARMS:
            serial = self._serials[arm]
            if serial:
                matches = [ep for ep in found if ep.firmware_sn == serial]
            else:
                side = getattr(self._taccap.Side, SIDE_OF_ARM[arm])
                role = self._taccap.Role.Follower
                matches = [ep for ep in found if ep.side == side and ep.role == role]
            if len(matches) != 1:
                wanted = f"SN {serial}" if serial else f"the {SIDE_OF_ARM[arm]} follower"
                raise RuntimeError(
                    f"arm {arm} expects exactly one gripper matching {wanted}, found "
                    f"{len(matches)}; connected: {listing}"
                )
            endpoints[arm] = matches[0]
        return endpoints

    def apertures(self) -> dict[str, float]:
        now = time.monotonic()
        values = {}
        for arm in ARMS:
            loop = self._loops.get(arm)
            if loop is not None:
                observation = loop.observation()
                if not observation.valid:
                    raise RuntimeError(f"arm {arm} gripper observation is invalid")
                value = float(observation.position)
            else:
                cached = self._cached.get(arm)
                if cached is not None and now - cached[0] < self._settings.read_period_s:
                    value = cached[1]
                else:
                    value = float(self._require_device(arm).position())
                    self._cached[arm] = (now, value)
            if not 0.0 <= value <= 1.0:
                raise RuntimeError(f"arm {arm} gripper aperture {value} is outside [0, 1]")
            values[arm] = value
        return values

    def _require_device(self, arm: str) -> Any:
        device = self._devices.get(arm)
        if device is None:
            raise RuntimeError(f"arm {arm} gripper is not started")
        return device

    def set_target(self, arm: str, aperture: float) -> float:
        loop = self._loops.get(arm)
        if loop is None:
            raise RuntimeError(f"arm {arm} gripper is not under control")
        target = min(1.0, max(0.0, float(aperture)))
        observation = loop.observation()
        if observation.valid:
            margin = self._settings.grip_margin
            target = max(observation.position - margin, min(observation.position + margin, target))
        loop.set_target(target)
        return target

    def hold(self) -> None:
        for loop in self._loops.values():
            observation = loop.observation()
            if observation.valid:
                loop.set_target(float(observation.position))

    def check(self) -> str | None:
        """Latch and return the first protection fault of a controlled gripper."""

        if self.fault is not None:
            return self.fault
        for arm, loop in self._loops.items():
            observation = loop.observation()
            if not observation.valid:
                self.fault = f"arm {arm} gripper observation is invalid"
            elif observation.age_ms > self._settings.stale_ms:
                self.fault = (
                    f"arm {arm} gripper observation is {observation.age_ms:.0f} ms old "
                    f"(limit {self._settings.stale_ms:.0f} ms)"
                )
            elif abs(observation.torque) > self._settings.max_torque_nm:
                self.fault = (
                    f"arm {arm} gripper torque {observation.torque:.3f} N*m exceeds "
                    f"{self._settings.max_torque_nm:.2f}"
                )
            if self.fault is not None:
                break
        return self.fault

    def close(self) -> None:
        errors: list[BaseException] = []
        for loop in list(self._loops.values()):
            try:
                loop.stop()
            except BaseException as exc:
                errors.append(exc)
        self._loops.clear()
        for arm in list(self._enabled):
            try:
                self._devices[arm].motor.disable()
            except BaseException as exc:
                errors.append(exc)
        self._enabled.clear()
        for device in self._devices.values():
            try:
                device.transport.stop()
            except BaseException as exc:
                errors.append(exc)
        self._devices.clear()
        self._cached.clear()
        if errors:
            raise RuntimeError("TacCap gripper shutdown did not complete cleanly") from errors[0]
