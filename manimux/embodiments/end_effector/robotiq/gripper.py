"""Independent Robotiq 2F Modbus RTU component, using measured register travel."""

import threading
import time
from pathlib import Path

import numpy as np

from manimux.clock import SystemClock
from manimux.embodiments.end_effector.base import EndEffectorModel
from manimux.embodiments.end_effector.gripper import GripperBase, GripperState
from manimux.kinematics.base import KinematicCoordinate
from manimux.kinematics.end_effector import load_end_effector
from manimux.kinematics.tool import FixedToolGeometry


class RobotiqGripper(GripperBase):
    """Already-activated 2F gripper; connect never activates or calibrates it.

    Zero/one opening maps to calibrated closed/open register positions, not
    pad distance. Stop clears rGTO while retaining rACT; next send resumes.
    Serial reads/writes are acknowledged synchronously; physical travel is not.
    """

    def __init__(
        self,
        *,
        port,
        raw_closed,
        raw_open,
        clock=None,
        execute=False,
        baudrate=115200,
        device_id=9,
        speed=0.2,
        force=0.2,
        timeout_s=0.1,
        connect_timeout_s=0.5,
    ):
        if not isinstance(port, str) or not port:
            raise ValueError("Robotiq requires a serial port")
        if any(not isinstance(v, int) or not 0 <= v <= 255 for v in (raw_closed, raw_open)):
            raise ValueError("Robotiq endpoints must be integer register positions in [0,255]")
        if raw_closed <= raw_open:
            raise ValueError("Robotiq closed register must exceed open register")
        if not np.isfinite([speed, force, timeout_s, connect_timeout_s]).all() or not (
            0 < speed <= 1 and 0 <= force <= 1 and min(timeout_s, connect_timeout_s) > 0
        ):
            raise ValueError("invalid Robotiq speed, force or timeout")
        if not isinstance(device_id, int) or not 1 <= device_id <= 247:
            raise ValueError("Modbus device_id must be in [1,247]")
        self.port, self.raw_closed, self.raw_open = port, raw_closed, raw_open
        self.clock = clock or SystemClock()
        self.execute = execute is True
        self.baudrate, self.device_id = baudrate, device_id
        self.speed, self.force, self.timeout_s = speed, force, timeout_s
        self.connect_timeout_s = connect_timeout_s
        self._client = None
        self._target = None
        self._commanded = False
        self._lock = threading.RLock()

    @classmethod
    def load_model(
        cls, *, tcp_transform, visual_model="frames",
        base_frame="robotiq_base", tcp_frame="robotiq_tcp",
    ):
        # Model selection changes only the nominal display, never device commands or TCP.
        if visual_model not in {"frames", "2f_85"}:
            raise ValueError("Robotiq visual_model must be 'frames' or '2f_85'")
        directory = Path(__file__).parent / "assets"
        visual = load_end_effector(
            directory if visual_model == "frames" else directory / visual_model
        )
        geometry = FixedToolGeometry(
            tcp_transform,
            coordinates=(KinematicCoordinate("gripper", "normalized"),),
            base_frame=base_frame,
            tcp_frame=tcp_frame,
        )
        return EndEffectorModel(geometry, visual)

    def connect(self):
        with self._lock:
            if self._client is not None:
                return
            from pymodbus.client import ModbusSerialClient

            self._client = ModbusSerialClient(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=8,
                parity="N",
                stopbits=1,
                timeout=self.timeout_s,
                retries=0,
            )
            try:
                if not self._client.connect():
                    raise ConnectionError("Robotiq serial connection failed")
                self._read_status(communication_deadline=time.monotonic() + self.connect_timeout_s)
            except BaseException:
                self._client.close()
                self._client = None
                raise

    def _read_status(self, *, communication_deadline=None):
        if self._client is None:
            raise RuntimeError("Robotiq is not connected")
        while True:
            response = self._client.read_holding_registers(
                address=0x07D0, count=3, device_id=self.device_id
            )
            if response.isError() or len(response.registers) != 3:
                raise OSError(f"Robotiq status read failed: {response}")
            first, second, third = response.registers
            fault = second >> 8
            # On reconnect, the first reply may still report the expired
            # communication watchdog. Read again until the device clears it;
            # never acknowledge it as healthy or write reset/activation bits.
            if (
                fault == 0x09
                and communication_deadline is not None
                and time.monotonic() < communication_deadline
            ):
                time.sleep(min(0.02, max(0, communication_deadline - time.monotonic())))
                continue
            break
        status = first >> 8
        if fault:
            raise RuntimeError(f"Robotiq fault 0x{fault:02x}; recovery is explicit")
        if self.execute and (status & 1 != 1 or (status >> 4) & 3 != 3):
            raise RuntimeError("Robotiq is not activated; activate explicitly before control")
        # gOBJ is meaningful only with gGTO and completed activation. Keep the
        # device's echoed target separate from actual travel; an old command's
        # contact must not complete a new close request.
        active = status & 1 == 1 and (status >> 4) & 3 == 3 and bool(status & 8)
        echo = second & 255
        target = None
        if active and self.raw_open <= echo <= self.raw_closed:
            target = (self.raw_closed - echo) / (self.raw_closed - self.raw_open)
        contact = {1: "opening", 2: "closing"}.get((status >> 6) & 3) if active else None
        return (third >> 8) & 255, target, contact

    def get_state(self):
        with self._lock:
            position, target, contact = self._read_status()
            if position < self.raw_open - 5 or position > self.raw_closed + 5:
                raise ValueError("Robotiq measured position exceeds calibrated endpoints")
            opening = (self.raw_closed - position) / (self.raw_closed - self.raw_open)
            return GripperState(
                float(np.clip(opening, 0, 1)), self.clock.now_ns() / 1e9,
                target_opening=target, contact=contact,
            )

    def _write(self, action, target):
        if self._client is None:
            raise RuntimeError("Robotiq is not connected")
        self._commanded = True  # A transport error can follow physical submission.
        response = self._client.write_registers(
            address=0x03E8,
            values=[action << 8, target, round(255 * self.speed) << 8 | round(255 * self.force)],
            device_id=self.device_id,
        )
        if response.isError():
            raise OSError(f"Robotiq command acknowledgement failed: {response}")

    def send_command(self, command):
        if not self.execute:
            raise PermissionError("Robotiq session is read-only")
        with self._lock:
            target = round(self.raw_closed + command.opening * (self.raw_open - self.raw_closed))
            if target == self._target:
                return  # Runtime get_state continues to monitor measured feedback/faults.
            self._read_status()
            self._write(0x09, target)
            self._target = target

    def stop(self):
        with self._lock:
            if self.execute and self._client is not None and self._commanded:
                self._write(0x01, self._target if self._target is not None else self.raw_open)
                self._target = None
                self._commanded = False

    def close(self):
        with self._lock:
            self.stop()
            if self._client is not None:
                self._client.close()
                self._client = None
