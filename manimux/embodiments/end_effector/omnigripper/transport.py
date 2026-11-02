"""DM4310 communication over a Damiao USB2CAN serial adapter.

The adapter's CAN bitrate must be configured externally. The serial link uses
921600 baud. Frame receipt timestamps use the station monotonic clock.
"""

from __future__ import annotations

import importlib
import math
import os
import select
import struct
import threading
import time
from dataclasses import dataclass

from manimux.clock import SystemClock


class FeedbackTimeout(TimeoutError):
    """Only a missing feedback reply; write failures must never be retried."""


@dataclass(frozen=True, slots=True)
class MotorFeedback:
    position: float
    velocity: float
    torque: float
    status: int
    timestamp: float
    mos_c: int = 0
    rotor_c: int = 0


class Usb2CanTransport:
    """Own one adapter and submit individual CAN frames for one motor."""

    def __init__(self, port, *, motor_id, master_id, clock=None, timeout_s=0.3):
        if not port or not 0 <= motor_id <= 15 or not 0 <= master_id <= 0x7FF:
            raise ValueError("a port and valid CAN IDs are required")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        self.port, self.motor_id, self.master_id = port, motor_id, master_id
        self.clock = clock if clock is not None else SystemClock()
        self.timeout_s = timeout_s
        self._serial = None
        self._lock = threading.RLock()

    def connect(self):
        with self._lock:
            if self._serial is not None:
                return
            serial = importlib.import_module("serial")
            self._serial = serial.Serial(self.port, 921600, timeout=0.005,
                                         write_timeout=self.timeout_s, exclusive=True)
            try:
                self._write_frame(bytes.fromhex("55 05 00 aa 55"), "adapter initialization")
                time.sleep(0.1)
                self._serial.reset_input_buffer()
            except BaseException:
                self.close()
                raise

    def close(self):
        with self._lock:
            if self._serial is not None:
                self._serial.close()
                self._serial = None

    def _write_frame(self, frame, context):
        """Submit one frame through the nonblocking POSIX descriptor.

        pyserial 3.5 waits for a subsequent writable notification even after a
        complete write. That can report a timeout after all bytes were accepted.
        Return immediately on completion, and retry only an unwritten suffix.
        """
        from serial import SerialTimeoutException

        offset = 0
        deadline = time.monotonic() + self.timeout_s
        fd = self._serial.fileno()  # pyserial opens this CDC descriptor O_NONBLOCK.

        def timed_out():
            return SerialTimeoutException(
                f'{self.port}: {context}: Write timeout ({offset}/{len(frame)} bytes submitted)')

        while offset < len(frame):
            if time.monotonic() >= deadline:
                raise timed_out()
            try:
                written = os.write(fd, frame[offset:])
            except BlockingIOError:
                written = 0
            except InterruptedError:
                continue
            if not 0 <= written <= len(frame) - offset:
                raise OSError(f'{self.port}: {context}: invalid serial write count {written}')
            offset += written
            if offset == len(frame):
                return offset
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise timed_out()
            try:
                _, ready, _ = select.select([], [fd], [], remaining)
            except InterruptedError:
                continue
            if not ready:
                raise timed_out()

    def send(self, can_id, payload):
        """Submit a standard eight-byte CAN frame; return does not confirm delivery."""
        if not 0 <= can_id <= 0x7FF or len(payload) != 8:
            raise ValueError("expected an 11-bit CAN ID and eight payload bytes")
        with self._lock:
            if self._serial is None:
                raise RuntimeError("USB2CAN is not connected")
            frame = bytearray.fromhex(
                "55aa1e03010000000a000000000000000000080000000000000000000000"
            )
            frame[13:17] = struct.pack("<I", can_id)
            frame[21:29] = payload
            self._write_frame(frame, "CAN frame")

    def _query(self, operation, register=0):
        with self._lock:
            if self._serial is None:
                raise RuntimeError("USB2CAN is not connected")
            self._serial.reset_input_buffer()
            self.send(0x7FF, struct.pack("<HBBI", self.motor_id, operation, register, 0))
            deadline = time.monotonic() + self.timeout_s
            data = bytearray()
            while time.monotonic() < deadline:
                waiting = getattr(self._serial, "in_waiting", 0)
                data.extend(self._serial.read(max(1, min(waiting, 256))))
                while len(data) >= 16:
                    if data[0] != 0xAA or data[15] != 0x55:
                        del data[0]
                        continue
                    packet = bytes(data[:16])
                    del data[:16]
                    can_id = int.from_bytes(packet[3:7], "little")
                    if packet[2] != 8 or can_id > 0x7FF:
                        continue
                    if packet[1] in (1, 2):
                        raise OSError("USB2CAN reported CAN transmit/receive failure")
                    payload = packet[7:15]
                    if packet[1] != 0x11 or can_id != self.master_id:
                        continue
                    if operation == 0x33:
                        if int.from_bytes(payload[:2], "little") != self.motor_id:
                            continue
                        if payload[2:4] != bytes((0x33, register)):
                            continue
                    elif payload[0] & 0x0F != self.motor_id & 0x0F:
                        continue
                    return payload, self.clock.now_ns() / 1e9
            error = FeedbackTimeout if operation == 0xCC else TimeoutError
            raise error(f"no motor response on {self.port}")

    def read_register(self, register):
        if not isinstance(register, int) or not 0 <= register <= 255:
            raise ValueError("register must be in [0, 255]")
        payload, _ = self._query(0x33, register)
        return payload[4:8]

    def read_feedback(self, *, pmax, vmax, tmax):
        if not all(math.isfinite(x) and x > 0 for x in (pmax, vmax, tmax)):
            raise ValueError("feedback scaling limits must be finite and positive")
        d, timestamp = self._query(0xCC)
        q = int.from_bytes(d[1:3], "big")
        v = (d[3] << 4) | (d[4] >> 4)
        tau = ((d[4] & 0x0F) << 8) | d[5]
        return MotorFeedback((q / 65535 * 2 - 1) * pmax,
                             (v / 4095 * 2 - 1) * vmax,
                             (tau / 4095 * 2 - 1) * tmax,
                             d[0] >> 4, timestamp, d[6], d[7])
