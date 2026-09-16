"""Read-only, per-motor SocketCAN feedback capture for YAM diagnostics.

No controller polling, command sends, or changes to the running i2rt threads.
Decode with the installed i2rt parser and snapshot its calibrated joint mapping.
"""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

# Linux asm-generic/socket.h, OLD timespec ABI on this 64-bit SocketCAN platform.
_TIMESTAMPNS = 35
_RXQ_OVFL = 40


@dataclass
class NativeJointSource:
    stream: str
    channel: str
    motor_ids: list[int]
    feedback_ids: list[int]
    motor_types: list[str]
    offsets: np.ndarray
    directions: np.ndarray
    absolute_positions: np.ndarray
    position_periods: np.ndarray
    parser: Any

    @classmethod
    def from_robot(cls, native, stream: str, joints: int = 6):
        from i2rt.motor_drivers.utils import MotorType

        chain = native.motor_chain
        if not chain.running:
            raise RuntimeError(f"{stream}: motor feedback is not running")
        if len(chain.motor_list) < joints:
            raise RuntimeError(f"{stream}: expected {joints} arm joints")
        motors = chain.motor_list[:joints]
        with chain.state_lock:
            offsets = np.array(chain.motor_offset[:joints], dtype=float, copy=True)
            directions = np.array(chain.motor_direction[:joints], dtype=float, copy=True)
            absolute = np.array(chain.absolute_positions[:joints], dtype=float, copy=True)
        periods = []
        for _, motor_type in motors:
            limits = MotorType.get_motor_constants(motor_type)
            periods.append(limits.POSITION_MAX - limits.POSITION_MIN)
        return cls(
            stream, chain.channel, [m[0] for m in motors],
            [chain.motor_interface.receive_mode.get_receive_id(m[0]) for m in motors],
            [m[1] for m in motors], offsets, directions, absolute, np.array(periods),
            chain.motor_interface.parse_recv_message,
        )

    def metadata(self) -> dict:
        return {
            "stream": self.stream, "channel": self.channel,
            "joint_names": [f"joint{i + 1}" for i in range(len(self.motor_ids))],
            "motor_ids": self.motor_ids, "feedback_ids": self.feedback_ids,
            "motor_types": self.motor_types, "offsets_rad": self.offsets.tolist(),
            "directions": self.directions.tolist(),
            "initial_absolute_positions_rad": self.absolute_positions.tolist(),
            "position_units": "radian", "velocity_units": "radian/second",
            "scope": "measured arm joints; excludes gripper and commanded action",
        }

    def open(self):
        return NativeJointReader(self)


class NativeJointReader:
    def __init__(self, source: NativeJointSource):
        if struct.calcsize("l") != 8:
            raise RuntimeError("native YAM capture requires a 64-bit Linux SocketCAN host")
        self.source = source
        self.dropped_frames = 0
        self._absolute = source.absolute_positions.copy()
        self._indices = {identifier: i for i, identifier in enumerate(source.feedback_ids)}
        self._sequences = [0] * len(source.motor_ids)
        self._socket = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        try:
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
            self._socket.setsockopt(socket.SOL_SOCKET, _TIMESTAMPNS, 1)
            self._socket.setsockopt(socket.SOL_SOCKET, _RXQ_OVFL, 1)
            self._socket.setsockopt(
                socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER,
                b"".join(struct.pack("=II", identifier, 0x7FF)
                         for identifier in source.feedback_ids),
            )
            self._socket.bind((source.channel,))
            self._socket.settimeout(0.1)
        except BaseException:
            self._socket.close()
            raise

    def decode(self, frame: bytes, timestamp_ns: int, flags: int) -> dict | None:
        import can

        if flags & socket.MSG_DONTROUTE:  # outgoing local echo is not measured feedback
            return None
        identifier, length, payload = struct.unpack("=IB3x8s", frame)
        if identifier not in self._indices or length != 8:
            return None
        index = self._indices[identifier]
        feedback = self.source.parser(
            can.Message(arbitration_id=identifier, data=payload, is_extended_id=False),
            self.source.motor_types[index], ignore_error=True,
        )
        period = self.source.position_periods[index]
        delta = (feedback.position - self._absolute[index] + period / 2) % period - period / 2
        self._absolute[index] += delta
        direction = self.source.directions[index]
        self._sequences[index] += 1
        return {
            "joint_index": index, "sequence": self._sequences[index],
            "timestamp_ns": timestamp_ns,
            "position_rad": float(
                (self._absolute[index] - self.source.offsets[index]) * direction
            ),
            "velocity_rad_s": float(feedback.velocity * direction),
            "effort_nm": float(feedback.torque * direction),
            "motor_error": str(feedback.error_code),
        }

    def read(self) -> dict | None:
        try:
            frame, ancillary, flags, _ = self._socket.recvmsg(
                16, socket.CMSG_SPACE(16) + socket.CMSG_SPACE(4)
            )
        except TimeoutError:
            return None
        if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
            raise RuntimeError("native CAN frame or timestamp was truncated")
        timestamp_ns = None
        for level, kind, data in ancillary:
            if level == socket.SOL_SOCKET and kind == _TIMESTAMPNS:
                seconds, nanoseconds = struct.unpack("@ll", data[:16])
                timestamp_ns = seconds * 1_000_000_000 + nanoseconds
            elif level == socket.SOL_SOCKET and kind == _RXQ_OVFL:
                self.dropped_frames = struct.unpack("=I", data[:4])[0]
        if timestamp_ns is None:
            raise RuntimeError("SocketCAN did not provide a kernel receive timestamp")
        return self.decode(frame, timestamp_ns, flags)

    def close(self):
        self._socket.close()
