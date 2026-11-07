"""DM4310 scalar position gripper over an independently owned USB2CAN."""

from __future__ import annotations

import math
import struct
import threading
import time
from pathlib import Path

import numpy as np

from manimux.clock import SystemClock
from manimux.embodiments.end_effector.base import EndEffectorModel
from manimux.embodiments.end_effector.gripper import GripperBase, GripperCommand, GripperState
from manimux.kinematics.base import KinematicCoordinate
from manimux.kinematics.end_effector import load_end_effector
from manimux.kinematics.tool import FixedToolGeometry

from .transport import Usb2CanTransport, FeedbackTimeout
from .force_position import ForcePosition

_ASSET_ROOT = Path(__file__).parent / "assets"


def _uint(value, lower, upper, bits):
    if not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError("MIT command outside protocol range")
    return int((value - lower) * ((1 << bits) - 1) / (upper - lower))


def _mit_frame(position, kp, kd, limits, *, velocity=0.0):
    pmax, vmax, tmax = limits
    p = _uint(position, -pmax, pmax, 16)
    v = _uint(velocity, -vmax, vmax, 12)
    k_p = _uint(kp, 0, 500, 12)
    k_d = _uint(kd, 0, 5, 12)
    tau = _uint(0, -tmax, tmax, 12)
    return bytes((p >> 8, p & 255, v >> 4, ((v & 15) << 4) | (k_p >> 8),
                  k_p & 255, k_d >> 4, ((k_d & 15) << 4) | (tau >> 8), tau & 255))


class OmniGripper(GripperBase):
    """Gripper with measured normalized motor travel, 0 closed and 1 open.

    Connect is read-only. The first command enables in MIT mode after reading
    feedback. Stop disables motor torque and can release a grasp. Serial sends
    are submissions, not proof of CAN delivery or arrival.
    """

    @classmethod
    def load_model(cls, *, visual_asset="flange_reference", tcp_transform=None,
                   base_frame="omnigripper_base", tcp_frame="tcp"):
        visual = load_end_effector(visual_asset, root=_ASSET_ROOT)
        transform = visual.spec.tcp.matrix() if tcp_transform is None else np.asarray(tcp_transform)
        geometry = FixedToolGeometry(
            transform, coordinates=(KinematicCoordinate("gripper", "normalized"),),
            base_frame=base_frame, tcp_frame=tcp_frame,
        )
        return EndEffectorModel(geometry, visual)

    def __init__(self, *, port, open_rad, closed_rad, motor_id=0, master_id=0,
                 kp, kd, max_follow_rad, max_torque_nm, max_error_rad, max_velocity_rad_s,
                 execute=False, timeout_s=0.3, clock=None, expected_sn=None,
                 can_bitrate_code=None, min_rad=None, max_rad=None,
                 control_mode="position", control_hz=30.0,
                 max_acceleration_rad_s2=6.0, grasp_torque_nm=0.3,
                 max_feedback_velocity_rad_s=1.5, max_temperature_c=65.0,
                 stale_timeout_s=0.2, feedback_retries=0):
        vals = (open_rad, closed_rad, kp, kd, max_follow_rad, max_torque_nm, max_error_rad,
                max_velocity_rad_s)
        if not all(math.isfinite(v) for v in vals):
            raise ValueError("gripper calibration and limits must be finite")
        if open_rad >= closed_rad or kp <= 0 or kp > 500 or not 0 <= kd <= 5:
            raise ValueError("invalid gripper travel or MIT gains")
        if min(max_follow_rad, max_torque_nm, max_error_rad, max_velocity_rad_s) <= 0:
            raise ValueError("gripper limits must be positive")
        if kp * max_follow_rad > max_torque_nm:
            raise ValueError("kp times max_follow_rad exceeds the torque limit")
        if control_mode not in ("position", "force_position"):
            raise ValueError("unsupported gripper control_mode")
        checks = (control_hz, max_acceleration_rad_s2, grasp_torque_nm,
                  max_feedback_velocity_rad_s, max_temperature_c, stale_timeout_s)
        if not all(math.isfinite(v) and v > 0 for v in checks):
            raise ValueError("gripper controller limits must be finite and positive")
        if grasp_torque_nm > max_torque_nm:
            raise ValueError("grasp torque exceeds measured torque trip limit")
        if control_mode == "force_position" and max_velocity_rad_s >= max_feedback_velocity_rad_s:
            raise ValueError("reference speed must be below the feedback velocity trip limit")
        if not isinstance(feedback_retries, int) or not 0 <= feedback_retries <= 1:
            raise ValueError("feedback_retries must be 0 or 1")
        for value in (expected_sn, can_bitrate_code):
            if value is not None and (not isinstance(value, int) or not 0 <= value <= 0xFFFFFFFF):
                raise ValueError("motor identity registers must be unsigned integers")
        if (min_rad is None) != (max_rad is None):
            raise ValueError("min_rad and max_rad must be provided together")
        if min_rad is not None and not (math.isfinite(min_rad) and math.isfinite(max_rad)
                                        and min_rad < open_rad < closed_rad < max_rad):
            raise ValueError("invalid calibrated motor position bounds")
        self._bounds = None if min_rad is None else (min_rad, max_rad)
        self._expected_sn, self._can_bitrate = expected_sn, can_bitrate_code
        self._mode, self._hz = control_mode, control_hz
        self._max_feedback_velocity = max_feedback_velocity_rad_s
        self._max_temperature = max_temperature_c
        self._stale_timeout, self._feedback_retries = stale_timeout_s, feedback_retries
        self._force_options = dict(kp=kp, kd=kd, speed=max_velocity_rad_s,
                                   acceleration=max_acceleration_rad_s2,
                                   hz=control_hz, grasp_torque=grasp_torque_nm)
        if control_mode == "force_position":
            ForcePosition(open_rad, **self._force_options)  # Validate encoding resolution now.
        self._force = None
        self._feedback = None
        self._identity = {}
        self._open, self._closed = open_rad, closed_rad
        self._kp, self._kd = kp, kd
        self._max_follow, self._max_torque, self._max_error = (
            max_follow_rad, max_torque_nm, max_error_rad
        )
        self._max_velocity = max_velocity_rad_s
        self._last_target = None
        self._last_command_time = None
        self._execute = execute is True
        self._clock = clock if clock is not None else SystemClock()
        self._transport = Usb2CanTransport(port, motor_id=motor_id, master_id=master_id,
                                          timeout_s=timeout_s, clock=self._clock)
        self._limits = None
        self._ready = False
        self._enabled = False
        self._lock = threading.RLock()
        self._worker = None
        self._worker_stop = threading.Event()
        self._target = None
        self._worker_error = None

    def connect(self):
        with self._lock:
            if self._ready:
                return
            if self._enabled or self._worker is not None:
                raise RuntimeError("cleanup required before reconnect")
            self._transport.connect()
            try:
                expected = {7: self._transport.master_id, 8: self._transport.motor_id, 10: 1}
                if self._expected_sn is not None:
                    expected[15] = self._expected_sn
                if self._can_bitrate is not None:
                    expected[35] = self._can_bitrate
                self._identity = {}
                for register, value in expected.items():
                    actual = struct.unpack("<I", self._transport.read_register(register))[0]
                    self._identity[register] = actual
                    if actual != value:
                        raise RuntimeError(f"motor register {register}: expected {value}, got {actual}")
                mode = self._identity[10]
                limits = tuple(struct.unpack('<f', self._transport.read_register(i))[0]
                               for i in (21, 22, 23))
                if mode != 1 or not all(math.isfinite(v) and v > 0 for v in limits):
                    raise RuntimeError("motor must report MIT mode and valid limits")
                if max(abs(self._open), abs(self._closed)) > limits[0]:
                    raise ValueError("gripper travel exceeds motor protocol limits")
                if self._bounds is not None and max(abs(v) for v in self._bounds) > limits[0]:
                    raise ValueError("calibration bounds exceed motor protocol range")
                if self._mode == "force_position" and limits != (12.5, 30.0, 10.0):
                    raise RuntimeError("force_position requires DM4310 scaling (12.5, 30, 10)")
                self._limits = limits
                feedback = self._read_feedback()
                if feedback.status not in (0, 1):
                    raise RuntimeError(f"motor fault status {feedback.status}")
                if feedback.status == 1:
                    raise RuntimeError("motor already enabled; disable it before connecting")
                self._check_feedback(feedback, expected_status=0)
                self._worker_error = None
                self._ready = True
            except Exception as error:
                try:
                    self._transport.close()
                except Exception as cleanup_error:
                    raise ExceptionGroup(
                        "gripper connect and cleanup failed", [error, cleanup_error]
                    ) from None
                self._limits = None
                raise

    def _read_feedback(self):
        if self._limits is None:
            raise RuntimeError("gripper not connected")
        p, v, t = self._limits
        for attempt in range(self._feedback_retries + 1):
            try:
                self._feedback = self._transport.read_feedback(pmax=p, vmax=v, tmax=t)
                return self._feedback
            except FeedbackTimeout:
                if attempt == self._feedback_retries:
                    raise

    def _check_feedback(self, f, *, expected_status):
        values = (f.position, f.velocity, f.torque, f.timestamp, f.mos_c, f.rotor_c)
        if not all(math.isfinite(v) for v in values):
            raise RuntimeError("non-finite gripper feedback")
        if f.status != expected_status:
            raise RuntimeError(f"unexpected motor status {f.status}, expected {expected_status}")
        age = self._clock.now_ns() / 1e9 - f.timestamp
        if not 0 <= age <= self._stale_timeout:
            raise RuntimeError("stale gripper feedback")
        if self._bounds is not None and not self._bounds[0] <= f.position <= self._bounds[1]:
            raise RuntimeError("gripper position outside calibrated bounds")
        if abs(f.torque) > self._max_torque:
            raise RuntimeError("gripper measured torque exceeds limit")
        if abs(f.velocity) > self._max_feedback_velocity:
            raise RuntimeError("gripper measured velocity exceeds limit")
        if max(f.mos_c, f.rotor_c) > self._max_temperature:
            raise RuntimeError("gripper motor temperature exceeds limit")

    @property
    def diagnostics(self):
        """Last real feedback and identity; this accessor never refreshes timestamps."""
        from dataclasses import asdict
        with self._lock:
            return {"registers": dict(self._identity), "control_mode": self._mode,
                    "feedback": None if self._feedback is None else asdict(self._feedback)}

    def get_state(self):
        with self._lock:
            if not self._ready:
                raise RuntimeError("gripper not connected")
            if self._worker_error is not None:
                raise RuntimeError("gripper controller failed") from self._worker_error
            f = self._feedback if self._worker is not None else self._read_feedback()
            self._check_feedback(f, expected_status=int(self._enabled))
            opening = min(1.0, max(0.0, (self._closed - f.position) / (self._closed - self._open)))
            return GripperState(opening, f.timestamp)

    def send_command(self, command: GripperCommand):
        try:
            with self._lock:
                if self._worker_error is not None:
                    raise RuntimeError("gripper controller failed") from self._worker_error
                if not self._execute:
                    raise PermissionError("gripper is read-only")
                if not self._ready:
                    raise RuntimeError("gripper not connected")
                if self._worker is None:
                    self._send_command(command)
                self._target = command
                if self._worker is None:
                    self._worker_stop.clear()
                    self._worker = threading.Thread(target=self._run, daemon=True)
                    self._worker.start()
        except Exception as error:
            try:
                self.stop()
            except Exception as cleanup_error:
                raise ExceptionGroup(
                    "gripper command and disable failed", [error, cleanup_error]
                ) from None
            raise

    def _run(self):
        try:
            deadline = time.monotonic() + 1 / self._hz
            while not self._worker_stop.wait(max(0.0, deadline - time.monotonic())):
                with self._lock:
                    if self._target is not None:
                        self._send_command(self._target)
                deadline += 1 / self._hz
                if deadline < time.monotonic():
                    deadline = time.monotonic() + 1 / self._hz
        except Exception as error:
            with self._lock:
                self._worker_error = error
            try:
                self.stop()
            except Exception as cleanup_error:
                with self._lock:
                    self._worker_error = ExceptionGroup(
                        "gripper controller and disable failed", [error, cleanup_error]
                    )

    def _send_command(self, command: GripperCommand):
        with self._lock:
            if not self._execute:
                raise PermissionError("gripper is read-only")
            if not self._ready or self._limits is None:
                raise RuntimeError("gripper not connected")
            if self._enabled:
                # A delayed worker must not conceal a stale hold by querying a new sample.
                self._check_feedback(self._feedback, expected_status=1)
            f = self._read_feedback()
            self._check_feedback(f, expected_status=int(self._enabled))
            if not self._enabled:
                if f.status != 0:
                    raise RuntimeError("motor enabled outside this session")
                # Retain the ownership flag if a send/feedback fails so cleanup retries disable.
                self._enabled = True
                self._transport.send(self._transport.motor_id, bytes([255] * 7 + [252]))
                enabled = self._read_feedback()
                if enabled.status != 1:
                    raise RuntimeError("motor enable not confirmed by feedback")
                self._check_feedback(enabled, expected_status=1)
                f = enabled
                if self._mode == "force_position":
                    self._force = ForcePosition(f.position, **self._force_options)
                else:
                    self._transport.send(self._transport.motor_id,
                                         _mit_frame(f.position, self._kp, self._kd, self._limits))
                self._last_target = f.position
                self._last_command_time = time.monotonic()
            if f.status != 1:
                raise RuntimeError("motor is no longer enabled")
            target = self._closed + command.opening * (self._open - self._closed)
            if self._mode == "force_position":
                if abs(self._last_target - f.position) > self._max_error:
                    raise RuntimeError("gripper target error exceeds limit")
                now = time.monotonic()
                dt = max(1e-6, now - self._last_command_time)
                command = self._force.step(target, f.position, f.velocity, dt)
                self._transport.send(self._transport.motor_id,
                                     _mit_frame(command.position, command.kp, command.kd,
                                                self._limits, velocity=command.velocity))
                self._last_target = command.position
                self._last_command_time = now
                return
            target = min(f.position + self._max_follow, max(f.position - self._max_follow, target))
            elapsed = max(0.0, time.monotonic() - self._last_command_time)
            step = self._max_velocity * elapsed
            target = min(self._last_target + step, max(self._last_target - step, target))
            if abs(target - f.position) > min(self._max_error, self._max_follow):
                raise RuntimeError("gripper target error exceeds limit")
            self._transport.send(self._transport.motor_id,
                                 _mit_frame(target, self._kp, self._kd, self._limits))
            self._last_target = target
            self._last_command_time = time.monotonic()

    def stop(self):
        self._worker_stop.set()
        worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=2.0)
            if worker.is_alive():
                raise RuntimeError("gripper control worker did not stop")
        with self._lock:
            self._target = None
            if not self._enabled:
                if worker is not threading.current_thread():
                    self._worker = None
                return
            self._transport.send(self._transport.motor_id, bytes([255] * 7 + [253]))
            f = self._read_feedback()
            if f.status != 0:
                raise RuntimeError("motor disable not confirmed by feedback")
            self._enabled = False
            self._force = None
            self._last_target = None
            self._last_command_time = None
            if worker is not threading.current_thread():
                self._worker = None

    def close(self):
        self.stop()
        with self._lock:
            self._ready = False
            self._transport.close()
            self._limits = None
            self._worker_error = None
