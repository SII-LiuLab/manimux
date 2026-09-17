"""One controller session assembling Tianji arms, grippers and TCP models."""

from __future__ import annotations

import importlib
import ipaddress
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from manimux.clock import Clock, SystemClock
from manimux.end_effectors.gripper import GripperBase, GripperCommand
from manimux.kinematics.base import KinematicCoordinate, ManipulatorKinematicsBase
from manimux.kinematics.robot import RobotKinematics
from manimux.robots.base import RobotBase
from manimux.types import FloatArray, RobotCommand, RobotState

_CONTROL_SESSION = threading.Lock()
_ARM = {"left": ("A", 0), "right": ("B", 1)}


@dataclass(frozen=True, slots=True)
class TianjiArmConfig:
    """One group's complete TCP model and control settings.

    Limits are (lower, upper), seven radians each; ratios are controller percent.
    An injected gripper is exclusively owned by this robot after connect.
    Its Clock must share the robot's domain. No TacCap-specific hardware branch
    exists here; use build_tianji_taccap_kinematics for a TacCap-equipped arm.
    """

    kinematics: ManipulatorKinematicsBase
    joint_limits: tuple[FloatArray, FloatArray]
    velocity_ratio: int
    acceleration_ratio: int
    gripper: GripperBase | None = None

    def __post_init__(self) -> None:
        for ratio in (self.velocity_ratio, self.acceleration_ratio):
            if isinstance(ratio, bool) or not isinstance(ratio, int) or not 1 <= ratio <= 100:
                raise ValueError("velocity/acceleration ratios must be integers in [1, 100]")
        if self.gripper is not None and not isinstance(self.gripper, GripperBase):
            raise TypeError("gripper must implement GripperBase")
        expected = tuple(KinematicCoordinate(f"joint_{i}", "rad") for i in range(1, 8))
        if self.gripper is not None:
            expected += (KinematicCoordinate("gripper", "normalized"),)
        if self.kinematics.coordinates != expected:
            raise ValueError("kinematics layout must match seven joints and optional gripper")
        lower, upper = (np.array(v, dtype=float, copy=True) for v in self.joint_limits)
        if (
            lower.shape != (7,)
            or upper.shape != (7,)
            or not np.isfinite(lower).all()
            or not np.isfinite(upper).all()
            or np.any(lower >= upper)
        ):
            raise ValueError("joint_limits must contain finite ordered seven-joint bounds")
        lower.setflags(write=False)
        upper.setflags(write=False)
        object.__setattr__(self, "joint_limits", (lower, upper))


class TianjiRobot(RobotBase):
    """One SDK session for configured left/right groups (either or both).

    Each group contains J1..J7 radians plus optional normalized gripper opening.
    kinematics uses those same groups and complete TCP models, without hardware
    access. Commands must include ALL configured groups; validation precedes any
    motion. Arm targets share one clear_set/send_cmd batch. Separate gripper links
    are dispatched afterward, not atomically with arms. Failures stop all owned
    components. No clipping, smoothing or implicit FK/IK is performed.

    connect requires configured arms disabled and advancing feedback, without
    enabling or clearing faults. First command enables position mode at measured
    joints and confirms speed settings before sending targets. stop disables owned
    arms and stops grippers, potentially releasing a grasp. Commands can re-enable.
    home is explicitly unsupported pending a configured trajectory. close retries
    incomplete cleanup and never releases a controller with unconfirmed disable.

    State time is oldest host receipt (not simultaneous acquisition); sequence
    counts returned snapshots. No background monitor is started: runtime must poll
    and stop on feedback errors. Only one native controller session per process.
    """

    def __init__(
        self,
        *,
        ip: str,
        arms: Mapping[str, TianjiArmConfig],
        clock: Clock | None = None,
        stale_timeout_s: float = 0.2,
        ready_timeout_s: float = 3.0,
    ) -> None:
        ipaddress.IPv4Address(ip)
        if not arms or set(arms) - _ARM.keys():
            raise ValueError("arms must configure left, right, or both")
        for timeout in (stale_timeout_s, ready_timeout_s):
            if not np.isfinite(timeout) or timeout <= 0:
                raise ValueError("timeouts must be finite and positive")
        configs = {name: arms[name] for name in _ARM if name in arms}
        for name, cfg in configs.items():
            if not isinstance(cfg, TianjiArmConfig):
                raise TypeError("arms values must be TianjiArmConfig")
            if cfg.kinematics.base_frame != f"tianji_{name}_base":
                raise ValueError(f"{name} kinematics has the wrong arm base frame")
        grippers = [cfg.gripper for cfg in configs.values() if cfg.gripper is not None]
        if len({id(g) for g in grippers}) != len(grippers):
            raise ValueError("each arm must have a distinct gripper instance")
        self._arms = MappingProxyType(configs)
        self._kinematics = RobotKinematics({name: cfg.kinematics for name, cfg in configs.items()})
        self._ip = ip
        self._clock = clock if clock is not None else SystemClock()
        self._stale_ns = int(stale_timeout_s * 1e9)
        self._ready_timeout = ready_timeout_s
        self._robot = self._buffer = None
        self._owns_session = self._ready = False
        self._gripper_open: set[str] = set()
        self._enabled: set[str] = set()
        self._serial: dict[str, int] = {}
        self._received: dict[str, int] = {}
        self._sequence = 0
        self._lock = threading.RLock()

    @property
    def arms(self) -> Mapping[str, TianjiArmConfig]:
        return self._arms

    @property
    def kinematics(self) -> RobotKinematics:
        return self._kinematics

    @staticmethod
    def _check(result: object, operation: str) -> None:
        if not result:
            raise RuntimeError(f"Marvin {operation} failed")

    def connect(self) -> None:
        with self._lock:
            if self._ready:
                return
            if self._owns_session or self._gripper_open:
                raise RuntimeError("cleanup incomplete; call close before reconnecting")
            if not _CONTROL_SESSION.acquire(blocking=False):
                raise RuntimeError("another TianjiRobot owns the process-wide SDK session")
            self._owns_session = True
            try:
                sdk = importlib.import_module("manimux.robots.tianji.vendor.marvin.fx_robot")
                self._robot, self._buffer = sdk.Marvin_Robot(), sdk.DCSS()
                self._check(self._robot.connect(self._ip), "connect")
                self._serial.clear()
                _, modes, _ = self._read()
                if any(mode != 0 for mode in modes.values()):
                    raise RuntimeError("configured arms must be disabled before connecting")
                initial = dict(self._serial)
                self._wait(
                    lambda modes, data: all(
                        self._serial[n] != initial[n] and modes[n] == 0 for n in self.arms
                    )
                )
                for name, cfg in self.arms.items():
                    if cfg.gripper is not None:
                        self._gripper_open.add(name)
                        cfg.gripper.connect()
                self._ready = True
                self.get_state()
            except Exception as error:
                try:
                    self.close()
                except Exception as cleanup_error:
                    raise ExceptionGroup(
                        "connect and cleanup failed", [error, cleanup_error]
                    ) from None
                raise

    def _read(self, *, allow_fault: bool = False):
        data = self._robot.subscribe(self._buffer)
        if not data:
            raise RuntimeError("Marvin feedback unavailable")
        joints, modes = {}, {}
        now = self._clock.now_ns()
        for name in self.arms:
            _, index = _ARM[name]
            output, status = data["outputs"][index], data["states"][index]
            serial = int(output["frame_serial"])
            if serial == 0:
                raise RuntimeError(f"{name}: no valid feedback frame")
            if serial != self._serial.get(name):
                self._serial[name], self._received[name] = serial, now
            if not 0 <= now - self._received[name] <= self._stale_ns:
                raise RuntimeError(f"{name}: stale arm feedback")
            modes[name] = int(status["cur_state"])
            if not allow_fault and (status["err_code"] or modes[name] == 100):
                raise RuntimeError(
                    f"{name}: controller fault {status['err_code']}, state {modes[name]}"
                )
            q = np.radians(np.asarray(output["fb_joint_pos"], dtype=float))
            if q.shape != (7,) or not np.isfinite(q).all():
                raise ValueError(f"{name}: invalid measured joints")
            joints[name] = q
        return joints, modes, data

    def _wait(self, predicate, *, allow_fault: bool = False) -> None:
        deadline = time.monotonic() + self._ready_timeout
        while True:
            _, modes, data = self._read(allow_fault=allow_fault)
            if predicate(modes, data):
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("Marvin transition/feedback timed out")
            time.sleep(0.01)

    def _require_ready(self) -> None:
        if not self._ready:
            raise RuntimeError("TianjiRobot is not connected")

    def get_state(self) -> RobotState:
        with self._lock:
            self._require_ready()
            groups, _, _ = self._read()
            timestamp = min(self._received.values())
            for name, cfg in self.arms.items():
                if cfg.gripper is not None:
                    state = cfg.gripper.get_state()
                    gripper_ns = int(state.timestamp * 1e9)
                    if not 0 <= self._clock.now_ns() - gripper_ns <= self._stale_ns:
                        raise RuntimeError(f"{name}: stale gripper feedback or clock mismatch")
                    groups[name] = np.r_[groups[name], state.opening]
                    timestamp = min(timestamp, gripper_ns)
            self._sequence += 1
            return RobotState(groups, timestamp, self._sequence)

    def _write(self, joints: Mapping[str, FloatArray], states: Mapping[str, int]) -> None:
        self._check(self._robot.clear_set(), "clear_set")
        for name in self.arms:
            arm, _ = _ARM[name]
            if name in joints:
                self._check(
                    self._robot.set_joint_cmd_pose(arm, np.degrees(joints[name]).tolist()),
                    "joint target",
                )
            if states.get(name) == 1:
                cfg = self.arms[name]
                self._check(
                    self._robot.set_vel_acc(arm, cfg.velocity_ratio, cfg.acceleration_ratio),
                    "speed ratios",
                )
            if name in states:
                self._check(self._robot.set_state(arm, states[name]), "set_state")
        self._check(self._robot.send_cmd(), "send_cmd")

    def send_command(self, command: RobotCommand) -> None:
        with self._lock:
            self._require_ready()
            if set(command.groups) != set(self.arms):
                raise ValueError("command must contain all configured groups, without extras")
            targets, grips = {}, {}
            for name, cfg in self.arms.items():
                q = np.array(command.groups[name], dtype=float, copy=True)
                if q.shape != (cfg.kinematics.num_coordinates,) or not np.isfinite(q).all():
                    raise ValueError(f"{name}: invalid command dimension/values")
                lower, upper = cfg.joint_limits
                if np.any(q[:7] < lower) or np.any(q[:7] > upper):
                    raise ValueError(f"{name}: target exceeds joint limits")
                targets[name] = q[:7]
                if cfg.gripper is not None:
                    grips[name] = GripperCommand(float(q[7]))
            try:
                self.get_state()
                measured, modes, feedback = self._read()
                pending = set(self.arms) - self._enabled
                for name, cfg in self.arms.items():
                    if modes[name] != (0 if name in pending else 1):
                        raise RuntimeError(f"{name}: unexpected control mode")
                    if name not in pending:
                        settings = feedback["inputs"][_ARM[name][1]]
                        if (
                            settings["joint_vel_ratio"] != cfg.velocity_ratio
                            or settings["joint_acc_ratio"] != cfg.acceleration_ratio
                        ):
                            raise RuntimeError(f"{name}: controller speed settings changed")
                    lower, upper = cfg.joint_limits
                    if np.any(measured[name] < lower) or np.any(measured[name] > upper):
                        raise ValueError(f"{name}: measured joints outside limits")
                if pending:
                    previous = dict(self._serial)
                    self._enabled.update(pending)  # cleanup even on partial enable failure
                    self._write({n: measured[n] for n in pending}, {n: 1 for n in pending})

                    def enabled(modes, data):
                        return all(
                            self._serial[n] != previous[n]
                            and modes[n] == 1
                            and data["inputs"][_ARM[n][1]]["joint_vel_ratio"]
                            == self.arms[n].velocity_ratio
                            and data["inputs"][_ARM[n][1]]["joint_acc_ratio"]
                            == self.arms[n].acceleration_ratio
                            for n in pending
                        )

                    self._wait(enabled)
                self._write(targets, {})
                for name, grip in grips.items():
                    self.arms[name].gripper.send_command(grip)
            except Exception as error:
                try:
                    self.stop()
                except Exception as stop_error:
                    raise ExceptionGroup("command and stop failed", [error, stop_error]) from None
                raise

    def home(self) -> None:
        raise NotImplementedError("home trajectory not configured; use the runtime motion planner")

    def stop(self) -> None:
        with self._lock:
            errors = []
            # Independent attempts ensure one arm's rejection cannot prevent a
            # disable request reaching the other arm or either gripper.
            for name in tuple(self._enabled):
                try:
                    previous = self._serial.get(name)
                    self._write({}, {name: 0})
                    self._wait(
                        lambda modes, data, name=name, previous=previous: (
                            self._serial[name] != previous and modes[name] == 0
                        ),
                        allow_fault=True,
                    )
                    self._enabled.remove(name)
                except Exception as error:
                    errors.append(error)
            for name in tuple(self._gripper_open):
                try:
                    self.arms[name].gripper.stop()
                except Exception as error:
                    errors.append(error)
            if errors:
                raise ExceptionGroup("robot stop incomplete", errors)

    def close(self) -> None:
        with self._lock:
            self._ready = False
            errors = []
            try:
                self.stop()
            except Exception as error:
                errors.append(error)
            for name in tuple(self._gripper_open):
                try:
                    self.arms[name].gripper.close()
                    self._gripper_open.remove(name)
                except Exception as error:
                    errors.append(error)
            if not self._enabled:
                try:
                    if self._robot is not None:
                        self._check(self._robot.release_robot(), "release_robot")
                        self._robot = self._buffer = None
                    if self._owns_session:
                        _CONTROL_SESSION.release()
                        self._owns_session = False
                except Exception as error:
                    errors.append(error)
            if errors:
                raise ExceptionGroup("robot cleanup incomplete", errors)
