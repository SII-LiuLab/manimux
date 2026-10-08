"""Tianji arm components and one shared official Marvin controller session."""

from __future__ import annotations

import importlib
import ipaddress
import threading
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from manimux.clock import Clock, SystemClock
from manimux.embodiments.arm.base import ArmBase, ArmController, ArmModel, ArmState
from manimux.embodiments.arm.tianji.kinematics import TianjiSDKKinematics
from manimux.kinematics.base import FlangeKinematicsBase, FloatArray, KinematicCoordinate

_CONTROL_SESSION = threading.Lock()
_ARM = {"left": ("A", 0), "right": ("B", 1)}
_TORQUE_MODE = 3
_DRAG_HZ = 250.0
_DRAG_TRACK_RATE_DEG_S = 15.0


def resolve_rate_contract(
    profile: dict, controller: dict, gripper_indices: dict[str, int] | None
) -> dict:
    """Expand one Tianji controller capability into runtime shaping and guard rates."""
    values = deepcopy(profile)
    contract = values.pop("rate_contract")
    margin = float(contract["command_margin"])
    if not 0 < margin <= 1:
        raise ValueError("Tianji command_margin must be in (0, 1]")
    rated = float(controller["rated_joint_velocity_rad_s"])
    ratio = controller["velocity_ratio"]
    if (
        not np.isfinite(rated)
        or rated <= 0
        or isinstance(ratio, bool)
        or not isinstance(ratio, int)
        or not 1 <= ratio <= 100
    ):
        raise ValueError("Tianji rated velocity and controller ratio are invalid")
    controller_velocity = rated * ratio / 100
    gripper = deepcopy(contract["gripper"])
    command_safety = values.setdefault("command_safety", {})
    if "motion_limits" in values or "max_velocity" in command_safety:
        raise ValueError("Tianji rate contract conflicts with authored runtime rates")
    groups = values["robot"]["group_dims"]
    if gripper_indices is None or set(gripper_indices) != set(groups):
        raise ValueError("Tianji rate contract requires one gripper index per robot group")
    command_safety["max_velocity"] = {}
    for name, dimension in groups.items():
        rates = [controller_velocity] * dimension
        rates[gripper_indices[name]] = gripper["max_velocity"] * (1.0 / margin)
        command_safety["max_velocity"][name] = rates
    values["motion_limits"] = {
        "arm": {
            "mode": contract["mode"],
            "max_velocity": controller_velocity * margin,
            "max_step_dt_s": contract["max_step_dt_s"],
            "max_acceleration": contract["max_acceleration"],
        },
        "gripper": gripper,
    }
    return values


@dataclass(frozen=True, slots=True)
class TianjiArmSettings:
    """Seven joint bounds in radians and official controller speed percentages.

    drag_tool is the mounted tool's Marvin load, {kine: XYZABC mm/deg (6),
    dynamic: mass, centre of mass and inertia (10)}; drag() requires it.
    """

    joint_limits: tuple[FloatArray, FloatArray]
    velocity_ratio: int
    acceleration_ratio: int
    drag_tool: Mapping[str, tuple[float, ...]] | None = None

    def __post_init__(self) -> None:
        for ratio in (self.velocity_ratio, self.acceleration_ratio):
            if isinstance(ratio, bool) or not isinstance(ratio, int) or not 1 <= ratio <= 100:
                raise ValueError("velocity/acceleration ratios must be integers in [1, 100]")
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
        if self.drag_tool is not None:
            if set(self.drag_tool) != {"kine", "dynamic"}:
                raise ValueError("drag_tool must contain exactly kine and dynamic")
            tool = {key: tuple(float(v) for v in values) for key, values in self.drag_tool.items()}
            if (
                len(tool["kine"]) != 6
                or len(tool["dynamic"]) != 10
                or not np.isfinite(tool["kine"] + tool["dynamic"]).all()
            ):
                raise ValueError("drag_tool must contain six finite kine and ten dynamic values")
            object.__setattr__(self, "drag_tool", MappingProxyType(tool))


class TianjiController(ArmController):
    """One Marvin session for left (A/0), right (B/1), or both.

    Connect requires disabled arms and advancing feedback. First command enables
    position mode at measured joints and confirms speed settings before targets.
    Each target batch uses one clear_set/send_cmd. Stop disables owned arms;
    close retains the connection if disabling cannot be confirmed. Imports and
    construction do not load the control SDK. This class never handles grippers.
    """

    def __init__(
        self,
        *,
        ip: str | None = None,
        settings: Mapping[str, TianjiArmSettings],
        clock: Clock | None = None,
        stale_timeout_s: float = 0.2,
        ready_timeout_s: float = 3.0,
        max_tracking_error_deg: float | None = None,
    ) -> None:
        if not settings or set(settings) - _ARM.keys():
            raise ValueError("settings must configure left, right, or both")
        for timeout in (stale_timeout_s, ready_timeout_s):
            if not np.isfinite(timeout) or timeout <= 0:
                raise ValueError("timeouts must be finite and positive")
        self.settings = MappingProxyType({n: settings[n] for n in _ARM if n in settings})
        self._ip = ip
        self._clock = clock if clock is not None else SystemClock()
        self._stale_ns = int(stale_timeout_s * 1e9)
        self._ready_timeout = ready_timeout_s
        self._max_tracking = (
            None if max_tracking_error_deg is None else np.radians(max_tracking_error_deg)
        )
        self._robot = self._buffer = None
        self._owns_session = self._ready = False
        self._enabled: set[str] = set()
        self._serial: dict[str, int] = {}
        self._received: dict[str, int] = {}
        self._lock = threading.RLock()

    @property
    def ip(self) -> str | None:
        """Controller address from the station binding; None while unbound."""
        return self._ip

    @staticmethod
    def _check(result: object, operation: str) -> None:
        if not result:
            raise RuntimeError(f"Marvin {operation} failed")

    def connect(self) -> None:
        with self._lock:
            if self._ready:
                return
            if self._owns_session:
                raise RuntimeError("cleanup incomplete; call close before reconnecting")
            # Hardware binding is needed only when opening the controller.
            ipaddress.IPv4Address(self._ip)
            if not _CONTROL_SESSION.acquire(blocking=False):
                raise RuntimeError("another TianjiController owns the process-wide SDK session")
            self._owns_session = True
            try:
                sdk = importlib.import_module("manimux.embodiments.arm.tianji.sdk.marvin.fx_robot")
                self._robot, self._buffer = sdk.Marvin_Robot(), sdk.DCSS()
                self._check(self._robot.connect(self._ip), "connect")
                self._serial.clear()
                _, modes, _ = self._read()
                if any(mode != 0 for mode in modes.values()):
                    raise RuntimeError("configured arms must be disabled before connecting")
                initial = dict(self._serial)
                self._wait(
                    lambda modes, data: all(
                        self._serial[n] != initial[n] and modes[n] == 0 for n in self.settings
                    )
                )
                self._ready = True
                self.get_states()
            except Exception as error:
                try:
                    self.close()
                except Exception as cleanup_error:
                    raise ExceptionGroup(
                        "connect and cleanup failed", [error, cleanup_error]
                    ) from None
                raise

    def clear_errors(self) -> None:
        """Clear configured arms' controller faults; call while disconnected.

        connect() rejects faulted arms, so recovery clears them over a short
        session first. Marvin can return false while accepting an asynchronous
        clear; fresh feedback, not that return value, is the confirmation. An
        engaged physical E-stop keeps the fault and fails after four attempts.
        """
        with self._lock:
            if self._owns_session:
                raise RuntimeError("clear_errors requires a disconnected controller")
            ipaddress.IPv4Address(self._ip)
            if not _CONTROL_SESSION.acquire(blocking=False):
                raise RuntimeError("another TianjiController owns the process-wide SDK session")
            try:
                sdk = importlib.import_module("manimux.embodiments.arm.tianji.sdk.marvin.fx_robot")
                robot, buffer = sdk.Marvin_Robot(), sdk.DCSS()
                self._check(robot.connect(self._ip), "connect")
                try:
                    for attempt in range(4):
                        data = robot.subscribe(buffer)
                        if not data:
                            raise RuntimeError("Marvin feedback unavailable")
                        faults = {}
                        for name in self.settings:
                            status = data["states"][_ARM[name][1]]
                            if int(status["err_code"]) or int(status["cur_state"]) == 100:
                                faults[name] = (int(status["err_code"]), int(status["cur_state"]))
                        if not faults:
                            return
                        if attempt == 3:
                            details = ", ".join(
                                f"{name}: fault {error}, state {state}"
                                for name, (error, state) in faults.items()
                            )
                            raise RuntimeError(
                                "controller did not confirm errors cleared; release the "
                                f"physical E-stop and retry ({details})"
                            )
                        for name in faults:
                            robot.clear_error(_ARM[name][0])
                        time.sleep(0.2)
                finally:
                    self._check(robot.release_robot(), "release_robot")
            finally:
                _CONTROL_SESSION.release()

    def _read(self, *, allow_fault: bool = False):
        data = self._robot.subscribe(self._buffer)
        if not data:
            raise RuntimeError("Marvin feedback unavailable")
        joints, modes = {}, {}
        now = self._clock.now_ns()
        for name in self.settings:
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
            raise RuntimeError("TianjiController is not connected")

    def _write(self, joints: Mapping[str, FloatArray], states: Mapping[str, int]) -> None:
        self._check(self._robot.clear_set(), "clear_set")
        for name in self.settings:
            arm, _ = _ARM[name]
            if name in joints:
                self._check(
                    self._robot.set_joint_cmd_pose(arm, np.degrees(joints[name]).tolist()),
                    "joint target",
                )
            if states.get(name) == 1:
                cfg = self.settings[name]
                self._check(
                    self._robot.set_vel_acc(arm, cfg.velocity_ratio, cfg.acceleration_ratio),
                    "speed ratios",
                )
            if name in states:
                self._check(self._robot.set_state(arm, states[name]), "set_state")
        self._check(self._robot.send_cmd(), "send_cmd")

    def get_states(self) -> Mapping[str, ArmState]:
        with self._lock:
            self._require_ready()
            groups, _, _ = self._read()
            return {
                name: ArmState(joints, self._received[name], self._serial[name])
                for name, joints in groups.items()
            }

    def validate_commands(self, targets: Mapping[str, FloatArray]) -> None:
        if not targets or set(targets) - self.settings.keys():
            raise ValueError("targets must name a non-empty subset of configured arms")
        for name, values in targets.items():
            q = np.asarray(values, dtype=float)
            if q.shape != (7,) or not np.isfinite(q).all():
                raise ValueError(f"{name}: expected seven finite arm targets")
            lower, upper = self.settings[name].joint_limits
            if np.any(q < lower) or np.any(q > upper):
                raise ValueError(f"{name}: target exceeds joint limits")

    def send_commands(self, targets: Mapping[str, FloatArray]) -> None:
        with self._lock:
            self._require_ready()
            targets = {name: np.array(q, dtype=float, copy=True) for name, q in targets.items()}
            self.validate_commands(targets)
            try:
                self.get_states()
                measured, modes, feedback = self._read()
                # Retain the old command-to-measured tracking bound before any write.
                if self._max_tracking is not None:
                    for name, target in targets.items():
                        if np.max(np.abs(target - measured[name])) > self._max_tracking:
                            raise RuntimeError(f"{name}: command exceeds tracking error limit")
                pending = set(targets) - self._enabled
                for name in targets:
                    cfg = self.settings[name]
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
                            == self.settings[n].velocity_ratio
                            and data["inputs"][_ARM[n][1]]["joint_acc_ratio"]
                            == self.settings[n].acceleration_ratio
                            for n in pending
                        )

                    self._wait(enabled)
                self._write(targets, {})
            except Exception as error:
                try:
                    self.stop()
                except Exception as stop_error:
                    raise ExceptionGroup("command and stop failed", [error, stop_error]) from None
                raise

    def drag(
        self,
        sides: tuple[str, ...],
        stop: threading.Event,
        on_active: Callable[[], None] | None = None,
    ) -> None:
        """Hand-guide disabled arms in joint-space drag until stop is set.

        on_active is called once drag is confirmed and the arms may be moved by hand.
        Uses torque mode with joint impedance and each arm's drag_tool load. Joint
        commands follow feedback at a bounded rate so leaving drag causes no jump.
        The arms count as enabled from the first mode write: after return or a
        failure, drag space is exited here and stop()/close() disables them.
        """
        with self._lock:
            self._require_ready()
            sides = tuple(dict.fromkeys(sides))
            if not sides or set(sides) - self.settings.keys():
                raise ValueError("drag sides must name configured arms")
            if self._enabled.intersection(sides):
                raise RuntimeError("drag requires disabled arms")
            for name in sides:
                if self.settings[name].drag_tool is None:
                    raise ValueError(f"{name}: drag_tool is not configured")
            self._enabled.update(sides)
            try:
                self._check(self._robot.clear_set(), "clear_set")
                for name in sides:
                    arm, tool = _ARM[name][0], self.settings[name].drag_tool
                    # Feedback readback below, not these return values, confirms the mode.
                    self._robot.set_state(arm, _TORQUE_MODE)
                    self._robot.set_impedance_type(arm, 1)
                    self._robot.set_tool(arm, list(tool["kine"]), list(tool["dynamic"]))
                    self._robot.set_joint_kd_params(arm, [1.0] * 7, [0.3] * 7)
                self._check(self._robot.send_cmd(), "send_cmd")
                time.sleep(0.5)
                _, modes, _ = self._read()
                for name in sides:
                    if modes[name] != _TORQUE_MODE:
                        raise RuntimeError(
                            f"{name}: failed to enter torque mode (state {modes[name]})"
                        )

                self._check(self._robot.clear_set(), "clear_set")
                for name in sides:
                    self._robot.set_drag_space(_ARM[name][0], 1)
                self._check(self._robot.send_cmd(), "send_cmd")
                time.sleep(0.2)
                measured, _, data = self._read()
                for name in sides:
                    if int(data["inputs"][_ARM[name][1]]["drag_sp_type"]) != 1:
                        raise RuntimeError(f"{name}: joint-drag readback failed")

                period_s = 1.0 / _DRAG_HZ
                max_step = _DRAG_TRACK_RATE_DEG_S * period_s
                commands = {name: np.degrees(measured[name]) for name in sides}
                if on_active is not None:
                    on_active()
                while not stop.is_set():
                    # _read also rejects faults and stale feedback.
                    measured, modes, _ = self._read()
                    self._check(self._robot.clear_set(), "clear_set")
                    for name in sides:
                        if modes[name] != _TORQUE_MODE:
                            raise RuntimeError(f"{name}: left drag torque mode")
                        step = np.degrees(measured[name]) - commands[name]
                        commands[name] += np.clip(step, -max_step, max_step)
                        self._check(
                            self._robot.set_joint_cmd_pose(
                                _ARM[name][0], commands[name].tolist()
                            ),
                            "joint target",
                        )
                    self._check(self._robot.send_cmd(), "send_cmd")
                    time.sleep(period_s)
            except Exception as error:
                try:
                    self._exit_drag(sides)
                except Exception as exit_error:
                    raise ExceptionGroup("drag and drag exit failed", [error, exit_error]) from None
                raise
            self._exit_drag(sides)

    def _exit_drag(self, sides: tuple[str, ...]) -> None:
        self._check(self._robot.clear_set(), "clear_set")
        for name in sides:
            self._robot.set_drag_space(_ARM[name][0], 0)
        self._check(self._robot.send_cmd(), "send_cmd")
        time.sleep(0.5)

    def stop(self) -> None:
        with self._lock:
            errors = []
            # Independent attempts ensure one arm's rejection cannot prevent a
            # disable request reaching the other arm or another owned arm.
            for name in tuple(self._enabled):
                try:
                    _, modes, _ = self._read(allow_fault=True)
                    if modes[name] in {0, 100}:
                        self._enabled.remove(name)
                        continue
                    previous = self._serial.get(name)
                    self._write({}, {name: 0})
                    self._wait(
                        lambda modes, data, name=name, previous=previous: (
                            self._serial[name] != previous and modes[name] in {0, 100}
                        ),
                        allow_fault=True,
                    )
                    self._enabled.remove(name)
                except Exception as error:
                    errors.append(error)
            if errors:
                raise ExceptionGroup("arm stop incomplete", errors)

    def close(self) -> None:
        with self._lock:
            self._ready = False
            errors = []
            try:
                self.stop()
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
                raise ExceptionGroup("arm cleanup incomplete", errors)


class TianjiArm(ArmBase):
    """One arm channel borrowing a TianjiController.

    The official flange model loads lazily on first FK/IK access. A configured
    official model can be supplied so control and robot assembly use one model.
    Direct legacy TCP models remain a separate compatibility construction path.
    """

    @classmethod
    def load_model(cls, *, side: str, **kinematics_options) -> ArmModel:
        model = TianjiSDKKinematics(side, **kinematics_options)
        return ArmModel(
            model,
            tuple(KinematicCoordinate(f"joint_{i}", "rad") for i in range(1, 8)),
            Path(__file__).parent / "assets" / side / "arm.urdf",
            "Flange_L" if side == "left" else "Flange_R",
            base_frame=f"tianji_{side}_base",
        )

    def __init__(
        self,
        controller: TianjiController,
        side: str,
        *,
        kinematics: FlangeKinematicsBase | None = None,
    ) -> None:
        if side not in controller.settings:
            raise ValueError("arm side must be configured on the controller")
        if kinematics is not None and kinematics.num_arm_joints != 7:
            raise ValueError("Tianji kinematics must have seven arm joints")
        if isinstance(kinematics, TianjiSDKKinematics) and kinematics.arm != side:
            raise ValueError("official kinematics must match the arm side")
        self._controller = controller
        self._side = side
        self._kinematics = kinematics

    @property
    def controller(self) -> TianjiController:
        return self._controller

    @property
    def channel(self) -> str:
        return self._side

    @property
    def num_joints(self) -> int:
        return 7

    @property
    def kinematics(self) -> FlangeKinematicsBase:
        if self._kinematics is None:
            self._kinematics = TianjiSDKKinematics(self._side)
        return self._kinematics
