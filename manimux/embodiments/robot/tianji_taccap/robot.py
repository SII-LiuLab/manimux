"""Tianji–TacCap assembly using public arm and end-effector interfaces."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from manimux.clock import Clock, SystemClock
from manimux.embodiments.arm.tianji.arm import TianjiArm, TianjiArmSettings, TianjiController
from manimux.embodiments.end_effector.gripper import GripperBase
from manimux.embodiments.robot.base import RobotBase, RobotModel
from manimux.kinematics.base import KinematicCoordinate, ManipulatorKinematicsBase
from manimux.types import FloatArray, RobotCommand

# Home profile: 9 deg/s peak cosine at 100 Hz.
_HOME_HZ = 100.0
_HOME_SPEED_DEG_S = 9.0
_HOME_TOLERANCE_DEG = 0.5
_SETTLE_TIMEOUT_S = 5.0


@dataclass(frozen=True, slots=True)
class TianjiArmConfig:
    """One group's complete TCP model and control settings.

    Limits are (lower, upper), seven radians each; ratios are controller percent.
    An injected gripper is exclusively owned by this robot after connect.
    Its Clock must share the robot's domain. No TacCap-specific hardware branch
    exists here; RobotModel loads TacCap-equipped groups from the assembly YAML.
    """

    kinematics: ManipulatorKinematicsBase
    joint_limits: tuple[FloatArray, FloatArray]
    velocity_ratio: int
    acceleration_ratio: int
    gripper: GripperBase | None = None

    def __post_init__(self) -> None:
        settings = TianjiArmSettings(
            self.joint_limits, self.velocity_ratio, self.acceleration_ratio
        )
        object.__setattr__(self, "joint_limits", settings.joint_limits)
        expected = tuple(KinematicCoordinate(f"joint_{i}", "rad") for i in range(1, 8))
        if self.gripper is not None:
            expected += (KinematicCoordinate("gripper", "normalized"),)
        if self.kinematics.coordinates != expected:
            raise ValueError("kinematics layout must match seven joints and optional gripper")


class TianjiTaccapRobot(RobotBase):
    """Bind Tianji arms and optional TacCap-compatible position grippers.

    Native connection, feedback, enabling and batch dispatch belong to the shared
    TianjiController. Generic robot lifecycle and tool coordination are inherited
    from RobotBase. The legacy constructor and per-arm TCP frame conventions are
    retained. Hardware remains disconnected until connect().
    """

    def __init__(
        self,
        *,
        ip: str | None = None,
        arms: Mapping[str, TianjiArmConfig] | None = None,
        model: RobotModel | None = None,
        hardware: Mapping[str, object] | None = None,
        component_hardware: Mapping[str, Mapping] | None = None,
        clock: Clock | None = None,
        stale_timeout_s: float = 0.2,
        ready_timeout_s: float = 3.0,
        execute: bool = True,
        end_effector_control: bool = True,
    ) -> None:
        if model is not None:
            if ip is not None or arms is not None:
                raise ValueError("model cannot be mixed with legacy ip/arms arguments")
            if stale_timeout_s != 0.2 or ready_timeout_s != 3.0:
                raise ValueError("model timeouts must be supplied through hardware")
            self._configure_model(
                model, clock, hardware, component_hardware, execute, end_effector_control
            )
            return
        if hardware is not None or component_hardware is not None:
            raise ValueError("hardware overrides require a robot model")
        if not arms or set(arms) - {"left", "right"}:
            raise ValueError("arms must configure left, right, or both")
        configs = {name: arms[name] for name in ("left", "right") if name in arms}
        for name, cfg in configs.items():
            if cfg.kinematics.base_frame != f"tianji_{name}_base":
                raise ValueError(f"{name} kinematics has the wrong arm base frame")
        clock = clock if clock is not None else SystemClock()
        self.controller = TianjiController(
            ip=ip,
            settings={
                name: TianjiArmSettings(
                    cfg.joint_limits, cfg.velocity_ratio, cfg.acceleration_ratio
                )
                for name, cfg in configs.items()
            },
            clock=clock,
            stale_timeout_s=stale_timeout_s,
            ready_timeout_s=ready_timeout_s,
        )
        self._arms = MappingProxyType(configs)
        super().__init__(
            arm_components={name: TianjiArm(self.controller, name) for name in configs},
            end_effectors={
                name: cfg.gripper for name, cfg in configs.items() if cfg.gripper is not None
            },
            models={name: cfg.kinematics for name, cfg in configs.items()},
            clock=clock,
            stale_timeout_s=stale_timeout_s,
            execute=execute,
            end_effector_control=end_effector_control,
        )

    def clear_errors(self) -> None:
        """Clear arm controller faults; call before connect(), which rejects faults."""
        with self._lock:
            if self._controller_open:
                raise RuntimeError("clear_errors requires a disconnected robot")
            self.controller.clear_errors()

    def drag(
        self,
        sides: Sequence[str],
        stop: threading.Event,
        on_active: Callable[[], None] | None = None,
    ) -> None:
        """Hand-guide arm sides ("left"/"right") until stop is set.

        on_active is called once the arms are in drag and may be moved by hand.
        Opens only the shared arm controller; grippers and sensors stay closed.
        Requires execute and a disconnected robot; the arms are disabled on return.
        """
        with self._lock:
            if not self._execute:
                raise RuntimeError("drag requires execute=true")
            if self._controller_open or self._end_effector_open:
                raise RuntimeError("drag requires a disconnected robot")
            self.controller.connect()
            try:
                self.controller.drag(tuple(sides), stop, on_active)
            except Exception as error:
                try:
                    self.controller.close()
                except Exception as cleanup_error:
                    raise ExceptionGroup(
                        "drag and cleanup failed", [error, cleanup_error]
                    ) from None
                raise
            self.controller.close()

    def home(self) -> None:
        """Move connected arms to model.home_joints, then fully open the grippers.

        The arms follow one cosine profile peaking at 9 deg/s and must settle within
        0.5 degrees. Grippers open only with end-effector control. No-op without execute.
        """
        if not self._execute:
            return
        targets = {} if self.model is None else dict(self.model.home_joints)
        if not targets:
            raise RuntimeError("Tianji-TacCap Home target is not configured")
        period_s = 1.0 / _HOME_HZ
        with self._lock:
            state = self.get_state()
            if set(targets) != set(state.groups):
                raise RuntimeError("Home targets do not match the robot groups")
            start = {name: state.groups[name][: len(q)].copy() for name, q in targets.items()}
            distance = max(float(np.max(np.abs(q - start[name]))) for name, q in targets.items())
            duration_s = (math.pi / 2.0) * distance / math.radians(_HOME_SPEED_DEG_S)
            deadline = time.monotonic() + duration_s * 2.0 + _SETTLE_TIMEOUT_S
            groups = {name: values.copy() for name, values in state.groups.items()}
            tick = 0
            while True:
                elapsed = tick * period_s
                fraction = (
                    1.0
                    if elapsed >= duration_s
                    else 0.5 * (1.0 - math.cos(math.pi * elapsed / duration_s))
                )
                for name, q in targets.items():
                    groups[name][: len(q)] = start[name] + (q - start[name]) * fraction
                self.send_command(RobotCommand(groups, self._clock.now_ns(), None))
                if fraction >= 1.0:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("Tianji-TacCap Home trajectory timed out")
                tick += 1
                time.sleep(period_s)

            deadline = time.monotonic() + _SETTLE_TIMEOUT_S
            tolerance = math.radians(_HOME_TOLERANCE_DEG)
            while True:
                state = self.get_state()
                if all(
                    np.max(np.abs(state.groups[name][: len(q)] - q)) <= tolerance
                    for name, q in targets.items()
                ):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("Tianji-TacCap did not reach Home within 0.5 degrees")
                self.send_command(RobotCommand(groups, self._clock.now_ns(), None))
                time.sleep(period_s)

            if not (self.end_effectors and self._end_effector_control):
                return
            for name in self.end_effectors:
                groups[name][-1] = 1.0
            self.send_command(RobotCommand(groups, self._clock.now_ns(), None))
            deadline = time.monotonic() + _SETTLE_TIMEOUT_S
            while not all(
                self.get_state().groups[name][-1] >= 0.98 for name in self.end_effectors
            ):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Tianji-TacCap grippers did not fully open")
                time.sleep(period_s)

    @property
    def arms(self) -> Mapping[str, TianjiArmConfig]:
        """Legacy construction settings; arm_components contains the actual arms."""
        return self._arms

    @classmethod
    def from_config(
        cls,
        path: Path | str,
        *,
        clock: Clock | None = None,
        hardware: Mapping[str, object] | None = None,
        component_hardware: Mapping[str, Mapping] | None = None,
        execute: bool = False,
        end_effector_control: bool = False,
    ) -> TianjiTaccapRobot:
        """Assemble configured components without connecting any hardware.

        RobotModel.from_config loads geometry; component constructors store device
        bindings. Missing bindings are reported when connecting or starting devices.
        Both config-based and direct construction keep poses in each arm base.
        Scene placement is excluded from FK/IK.
        """
        return cls(
            model=RobotModel.from_config(path),
            clock=clock,
            hardware=hardware,
            component_hardware=component_hardware,
            execute=execute,
            end_effector_control=end_effector_control,
        )

    def _configure_model(
        self, model, clock, hardware, component_hardware, execute, end_effector_control
    ) -> None:
        clock = clock if clock is not None else SystemClock()
        control = {**model.hardware, **(hardware or {})}
        overrides = dict(component_hardware or {})
        bound = {name: dict(entry["hardware"]) for name, entry in model.components.items()}
        # 按组件名应用本地绑定；拼错名称时由字典索引直接报错。
        for name, options in overrides.items():
            bound[name].update(options)
        # Both arms share one native controller session; groups retain their YAML order.
        settings, sides = {}, {}
        for name, group in model.groups.items():
            component = model.components[group.arm_name]
            side = component["options"]["side"]
            if side in settings:
                raise ValueError("one controller side cannot be used by multiple groups")
            sides[name] = side
            settings[side] = TianjiArmSettings(**bound[group.arm_name])
        controller = TianjiController(settings=settings, clock=clock, **control)
        arms, end_effectors, sensors = {}, {}, {}
        configs = {}
        for name, group in model.groups.items():
            arm_class = model.components[group.arm_name]["class"]
            side = sides[name]
            arms[name] = arm_class(controller, side, kinematics=group.arm.kinematics)
            if group.end_effector_name is not None:
                tool_name = group.end_effector_name
                component = model.components[tool_name]
                end_effectors[name] = component["class"](clock=clock, **bound[tool_name])
            cfg = settings[side]
            configs[name] = TianjiArmConfig(
                group.kinematics,
                cfg.joint_limits,
                cfg.velocity_ratio,
                cfg.acceleration_ratio,
                end_effectors.get(name),
            )
        # Sensors stay closed until explicitly started; FK/IK never opens devices.
        for name, component in model.components.items():
            if component["type"] == "sensor":
                options = {**component["options"], **bound[name]}
                sensors[name] = component["class"](name=name, clock=clock, **options)
        super().__init__(
            model=model,
            arm_components=arms,
            end_effectors=end_effectors,
            sensors=sensors,
            models={name: group.kinematics for name, group in model.groups.items()},
            clock=clock,
            stale_timeout_s=control.get("stale_timeout_s", 0.2),
            execute=execute,
            end_effector_control=end_effector_control,
        )
        self.controller = controller
        self._arms = MappingProxyType(configs)


# Compatibility name for existing direct-construction callers.
TianjiRobot = TianjiTaccapRobot
