"""Tianji–TacCap assembly using public arm and end-effector interfaces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from manimux.clock import Clock, SystemClock
from manimux.embodiments.arm.tianji.arm import TianjiArm, TianjiArmSettings, TianjiController
from manimux.embodiments.end_effector.gripper import GripperBase
from manimux.embodiments.robot.base import RobotBase, RobotModel
from manimux.kinematics.base import KinematicCoordinate, ManipulatorKinematicsBase
from manimux.types import FloatArray


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
        shared_arm_hardware = {
            name: control.pop(name) for name in ("velocity_ratio", "acceleration_ratio")
        }
        # Runtime shaping consumes the rated capability; the SDK only needs percentages.
        control.pop("rated_joint_velocity_rad_s")
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
            settings[side] = TianjiArmSettings(
                **{**shared_arm_hardware, **bound[group.arm_name]}
            )
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
