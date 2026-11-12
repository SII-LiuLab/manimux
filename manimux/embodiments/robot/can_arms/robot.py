"""Assemble independently owned CAN arm sessions using the common RobotBase."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from manimux.clock import Clock, SystemClock
from manimux.embodiments.robot.base import RobotBase, RobotModel
from manimux.plugins import load_plugin


class CanArmRobot(RobotBase):
    """Component YAML selects each controller; the assembly never calls vendor APIs."""

    def __init__(
        self,
        model: RobotModel,
        *,
        clock: Clock | None = None,
        hardware: Mapping[str, Any] | None = None,
        component_hardware: Mapping[str, Mapping[str, Any]] | None = None,
        execute: bool = False,
        home_on_close: bool = False,
    ) -> None:
        if home_on_close is not False:
            raise ValueError("CAN arm integration has no configured hardware Home operation")
        if any(component["type"] != "arm" for component in model.components.values()):
            raise ValueError(
                "CAN arm assembly currently contains arms only; cameras use sensor clients"
            )
        clock = clock if clock is not None else SystemClock()
        bindings = component_hardware or {}
        arms = {}
        channels = set()
        for name, group in model.groups.items():
            if group.end_effector is not None:
                raise ValueError("CAN arm assembly requires an integrated scalar gripper")
            component = model.components[group.arm_name]
            options = {
                **component["hardware"],
                **bindings.get(group.arm_name, {}),
            }
            controller_type = load_plugin(
                component["controller"], group="manimux.controllers", builtins={}
            )
            controller = controller_type(clock=clock, execute=execute, **options)
            if controller.channel in channels:
                raise ValueError("independent arm components must have distinct CAN channels")
            channels.add(controller.channel)
            arms[name] = component["class"](controller, kinematics=group.kinematics)
        settings = {**model.hardware, **(hardware or {})}
        super().__init__(
            arm_components=arms,
            models={name: group.kinematics for name, group in model.groups.items()},
            model=model,
            clock=clock,
            stale_timeout_s=settings.get("stale_timeout_s", 0.2),
            execute=execute,
        )

    @classmethod
    def from_config(cls, path: Path | str, **options: Any) -> CanArmRobot:
        return cls(RobotModel.from_config(path), **options)


def build_robot(config: Mapping[str, Any], clock: Clock) -> CanArmRobot:
    robot = CanArmRobot.from_config(config["config"], clock=clock, **config.get("options", {}))
    if config["group_dims"] != {
        name: model.num_coordinates for name, model in robot.kinematics.models.items()
    }:
        raise ValueError("group_dims must match the selected CAN arm assembly")
    return robot
