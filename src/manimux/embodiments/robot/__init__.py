"""Whole-robot construction. Factories assemble components without connecting them."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from manimux.clock import Clock
from manimux.embodiments.robot.base import RobotBase, RobotModel
from manimux.plugins import load_plugin

if TYPE_CHECKING:
    from manimux.robots.base import RobotDriver

RobotFactory = Callable[[dict, Clock], "RobotBase | RobotDriver"]


def _tianji_taccap(config: dict, clock: Clock) -> RobotBase:
    from .tianji_taccap import TianjiTaccapRobot

    robot = TianjiTaccapRobot.from_config(config["config"], clock=clock, **config["options"])
    # A wrong action layout can route values to the wrong joints; check once at entry.
    if config["group_dims"] != {
        name: model.num_coordinates for name, model in robot.kinematics.models.items()
    }:
        raise ValueError("experiment group_dims must match the robot assembly")
    return robot


def _mock(config, clock):
    from manimux.embodiments.robot.mock import MockDualArmDriver

    return MockDualArmDriver(config["group_dims"], clock)


def _maniunicon(config, clock):
    from manimux.robots.maniunicon import ManiUniConMeshcatDualArmDriver

    return ManiUniConMeshcatDualArmDriver.from_config_file(
        config["config"], config["group_dims"], clock
    )


_BUILTINS = {
    "tianji_taccap": _tianji_taccap,
    "yam_dual": "manimux.robots.yam:build_robot",
    "mock_dual_arm": _mock,
    "maniunicon_meshcat_dual_arm": _maniunicon,
}


def build_robot(config: dict, clock: Clock) -> RobotBase | RobotDriver:
    """Select an assembly by robot.type; only connect() may open its hardware."""
    factory = load_plugin(config["type"], group="manimux.robots", builtins=_BUILTINS)
    return factory(config, clock)


__all__ = ["RobotBase", "RobotModel", "RobotFactory", "build_robot"]


def robot_parameters(**options) -> dict:
    """整机选择和控制频率的默认值；兼容旧 YAML 中的 driver 字段。"""

    values = {
        "config": None,
        "control_hz": 100.0,
        **shared_robot_parameters(**options),
    }
    if values.get("config") is not None:
        values["config"] = Path(values["config"])
    if not values["group_dims"] or any(dim <= 0 for dim in values["group_dims"].values()):
        raise ValueError("robot group_dims must contain positive dimensions")
    return values


def shared_robot_parameters(**options) -> dict:
    """读取采集与推理共用的机器人参数；不添加实验级频率。"""

    values = {
        "options": {},
        **options,
    }
    if "driver" in values:
        values["type"] = values.pop("driver")
    return values
