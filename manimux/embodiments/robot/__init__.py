"""Whole-robot construction. Factories assemble components without connecting them."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from manimux.clock import Clock
from manimux.embodiments.robot.base import RobotBase, RobotModel
from manimux.plugins import load_plugin

RobotFactory = Callable[[dict, Clock], RobotBase]


def _tianji_taccap(config: dict, clock: Clock) -> RobotBase:
    from .tianji_taccap import TianjiTaccapRobot

    robot = TianjiTaccapRobot.from_config(config["config"], clock=clock, **config["options"])
    # A wrong action layout can route values to the wrong joints; check once at entry.
    if config["group_dims"] != {
        name: model.num_coordinates for name, model in robot.kinematics.models.items()
    }:
        raise ValueError("experiment group_dims must match the robot assembly")
    return robot


def _yam(config, clock):
    from .yam import YamRobot

    return YamRobot.from_config(config["config"], clock=clock, **config["options"])


_BUILTINS = {
    "tianji_taccap": _tianji_taccap,
    "yam": _yam,
}


def build_robot(config: dict, clock: Clock) -> RobotBase:
    """Select an assembly by robot.type; only connect() may open its hardware."""
    factory = load_plugin(config["type"], group="manimux.embodiments.robot", builtins=_BUILTINS)
    return factory(config, clock)


__all__ = ["RobotBase", "RobotModel", "RobotFactory", "build_robot"]


def robot_parameters(**options) -> dict:
    """整机选择和控制频率的默认值；实现由 robot.type 指定。"""

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
    return values
