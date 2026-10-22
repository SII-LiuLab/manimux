"""Whole-robot construction. Factories assemble components without connecting them."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from manimux.clock import Clock
from manimux.embodiments.layout import action_contract_group_indices, apply_action_contract
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
    "franka": "manimux.embodiments.robot.franka:build_robot",
    "tianji_taccap": _tianji_taccap,
    "yam": _yam,
}


def build_robot(config: dict, clock: Clock) -> RobotBase:
    """Select an assembly by robot.type; only connect() may open its hardware."""
    factory = load_plugin(config["type"], group="manimux.embodiments.robot", builtins=_BUILTINS)
    robot = factory(config, clock)
    declared = config.get("group_layouts")
    if declared is not None and robot.model is not None and declared != robot.model.action_layouts:
        raise ValueError("configured action layout does not match the assembled robot")
    return robot


__all__ = [
    "RobotBase",
    "RobotModel",
    "RobotFactory",
    "action_contract_group_indices",
    "apply_action_contract",
    "build_robot",
]


def robot_parameters(**options) -> dict:
    """Default robot selection and control frequency; robot.type selects the implementation."""

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
    """Read robot parameters shared by collection and inference without adding experiment-level frequencies."""

    from manimux.config_validation import merge_parameters
    values = merge_parameters({
        "options": {},
    }, options, "robot", extra=("type", "config", "control_hz", "group_dims", "group_layouts"))
    # Retired policy: normalize both historical and new configurations at the
    # same boundary. Runtime always requests Home before closing hardware.
    values["options"] = dict(values["options"])
    values["options"].pop("home_on_close", None)
    return values
