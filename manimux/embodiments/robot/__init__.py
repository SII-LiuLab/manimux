"""Whole-robot construction. Factories assemble components without connecting them."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

from manimux.clock import Clock
from manimux.embodiments.layout import gripper_indices, group_layouts
from manimux.embodiments.robot.base import RobotBase, RobotModel
from manimux.plugins import load_plugin

RobotFactory = Callable[[dict, Clock], RobotBase]


def _set_contract_value(target: dict, name: str, value: object, label: str) -> None:
    if name in target and target[name] != value:
        raise ValueError(f"{label} conflicts with embodiment action_contract")
    target[name] = deepcopy(value)


def action_contract_group_indices(config: dict) -> dict[str, int] | None:
    """Return each group's gripper coordinate from its resolved action contract."""

    dims = config.get("robot", {}).get("group_dims")
    options = config.get("policy", {}).get("adapter", {})
    if not dims or not ({"group_layouts", "gripper_dofs"} & options.keys()):
        return None
    return gripper_indices(group_layouts(dims, options))


def apply_action_contract(config: dict, contract: dict) -> dict[str, int]:
    """Apply one embodiment-owned action layout to policy and execution sections."""

    dimensions = contract.get("group_dims")
    if "group_layouts" in contract:
        dimensions = {
            name: item["arm_dofs"] + item["gripper_dofs"]
            for name, item in contract["group_layouts"].items()
        }
        if "group_dims" in contract and dimensions != contract["group_dims"]:
            raise ValueError("component action layout conflicts with assembly dimensions")
    layouts = group_layouts(dimensions, contract)
    robot = config.setdefault("robot", {})
    adapter = config.setdefault("policy", {}).setdefault("adapter", {})
    _set_contract_value(robot, "group_dims", dimensions, "robot.group_dims")
    _set_contract_value(robot, "group_layouts", layouts, "robot.group_layouts")
    for name, value in {
        "group_order": list(dimensions),
        "group_prefixes": contract.get("group_prefixes", {name: name for name in dimensions}),
        "group_layouts": layouts,
    }.items():
        _set_contract_value(adapter, name, value, f"policy.adapter.{name}")
    indices = action_contract_group_indices(config)
    assert indices is not None
    execution = config.setdefault("executor", {})
    motion = execution.get("motion_limits")
    if isinstance(motion, dict):
        _set_contract_value(
            motion.setdefault("gripper", {}),
            "group_indices",
            indices,
            "executor.motion_limits.gripper.group_indices",
        )
    smooth = execution.get("smooth")
    if isinstance(smooth, dict) and isinstance(smooth.get("gripper"), dict):
        _set_contract_value(
            smooth["gripper"],
            "group_indices",
            indices,
            "executor.smooth.gripper.group_indices",
        )
    return indices


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
