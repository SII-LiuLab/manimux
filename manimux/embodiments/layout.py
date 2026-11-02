"""Per-group action layout, independent of policy transports and hardware SDKs."""

from copy import deepcopy
from pathlib import Path

import yaml


def assembly_action_contract(path: str | Path) -> dict | None:
    """Resolve component action widths without loading geometry or opening hardware.

    Integrated arms declare both widths. A separate end effector contributes its
    own coordinates. Existing assemblies with action_contract remain readable.
    """
    source = Path(path)
    assembly = yaml.safe_load(source.read_text())
    contract = dict(assembly.get("action_contract") or {})
    layouts = {}
    for name, group in assembly["groups"].items():
        layout = {"arm_dofs": 0, "gripper_dofs": 0}
        for component_name in (group["arm"], group.get("end_effector")):
            if component_name is None:
                continue
            entry = assembly["components"][component_name]
            component = yaml.safe_load((source.parent / entry["config"]).read_text())
            declared = component.get("action_layout")
            if declared is None:
                # Compatibility for assemblies whose SDK/component metadata is not migrated.
                return contract or None
            for key in layout:
                layout[key] += declared.get(key, 0)
        layouts[name] = layout
    contract["group_layouts"] = layouts
    return contract


def group_layouts(group_dims: dict, options: dict) -> dict[str, dict[str, int]]:
    """Resolve explicit group layouts; accept the former uniform split at the boundary."""
    declared = options.get("group_layouts")
    if declared is None:
        gripper = options.get("gripper_dofs", 1)
        declared = {
            name: {"arm_dofs": dim - gripper, "gripper_dofs": gripper}
            for name, dim in group_dims.items()
        }
    if (
        "group_layouts" in options
        and "gripper_dofs" in options
        and any(item["gripper_dofs"] != options["gripper_dofs"] for item in declared.values())
    ):
        raise ValueError("uniform gripper_dofs conflicts with the per-group layout")
    if set(declared) != set(group_dims):
        raise ValueError("action layout groups must match robot groups")
    for name, layout in declared.items():
        arm, gripper = layout["arm_dofs"], layout["gripper_dofs"]
        if (
            not isinstance(arm, int)
            or not isinstance(gripper, int)
            or isinstance(arm, bool)
            or isinstance(gripper, bool)
            or arm <= 0
            or gripper < 0
            or arm + gripper != group_dims[name]
        ):
            raise ValueError(f"invalid action layout for {name!r}")
    return {name: dict(declared[name]) for name in group_dims}


def gripper_indices(layouts: dict) -> dict[str, int]:
    """Current executors support an optional scalar opening for each arm group."""
    if any(layout["gripper_dofs"] > 1 for layout in layouts.values()):
        raise ValueError("multi-coordinate end effectors require a matching executor capability")
    return {name: layout["arm_dofs"] for name, layout in layouts.items() if layout["gripper_dofs"]}


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
