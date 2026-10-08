"""Per-group action layout, independent of policy transports and hardware SDKs."""

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
