"""Translation between ManiMux canonical types and XPolicyLab dictionaries.

These are pure functions over numpy so that the mapping -- the part that is easy
to get subtly wrong -- can be unit tested without a server, a robot, or the
msgpack/websockets dependencies.

XPolicyLab splits an arm from its gripper; ManiMux carries one named group per
arm with the gripper as its trailing value(s). ``GroupLayout`` is the whole of
that difference, and it is declared in the run config rather than inferred.

The observation dictionary follows XPolicyLab's ``data_format_version`` ``v1.0``::

    {"vision": {<camera>: {"color": uint8 (H, W, 3), "shape": [H, W]}},
     "instruction": str,
     "state": {"<prefix>_arm_joint_state": f32, "<prefix>_ee_joint_state": f32},
     "additional_info": {"frequency": float},
     "data_format_version": "v1.0",
     "env_idx": 0}

Depth, intrinsics and extrinsics are omitted: ManiMux publishes colour frames
only, and XPolicyLab's own model template guarantees just ``color``. A policy
that genuinely needs depth has to say so, and will fail loudly on the missing
key rather than silently receive zeros.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from manimux.embodiments.layout import group_layouts
from manimux.kinematics.poses import matrix_pose as matrix_pose
from manimux.kinematics.poses import pose_matrix as pose_matrix
from manimux.types import FloatArray, ObservationSnapshot

DATA_FORMAT_VERSION = "v1.0"


@dataclass(frozen=True, slots=True)
class GroupLayout:
    """How one ManiMux group maps onto XPolicyLab's arm/gripper key pair."""

    group: str
    prefix: str
    arm_dofs: int
    gripper_dofs: int

    def __post_init__(self) -> None:
        if self.arm_dofs <= 0:
            raise ValueError(f"group {self.group!r} must have a positive arm_dofs")
        if self.gripper_dofs < 0:
            raise ValueError(f"group {self.group!r} cannot have negative gripper_dofs")

    @property
    def dim(self) -> int:
        return self.arm_dofs + self.gripper_dofs

    @property
    def arm_key(self) -> str:
        return f"{self.prefix}_arm_joint_state" if self.prefix else "arm_joint_state"

    @property
    def gripper_key(self) -> str:
        return f"{self.prefix}_ee_joint_state" if self.prefix else "ee_joint_state"


def encode_observation(
    snapshot: ObservationSnapshot,
    *,
    layouts: Sequence[GroupLayout],
    camera_map: Mapping[str, str],
    instruction: str,
    frequency: float,
) -> dict[str, Any]:
    """Build one XPolicyLab observation from a canonical ManiMux snapshot.

    ``camera_map`` is keyed by the XPolicyLab camera name and valued by the
    ManiMux frame name, matching how the other integrations spell it.
    """

    state: dict[str, Any] = {}
    for layout in layouts:
        values = snapshot.state.groups.get(layout.group)
        if values is None:
            raise ValueError(f"observation is missing group {layout.group!r}")
        if values.shape != (layout.dim,):
            raise ValueError(
                f"group {layout.group!r} must have {layout.dim} values, got {values.shape}"
            )
        state[layout.arm_key] = np.asarray(values[: layout.arm_dofs], dtype=np.float32)
        if layout.gripper_dofs:
            state[layout.gripper_key] = np.asarray(values[layout.arm_dofs :], dtype=np.float32)

    vision: dict[str, Any] = {}
    for wire_name, frame_name in camera_map.items():
        frame = snapshot.frames.get(frame_name)
        if frame is None:
            raise ValueError(f"observation is missing camera {frame_name!r}")
        color = np.ascontiguousarray(frame.data, dtype=np.uint8)
        vision[wire_name] = {
            "color": color,
            # A list, not a tuple: msgpack has no tuple type, so a tuple would
            # arrive as a list anyway. Emitting one keeps the local value and
            # the value the server actually sees identical.
            "shape": [int(color.shape[0]), int(color.shape[1])],
        }

    return {
        "vision": vision,
        "instruction": instruction,
        "state": state,
        # The environment owns the control rate; the policy is told what it is
        # rather than deciding it. See docs/xpolicylab-runbook.md.
        "additional_info": {"frequency": float(frequency)},
        "data_format_version": DATA_FORMAT_VERSION,
        "env_idx": 0,
    }


def decode_action_steps(
    raw: object,
    *,
    layouts: Sequence[GroupLayout],
) -> dict[str, FloatArray]:
    """Stack XPolicyLab's per-step action dictionaries into group trajectories.

    ``get_action`` returns one dictionary per future step; ManiMux wants one
    ``(horizon, dim)`` matrix per named group.
    """

    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raise ValueError("XPolicyLab actions must be a sequence of per-step dictionaries")
    steps = list(raw)
    if not steps:
        raise ValueError("XPolicyLab returned an empty action chunk")

    columns: dict[str, list[FloatArray]] = {layout.group: [] for layout in layouts}
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise ValueError(f"XPolicyLab action step {index} is not a mapping")
        for layout in layouts:
            parts = [_step_value(step, layout.arm_key, layout.arm_dofs, index)]
            if layout.gripper_dofs:
                parts.append(_step_value(step, layout.gripper_key, layout.gripper_dofs, index))
            columns[layout.group].append(np.concatenate(parts))

    groups: dict[str, FloatArray] = {}
    for layout in layouts:
        matrix = np.ascontiguousarray(np.stack(columns[layout.group], axis=0))
        if not np.isfinite(matrix).all():
            raise ValueError(f"XPolicyLab actions for {layout.group!r} are not finite")
        groups[layout.group] = matrix
    return groups


def _step_value(step: Mapping[str, Any], key: str, dofs: int, index: int) -> FloatArray:
    values = np.asarray(step[key], dtype=np.float64).reshape(-1)
    if values.size != dofs:
        raise ValueError(
            f"XPolicyLab action step {index} key {key!r} must have {dofs} values, got {values.size}"
        )
    return values


def build_layouts(
    group_order: Sequence[str],
    prefixes: Mapping[str, str],
    group_dims: Mapping[str, int],
    *,
    gripper_dofs: int,
) -> tuple[GroupLayout, ...]:
    """Derive one layout per configured group, failing on anything unmapped."""

    layouts: list[GroupLayout] = []
    for group in group_order:
        dim = int(group_dims[group])
        if dim <= gripper_dofs:
            raise ValueError(
                f"group {group!r} has {dim} values, which leaves no arm joints "
                f"after {gripper_dofs} gripper value(s)"
            )
        layouts.append(
            GroupLayout(
                group=group,
                prefix=prefixes[group],
                arm_dofs=dim - gripper_dofs,
                gripper_dofs=gripper_dofs,
            )
        )
    return tuple(layouts)


# 标准位姿字典使用 xyz + wxyz；只转换表示，不变换参考坐标系。
# Wire names and the existing numeric options are shared by transport and adapters.
DEFAULT_SERVER = "ws://127.0.0.1:8500"
DEFAULT_GROUP_ORDER = ("left_arm", "right_arm")
DEFAULT_GROUP_PREFIXES = {"left_arm": "left", "right_arm": "right"}
DEFAULT_CAMERA_MAP = {
    "cam_head": "front_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
}
DEFAULT_GRIPPER_DOFS = 1


def _positive_float_option(options: Mapping[str, object], name: str, default: float) -> float:
    value = options.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"policy.options.{name} must be a number")
    if value <= 0:
        raise ValueError(f"policy.options.{name} must be positive")
    return float(value)


def _layouts_from_options(
    options: Mapping[str, object],
    group_dims: Mapping[str, int],
) -> tuple[GroupLayout, ...]:
    layouts = group_layouts(group_dims, options)
    prefixes = options.get("group_prefixes", DEFAULT_GROUP_PREFIXES)
    return tuple(
        GroupLayout(name, prefixes[name], **layouts[name])
        for name in options.get("group_order", tuple(group_dims))
    )


def decode_policy_actions(raw, *, layouts, format):
    """Translate XPolicyLab step dictionaries at the backend boundary."""
    if format == "native":
        return raw
    metadata = dict(raw) if isinstance(raw, Mapping) else {}
    steps = metadata.pop("actions", raw)
    if format == "joint":
        groups = decode_action_steps(steps, layouts=layouts)
    elif format == "pose":
        if not isinstance(steps, Sequence) or not steps:
            raise ValueError("pose actions must contain per-step dictionaries")
        groups = {}
        for layout in layouts:
            pose_key = f"{layout.prefix}_ee_pose" if layout.prefix else "ee_pose"
            rows = []
            for index, step in enumerate(steps):
                pose = _step_value(step, pose_key, 7, index)
                tool = (
                    _step_value(step, layout.gripper_key, layout.gripper_dofs, index)
                    if layout.gripper_dofs
                    else np.empty(0)
                )
                rows.append(np.r_[pose, tool])
            groups[layout.group] = np.stack(rows)
    else:
        raise ValueError(f"unsupported policy action_format {format!r}")
    return {**metadata, "format": format, "actions": groups}
