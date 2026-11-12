"""Map robot groups to StarVLA's raw observations and native action arrays."""

from dataclasses import dataclass

import numpy as np

from manimux.policies.starvla.eef import EefCodec


@dataclass(frozen=True)
class GroupLayout:
    group: str
    arm_dofs: int
    gripper_dofs: int

    @property
    def dim(self):
        return self.arm_dofs + self.gripper_dofs


class StarVlaCodec:
    """Keep native checkpoint ordering separate from canonical robot ordering.

    action_indices maps native output to canonical actions; state_indices maps
    canonical observations to native input. Conditions use the inverse action map.
    """

    def __init__(self, adapter, contract):
        self.contract = dict(contract)
        self.cameras = dict(adapter["camera_map"])
        if set(self.cameras) != set(contract["camera_names"]):
            raise ValueError("Camera mapping differs from the serving contract")
        self.layouts = tuple(
            GroupLayout(group, **layout) for group, layout in adapter["group_layouts"].items()
        )
        self.prefixes = adapter.get("group_prefixes", {})
        self.action_order = np.asarray(contract["action_indices"])
        self.state_order = np.asarray(contract["state_indices"])
        for order, width in [
            (self.action_order, contract["action_dim"]),
            (self.state_order, contract["state_dim"]),
        ]:
            if (order.size > 0 and order.dtype.kind not in "iu") or not np.array_equal(
                np.sort(order), np.arange(width)
            ):
                raise ValueError("State/action order must be a full permutation")
        self.action_order = self.action_order.astype(np.intp)
        self.state_order = self.state_order.astype(np.intp)
        self.eef = None
        if contract["action_type"] == "ee":
            self.eef = EefCodec(contract["eef"], [layout.gripper_dofs for layout in self.layouts])
            width = self.eef.native_dim
        elif contract["action_type"] == "joint":
            width = sum(p.dim for p in self.layouts)
        else:
            raise ValueError("Unknown action_type")
        if width != contract["action_dim"]:
            raise ValueError("Robot layout differs from native action dimensions")

    def encode(self, request):
        observation = request.observation
        if tuple(observation.state.groups) != tuple(p.group for p in self.layouts):
            raise ValueError("Observation groups differ from configured action order")
        images = []
        for camera in self.contract["camera_names"]:
            image = observation.frames[self.cameras[camera]].data
            if (
                not isinstance(image, np.ndarray)
                or image.dtype != np.uint8
                or image.ndim != 3
                or image.shape[-1] != 3
            ):
                raise ValueError("StarVLA observations require HWC uint8 RGB")
            images.append(image)
        example = {"image": images, "lang": request.instruction}
        if self.contract["include_state"]:
            parts = []
            for layout in self.layouts:
                group = observation.state.groups[layout.group]
                if self.contract["state_type"] == "joint":
                    parts.append(group)
                elif self.contract["state_type"] == "ee":
                    prefix = self.prefixes[layout.group]
                    key = f"{prefix}_ee_pose" if prefix else "ee_pose"
                    pose = request.model_state[key]
                    parts.append(np.r_[pose, group[layout.arm_dofs :]])
                else:
                    raise ValueError("Unknown state_type")
            state = np.concatenate(parts)
            if state.shape != (len(self.state_order),) or not np.isfinite(state).all():
                raise ValueError("Observation state differs from checkpoint dimensions")
            example["state"] = state[None, self.state_order].astype(np.float32)
        return {"examples": [example], "unnorm_key": self.contract["unnorm_key"]}

    def decode(self, rows):
        """Convert one complete native chunk into canonical robot groups."""
        actions = np.asarray(rows, dtype=np.float32)
        horizon = self.contract["action_horizon"]
        if (
            actions.ndim != 2
            or actions.shape[1] != len(self.action_order)
            or not np.isfinite(actions).all()
        ):
            raise ValueError("Invalid native action shape or values")
        if len(actions) != horizon:
            raise ValueError("Native action horizon differs from deployment")
        actions = actions[:, self.action_order]
        if self.eef is not None:
            actions = self.eef.convert(actions)
        groups, offset = {}, 0
        for layout in self.layouts:
            width = 7 + layout.gripper_dofs if self.eef is not None else layout.dim
            groups[layout.group] = actions[:, offset : offset + width].copy()
            offset += width
        return {
            "format": "pose" if self.eef is not None else "joint",
            "actions": groups,
            "action_semantics": self.contract["action_semantics"],
        }

    def native_condition(self, rows):
        if self.eef is not None:
            raise ValueError("Conditioned EEF sampling is not implemented")
        values = np.asarray(rows, dtype=np.float32)
        if (
            values.ndim != 2
            or values.shape[1] != len(self.action_order)
            or not np.isfinite(values).all()
        ):
            raise ValueError("Invalid joint action condition")
        return values[:, np.argsort(self.action_order)]
