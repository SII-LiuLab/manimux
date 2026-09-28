"""RoboDojo joint actions with its normalized gripper command convention."""

from __future__ import annotations

import numpy as np

from manimux.policies.actions import action_groups
from manimux.policy_adapter.joint import JointAdapter
from manimux.types import ActionChunk, ActionContext


class RoboDojoJointAdapter(JointAdapter):
    """Clip gripper targets as RoboDojo's synchronous joint evaluator does."""

    def __init__(self, robot: dict, policy: dict, *, kinematics=None) -> None:
        super().__init__(robot, policy, kinematics=kinematics)
        if any(layout["gripper_dofs"] != 1 for layout in self._layouts.values()):
            raise ValueError("RoboDojo joint actions require one gripper value per arm")

    def decode_action(self, raw: object, context: ActionContext) -> ActionChunk:
        groups = action_groups(raw, self._dimensions, format="joint")
        for name, values in groups.items():
            gripper_index = self._layouts[name]["arm_dofs"]
            clipped = values.copy()
            clipped[:, gripper_index] = np.clip(values[:, gripper_index], 0.0, 1.0)
            groups[name] = clipped
        return super().decode_action({**raw, "actions": groups}, context)
