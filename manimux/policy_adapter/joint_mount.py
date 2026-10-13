"""Absolute joint targets for a model trained with its arm bases mounted elsewhere.

A joint-space model encodes gripper poses relative to its training base. When this
robot's base sits at ``base_offset_m`` from that training base (same orientation, metres,
arm base frame), one physical gripper pose has base-frame translation
``p_robot = p_model - offset``. Requests convert measured joints and RTC/PAINT rows
robot -> model; decoded actions convert model -> robot. Each conversion is FK, the
translation, then IK seeded at the input joints, so identical rows map identically and
orientation and gripper are unchanged. An IK failure rejects the request or chunk.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from manimux.policy_adapter.joint import JointAdapter
from manimux.types import ActionChunk, ActionContext, InferenceRequest


class MountOffsetJointAdapter(JointAdapter):
    """``JointAdapter`` with a fixed base translation between model and robot."""

    def __init__(self, robot: dict, policy: dict, *, kinematics=None) -> None:
        super().__init__(robot, policy, kinematics=kinematics)
        if kinematics is None:
            from manimux.embodiments.robot.base import RobotModel

            kinematics = RobotModel.from_config(robot["config"]).kinematics
        self._kinematics = kinematics
        raw = policy["adapter"].get("base_offset_m")
        if not isinstance(raw, dict) or set(raw) != set(self._dimensions):
            raise ValueError(f"base_offset_m must map every group {sorted(self._dimensions)} to [x, y, z]")
        self._offsets = {}
        for group, value in raw.items():
            offset = np.asarray(value, dtype=float)
            if offset.shape != (3,) or not np.isfinite(offset).all():
                raise ValueError(f"base_offset_m[{group!r}] must be three finite metres")
            model = kinematics.models[group]
            if len(model.coordinates) != self._dimensions[group]:
                raise ValueError(f"kinematics for {group!r} does not match its {self._dimensions[group]} values")
            self._offsets[group] = offset
        # The first solve loads solver state (~0.1 s); keep it out of the control loop.
        for group in self._offsets:
            model = self._kinematics.models[group]
            q = np.zeros(len(model.coordinates))
            q[1:4] = (1.5, 0.4, 0.9)
            self._shift(group, q, 0.0)

    def _shift(self, group: str, joints: np.ndarray, sign: float) -> np.ndarray:
        model = self._kinematics.models[group]
        q = np.asarray(joints, dtype=float)
        target = model.fk(q)
        target[:3, 3] += sign * self._offsets[group]
        result = model.ik(target, q, fixed_coordinates={model.coordinates[-1].name: float(q[-1])})
        if not result.converged:
            raise ValueError(f"mount offset IK failed for {group}: {result.reason}")
        return np.asarray(result.joints, dtype=float)

    def _shift_rows(self, rows: np.ndarray, sign: float, active=None) -> np.ndarray:
        """Convert packed ``(N, sum(group_dims))`` rows in robot group order."""
        rows = np.asarray(rows, dtype=float)
        out = rows.copy()
        for index in range(len(rows)):
            if active is not None and not active[index]:
                continue
            offset = 0
            for group, dim in self._dimensions.items():
                out[index, offset : offset + dim] = self._shift(group, rows[index, offset : offset + dim], sign)
                offset += dim
        return out

    def prepare_request(self, request: InferenceRequest) -> InferenceRequest:
        state = request.observation.state
        groups = {
            group: self._shift(group, values, +1.0) if group in self._offsets else values
            for group, values in state.groups.items()
        }
        observation = replace(request.observation, state=replace(state, groups=groups))
        changes: dict[str, object] = {"observation": observation}
        condition = getattr(request, "action_condition", None)
        if condition is not None:
            weights = np.asarray(request.condition_weights)
            changes["action_condition"] = self._shift_rows(condition, +1.0, active=weights > 0)
        prefix = getattr(request, "paint_action_prefix", None)
        if prefix is not None:
            changes["paint_action_prefix"] = self._shift_rows(prefix, +1.0)
        return replace(request, **changes)

    def decode_action(self, raw: object, context: ActionContext) -> ActionChunk:
        chunk = super().decode_action(raw, context)
        groups = {
            group: np.stack([self._shift(group, row, -1.0) for row in rows])
            for group, rows in chunk.groups.items()
        }
        metadata = dict(chunk.metadata or {})
        metadata["mount_offset_m"] = {group: offset.tolist() for group, offset in self._offsets.items()}
        return replace(chunk, groups=groups, metadata=metadata)
