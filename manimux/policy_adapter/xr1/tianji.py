"""Decode Xiaomi Robotics 1 Cartesian deltas for Tianji–TacCap.

The checkpoint emits a ``(30, 60)`` packed action.  Both arm poses and
grippers are anchor-relative: every row is reconstructed from the observation
that initiated the request, then solved to the 7-joint Tianji groups.  Waist,
base, and reserved columns have no Tianji counterpart and are deliberately
ignored.
"""

from __future__ import annotations

import math
import uuid
from collections import OrderedDict
from collections.abc import Mapping

import numpy as np

from manimux.kinematics.robot import RobotKinematics
from manimux.kinematics.tianji_diff import rotation_matrix, rotation_vector
from manimux.policies.base import action_interval
from manimux.policy_adapter.base import PolicyAdapter
from manimux.types import ActionChunk, ActionContext, InferenceRequest, ObservationSnapshot

ACTION_DIM = 60
ACTION_SEMANTICS = "anchor_relative_ee_delta"
OBSERVATION_PROFILE = "tianji_taccap_two_wrist_black_ego"
GROUP_DIMS = {"left_arm": 8, "right_arm": 8}
ARM_SLICES = {
    "left_arm": {"position": slice(0, 3), "axis_angle": slice(3, 6), "gripper": 6},
    "right_arm": {"position": slice(8, 11), "axis_angle": slice(11, 14), "gripper": 14},
}


def _state_vector(value: object, *, label: str) -> np.ndarray:
    state = np.asarray(value, dtype=np.float64)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError(f"{label} must contain seven finite joints and one gripper value")
    if not 0.0 <= state[-1] <= 1.0:
        raise ValueError(f"{label} gripper value must be in [0, 1]")
    return np.ascontiguousarray(state)


def _axis_angle_to_rotation(axis_angle: np.ndarray) -> np.ndarray:
    value = np.asarray(axis_angle, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("XR-1 axis-angle delta must be a finite 3-vector")
    return rotation_matrix(value)


class XR1TianjiTacCapAdapter(PolicyAdapter):
    """Convert native XR-1 actions to safe, continuous Tianji joint chunks."""

    def __init__(self, robot: dict, policy: dict, *, kinematics=None) -> None:
        if not isinstance(kinematics, RobotKinematics):
            raise ValueError("XR-1 Tianji requires the assembled robot kinematics")
        self.kinematics = kinematics
        self.policy = policy
        self._group_order = tuple(policy["adapter"].get("group_order", GROUP_DIMS))
        self._camera_map = dict(policy["adapter"].get("camera_map", {}))
        self._required_cameras = tuple(self._camera_map.values())
        self._horizon_steps = int(policy["horizon_policy_steps"])
        self._action_dt_s = action_interval(policy)
        self._action_dt_ns = round(self._action_dt_s * 1e9)
        self._anchors: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()

    def validate(self, robot: dict, policy: dict) -> None:
        if policy["worker"] != "xpolicylab_ws":
            raise ValueError("XR-1 Tianji must use the shared xpolicylab_ws worker")
        if policy.get("action_decoding") != "inline":
            raise ValueError("XR-1 Tianji requires inline decoding with injected robot kinematics")
        if robot["group_dims"] != GROUP_DIMS or self._group_order != tuple(GROUP_DIMS):
            raise ValueError("XR-1 Tianji requires left_arm/right_arm with 7+1 values")
        if set(self.kinematics.models) != set(GROUP_DIMS):
            raise ValueError("assembled kinematics must provide left_arm and right_arm")
        if self._horizon_steps != 30:
            raise ValueError("the pass-ball XR-1 checkpoint requires a 30-step horizon")
        expected = policy.get("expected_backend") or {}
        identity = expected.get("model", {})
        required = {
            "policy_name": "Xiaomi_Robotics_1",
            "observation_profile": OBSERVATION_PROFILE,
            "output_format": "packed_ee_delta",
            "ego_view_mode": "black",
            "action_semantics": ACTION_SEMANTICS,
        }
        mismatches = {
            name: (identity.get(name), value)
            for name, value in required.items()
            if identity.get(name) != value
        }
        if mismatches:
            raise ValueError(f"XR-1 Tianji backend identity mismatch: {mismatches}")

    def build_observation(self, snapshot: ObservationSnapshot) -> ObservationSnapshot:
        missing = [name for name in self._required_cameras if name not in snapshot.frames]
        if missing:
            raise ValueError(f"XR-1 Tianji adapter is missing cameras: {missing}")
        return snapshot

    def prepare_request(self, request: InferenceRequest) -> InferenceRequest:
        if getattr(request, "action_condition", None) is not None:
            raise ValueError("XR-1 Tianji RTC conditioning is not implemented")
        anchor = {
            name: _state_vector(
                request.observation.state.groups[name], label=f"observation {name}"
            ).copy()
            for name in self._group_order
        }
        self._anchors[request.request_seq] = anchor
        while len(self._anchors) > 8:
            self._anchors.popitem(last=False)
        return request

    def _solve_knot(self, model, current, target, gripper):
        current = _state_vector(current, label="IK seed").copy()
        current[-1] = gripper
        start = model.fk(current)
        rotation = rotation_vector(start[:3, :3].T @ target[:3, :3])

        step_dt_s = float(self.policy["adapter"].get("ik_validation_dt_s", 0.004))
        if not np.isfinite(step_dt_s) or step_dt_s <= 0:
            raise ValueError("policy.adapter.ik_validation_dt_s must be finite and positive")
        backend_limit = getattr(model.arm, "max_step_duration_s", None)
        if backend_limit is not None:
            step_dt_s = min(step_dt_s, backend_limit)
        substeps = max(1, math.ceil(self._action_dt_s / step_dt_s))

        for index in range(1, substeps + 1):
            alpha = index / substeps
            waypoint = np.eye(4, dtype=np.float64)
            waypoint[:3, 3] = (1.0 - alpha) * start[:3, 3] + alpha * target[:3, 3]
            waypoint[:3, :3] = start[:3, :3] @ rotation_matrix(alpha * rotation)
            result = model.ik(
                waypoint,
                current,
                fixed_coordinates={"gripper": gripper},
            )
            if not result.converged:
                raise ValueError(f"Tianji IK {result.reason}")
            current = _state_vector(result.joints, label="IK result").copy()
        return current

    def decode_action(self, raw: object, context: ActionContext) -> ActionChunk:
        payload = raw.get("actions") if isinstance(raw, Mapping) else raw
        actions = np.asarray(payload, dtype=np.float64)
        if actions.shape != (self._horizon_steps, ACTION_DIM):
            raise ValueError(
                f"XR-1 actions must have shape ({self._horizon_steps}, {ACTION_DIM}), "
                f"got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("XR-1 actions must be finite")
        anchor = self._anchors.pop(context.request_seq, None)
        if anchor is None:
            raise ValueError(
                f"XR-1 Tianji has no observation anchor for request {context.request_seq}"
            )

        measured = None if context.measured_state is None else context.measured_state.groups
        groups: dict[str, np.ndarray] = {}
        for name in self._group_order:
            model = self.kinematics.models[name]
            anchor_state = _state_vector(anchor[name], label=f"anchor {name}")
            anchor_pose = model.fk(anchor_state)
            anchor_rotation = anchor_pose[:3, :3]
            anchor_position = anchor_pose[:3, 3]
            seed_source = anchor_state if measured is None else measured[name]
            current = _state_vector(seed_source, label=f"measured {name}").copy()
            columns = ARM_SLICES[name]
            rows: list[np.ndarray] = []
            for row in actions:
                gripper = float(anchor_state[-1] + row[columns["gripper"]])
                if not 0.0 <= gripper <= 1.0:
                    raise ValueError(f"XR-1 {name} gripper target {gripper} is outside [0, 1]")
                target = np.eye(4, dtype=np.float64)
                target[:3, 3] = (
                    anchor_position + anchor_rotation @ row[columns["position"]]
                )
                target[:3, :3] = anchor_rotation @ _axis_angle_to_rotation(
                    row[columns["axis_angle"]]
                )
                current = self._solve_knot(model, current, target, gripper)
                rows.append(current.copy())
            groups[name] = np.ascontiguousarray(np.stack(rows))

        ignored = actions[:, 16:20]
        return ActionChunk(
            plan_id=f"xr1-tianji-{context.request_seq}-{uuid.uuid4().hex[:8]}",
            request_seq=context.request_seq,
            observation_time_ns=context.observation_time_ns,
            created_time_ns=context.created_time_ns,
            action_space="joint_position",
            dt_ns=self._action_dt_ns,
            groups=groups,
            metadata={
                "source_action_semantics": ACTION_SEMANTICS,
                "ignored_waist_base_max_abs": float(np.max(np.abs(ignored))),
            },
        )


def build_adapter(
    robot: dict, policy: dict, *, kinematics=None
) -> XR1TianjiTacCapAdapter:
    return XR1TianjiTacCapAdapter(robot, policy, kinematics=kinematics)
