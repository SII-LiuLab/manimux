"""ManiMux policy plugins for the Xiaomi Robotics 1 (XR-1) checkpoint.

The adapter interprets the service's Cartesian deltas in the measured
end-effector anchor frame, runs FK/IK, and returns joint targets. Checkpoint
loading and sampling belong to XPolicyLab; this module has no model dependency.

Action layout (``mibot.utils.io.ACTION_PARTS``), 60 columns per step::

     0: 3  left  end-effector position delta, in the current left EE frame
     3: 6  left  end-effector rotation delta, axis-angle, same frame
     6: 7  left  gripper delta
     8:11  right end-effector position delta
    11:14  right end-effector rotation delta
    14:15  right gripper delta
    16:17  waist delta          -- YAM has no waist, dropped
    17:20  base velocity        -- YAM has no base,  dropped
    rest   reserved, always zero

All 30 steps are deltas against the *same* anchor: the pose measured when the
chunk was requested. They are absolute waypoints once reconstructed, not
incremental step-to-step offsets.
"""

from __future__ import annotations

import logging
import uuid
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from manimux.policies.base import action_interval
from manimux.policy_adapter.kinematics import KinematicAdapter
from manimux.runtime.rtc.request import RtcInferenceRequest
from manimux.types import (
    ActionChunk,
    ActionContext,
    InferenceRequest,
    ObservationSnapshot,
)

log = logging.getLogger("manimux.policies.xr1")

DEFAULT_SERVER = "http://127.0.0.1:8400"
DEFAULT_GROUP_ORDER = ("left_arm", "right_arm")
DEFAULT_CAMERA_MAP = {
    "top_cam": "front_camera",
    "left_cam": "left_camera",
    "right_cam": "right_camera",
}
ACTION_DIM = 60
ARM_JOINTS = 6
GROUP_DIM = ARM_JOINTS + 1

# Column slices, per arm, inside one 60-wide action row.
ARM_SLICES = {
    "left_arm": {"pos": slice(0, 3), "aa": slice(3, 6), "gripper": slice(6, 7)},
    "right_arm": {"pos": slice(8, 11), "aa": slice(11, 14), "gripper": slice(14, 15)},
}


def joint_condition_to_xr1_actions(
    condition: np.ndarray,
    anchor_groups: Mapping[str, np.ndarray],
    *,
    group_order: Sequence[str],
    kinematics: Any,
) -> np.ndarray:
    """Encode joint-position waypoints as XR-1's anchor-relative 60-D actions."""
    from manimux.policy_adapter.xr1.rotation import rotm2aa_batch

    condition = np.asarray(condition, dtype=np.float64)
    expected_dim = sum(np.asarray(anchor_groups[name]).size for name in group_order)
    if condition.ndim != 2 or condition.shape[1] != expected_dim:
        raise ValueError(
            f"XR-1 joint condition must have shape (horizon, {expected_dim}), got {condition.shape}"
        )
    if not np.isfinite(condition).all():
        raise ValueError("XR-1 joint condition must be finite")
    if tuple(group_order) != DEFAULT_GROUP_ORDER:
        raise ValueError(f"XR-1 condition codec requires group order {list(DEFAULT_GROUP_ORDER)}")
    actions = np.zeros((condition.shape[0], ACTION_DIM), dtype=np.float64)
    offset = 0
    for group in group_order:
        anchor = np.asarray(anchor_groups[group], dtype=np.float64).reshape(-1)
        if anchor.shape != (GROUP_DIM,):
            raise ValueError(f"XR-1 anchor group {group!r} must have shape ({GROUP_DIM},)")
        targets = condition[:, offset : offset + GROUP_DIM]
        offset += GROUP_DIM

        anchor_pose = kinematics.models[group].fk(anchor)
        target_poses = np.stack(
            [kinematics.models[group].fk(row) for row in targets]
        )
        anchor_rotation = anchor_pose[:3, :3]
        columns = ARM_SLICES[group]
        actions[:, columns["pos"]] = (
            anchor_rotation.T @ (target_poses[:, :3, 3] - anchor_pose[:3, 3]).T
        ).T
        delta_rotations = anchor_rotation.T @ target_poses[:, :3, :3]
        actions[:, columns["aa"]] = rotm2aa_batch(delta_rotations)
        actions[:, columns["gripper"]] = targets[:, -1:] - anchor[-1]

    return np.ascontiguousarray(actions, dtype=np.float32)


def _axis_angle_to_rotation(axis_angle: np.ndarray) -> np.ndarray:
    """Rodrigues' formula, matching ``mibot.utils.io.aa2rotm``."""
    theta = float(np.linalg.norm(axis_angle))
    if theta < 1e-8:
        return np.eye(3)
    axis = axis_angle / theta
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(theta) * cross + (1.0 - np.cos(theta)) * (cross @ cross)


class XR1YamAdapter(KinematicAdapter):
    """End-effector deltas -> absolute poses -> YAM joint groups via IK."""

    def __init__(self, robot: dict, policy: dict, *, kinematics=None) -> None:
        super().__init__(robot, policy, kinematics=kinematics)

        self._group_order = tuple(policy["adapter"].get("group_order", DEFAULT_GROUP_ORDER))
        self._group_dims = dict(robot["group_dims"])
        self._action_dt_ns = int(action_interval(policy) * 1_000_000_000)
        camera_map = policy["adapter"].get("camera_map", DEFAULT_CAMERA_MAP)
        self._required_cameras = tuple(str(value) for value in camera_map.values())

        self._anchors: OrderedDict[int, np.ndarray] = OrderedDict()

    def build_observation(self, snapshot: ObservationSnapshot) -> ObservationSnapshot:
        missing = [name for name in self._required_cameras if name not in snapshot.frames]
        if missing:
            raise ValueError(f"XR-1 YAM adapter is missing cameras: {missing}")
        return snapshot

    def prepare_request(self, request: InferenceRequest) -> InferenceRequest:
        anchor = np.concatenate(
            [
                np.asarray(request.observation.state.groups[name], dtype=np.float64)
                for name in self._group_order
            ]
        )
        self._anchors[request.request_seq] = np.ascontiguousarray(anchor)
        while len(self._anchors) > 8:
            self._anchors.popitem(last=False)
        if not isinstance(request, RtcInferenceRequest) or request.action_condition is None:
            return request
        native_condition = joint_condition_to_xr1_actions(
            request.action_condition,
            request.observation.state.groups,
            group_order=self._group_order,
            kinematics=self.kinematics,
        )
        return replace(request, action_condition=native_condition)

    def decode_action(self, raw: object, context: ActionContext) -> ActionChunk:
        # {"actions", "state"} is test-only. Live inference returns a bare
        # (horizon, 60) matrix; the observation anchor is looked up below.
        if isinstance(raw, dict) and "actions" in raw and "state" in raw:
            actions = np.asarray(raw["actions"], dtype=np.float64)
            anchor = np.asarray(raw["state"], dtype=np.float64).reshape(-1)
        else:
            actions = np.asarray(raw, dtype=np.float64)
            stored = self._anchors.pop(context.request_seq, None)
            if stored is None:
                raise ValueError(
                    f"XR-1 adapter has no observation anchor for request {context.request_seq}"
                )
            anchor = stored
        expected_state = sum(self._group_dims[name] for name in self._group_order)
        if anchor.shape != (expected_state,):
            raise ValueError(f"XR-1 anchor state must have shape ({expected_state},)")
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or not actions.shape[0]:
            raise ValueError(
                f"XR-1 actions must have shape (horizon, {ACTION_DIM}), got {actions.shape}"
            )

        groups: dict[str, np.ndarray] = {}
        start = 0
        for name in self._group_order:
            width = self._group_dims[name]
            joints = anchor[start : start + ARM_JOINTS]
            gripper = float(anchor[start + ARM_JOINTS])
            groups[name] = self._solve_arm(name, actions, joints, gripper, width)
            start += width

        return ActionChunk(
            plan_id=f"xr1-{context.request_seq}-{uuid.uuid4().hex[:8]}",
            request_seq=context.request_seq,
            observation_time_ns=context.observation_time_ns,
            created_time_ns=context.created_time_ns,
            action_space="joint_position",
            dt_ns=self._action_dt_ns,
            groups=groups,
        )

    def _solve_arm(
        self,
        group: str,
        actions: np.ndarray,
        joints: np.ndarray,
        gripper: float,
        width: int,
    ) -> np.ndarray:
        columns = ARM_SLICES[group]
        anchor_pose = self._fk(group, joints, gripper)
        anchor_rotation = anchor_pose[:3, :3]
        anchor_position = anchor_pose[:3, 3]

        horizon = actions.shape[0]
        out = np.empty((horizon, width), dtype=np.float64)
        seed = joints.astype(np.float64).copy()
        failures = 0

        for step in range(horizon):
            row = actions[step]
            target = np.eye(4)
            # Deltas are expressed in the anchor end-effector frame.
            target[:3, 3] = anchor_position + anchor_rotation @ row[columns["pos"]]
            target[:3, :3] = anchor_rotation @ _axis_angle_to_rotation(row[columns["aa"]])

            converged, solved = self._ik(group, target, seed, gripper)
            if converged:
                # Seeding the next step with this solution keeps the chunk on one
                # IK branch, so the joint trajectory stays continuous.
                seed = solved
            else:
                # A non-converged solve is wherever the differential solver ran out
                # of iterations, not a pose the model asked for. Hold the last good
                # configuration instead of commanding it, and keep the seed clean so
                # one unreachable step cannot poison the rest of the chunk.
                failures += 1
                solved = seed
            out[step, :ARM_JOINTS] = solved
            out[step, ARM_JOINTS] = gripper + float(row[columns["gripper"]][0])

        if failures:
            log.warning("%s: IK did not converge on %d/%d chunk steps", group, failures, horizon)
        if not np.isfinite(out).all():
            raise ValueError(f"XR-1 IK produced non-finite joints for {group}")
        return np.ascontiguousarray(out)

    def validate(self, robot: dict, policy: dict) -> None:
        del policy
        if tuple(robot["group_dims"]) != self._group_order:
            raise ValueError(
                "XR-1 YAM requires robot groups in order "
                f"{list(self._group_order)}, got {list(robot['group_dims'])}"
            )
        if any(robot["group_dims"][name] != GROUP_DIM for name in self._group_order):
            raise ValueError(f"XR-1 YAM requires two {GROUP_DIM}-value arm+gripper groups")
