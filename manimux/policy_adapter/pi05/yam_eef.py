"""Observation-anchored Pi05 EEF targets converted to YAM joint commands."""

import json
import uuid
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from manimux.policy_adapter.base import PolicyAdapter
from manimux.types import ActionChunk, InferenceRequest


@dataclass(slots=True)
class EefRequest(InferenceRequest):
    xpolicylab_state: dict = field(default_factory=dict)


class Pi05YamEefAdapter(PolicyAdapter):
    def __init__(self, robot, policy, *, kinematics=None):
        from manimux.embodiments.robot.base import RobotModel

        super().__init__(robot, policy, kinematics=kinematics)
        self.kin = (
            kinematics
            if kinematics is not None
            else RobotModel.from_config(robot["config"]).kinematics
        )
        self.groups = ("left_arm", "right_arm")
        self.dt = round(policy["action_dt_s"] * 1e9)
        self.horizon = policy["horizon_policy_steps"]
        self.decode_steps = int(policy["adapter"].get("decode_policy_steps", self.horizon))
        if not 0 < self.decode_steps <= self.horizon:
            raise ValueError("Invalid Pi05 EEF decode prefix")
        self.anchors = {}
        if robot["group_dims"] != dict(left_arm=7, right_arm=7):
            raise ValueError("Pi05 YAM EEF requires two 6+1 joint groups")

    def prepare_request(self, request):
        if getattr(request, "action_condition", None) is not None:
            raise ValueError("Pi05 YAM EEF does not support RTC conditions")
        state = {}
        poses = self.kin.fk(request.observation.state.groups)
        for side, group in zip(("left", "right"), self.groups, strict=True):
            pose = poses[group]
            quat = Rotation.from_matrix(pose[:3, :3]).as_quat()
            state[f"{side}_ee_pose"] = np.r_[pose[:3, 3], quat[3], quat[:3]]
        self.anchors[request.request_seq] = request.observation.state
        while len(self.anchors) > 8:
            self.anchors.pop(next(iter(self.anchors)))
        return EefRequest(
            request.session_id,
            request.request_seq,
            request.observation_time_ns,
            request.deadline_ns,
            request.observation,
            request.instruction,
            state,
        )

    def decode_action(self, raw, context):
        actions = raw.get("actions") if isinstance(raw, dict) else raw
        if len(actions) != self.horizon:
            raise ValueError("Pi05 EEF action horizon mismatch")
        actions = actions[: self.decode_steps]
        anchor = self.anchors.pop(context.request_seq, None)
        seed_state = context.measured_state if context.measured_state is not None else anchor
        if seed_state is None:
            raise ValueError("Pi05 EEF IK requires measured joints")
        groups = {}
        for side, group in zip(("left", "right"), self.groups, strict=True):
            seed = np.asarray(seed_state.groups[group]).copy()
            rows = []
            for step, action in enumerate(actions):
                pose = np.asarray(action[f"{side}_ee_pose"], dtype=float)
                grip = np.asarray(action[f"{side}_ee_joint_state"], dtype=float)
                if (
                    pose.shape != (7,)
                    or grip.shape != (1,)
                    or not np.isfinite(np.r_[pose, grip]).all()
                ):
                    raise ValueError("Pi05 returned an invalid EEF target")
                target = np.eye(4)
                target[:3, 3] = pose[:3]
                target[:3, :3] = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
                aperture = float(np.clip(grip[0], 0, 1))
                result = self.kin.ik(
                    {group: target},
                    {group: seed},
                    fixed_coordinates={group: {"gripper": aperture}},
                )[group]
                if not result.converged:
                    diagnostic = dict(
                        group=group,
                        step=step,
                        target=pose.tolist(),
                        seed=seed.tolist(),
                        gripper=aperture,
                        reason=result.reason,
                    )
                    raise ValueError(
                        "Pi05 EEF IK failed: " + json.dumps(diagnostic, separators=(",", ":"))
                    )
                seed = np.asarray(result.joints).copy()
                rows.append(seed)
            groups[group] = np.asarray(rows)
        return ActionChunk(
            f"pi05-eef-{uuid.uuid4().hex[:8]}",
            context.request_seq,
            context.observation_time_ns,
            context.created_time_ns,
            "joint_position",
            self.dt,
            groups,
            metadata={
                "raw_model_eef": {
                    side: np.stack([a[f"{side}_ee_pose"] for a in actions]).tolist()
                    for side in ("left", "right")
                }
            },
        )
