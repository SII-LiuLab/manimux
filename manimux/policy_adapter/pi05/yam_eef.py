"""Observation-anchored Pi05 EEF targets converted to YAM joint commands."""

import uuid
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from manimux.policies.actions import action_groups
from manimux.policy_adapter.base import PolicyAdapter, require_request_type
from manimux.policy_adapter.gripper_mapping import clip_aperture
from manimux.policy_adapter.pose_output import pose_chunk
from manimux.types import ActionChunk, InferenceRequest


@dataclass(slots=True)
class EefRequest(InferenceRequest):
    model_state: dict = field(default_factory=dict)


class Pi05YamEefAdapter(PolicyAdapter):
    supports_step_ik = True

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
        require_request_type(request, (InferenceRequest,), "Pi05 YAM EEF")
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
        actions = action_groups(raw, {g: 8 for g in self.groups}, format="pose")
        if any(len(rows) != self.horizon for rows in actions.values()):
            raise ValueError("Pi05 EEF action horizon mismatch")
        actions = {group: rows[: self.decode_steps] for group, rows in actions.items()}
        anchor = self.anchors.pop(context.request_seq, None)
        seed_state = context.ik_seed if context.ik_seed is not None else anchor
        if seed_state is None:
            raise ValueError("Pi05 EEF IK requires a joint seed")
        apertures, gripper_clip = clip_aperture(
            {name: np.asarray(rows[:, 7], dtype=float) for name, rows in actions.items()}
        )
        if self.decode_pose_only:
            poses = {name: rows.copy() for name, rows in actions.items()}
            for name, rows in poses.items():
                rows[:, 7] = apertures[name]
            return pose_chunk(context, poses, self.dt, metadata=gripper_clip)
        groups, diagnostics = {}, {}
        for group in self.groups:
            seed = np.asarray(seed_state.groups[group]).copy()
            sequence = self.ik_sequence(group, seed)
            rows = []
            for action, aperture in zip(actions[group], apertures[group], strict=True):
                pose = np.asarray(action[:7], dtype=float)
                grip = np.asarray(action[7:], dtype=float)
                if (
                    pose.shape != (7,)
                    or grip.shape != (1,)
                    or not np.isfinite(np.r_[pose, grip]).all()
                ):
                    raise ValueError("Pi05 returned an invalid EEF target")
                target = np.eye(4)
                target[:3, 3] = pose[:3]
                target[:3, :3] = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
                aperture = float(aperture)
                result = self.kin.ik(
                    {group: target},
                    {group: sequence.current},
                    fixed_coordinates={group: {"gripper": aperture}},
                )[group]
                seed = sequence.resolve(result.converged, result.joints, reason=result.reason)
                rows.append(seed)
            diagnostics[group] = sequence.metadata()
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
                "ik": diagnostics,
                **gripper_clip,
                "raw_model_eef": {
                    side: actions[f"{side}_arm"][:, :7].tolist() for side in ("left", "right")
                },
            },
        )
