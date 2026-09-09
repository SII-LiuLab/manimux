"""YAM embodiment boundary for XPolicyLab absolute per-arm base poses."""

import uuid
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from manimux.integrations.xpolicylab.policy_plugin import DEFAULT_CAMERA_MAP
from manimux.types import ActionChunk, InferenceRequest

SEMANTICS = "absolute_per_arm_base_xyz_wxyz"


@dataclass(slots=True)
class XPolicyPoseRequest(InferenceRequest):
    xpolicylab_state: dict | None = None


def pose_matrix(value):
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("EE pose must be finite [xyz, quaternion wxyz]")
    if not np.isclose(np.linalg.norm(pose[3:]), 1.0, atol=1e-4):
        raise ValueError("EE quaternion must be unit length")
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    result[:3, 3] = pose[:3]
    return result


class OpenWAMYamAdapter:
    def __init__(self, robot, policy):
        from manimux.kinematics import build_kinematics

        self.validate(robot, policy)
        self.cameras = policy.options.get("camera_map", DEFAULT_CAMERA_MAP)
        if set(self.cameras) != set(DEFAULT_CAMERA_MAP):
            raise ValueError("OpenWAM requires the three standard XPolicy cameras")
        self.horizon = policy.horizon_steps
        self.dt = int(policy.effective_action_dt_s * 1e9)
        self.kin = build_kinematics(
            policy.options.get("kinematics", "yam"),
            **policy.options.get("kinematics_options", {}),
        )
        if self.kin.num_arm_joints != 6:
            raise ValueError("YAM requires six joints per arm")
        self.anchors = OrderedDict()

    def validate(self, robot, policy):
        if policy.worker != "xpolicylab_ws":
            raise ValueError("OpenWAM requires xpolicylab_ws")
        if list(robot.group_dims.items()) != [("left_arm", 7), ("right_arm", 7)]:
            raise ValueError("OpenWAM YAM requires left_arm/right_arm with 6+1 values")
        if robot.driver == "yam_dual":
            expected = policy.expected_backend
            identity = {} if expected is None else expected.model
            required = ("checkpoint_path", "checkpoint_file", "checkpoint_sha256",
                        "training_config_sha256", "norm_stats_path", "norm_stats_sha256")
            if not policy.options.get("deployment_bound") or any(not identity.get(k) for k in required):
                raise ValueError("Bind OpenWAM deployment identity before using the YAM driver")
            if identity.get("action_horizon") != policy.horizon_steps:
                raise ValueError("Bound OpenWAM horizon does not match runtime horizon")

    def build_observation(self, snapshot):
        if any(name not in snapshot.frames for name in self.cameras.values()):
            raise ValueError("OpenWAM observation is missing cameras")
        return snapshot

    def prepare_request(self, request):
        state, anchors = {}, {}
        for side in ("left", "right"):
            group = f"{side}_arm"
            values = np.asarray(request.observation.state.groups[group]).copy()
            if values.shape != (7,) or not np.isfinite(values).all():
                raise ValueError("YAM state must contain seven finite values per arm")
            if not 0 <= values[-1] <= 1:
                raise ValueError("YAM gripper must be in [0, 1]")
            matrix = self.kin.fk(values[:6], float(values[-1]))
            quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()
            state[f"{side}_ee_pose"] = np.r_[matrix[:3, 3], quat[[3, 0, 1, 2]]]
            anchors[group] = values
        self.anchors[request.request_seq] = anchors
        while len(self.anchors) > 8:
            self.anchors.popitem(last=False)
        return XPolicyPoseRequest(
            session_id=request.session_id,
            request_seq=request.request_seq,
            observation_time_ns=request.observation_time_ns,
            deadline_ns=request.deadline_ns,
            observation=request.observation,
            instruction=request.instruction,
            xpolicylab_state=state,
        )

    def decode_action(self, raw, context):
        if not isinstance(raw, Mapping) or raw.get("action_semantics") != SEMANTICS:
            raise ValueError("OpenWAM actions must declare absolute per-arm base pose semantics")
        steps = raw.get("actions")
        if not isinstance(steps, Sequence) or len(steps) != self.horizon:
            raise ValueError(f"OpenWAM action horizon must be {self.horizon}")
        anchors = self.anchors.pop(context.request_seq, None)
        if anchors is None:
            raise ValueError("OpenWAM action has no matching observation anchor")
        groups = {}
        for side in ("left", "right"):
            group = f"{side}_arm"
            seed = (
                anchors[group]
                if context.measured_state is None
                else context.measured_state.groups[group]
            )
            seed = np.asarray(seed, dtype=float)
            if seed.shape != (7,) or not np.isfinite(seed).all():
                raise ValueError("Invalid measured YAM state")
            current, rows = seed[:6].copy(), []
            for step in steps:
                target = pose_matrix(step[f"{side}_ee_pose"])
                grip = np.asarray(step[f"{side}_ee_joint_state"], dtype=float)
                if grip.shape != (1,) or not np.isfinite(grip).all() or not 0 <= grip[0] <= 1:
                    raise ValueError("OpenWAM gripper must be one value in [0, 1]")
                ok, solved = self.kin.ik(target, current, float(grip[0]))
                solved = np.asarray(solved)
                if not ok or solved.shape != (6,) or not np.isfinite(solved).all():
                    raise ValueError("OpenWAM IK failed; rejecting the entire action chunk")
                current = solved.copy()
                rows.append(np.r_[current, grip])
            groups[group] = np.asarray(rows)
        return ActionChunk(
            plan_id=f"openwam-{uuid.uuid4().hex}",
            request_seq=context.request_seq,
            observation_time_ns=context.observation_time_ns,
            created_time_ns=context.created_time_ns,
            action_space="joint_position",
            dt_ns=self.dt,
            groups=groups,
            metadata={"native_action_semantics": SEMANTICS},
        )


def build_adapter(robot, policy):
    return OpenWAMYamAdapter(robot, policy)
