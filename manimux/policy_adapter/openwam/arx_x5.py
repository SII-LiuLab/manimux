"""RoboDojo OpenWAM absolute link6 poses to canonical X5 joint chunks."""

import uuid
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np

from manimux.kinematics.poses import pose_matrix
from manimux.policies.actions import action_groups
from manimux.policies.base import action_interval
from manimux.policy_adapter.kinematics import KinematicAdapter
from manimux.types import ActionChunk, InferenceRequest, RobotState, copy_group_vector

GROUPS = ("left_arm", "right_arm")
CAMERAS = {"cam_head", "cam_left_wrist", "cam_right_wrist"}
SEMANTICS = "absolute_env_relative_xyz_wxyz"


@dataclass(slots=True)
class OpenWAMX5Request(InferenceRequest):
    model_state: dict | None = None


class OpenWAMArxX5Adapter(KinematicAdapter):
    """Keep policy poses/control openings separate from measured joint feedback.

    The backend converts environment-relative poses to its EEF20 base-frame
    representation. Here only its returned world-frame poses are converted back
    to the configured arm bases for IK; no MiBot reframe or TCP offset applies.
    """

    def __init__(self, robot, policy, *, kinematics=None):
        super().__init__(robot, policy, kinematics=kinematics)
        self.validate(robot, policy)
        spec = policy["adapter"]
        self._cameras = tuple(spec["camera_map"].values())
        self._pose_stream = spec["pose_state_stream"]
        self._gripper_stream = spec["gripper_state_stream"]
        self._env_index = robot["options"].get("env_index", 0)
        self._bases = {name: pose_matrix(spec["base_poses"][name]) for name in GROUPS}
        self._base_inverses = {name: np.linalg.inv(pose) for name, pose in self._bases.items()}
        self._horizon = policy["horizon_policy_steps"]
        self._dt_ns = round(action_interval(policy) * 1e9)
        self._anchors = OrderedDict()
        self._session_id = None

    def validate(self, robot, policy):
        if robot["type"] != "asyncsim" or robot["group_dims"] != dict.fromkeys(GROUPS, 7):
            raise ValueError(
                "OpenWAM X5 requires AsyncSim with two six-joint/scalar-gripper groups"
            )
        if policy["options"].get("action_format") != "pose":
            raise ValueError("OpenWAM X5 requires the XPolicyLab pose action format")
        spec = policy["adapter"]
        if spec.get("action_semantics") != SEMANTICS:
            raise ValueError("OpenWAM X5 requires absolute environment-relative link6 poses")
        if set(spec.get("camera_map", {})) != CAMERAS:
            raise ValueError("OpenWAM X5 requires the three RoboDojo RGB cameras")
        if set(spec.get("base_poses", {})) != set(GROUPS):
            raise ValueError("OpenWAM X5 requires both environment-relative base poses")
        if set(self.kinematics.models) != set(GROUPS):
            raise ValueError("OpenWAM X5 requires both configured arm models")
        for model in self.kinematics.models.values():
            if (model.base_frame, model.tcp_frame) != ("base_link", "link6"):
                raise ValueError("OpenWAM X5 requires base_link -> link6 kinematics")
        identity = (policy.get("expected_backend") or {}).get("model", {})
        if identity.get("policy_name") != "OpenWAM" or identity.get("action_type") != "ee":
            raise ValueError("OpenWAM X5 requires a pinned OpenWAM EE backend identity")
        if identity.get("action_horizon") != policy["horizon_policy_steps"]:
            raise ValueError("OpenWAM X5 backend and runtime horizons must match")

    def _state_packet(self, snapshot, stream):
        packet = snapshot.stream_packets.get(stream)
        if packet is None:
            raise ValueError(f"OpenWAM X5 observation is missing {stream}")
        payload = packet["payload"]
        if not isinstance(payload, Mapping):
            raise ValueError("OpenWAM X5 state packet must be an environment mapping")
        state = payload.get(self._env_index, payload.get(str(self._env_index)))
        if not isinstance(state, Mapping):
            raise ValueError("OpenWAM X5 state packet lacks the configured environment")
        return state

    def build_observation(self, snapshot):
        if any(name not in snapshot.frames for name in self._cameras):
            raise ValueError("OpenWAM X5 observation is missing a configured camera")
        poses = self._state_packet(snapshot, self._pose_stream)
        control = self._state_packet(snapshot, self._gripper_stream)
        groups = copy_group_vector(snapshot.state.groups)
        for name in GROUPS:
            prefix = name.removesuffix("_arm")
            pose_matrix(poses[f"{prefix}_ee_pose"])
            opening = np.asarray(control.get(f"{prefix}_ee_joint_state"), dtype=float).reshape(-1)
            if (
                opening.shape != (1,)
                or not np.isfinite(opening).all()
                or not -1e-8 <= opening[0] <= 1 + 1e-8
            ):
                raise ValueError(f"OpenWAM X5 requires one control opening in [0, 1] for {name}")
            groups[name][-1] = np.clip(opening[0], 0, 1)
        state = RobotState(groups, snapshot.state.monotonic_ns, snapshot.state.sequence)
        return replace(snapshot, state=state)

    @staticmethod
    def _geometry_state(values):
        result = np.array(values, dtype=float, copy=True)
        if result.shape != (7,) or not np.isfinite(result).all():
            raise ValueError("OpenWAM X5 IK seed must contain seven finite coordinates")
        result[-1] = np.clip(result[-1], 0, 1)
        return result

    def prepare_request(self, request):
        if request.session_id != self._session_id:
            self._anchors.clear()
            self._session_id = request.session_id
        poses = self._state_packet(request.observation, self._pose_stream)
        model_state = {}
        for name in GROUPS:
            key = f"{name.removesuffix('_arm')}_ee_pose"
            pose = np.asarray(poses[key], dtype=float)
            pose_matrix(pose)
            model_state[key] = pose.copy()
        anchor = {
            name: self._geometry_state(request.observation.state.groups[name]) for name in GROUPS
        }
        self._anchors[request.request_seq] = (request.observation_time_ns, anchor)
        while len(self._anchors) > 8:
            self._anchors.popitem(last=False)
        return OpenWAMX5Request(
            session_id=request.session_id,
            request_seq=request.request_seq,
            observation_time_ns=request.observation_time_ns,
            deadline_ns=request.deadline_ns,
            observation=request.observation,
            instruction=request.instruction,
            model_state=model_state,
        )

    def decode_action(self, raw, context):
        saved = self._anchors.pop(context.request_seq, None)
        if saved is None or saved[0] != context.observation_time_ns:
            raise ValueError("OpenWAM X5 action has no matching observation anchor")
        actions = action_groups(raw, dict.fromkeys(GROUPS, 8), format="pose")
        if any(len(rows) != self._horizon for rows in actions.values()):
            raise ValueError(f"OpenWAM X5 action horizon must be {self._horizon}")
        seed_groups = saved[1] if context.ik_seed is None else context.ik_seed.groups
        groups = {}
        for name in GROUPS:
            model = self.kinematics.models[name]
            seed = self._geometry_state(seed_groups[name])
            rows = []
            for index, row in enumerate(actions[name]):
                target = self._base_inverses[name] @ pose_matrix(row[:7])
                opening = float(np.clip(row[7], 0, 1))
                result = model.ik(target, seed, fixed_coordinates={"gripper": opening})
                if not result.converged:
                    raise ValueError(f"OpenWAM X5 IK rejected {name} step {index}: {result.reason}")
                seed = result.joints.copy()
                rows.append(seed)
            groups[name] = np.ascontiguousarray(rows)
        return ActionChunk(
            plan_id=f"openwam-x5-{context.request_seq}-{uuid.uuid4().hex[:8]}",
            request_seq=context.request_seq,
            observation_time_ns=context.observation_time_ns,
            created_time_ns=context.created_time_ns,
            action_space="joint_position",
            dt_ns=self._dt_ns,
            groups=groups,
            metadata={"native_action_semantics": SEMANTICS},
        )
