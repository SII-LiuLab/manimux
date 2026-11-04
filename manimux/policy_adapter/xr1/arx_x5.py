"""XR1 RoboDojo X5 deltas -> anchored link6 poses -> canonical joint chunks."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from collections import OrderedDict
from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from manimux.policies.base import action_interval
from manimux.policy_adapter.kinematics import KinematicAdapter
from manimux.runtime.rtc.request import RtcInferenceRequest
from manimux.types import ActionChunk, RobotState, copy_group_vector

# RoboDojo's XR1 training conversion uses R_mibot = R_link6 @ P.
EEF_REFRAME = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
GROUPS = ("left_arm", "right_arm")
SLICES = {"left_arm": (slice(0, 3), slice(3, 6), 6),
          "right_arm": (slice(8, 11), slice(11, 14), 14)}


class XR1ArxX5Adapter(KinematicAdapter):
    """Decode 60-D actions from measured arm joints and the gripper control anchor.

    Base placement cancels for anchor-local deltas, so FK/IK stay in each arm's
    base_link frame. Whole chunks are rejected on IK failure. Gripper commands
    are clipped at the output boundary as in RoboDojo's synchronous evaluator.
    """

    def __init__(self, robot, policy, *, kinematics=None):
        super().__init__(robot, policy, kinematics=kinematics)
        self._dt_ns = round(action_interval(policy) * 1e9)
        self._anchors = OrderedDict()
        self._session_id = None
        self._camera_names = tuple(policy["adapter"]["camera_map"].values())
        self._gripper_stream = policy["adapter"].get("gripper_state_stream")
        self._env_index = robot["options"].get("env_index", 0)

    def validate(self, robot, policy):
        if robot["group_dims"] != {"left_arm": 7, "right_arm": 7}:
            raise ValueError("XR1 X5 requires two six-joint plus scalar-gripper groups")
        if policy["options"].get("action_format") != "native":
            raise ValueError("XR1 X5 requires native packed_ee_delta model output")
        if set(self.kinematics.models) != set(GROUPS):
            raise ValueError("XR1 X5 requires both configured arm models")
        for model in self.kinematics.models.values():
            if (model.base_frame, model.tcp_frame) != ("base_link", "link6"):
                raise ValueError("XR1 X5 requires base_link -> link6 kinematics")

    def build_observation(self, snapshot):
        if any(name not in snapshot.frames for name in self._camera_names):
            raise ValueError("XR1 X5 observation is missing a configured camera")
        if self._gripper_stream is None:
            return snapshot
        packet = snapshot.stream_packets.get(self._gripper_stream)
        if packet is None:
            raise ValueError("XR1 X5 observation is missing its gripper control state")
        payload = packet["payload"]
        if not isinstance(payload, Mapping):
            raise ValueError("XR1 X5 gripper control payload must be an environment mapping")
        source = payload.get(self._env_index, payload.get(str(self._env_index)))
        if not isinstance(source, Mapping):
            raise ValueError("XR1 X5 gripper control state lacks the configured environment")
        groups = copy_group_vector(snapshot.state.groups)
        for name in GROUPS:
            key = name.replace("_arm", "_ee") + "_joint_state"
            opening = np.asarray(source.get(key), dtype=float).reshape(-1)
            if opening.shape != (1,) or not np.isfinite(opening).all():
                raise ValueError(f"XR1 X5 requires one finite control opening for {name}")
            groups[name][-1] = opening[0]
        # Only the policy view changes; executor feedback remains measured.
        state = RobotState(groups, snapshot.state.monotonic_ns, snapshot.state.sequence)
        return replace(snapshot, state=state)

    @staticmethod
    def _geometry_state(values):
        q = np.array(values, dtype=float, copy=True)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError("XR1 X5 anchor must contain seven finite coordinates")
        # Finger tracking can overshoot the normalized interval slightly. The
        # fixed link6 geometry does not depend on opening. Clip only this FK/IK
        # argument; the policy anchor retains the original control semantics.
        q[-1] = np.clip(q[-1], 0, 1)
        return q

    def prepare_request(self, request):
        if request.session_id != self._session_id:
            self._anchors.clear()
            self._session_id = request.session_id
        anchor = {name: np.array(request.observation.state.groups[name], copy=True)
                  for name in GROUPS}
        for q in anchor.values():
            self._geometry_state(q)
        self._anchors[request.request_seq] = anchor
        while len(self._anchors) > 8:
            self._anchors.popitem(last=False)
        if isinstance(request, RtcInferenceRequest) and request.action_condition is not None:
            return replace(request, action_condition=self._encode_condition(
                request.action_condition, anchor
            ))
        return request

    def _encode_condition(self, condition, anchor):
        condition = np.asarray(condition, dtype=float)
        if condition.ndim != 2 or condition.shape[1] != 14 or not np.isfinite(condition).all():
            raise ValueError("XR1 X5 RTC condition must be finite (horizon, 14) joints")
        output = np.zeros((len(condition), 60), dtype=np.float32)
        for arm_index, name in enumerate(GROUPS):
            model = self.kinematics.models[name]
            pose = model.fk(self._geometry_state(anchor[name]))
            rotation = pose[:3, :3] @ EEF_REFRAME
            pos_slice, rot_slice, grip_index = SLICES[name]
            for index, q in enumerate(condition[:, arm_index * 7:(arm_index + 1) * 7]):
                target = model.fk(self._geometry_state(q))
                output[index, pos_slice] = rotation.T @ (target[:3, 3] - pose[:3, 3])
                output[index, rot_slice] = Rotation.from_matrix(
                    rotation.T @ target[:3, :3] @ EEF_REFRAME
                ).as_rotvec()
                output[index, grip_index] = q[-1] - anchor[name][-1]
        return output

    def decode_action(self, raw, context):
        anchor = self._anchors.pop(context.request_seq, None)
        if anchor is None:
            raise ValueError(f"XR1 X5 has no observation anchor for request {context.request_seq}")
        actions = np.asarray(raw, dtype=float)
        if (actions.ndim != 2 or actions.shape[1] != 60 or not len(actions)
                or not np.isfinite(actions).all()):
            raise ValueError("XR1 X5 actions must be a nonempty finite (horizon, 60) array")
        groups = {}
        for name in GROUPS:
            model = self.kinematics.models[name]
            seed = self._geometry_state(anchor[name])
            pose = model.fk(seed)
            rotation = pose[:3, :3] @ EEF_REFRAME
            pos_slice, rot_slice, grip_index = SLICES[name]
            trajectory = []
            for step, row in enumerate(actions):
                target = np.eye(4)
                target[:3, 3] = pose[:3, 3] + rotation @ row[pos_slice]
                target[:3, :3] = (
                    rotation @ Rotation.from_rotvec(row[rot_slice]).as_matrix() @ EEF_REFRAME.T
                )
                opening = float(np.clip(anchor[name][-1] + row[grip_index], 0, 1))
                result = model.ik(target, seed, fixed_coordinates={"gripper": opening})
                if not result.converged:
                    raise ValueError(f"XR1 X5 IK rejected {name} step {step}: {result.reason}")
                seed = result.joints.copy()
                trajectory.append(seed)
            groups[name] = np.ascontiguousarray(trajectory)
        return ActionChunk(
            plan_id=f"xr1-x5-{context.request_seq}-{uuid.uuid4().hex[:8]}",
            request_seq=context.request_seq,
            observation_time_ns=context.observation_time_ns,
            created_time_ns=context.created_time_ns,
            action_space="joint_position", dt_ns=self._dt_ns, groups=groups,
        )
