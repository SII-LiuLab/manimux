"""Convert LingBot-VLA2 native relative joints into ManiMux joint chunks.

The fine-tuned YAM policy predicts every arm waypoint relative to the robot
state in the observation that triggered that inference request. Gripper values
remain absolute. The XPolicy server transports that model-native result; this
adapter owns the embodiment conversion into ManiMux's canonical absolute-joint
contract.
"""

from __future__ import annotations

import uuid
from collections import OrderedDict
from collections.abc import Mapping

import numpy as np

from manimux.policies.base import action_interval
from manimux.policies.xpolicylab.codec import build_layouts, decode_action_steps
from manimux.policy_adapter.base import PolicyAdapter
from manimux.types import ActionChunk, ActionContext, InferenceRequest, ObservationSnapshot

ACTION_SEMANTICS = "anchor_relative_arm_absolute_gripper"
DEFAULT_GROUP_ORDER = ("left_arm", "right_arm")
DEFAULT_GROUP_PREFIXES = {"left_arm": "left", "right_arm": "right"}
DEFAULT_CAMERA_MAP = {
    "cam_head": "front_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
}


class LingBotVLA2YamAdapter(PolicyAdapter):
    """Anchor-relative arm joints plus absolute grippers -> absolute YAM joints."""

    def __init__(self, robot: dict, policy: dict, *, kinematics=None) -> None:
        group_order = tuple(policy["adapter"].get("group_order", DEFAULT_GROUP_ORDER))
        prefixes = dict(policy["adapter"].get("group_prefixes", DEFAULT_GROUP_PREFIXES))
        gripper_dofs = int(policy["adapter"].get("gripper_dofs", 1))
        if gripper_dofs != 1:
            raise ValueError("LingBot-VLA2 YAM requires exactly one gripper value per arm")
        self._layouts = build_layouts(
            group_order,
            prefixes,
            robot["group_dims"],
            gripper_dofs=gripper_dofs,
        )
        self._camera_map = dict(policy["adapter"].get("camera_map", DEFAULT_CAMERA_MAP))
        self._required_cameras = tuple(self._camera_map.values())
        self._action_dt_ns = int(action_interval(policy) * 1_000_000_000)
        self._horizon_steps = policy["horizon_steps"]
        self._anchors: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self.validate(robot, policy)

    def build_observation(self, snapshot: ObservationSnapshot) -> ObservationSnapshot:
        missing = [name for name in self._required_cameras if name not in snapshot.frames]
        if missing:
            raise ValueError(f"LingBot-VLA2 YAM adapter is missing cameras: {missing}")
        return snapshot

    def prepare_request(self, request: InferenceRequest) -> InferenceRequest:
        anchors: dict[str, np.ndarray] = {}
        for layout in self._layouts:
            values = request.observation.state.groups.get(layout.group)
            if values is None:
                raise ValueError(f"LingBot-VLA2 observation is missing group {layout.group!r}")
            vector = np.asarray(values, dtype=np.float64).reshape(-1)
            if vector.shape != (layout.dim,) or not np.isfinite(vector).all():
                raise ValueError(
                    f"LingBot-VLA2 observation group {layout.group!r} must be "
                    f"finite with shape ({layout.dim},)"
                )
            anchors[layout.group] = np.ascontiguousarray(vector.copy())

        self._anchors[request.request_seq] = anchors
        while len(self._anchors) > 8:
            self._anchors.popitem(last=False)
        return request

    def decode_action(self, raw: object, context: ActionContext) -> ActionChunk:
        if not isinstance(raw, Mapping):
            raise ValueError(
                "LingBot-VLA2 relative actions must include explicit action_semantics metadata"
            )
        semantics = raw.get("action_semantics")
        if semantics != ACTION_SEMANTICS:
            raise ValueError(
                f"LingBot-VLA2 action_semantics must be {ACTION_SEMANTICS!r}, got {semantics!r}"
            )
        anchors = self._anchors.pop(context.request_seq, None)
        if anchors is None:
            raise ValueError(
                f"LingBot-VLA2 adapter has no observation anchor for request {context.request_seq}"
            )

        native_groups = decode_action_steps(raw.get("actions"), layouts=self._layouts)
        horizons = {values.shape[0] for values in native_groups.values()}
        if horizons != {self._horizon_steps}:
            raise ValueError(
                f"LingBot-VLA2 action horizon must be {self._horizon_steps}, got {sorted(horizons)}"
            )

        groups: dict[str, np.ndarray] = {}
        for layout in self._layouts:
            native = native_groups[layout.group]
            absolute = native.copy()
            absolute[:, : layout.arm_dofs] += anchors[layout.group][None, : layout.arm_dofs]
            groups[layout.group] = np.ascontiguousarray(absolute)

        return ActionChunk(
            plan_id=f"lingbot-vla2-{context.request_seq}-{uuid.uuid4().hex[:8]}",
            request_seq=context.request_seq,
            observation_time_ns=context.observation_time_ns,
            created_time_ns=context.created_time_ns,
            action_space="joint_position",
            dt_ns=self._action_dt_ns,
            groups=groups,
            metadata={
                "native_action_semantics": ACTION_SEMANTICS,
                "anchor_request_seq": context.request_seq,
            },
        )

    def validate(self, robot: dict, policy: dict) -> None:
        del policy
        expected_groups = tuple(layout.group for layout in self._layouts)
        if tuple(robot["group_dims"]) != expected_groups:
            raise ValueError(
                "LingBot-VLA2 YAM requires robot groups in order "
                f"{list(expected_groups)}, got {list(robot['group_dims'])}"
            )
        for layout in self._layouts:
            if layout.arm_dofs != 6 or layout.gripper_dofs != 1:
                raise ValueError("LingBot-VLA2 YAM requires two 6+1 arm groups")
