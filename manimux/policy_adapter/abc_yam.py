"""将原有服务的绝对关节矩阵按配置分组，保留数值单位与动作间隔。

服务端已完成模型输出恢复；这里不处理 checkpoint 归一化或硬件通信。
"""

from __future__ import annotations

import uuid

import numpy as np

from manimux.policies.base import action_interval
from manimux.policy_adapter.base import PolicyAdapter
from manimux.types import ActionChunk, ActionContext, ObservationSnapshot

DEFAULT_SERVER = "http://127.0.0.1:8300"
DEFAULT_GROUP_ORDER = ("left_arm", "right_arm")
DEFAULT_CAMERA_MAP = {
    "left_cam": "left_camera",
    "top_cam": "front_camera",
    "right_cam": "right_camera",
}


class AbcYamAdapter(PolicyAdapter):
    """Translate canonical YAM snapshots and raw ABC matrices."""

    def __init__(self, robot: dict, policy: dict, *, kinematics=None) -> None:
        self._group_order = tuple(policy["adapter"].get("group_order", DEFAULT_GROUP_ORDER))
        self._group_dims = dict(robot["group_dims"])
        self._action_dt_ns = int(action_interval(policy) * 1_000_000_000)
        camera_map = policy["adapter"].get("camera_map", DEFAULT_CAMERA_MAP)
        self._required_cameras = tuple(str(value) for value in camera_map.values())

    def build_observation(self, snapshot: ObservationSnapshot) -> ObservationSnapshot:
        missing = [name for name in self._required_cameras if name not in snapshot.frames]
        if missing:
            raise ValueError(f"ABC YAM adapter is missing cameras: {missing}")
        return snapshot

    def decode_action(self, raw: object, context: ActionContext) -> ActionChunk:
        actions = np.asarray(raw, dtype=np.float64)
        expected_dim = sum(self._group_dims[name] for name in self._group_order)
        if actions.ndim != 2 or actions.shape[1] != expected_dim:
            raise ValueError(
                f"ABC actions must have shape (horizon, {expected_dim}), got {actions.shape}"
            )
        if not actions.shape[0] or not np.isfinite(actions).all():
            raise ValueError("ABC actions must be non-empty and finite")
        groups: dict[str, np.ndarray] = {}
        start = 0
        for name in self._group_order:
            end = start + self._group_dims[name]
            groups[name] = np.ascontiguousarray(actions[:, start:end])
            start = end
        return ActionChunk(
            plan_id=f"abc-{context.request_seq}-{uuid.uuid4().hex[:8]}",
            request_seq=context.request_seq,
            observation_time_ns=context.observation_time_ns,
            created_time_ns=context.created_time_ns,
            action_space="joint_position",
            dt_ns=self._action_dt_ns,
            groups=groups,
        )

    def validate(self, robot: dict, policy: dict) -> None:
        del policy
        if tuple(robot["group_dims"]) != self._group_order:
            raise ValueError(
                "ABC YAM requires robot groups in order "
                f"{list(self._group_order)}, got {list(robot['group_dims'])}"
            )
        if any(robot["group_dims"][name] != 7 for name in self._group_order):
            raise ValueError("ABC YAM requires two 7-value arm+gripper groups")
