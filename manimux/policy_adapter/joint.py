"""标准绝对关节动作适配；臂爪分组和时间信息来自实验配置。

这里只组织服务端已经解码的关节目标，不做归一化恢复、增量恢复或 IK。
"""

from __future__ import annotations

import uuid

from manimux.embodiments.layout import group_layouts
from manimux.policies.actions import action_groups
from manimux.policies.base import action_interval
from manimux.policy_adapter.base import PolicyAdapter
from manimux.types import ActionChunk, ActionContext, ObservationSnapshot


class JointAdapter(PolicyAdapter):
    """Translate canonical grouped joint targets into timed robot chunks."""

    def __init__(self, robot: dict, policy: dict, *, kinematics=None) -> None:
        self._dimensions = dict(robot["group_dims"])
        self._layouts = group_layouts(self._dimensions, policy["adapter"])
        self._camera_map = dict(policy["adapter"].get("camera_map", {}))
        self._required_cameras = tuple(self._camera_map.values())
        self._action_dt_ns = int(action_interval(policy) * 1_000_000_000)
        self._horizon_steps = policy["horizon_policy_steps"]
        self._allow_short_horizon = policy["adapter"].get("allow_short_horizon", False)
        if not isinstance(self._allow_short_horizon, bool):
            raise ValueError("allow_short_horizon must be boolean")

    def build_observation(self, snapshot: ObservationSnapshot) -> ObservationSnapshot:
        missing = [name for name in self._required_cameras if name not in snapshot.frames]
        if missing:
            raise ValueError(f"Policy adapter is missing cameras: {missing}")
        return snapshot

    def decode_action(self, raw: object, context: ActionContext) -> ActionChunk:
        groups = action_groups(raw, self._dimensions, format="joint")
        if raw.get("action_semantics") not in (None, "absolute_joint_position"):
            raise ValueError("JointAdapter requires absolute joint-position semantics")
        horizons = {values.shape[0] for values in groups.values()}
        horizon = next(iter(horizons)) if len(horizons) == 1 else None
        valid_short_horizon = (
            self._allow_short_horizon
            and horizon is not None
            and 2 <= horizon <= self._horizon_steps
        )
        if horizons != {self._horizon_steps} and not valid_short_horizon:
            raise ValueError(
                f"Policy action horizon must be {self._horizon_steps}"
                f"{' or 2..' + str(self._horizon_steps) if self._allow_short_horizon else ''}, "
                f"got {sorted(horizons)}"
            )
        return ActionChunk(
            plan_id=f"joint-{context.request_seq}-{uuid.uuid4().hex[:8]}",
            request_seq=context.request_seq,
            observation_time_ns=context.observation_time_ns,
            created_time_ns=context.created_time_ns,
            action_space="joint_position",
            dt_ns=self._action_dt_ns,
            groups=groups,
        )

    def validate(self, robot: dict, policy: dict) -> None:
        del policy
        if robot["group_dims"] != self._dimensions:
            raise ValueError("adapter action layout does not match robot groups")
