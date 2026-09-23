"""标准绝对关节动作适配；臂爪分组和时间信息来自实验配置。

这里只组织服务端已经解码的关节目标，不做归一化恢复、增量恢复或 IK。
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping

from manimux.policies.base import action_interval
from manimux.policies.xpolicylab.codec import (
    DEFAULT_CAMERA_MAP,
    _bool_option,
    _layouts_from_options,
    decode_action_steps,
)
from manimux.policy_adapter.base import PolicyAdapter
from manimux.types import ActionChunk, ActionContext, ObservationSnapshot


class JointAdapter(PolicyAdapter):
    """Translate canonical snapshots and XPolicyLab per-step action dictionaries."""

    def __init__(self, robot: dict, policy: dict, *, kinematics=None) -> None:
        self._layouts = _layouts_from_options(policy["adapter"], robot["group_dims"])
        self._camera_map = dict(policy["adapter"].get("camera_map", DEFAULT_CAMERA_MAP))
        self._required_cameras = tuple(self._camera_map.values())
        self._action_dt_ns = int(action_interval(policy) * 1_000_000_000)
        self._horizon_steps = policy["horizon_policy_steps"]
        self._allow_short_horizon = _bool_option(policy["adapter"], "allow_short_horizon", False)

    def build_observation(self, snapshot: ObservationSnapshot) -> ObservationSnapshot:
        missing = [name for name in self._required_cameras if name not in snapshot.frames]
        if missing:
            raise ValueError(f"XPolicyLab adapter is missing cameras: {missing}")
        return snapshot

    def decode_action(self, raw: object, context: ActionContext) -> ActionChunk:
        action_payload = raw.get("actions") if isinstance(raw, Mapping) else raw
        groups = decode_action_steps(action_payload, layouts=self._layouts)
        horizons = {values.shape[0] for values in groups.values()}
        horizon = next(iter(horizons)) if len(horizons) == 1 else None
        valid_short_horizon = (
            self._allow_short_horizon
            and horizon is not None
            and 2 <= horizon <= self._horizon_steps
        )
        if horizons != {self._horizon_steps} and not valid_short_horizon:
            raise ValueError(
                f"XPolicyLab action horizon must be {self._horizon_steps}"
                f"{' or 2..' + str(self._horizon_steps) if self._allow_short_horizon else ''}, "
                f"got {sorted(horizons)}"
            )
        return ActionChunk(
            plan_id=f"xpolicylab-{context.request_seq}-{uuid.uuid4().hex[:8]}",
            request_seq=context.request_seq,
            observation_time_ns=context.observation_time_ns,
            created_time_ns=context.created_time_ns,
            action_space="joint_position",
            dt_ns=self._action_dt_ns,
            groups=groups,
        )

    def validate(self, robot: dict, policy: dict) -> None:
        del policy
        expected = tuple(layout.group for layout in self._layouts)
        if tuple(robot["group_dims"]) != expected:
            raise ValueError(
                "XPolicyLab requires robot groups in order "
                f"{list(expected)}, got {list(robot['group_dims'])}"
            )
        for layout in self._layouts:
            if robot["group_dims"][layout.group] != layout.dim:
                raise ValueError(
                    f"group {layout.group!r} is {robot['group_dims'][layout.group]} values "
                    f"but the layout describes {layout.dim}"
                )
