from __future__ import annotations

import time
import uuid

import numpy as np

from manimux.policies.capabilities import PolicyCapabilities
from manimux.policy_adapter.base import PolicyAdapter
from manimux.types import ActionChunk, ActionContext, InferenceRequest, ObservationSnapshot


class FakePolicyAdapter(PolicyAdapter):
    def build_observation(self, snapshot: ObservationSnapshot) -> ObservationSnapshot:
        return snapshot

    def decode_action(self, raw: object, context: ActionContext) -> ActionChunk:
        del context
        if not isinstance(raw, ActionChunk):
            raise TypeError("identity adapter requires an ActionChunk")
        return raw

    def validate(self, robot: dict, policy: dict) -> None:
        if not robot["group_dims"]:
            raise ValueError("fake policy requires robot groups")
        if policy["horizon_policy_steps"] < 2:
            raise ValueError("fake policy requires at least two horizon steps")


class FakePolicyModel:
    def __init__(self, action_dt_ns: int, horizon_steps: int, delay_s: float) -> None:
        self._action_dt_ns = action_dt_ns
        self._horizon_steps = horizon_steps
        self._delay_s = delay_s
        self._session_id: str | None = None

    def reset(self, session_id: str) -> None:
        self._session_id = session_id

    def infer(self, request: InferenceRequest) -> ActionChunk:
        if request.session_id != self._session_id:
            raise RuntimeError("fake policy session is not initialized")
        if self._delay_s:
            time.sleep(self._delay_s)
        phase = request.request_seq * 0.35
        step_axis = np.linspace(0.0, 1.0, self._horizon_steps, dtype=np.float64)
        groups: dict[str, np.ndarray] = {}
        for group_name, current in request.observation.state.groups.items():
            amplitude = 0.12 if "arm" in group_name else 0.04
            direction = -1.0 if group_name.startswith("right") else 1.0
            wave = direction * amplitude * np.sin(phase + step_axis * np.pi)
            groups[group_name] = current[None, :] + wave[:, None]
        now_ns = time.monotonic_ns()
        return ActionChunk(
            plan_id=f"plan-{request.request_seq}-{uuid.uuid4().hex[:8]}",
            request_seq=request.request_seq,
            observation_time_ns=request.observation_time_ns,
            created_time_ns=now_ns,
            action_space="joint_position",
            dt_ns=self._action_dt_ns,
            groups=groups,
        )

    def capabilities(self) -> PolicyCapabilities:
        return PolicyCapabilities(sampling_modes=frozenset({"default", "rtc"}))

    def close(self) -> None:
        self._session_id = None
