from __future__ import annotations

from typing import Protocol

from manimux.policies.capabilities import PolicyCapabilities
from manimux.types import InferenceRequest


class PolicyModel(Protocol):
    def reset(self, session_id: str) -> None: ...

    def infer(self, request: InferenceRequest) -> object: ...

    def capabilities(self) -> PolicyCapabilities: ...

    def close(self) -> None: ...


def expected_backend_parameters(**options) -> dict:
    """保留模型服务与 checkpoint 身份约定，防止连接错误模型。"""

    values = {
        "server": None,
        "model": {},
        **options,
    }
    if values["server"] is None and (not values["model"]):
        raise ValueError("policy.expected_backend must declare server or model identity")
    return values


def policy_parameters(**options) -> dict:
    """补齐推理客户端和动作解码参数；模型实现仍在 XPolicyLab。"""

    values = {
        "device": "cpu",
        "action_dt_s": 0.05,
        "trajectory_duration_s": None,
        "timeout_s": 1.0,
        "horizon_steps": 20,
        "inference_delay_s": 0.04,
        "startup_timeout_s": 30.0,
        "action_decoding": "inline",
        "expected_backend": None,
        "options": {},
        **options,
    }
    if values.get("expected_backend") is not None:
        values["expected_backend"] = expected_backend_parameters(**values["expected_backend"])
    return values


def action_interval(policy: dict) -> float:
    """保留原动作间隔：指定总时长时，用总时长除以相邻点的间隔数。"""
    duration = policy.get("trajectory_duration_s")
    return policy["action_dt_s"] if duration is None else duration / (policy["horizon_steps"] - 1)
