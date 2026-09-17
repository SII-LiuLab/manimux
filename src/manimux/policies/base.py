from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Protocol, cast

from manimux.policies.capabilities import PolicyCapabilities
from manimux.types import ActionChunk, ActionContext, InferenceRequest, ObservationSnapshot


class PolicyModel(Protocol):
    def reset(self, session_id: str) -> None: ...

    def infer(self, request: InferenceRequest) -> object: ...

    def capabilities(self) -> PolicyCapabilities: ...

    def close(self) -> None: ...


class PolicyAdapterBase(ABC):
    """Class-based interface for new embodiment adapters; no hardware ownership."""

    @abstractmethod
    def build_observation(self, snapshot: ObservationSnapshot) -> ObservationSnapshot:
        raise NotImplementedError

    @abstractmethod
    def decode_action(self, raw: object, context: ActionContext) -> ActionChunk:
        raise NotImplementedError

    @abstractmethod
    def validate(self, robot: dict, policy: dict) -> None:
        raise NotImplementedError


class PolicyAdapter(Protocol):
    def build_observation(self, snapshot: ObservationSnapshot) -> ObservationSnapshot: ...

    def decode_action(self, raw: object, context: ActionContext) -> ActionChunk: ...

    def validate(self, robot: dict, policy: dict) -> None: ...


def decode_policy_action(
    adapter: PolicyAdapter,
    raw: object,
    context: ActionContext,
) -> ActionChunk:
    """Call a context-aware adapter while retaining the original one-argument API."""
    method = adapter.decode_action
    if len(inspect.signature(method).parameters) == 1:
        legacy = cast(Callable[[object], ActionChunk], method)
        return legacy(raw)
    return method(raw, context)


def prepare_policy_request(
    adapter: PolicyAdapter,
    request: InferenceRequest,
) -> InferenceRequest:
    """Apply an adapter's optional request transformation hook."""
    method = getattr(adapter, "prepare_request", None)
    if not callable(method):
        return request
    return method(request)


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
