from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Protocol, cast

from manimux.policies.capabilities import PolicyCapabilities
from manimux.types import ActionChunk, ActionContext, InferenceRequest, ObservationSnapshot


class PolicyModel(Protocol):
    """Inference-backend contract; implementations do not own robot hardware."""

    def reset(self, session_id: str) -> None: ...

    def infer(self, request: InferenceRequest) -> object: ...

    def capabilities(self) -> PolicyCapabilities: ...

    def close(self) -> None: ...


class PolicyAdapterBase(ABC):
    """Translate between an embodiment and a policy without owning hardware."""

    @abstractmethod
    def build_observation(self, snapshot: ObservationSnapshot) -> ObservationSnapshot:
        """Map a runtime snapshot to the observation contract expected by the policy."""
        raise NotImplementedError

    @abstractmethod
    def decode_action(self, raw: object, context: ActionContext) -> ActionChunk:
        """Convert one raw policy response into a runtime-ready action chunk."""
        raise NotImplementedError

    @abstractmethod
    def validate(self, robot: dict, policy: dict) -> None:
        """Reject incompatible robot/policy configuration before runtime starts."""
        raise NotImplementedError


class PolicyAdapter(Protocol):
    """Structural adapter contract, including adapters not derived from the ABC."""

    def build_observation(self, snapshot: ObservationSnapshot) -> ObservationSnapshot: ...

    def decode_action(self, raw: object, context: ActionContext) -> ActionChunk: ...

    def validate(self, robot: dict, policy: dict) -> None: ...


def decode_policy_action(
    adapter: PolicyAdapter,
    raw: object,
    context: ActionContext,
) -> ActionChunk:
    """Decode an action while retaining the legacy ``decode_action(raw)`` API."""
    method = adapter.decode_action
    # A bound legacy method exposes only ``raw``; current adapters also expose
    # ``context`` for timing, measured-state, and decode-budget decisions.
    if len(inspect.signature(method).parameters) == 1:
        legacy = cast(Callable[[object], ActionChunk], method)
        return legacy(raw)
    return method(raw, context)


def prepare_policy_request(
    adapter: PolicyAdapter,
    request: InferenceRequest,
) -> InferenceRequest:
    """Apply the optional request hook used by adapters needing request context."""
    method = getattr(adapter, "prepare_request", None)
    # ``prepare_request`` is intentionally optional so existing adapters remain
    # valid implementations of PolicyAdapter.
    if not callable(method):
        return request
    return method(request)


def expected_backend_parameters(**options) -> dict:
    """Build the server/checkpoint identity used to reject a mismatched backend."""

    # Explicit options override these defaults.
    values = {
        "server": None,
        "model": {},
        **options,
    }
    if values["server"] is None and (not values["model"]):
        raise ValueError("policy.expected_backend must declare server or model identity")
    return values


def policy_parameters(**options) -> dict:
    """Fill runtime/client defaults; model implementation remains in XPolicyLab."""

    # Keep task-, checkpoint-, and station-specific values in configuration.
    # Explicit options are expanded last so callers can override every default.
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
    """Return seconds between samples, deriving it from trajectory duration if set."""
    duration = policy.get("trajectory_duration_s")
    # A horizon with N samples contains N - 1 time intervals.
    return policy["action_dt_s"] if duration is None else duration / (policy["horizon_steps"] - 1)
