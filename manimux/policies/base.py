from __future__ import annotations

from copy import deepcopy
from typing import Protocol

from manimux.policies.capabilities import PolicyCapabilities
from manimux.types import InferenceRequest


class PolicyModel(Protocol):
    def reset(self, session_id: str) -> None: ...

    def infer(self, request: InferenceRequest) -> object: ...

    def capabilities(self) -> PolicyCapabilities: ...

    def close(self) -> None: ...


def expected_backend_parameters(**options) -> dict:
    """Preserve model service and checkpoint identity contracts to prevent connecting to the wrong model."""

    values = {
        "server": None,
        "model": {},
        **options,
    }
    if values["server"] is None and (not values["model"]):
        raise ValueError("policy.expected_backend must declare server or model identity")
    return values


def backend_identity_from_recipe(server: dict, spec: dict) -> dict:
    """Build the runtime handshake identity from one policy-server recipe."""

    model = deepcopy(spec.get("model", {}))
    for field in spec.get("fields", []):
        if field not in server:
            raise ValueError(f"policy_server identity field {field!r} is missing")
        model[field] = deepcopy(server[field])
    for target, source in spec.get("aliases", {}).items():
        if source not in server:
            raise ValueError(f"policy_server identity source {source!r} is missing")
        model[target] = deepcopy(server[source])
    return {
        "server": spec.get("server", "xpolicylab_policy_server"),
        "model": model,
    }


def policy_parameters(*, action_dt_s: float, horizon_policy_steps: int, **options) -> dict:
    """Require experiment timing and fill client and action-decoding options."""

    if "horizon_steps" in options:
        raise ValueError("unsupported policy field: horizon_steps")
    values = {
        "device": "cpu",
        "action_dt_s": action_dt_s,
        "trajectory_duration_s": None,
        "timeout_s": 1.0,
        "horizon_policy_steps": horizon_policy_steps,
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
    """Preserve action timing: when total duration is specified, divide it by the number of intervals between points."""
    duration = policy.get("trajectory_duration_s")
    return (
        policy["action_dt_s"]
        if duration is None
        else duration / (policy["horizon_policy_steps"] - 1)
    )
