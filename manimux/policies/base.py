from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from math import isfinite
from numbers import Real
from typing import Protocol

from manimux.policies.capabilities import PolicyCapabilities
from manimux.types import ActionChunk, InferenceRequest


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
        "playback_hz": None,
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
    action_interval(values)
    return values


def _positive_timing(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def native_action_interval(policy: dict) -> float:
    """Checkpoint action spacing, independent of runtime playback speed."""
    return _positive_timing(policy["action_dt_s"], "policy.action_dt_s")


def action_interval(policy: dict) -> float:
    """Runtime spacing of source action rows; never changes the servo rate."""
    native_dt = native_action_interval(policy)
    playback_hz = policy.get("playback_hz")
    duration = policy.get("trajectory_duration_s")
    if playback_hz is not None and duration is not None:
        raise ValueError("policy.playback_hz and trajectory_duration_s are mutually exclusive")
    if playback_hz is not None:
        dt = 1.0 / _positive_timing(playback_hz, "policy.playback_hz")
    elif duration is not None:
        if policy["horizon_policy_steps"] < 2:
            raise ValueError("trajectory_duration_s requires at least two action points")
        dt = _positive_timing(duration, "policy.trajectory_duration_s") / (
            policy["horizon_policy_steps"] - 1
        )
    else:
        dt = native_dt
    if not isfinite(dt * 1e9) or dt * 1e9 < 1:
        raise ValueError("policy action interval must be representable in nanoseconds")
    return dt


def action_interval_ns(policy: dict) -> int:
    """Use the same rounded action grid for decoding, scheduling and validation."""
    return round(action_interval(policy) * 1e9)


def action_offset_s(policy: dict) -> float:
    """Phase of source row zero on the playback clock, relative to observation."""
    options = policy.get("adapter", {})
    if "execution_offset_s" in options:
        return float(options["execution_offset_s"])
    identity = (policy.get("expected_backend") or {}).get("model", {})
    scale = action_interval(policy) / native_action_interval(policy)
    return float(identity.get("first_action_offset_s", 0.0)) * scale


def attach_sampling_stats(chunk: ActionChunk, raw: object) -> ActionChunk:
    """Preserve optional model counters through inline and process decoding.

    Only the dedicated metadata field crosses this boundary, never raw tensors
    or action arrays. Recording already persists chunk.metadata in plan traces.
    """
    if isinstance(raw, Mapping) and "sampling_stats" in raw:
        stats = raw["sampling_stats"]
        if not isinstance(stats, Mapping):
            raise ValueError("sampling_stats must be a mapping")
        chunk.metadata["sampling_stats"] = deepcopy(dict(stats))
    return chunk
