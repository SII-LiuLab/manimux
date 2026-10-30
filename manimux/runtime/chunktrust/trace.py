"""Validate evidence reported by a live backend; never manufacture capabilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from manimux.policies.capabilities import PolicyCapabilities

TRACE_FORK_URL = "https://github.com/hf618/XPolicyLab.git"
TRACE_FORK_REVISION = "53ac927bd485c97d068586b0b9116b14b5e07126"
_INDICES = tuple(range(14))


@dataclass(frozen=True, slots=True)
class TraceProfile:
    horizon: int
    model_dim: int
    num_steps: int
    indices: tuple[int, ...]
    sampling_modes: frozenset[str]
    model_identity: dict[str, Any]


def _integer(value: object, name: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}; got {value!r}")
    return value


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping; got {type(value).__name__}")
    return value


def _equal(actual: object, expected: object, name: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{name} expected {expected!r}; got {actual!r}")


def _indices(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or any(type(index) is not int for index in value):
        raise ValueError(f"{name} must contain integer action indices 0..13; got {value!r}")
    if tuple(value) != _INDICES:
        raise ValueError(f"{name} must be the ordered action indices 0..13; got {value!r}")
    return tuple(value)


def _subset(expected: Mapping[str, Any], actual: Mapping[str, Any], path: str) -> None:
    for key, value in expected.items():
        if key not in actual:
            raise ValueError(f"{path}.{key} is missing; expected {value!r}")
        if isinstance(value, Mapping):
            _subset(value, _mapping(actual[key], f"{path}.{key}"), f"{path}.{key}")
        else:
            _equal(actual[key], value, f"{path}.{key}")


def validate_capabilities(
    capabilities: PolicyCapabilities,
    config: dict[str, Any],
    *,
    rtc: bool = False,
) -> TraceProfile:
    """Check the joint Pi05 H50 contract before connecting hardware or previewing.

    RTC requires evidence that *both* sampling paths return raw trace and that
    guidance excludes model padding. A callable RTC hook alone is insufficient.
    """
    policy = config["policy"]
    _equal(policy.get("worker"), "xpolicylab_ws", "policy.worker")
    horizon = _integer(policy.get("horizon_policy_steps"), "policy.horizon_policy_steps")
    _equal(horizon, 50, "ChunkTrust prediction horizon")
    metadata = _mapping(capabilities.backend_metadata, "backend")
    model = _mapping(metadata.get("model"), "backend.model")
    for key, expected in {"policy_family": "pi05", "action_type": "joint"}.items():
        _equal(model.get(key), expected, f"backend.model.{key}")
    observation_profile = model.get("observation_profile")
    if type(observation_profile) is not str or observation_profile not in {"aloha", "yam_native"}:
        raise ValueError("ChunkTrust requires a live Pi05 ALOHA/YAM native-joint profile")
    execution = _mapping(model.get("execution"), "backend.model.execution")
    _equal(execution.get("backend"), "jax", "backend.model.execution.backend")
    _equal(model.get("action_horizon"), horizon, "backend.model.action_horizon")
    num_steps = _integer(model.get("num_steps"), "backend.model.num_steps")
    seed = _integer(model.get("inference_seed"), "backend.model.inference_seed", minimum=0)
    if seed >= 2**32:
        raise ValueError("backend.model.inference_seed must be below 2**32")
    trace = _mapping(model.get("denoise_trace"), "backend.model.denoise_trace")
    schema = {
        "enabled": True,
        "schema_version": 1,
        "field": "v",
        "axes": ["denoising_step", "action_step", "model_action"],
        "coordinate_space": "normalized_model",
        "velocity_kind": "raw_network_velocity",
        "dtype": "float32",
    }
    _subset(schema, trace, "backend.model.denoise_trace")
    model_dim = _integer(trace.get("model_action_dim"), "trace.model_action_dim", minimum=14)
    _equal(model.get("action_dim"), model_dim, "backend.model.action_dim")
    indices = _indices(trace.get("valid_action_indices"), "trace.valid_action_indices")
    modes = trace.get("supported_modes")
    if (
        not isinstance(modes, list)
        or not modes
        or any(type(mode) is not str for mode in modes)
        or len(set(modes)) != len(modes)
    ):
        raise ValueError("trace.supported_modes must be a nonempty unique list of strings")
    required = {"default", "rtc"} if rtc else {"default"}
    if not required.issubset(capabilities.sampling_modes) or not required.issubset(modes):
        raise ValueError(
            f"ChunkTrust requires joint sampling and raw-trace support for {sorted(required)}; "
            f"sampling={sorted(capabilities.sampling_modes)}, trace={modes}. "
            "The published 53ac927 trace backend supports default only."
        )
    if rtc:
        guidance = _mapping(model.get("rtc_guidance"), "backend.model.rtc_guidance")
        _equal(guidance.get("padding_mask"), "valid_action_indices", "rtc_guidance.padding_mask")
        _indices(guidance.get("valid_action_indices"), "rtc_guidance.valid_action_indices")
    expected = policy.get("expected_backend")
    if expected is not None:
        _subset(_mapping(expected, "policy.expected_backend"), metadata, "backend")
    return TraceProfile(
        horizon, model_dim, num_steps, indices, frozenset(modes), deepcopy(dict(model))
    )


def decode_trace(
    raw_action: object,
    *,
    expected_horizon: int,
    expected_dim: int,
    indices: Sequence[int],
    num_steps: int,
) -> NDArray[np.float32]:
    """Return an owned [T,H,14] array after validating the complete model trace."""
    horizon = _integer(expected_horizon, "expected_horizon")
    dim = _integer(expected_dim, "expected_dim", minimum=14)
    steps = _integer(num_steps, "num_steps")
    selected = _indices(indices, "indices")
    raw = _mapping(raw_action, "response.raw_action")
    if "actions" not in raw:
        raise ValueError("response.raw_action.actions is missing")
    trace = _mapping(raw.get("denoise_trace"), "response.raw_action.denoise_trace")
    velocity = trace.get("v")
    expected_shape = (steps, horizon, dim)
    finite = None
    if isinstance(velocity, np.ndarray) and velocity.dtype.kind in "biufc":
        finite = int(np.count_nonzero(~np.isfinite(velocity)))
    if (
        not isinstance(velocity, np.ndarray)
        or velocity.dtype != np.dtype("float32")
        or velocity.shape != expected_shape
        or finite != 0
    ):
        raise ValueError(
            f"denoise_trace.v expected finite float32 shape={expected_shape}; "
            f"actual type={type(velocity).__name__}, dtype={getattr(velocity, 'dtype', None)}, "
            f"shape={getattr(velocity, 'shape', None)}, nonfinite_count={finite}"
        )
    return np.array(velocity[:, :, selected], dtype=np.float32, copy=True)


def extract_velocity(raw_action: object, profile: TraceProfile) -> NDArray[np.float32]:
    return decode_trace(
        raw_action,
        expected_horizon=profile.horizon,
        expected_dim=profile.model_dim,
        indices=profile.indices,
        num_steps=profile.num_steps,
    )
