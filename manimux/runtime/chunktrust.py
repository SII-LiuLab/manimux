"""Opt-in synchronous PI-family AHS using the ordinary serial runtime."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from manimux.embodiments.layout import group_layouts
from manimux.policies.base import action_interval
from manimux.policies.capabilities import metadata_mismatches
from manimux.runtime.feedback import CommandFeedback
from manimux.runtime.synchronous import SynchronousChunkStrategy
from manimux.runtime.timeline import CommitResult
from manimux.types import ActionChunk, InferenceRequest, InferenceResponse, copy_action_chunk

_GROUP_ORDER = ("left_arm", "right_arm")
_ACTION_DIM = 14
_TRACE_SCHEMA: dict[str, object] = {
    "enabled": True,
    "schema_version": 1,
    "field": "v",
    "axes": ["denoising_step", "action_step", "model_action"],
    "coordinate_space": "normalized_model",
    "velocity_kind": "raw_network_velocity",
    "valid_action_indices": list(range(_ACTION_DIM)),
    "dtype": "float32",
    "supported_modes": ["default"],
}


def _json_value(value: Any) -> Any:
    """Small selector diagnostics include unavailable inter-history scores."""
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ValueError(f"unsupported ChunkTrust diagnostic type: {type(value).__name__}")


@dataclass(slots=True)
class _PendingSelection:
    prefix: ActionChunk
    selector_state: dict[str, Any]


@dataclass(slots=True)
class _ExecutionWindow:
    plan_id: str
    start_ns: int
    dt_ns: int
    steps: int
    last_row: int = -1


class ChunkTrustStrategy(SynchronousChunkStrategy):
    """Select on a copy; accept posterior state separately from command history.

    History samples the first successfully submitted canonical command in each
    policy interval. This is neither measured arrival nor a CAN-send receipt.
    Preview follows the same rule in its independent strategy instance.
    """

    name = "chunktrust"
    label = "ChunkTrust AHS"
    request_type = InferenceRequest
    discard_plans_while_paused = True
    warmup_required_calls = {"ordinary": 2}

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._validate_config(config)
        options = config["inference"].get("chunktrust")
        if not isinstance(options, Mapping):
            raise ValueError("inference.chunktrust must be a mapping")
        unknown = set(options) - {"candidates", "temperature", "seed"}
        if unknown:
            raise ValueError(f"unknown inference.chunktrust fields: {sorted(unknown)}")
        self._seed = options.get("seed", 0)
        if type(self._seed) is not int or not 0 <= self._seed < 2**32:
            raise ValueError("inference.chunktrust.seed must be an integer in [0, 2**32)")
        candidates = options.get("candidates", (10, 20, 30, 40, 50))
        if not isinstance(candidates, (list, tuple)) or not candidates:
            raise ValueError("inference.chunktrust.candidates must be a nonempty integer list")
        if any(type(value) is not int for value in candidates):
            raise ValueError("inference.chunktrust.candidates must contain integers")
        temperature = options.get("temperature", 0.8)
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature <= 0
        ):
            raise ValueError("inference.chunktrust.temperature must be finite and positive")
        # Import only after this strategy was selected and its runtime was checked.
        try:
            from chunktrust import AHS, AHSConfig, __version__  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "ChunkTrust strategy requires the separately installed chunktrust package"
            ) from exc
        self._selector_type = AHS
        self._selector_config = AHSConfig(
            horizon=self._horizon,
            action_dim=_ACTION_DIM,
            candidates=tuple(candidates),
            temperature=float(temperature),
        )
        self._effective_config = {
            "implementation": "chunktrust.AHS",
            "package_version": __version__,
            "seed": self._seed,
            **asdict(self._selector_config),
            "history_semantics": "first_submitted_command_per_policy_interval",
        }
        self.reset()

    def _validate_config(self, config: dict[str, Any]) -> None:
        inference = config["inference"]
        required = {
            "algorithm": "serial",
            "inference_schedule": "serial",
            "request_trigger": "algorithm",
            "action_start_mode": "first_step_when_ready",
            "chunk_policy_steps": None,
            "max_chunk_policy_steps": None,
            "min_plan_hold_policy_steps": 0,
            "blend_policy_steps": 0,
            "handoff_skip_steps": 0,
            "handoff": "blend",
            "independent_group_decoding": False,
        }
        if any(inference.get(key) != value for key, value in required.items()):
            raise ValueError(
                "ChunkTrust requires uncapped Serial, first_step_when_ready, zero "
                "blend/skip/minimum hold and complete inline joint decoding"
            )
        if (inference.get("temporal_ensemble") or {}).get("enabled") or inference.get("history"):
            raise ValueError(
                "ChunkTrust does not support temporal ensembling or observation-history wrappers"
            )
        if (
            config["executor"]["type"] != "direct"
            or config["executor"].get("motion_limits") is not None
        ):
            raise ValueError("ChunkTrust requires Direct without motion limits")
        if config.get("execution", {}).get("ik", {}).get("mode", "chunk") != "chunk":
            raise ValueError("ChunkTrust requires complete joint chunks without step IK")
        policy = config["policy"]
        adapter = policy["adapter"]
        dimensions = config["robot"]["group_dims"]
        if (
            policy["worker"] != "xpolicylab_ws"
            or policy["action_decoding"] != "inline"
            or policy.get("options", {}).get("action_format") != "joint"
            or adapter["type"] != "manimux.policy_adapter.joint:JointAdapter"
            or dimensions != {"left_arm": 7, "right_arm": 7}
            or tuple(adapter.get("group_order", _GROUP_ORDER)) != _GROUP_ORDER
            or adapter.get("group_prefixes", {"left_arm": "left", "right_arm": "right"})
            != {"left_arm": "left", "right_arm": "right"}
            or any(
                layout != {"arm_dofs": 6, "gripper_dofs": 1}
                for layout in group_layouts(dimensions, adapter).values()
            )
            or (adapter.get("gripper_mapping") or {}).get("mode", "continuous") != "continuous"
            or adapter.get("allow_short_horizon", False)
        ):
            raise ValueError(
                "ChunkTrust requires XPolicyLab inline joint14 with the unmodified JointAdapter"
            )
        if config["robot"].get("options", {}).get("end_effector_control", True) is not True:
            raise ValueError("ChunkTrust joint14 history requires both grippers to be controlled")
        self._horizon = policy["horizon_policy_steps"]
        if type(self._horizon) is not int or self._horizon < 1:
            raise ValueError("ChunkTrust requires a positive policy horizon")
        self._dt_ns = int(action_interval(policy) * 1e9)
        if self._dt_ns <= 0:
            raise ValueError("ChunkTrust requires a positive policy interval")
        expected = policy.get("expected_backend") or {}
        model = expected.get("model", {})
        required_model: dict[str, object] = {
            "policy_family": "pi05",
            "action_type": "joint",
            "execution": {"backend": "jax"},
        }
        mismatches = metadata_mismatches(required_model, model)
        trace = model.get("denoise_trace")
        if not isinstance(trace, dict):
            raise ValueError("ChunkTrust requires expected_backend.model.denoise_trace")
        indices = trace.get("valid_action_indices")
        if (
            not isinstance(indices, list)
            or any(type(index) is not int for index in indices)
            or indices != list(range(_ACTION_DIM))
        ):
            raise ValueError(
                "ChunkTrust denoise-trace valid_action_indices must be the integer "
                f"list 0..13, got {indices!r}"
            )
        mismatches.extend(metadata_mismatches(_TRACE_SCHEMA, trace))
        self._model_dim = trace.get("model_action_dim")
        self._denoise_steps = model.get("num_steps")
        if (
            mismatches
            or trace.get("enabled") is not True
            or type(trace.get("schema_version")) is not int
            or type(self._model_dim) is not int
            or self._model_dim < _ACTION_DIM
            or type(self._denoise_steps) is not int
            or self._denoise_steps < 1
            or model.get("action_horizon") != self._horizon
        ):
            raise ValueError(
                "ChunkTrust requires a complete Pi05 JAX default velocity-trace identity: "
                + "; ".join(mismatches)
            )

    @property
    def required_sampling_modes(self) -> frozenset[str]:
        return frozenset({"default"})

    def reset(self) -> None:
        self._selector = self._selector_type(self._selector_config, self._seed)
        self._pending: dict[int, _PendingSelection] = {}
        self._active: _ExecutionWindow | None = None
        self._history = np.zeros((0, _ACTION_DIM), dtype=np.float32)
        self._history_sources = {"submitted": 0, "preview": 0}
        self._events: list[tuple[str, dict[str, object]]] = []

    def prepare_chunk(
        self, *, chunk: ActionChunk, response: InferenceResponse, now_ns: int
    ) -> ActionChunk:
        del now_ns
        cached = self._pending.get(response.request_seq)
        if cached is not None:
            if cached.prefix.plan_id != chunk.plan_id:
                raise ValueError("ChunkTrust response sequence was reused for a different plan")
            return copy_action_chunk(cached.prefix)
        if (
            chunk.request_seq != response.request_seq
            or chunk.action_space != "joint_position"
            or chunk.source_offset_steps
            or chunk.horizon_steps != self._horizon
            or chunk.dt_ns != self._dt_ns
            or chunk.runtime_trajectory is not None
            or chunk.handoff is not None
            or chunk.hold_from_step
            or set(chunk.groups) != set(_GROUP_ORDER)
            or any(values.shape != (self._horizon, 7) for values in chunk.groups.values())
        ):
            raise ValueError("ChunkTrust requires a complete, untrimmed H-by-joint14 prediction")
        raw = response.raw_action
        trace = raw.get("denoise_trace") if isinstance(raw, Mapping) else None
        velocity = None if not isinstance(trace, Mapping) else trace.get("v")
        expected_shape = (self._denoise_steps, self._horizon, self._model_dim)
        nonfinite: int | None = None
        if isinstance(velocity, np.ndarray) and velocity.dtype.kind in "biufc":
            nonfinite = int(np.count_nonzero(~np.isfinite(velocity)))
        if (
            not isinstance(velocity, np.ndarray)
            or velocity.dtype != np.dtype("float32")
            or velocity.shape != expected_shape
            or nonfinite != 0
        ):
            raise ValueError(
                f"ChunkTrust denoise_trace.v expected finite float32 shape={expected_shape}; "
                f"actual type={type(velocity).__name__}, "
                f"dtype={getattr(velocity, 'dtype', None)}, "
                f"shape={getattr(velocity, 'shape', None)}, nonfinite_count={nonfinite}"
            )
        actions = np.concatenate([chunk.groups[name] for name in _GROUP_ORDER], axis=1).astype(
            np.float32
        )
        trial = self._selector_type(self._selector_config, self._seed)
        before = self._selector.state_dict()
        trial.load_state_dict(before)
        history_for_selection = self._history.copy()
        selected, diagnostics = trial.select(
            velocity[:, :, :_ACTION_DIM], actions, history_for_selection
        )
        selected = int(selected)
        if not 1 <= selected <= self._horizon:
            raise ValueError("ChunkTrust selected an execution length outside the prediction")
        after = trial.state_dict()
        prefix = copy_action_chunk(chunk)
        prefix.groups = {name: values[:selected].copy() for name, values in chunk.groups.items()}
        prefix.metadata["chunktrust"] = _json_value(
            {
                "selected_horizon_steps": selected,
                "history_rows": len(self._history),
                "effective_config": self._effective_config,
                "evidence": diagnostics,
                "selector_state_before": before,
                "selector_state_after": after,
                "history_targets": history_for_selection,
            }
        )
        self._pending[response.request_seq] = _PendingSelection(prefix, after)
        return copy_action_chunk(prefix)

    def on_plan_accepted(
        self,
        *,
        chunk: ActionChunk,
        result: CommitResult,
        response: InferenceResponse,
        now_ns: int,
    ) -> dict[str, object]:
        pending = self._pending.pop(response.request_seq, None)
        if (
            pending is None
            or not result.accepted
            or result.trimmed_steps
            or pending.prefix.plan_id != chunk.plan_id
            or pending.prefix.horizon_steps != chunk.horizon_steps
        ):
            raise ValueError(
                "ChunkTrust acceptance does not match its pending, untrimmed selection"
            )
        self._selector.load_state_dict(pending.selector_state)
        self._active = _ExecutionWindow(chunk.plan_id, now_ns, chunk.dt_ns, chunk.horizon_steps)
        return {
            "selected_horizon_steps": chunk.horizon_steps,
            "chunktrust_history_rows": len(self._history),
        }

    def on_response_rejected(self, response: InferenceResponse) -> None:
        self._pending.pop(response.request_seq, None)

    def on_command_feedback(self, feedback: CommandFeedback) -> None:
        if feedback.source == "validated_only":
            return
        if feedback.source not in self._history_sources:
            raise ValueError("unknown ChunkTrust command-feedback source")
        active, command = self._active, feedback.command
        if (
            active is None
            or feedback.reference_plan_id != active.plan_id
            or command.plan_id != active.plan_id
            or command.monotonic_ns < active.start_ns
        ):
            return
        row = (command.monotonic_ns - active.start_ns) // active.dt_ns
        if not active.last_row < row < active.steps:
            return
        if set(command.groups) != set(_GROUP_ORDER) or any(
            values.shape != (7,) for values in command.groups.values()
        ):
            raise ValueError("ChunkTrust feedback requires complete canonical joint14 targets")
        other_source = "preview" if feedback.source == "submitted" else "submitted"
        if self._history_sources[other_source]:
            raise ValueError(
                "ChunkTrust preview and formal history must use independent strategy instances"
            )
        if row > active.last_row + 1:
            self._events.append(
                (
                    "chunktrust_history_rows_skipped",
                    {
                        "plan_id": active.plan_id,
                        "first_skipped_row": active.last_row + 1,
                        "last_skipped_row": int(row) - 1,
                    },
                )
            )
        target = np.concatenate([command.groups[name] for name in _GROUP_ORDER]).astype(np.float32)
        if not np.isfinite(target).all():
            raise ValueError("ChunkTrust feedback targets must remain finite in float32")
        self._history = np.concatenate([self._history, target[None, :]], axis=0)[
            -self._selector_config.history_window :
        ]
        self._history_sources[feedback.source] += 1
        active.last_row = int(row)

    def take_runtime_events(self, *, step: int) -> list[tuple[str, dict[str, object]]]:
        events, self._events = self._events, []
        return [(name, {"step": step, **fields}) for name, fields in events]

    def state_dict(self) -> dict[str, Any]:
        """Copied audit snapshot; history is command submission, never measured arrival."""
        return {
            "selector": self._selector.state_dict(),
            "history": self._history.copy(),
            "history_sources": dict(self._history_sources),
            "active": None if self._active is None else asdict(self._active),
            "pending": sorted(self._pending),
            "effective_config": deepcopy(self._effective_config),
        }


def build_strategy(config: dict[str, Any]) -> ChunkTrustStrategy:
    return ChunkTrustStrategy(config)
