from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from manimux.config import ManiMuxConfig
from manimux.plugins import load_plugin
from manimux.policies.base import PolicyAdapter
from manimux.runtime.safety import RuntimeState
from manimux.runtime.timeline import ActionTimeline, CommitResult
from manimux.types import (
    ActionChunk,
    GroupVector,
    InferenceRequest,
    InferenceResponse,
    ObservationSnapshot,
    copy_group_vector,
)


@dataclass(frozen=True, slots=True)
class RequestState:
    in_flight: bool
    last_submitted_seq: int
    last_deadline_ns: int


@dataclass(frozen=True, slots=True)
class InferenceSubmission:
    request: InferenceRequest
    event_fields: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CommitSettings:
    current_command: GroupVector
    blend_steps: int
    anchor_source: str


class InferenceStrategy(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def control_mode(self) -> str: ...

    @property
    def required_sampling_modes(self) -> frozenset[str]: ...

    def reset(self) -> None: ...

    def build_submission(
        self,
        *,
        session_id: str,
        request_seq: int,
        now_ns: int,
        snapshot: ObservationSnapshot,
        adapter: PolicyAdapter,
        timeline: ActionTimeline,
        request_state: RequestState,
        runtime_state: RuntimeState,
    ) -> InferenceSubmission | None: ...

    def commit_settings(
        self,
        *,
        response: InferenceResponse,
        measured: GroupVector,
        last_command: GroupVector,
    ) -> CommitSettings: ...

    def prepare_chunk(
        self,
        *,
        chunk: ActionChunk,
        response: InferenceResponse,
        now_ns: int,
    ) -> ActionChunk: ...

    def on_plan_accepted(
        self,
        *,
        chunk: ActionChunk,
        result: CommitResult,
        response: InferenceResponse,
        now_ns: int,
    ) -> dict[str, object]: ...

    def on_response_rejected(self, response: InferenceResponse) -> None: ...

    def take_runtime_events(self, *, step: int) -> list[tuple[str, dict[str, object]]]: ...

    def on_tick(self, *, steps: int, loop_ms: float, control_dt_ns: int) -> None: ...


class DefaultChunkStrategy:
    """Original ManiMux latest-chunk scheduling, isolated from the control loop."""

    def __init__(self, config: ManiMuxConfig) -> None:
        self._config = config

    @property
    def name(self) -> str:
        return "manimux"

    @property
    def control_mode(self) -> str:
        return "managed"

    @property
    def required_sampling_modes(self) -> frozenset[str]:
        return frozenset({"default"})

    def reset(self) -> None:
        return None

    def build_submission(
        self,
        *,
        session_id: str,
        request_seq: int,
        now_ns: int,
        snapshot: ObservationSnapshot,
        adapter: PolicyAdapter,
        timeline: ActionTimeline,
        request_state: RequestState,
        runtime_state: RuntimeState,
    ) -> InferenceSubmission | None:
        schedule = self._config.execution.inference_schedule
        if schedule == "serial":
            if (runtime_state != RuntimeState.RUNNING or request_state.in_flight
                    or timeline.remaining_ns(now_ns) > 0):
                return None
        else:
            refill_ns = int(self._config.execution.refill_threshold_s * 1_000_000_000)
            request_expired = now_ns > request_state.last_deadline_ns
            request_ready = (
                not request_state.in_flight
                if schedule == "single_inflight"
                else request_state.last_submitted_seq < 0 or request_expired
            )
            if timeline.remaining_ns(now_ns) >= refill_ns or not request_ready:
                return None
        deadline_ns = now_ns + int(self._config.policy.timeout_s * 1_000_000_000)
        request = InferenceRequest(
            session_id=session_id,
            request_seq=request_seq,
            observation_time_ns=snapshot.state.monotonic_ns,
            deadline_ns=deadline_ns,
            observation=adapter.build_observation(snapshot),
            instruction=self._config.run.task,
        )
        return InferenceSubmission(request=request)

    def commit_settings(
        self,
        *,
        response: InferenceResponse,
        measured: GroupVector,
        last_command: GroupVector,
    ) -> CommitSettings:
        del response, last_command
        return CommitSettings(
            current_command=copy_group_vector(measured),
            blend_steps=self._config.execution.blend_steps,
            anchor_source="measured_state",
        )

    def prepare_chunk(
        self,
        *,
        chunk: ActionChunk,
        response: InferenceResponse,
        now_ns: int,
    ) -> ActionChunk:
        del response, now_ns
        return chunk

    def on_plan_accepted(
        self,
        *,
        chunk: ActionChunk,
        result: CommitResult,
        response: InferenceResponse,
        now_ns: int,
    ) -> dict[str, object]:
        del chunk, response, now_ns
        return {"trimmed_steps": result.trimmed_steps}

    def on_response_rejected(self, response: InferenceResponse) -> None:
        del response

    def take_runtime_events(self, *, step: int) -> list[tuple[str, dict[str, object]]]:
        del step
        return []

    def on_tick(self, *, steps: int, loop_ms: float, control_dt_ns: int) -> None:
        del steps, loop_ms, control_dt_ns


InferenceStrategyFactory = Callable[[ManiMuxConfig], InferenceStrategy]


def _rtc_strategy_factory(config: ManiMuxConfig) -> InferenceStrategy:
    from manimux.runtime.rtc.strategy import RtcInferenceStrategy

    return RtcInferenceStrategy(config)


def _act_temporal_ensemble_factory(config: ManiMuxConfig) -> InferenceStrategy:
    from manimux.runtime.temporal_ensemble import ACTTemporalEnsembleStrategy

    return ACTTemporalEnsembleStrategy(config)


def _aac_strategy_factory(config: ManiMuxConfig) -> InferenceStrategy:
    from manimux.runtime.aac import AacInferenceStrategy

    return AacInferenceStrategy(config)


def _paint_strategy_factory(config: ManiMuxConfig) -> InferenceStrategy:
    from manimux.runtime.paint import PaintInferenceStrategy

    return PaintInferenceStrategy(config)


def _autohorizon_strategy_factory(config: ManiMuxConfig) -> InferenceStrategy:
    from manimux.runtime.autohorizon import AutoHorizonInferenceStrategy

    return AutoHorizonInferenceStrategy(config)


def _dvac_strategy_factory(config: ManiMuxConfig) -> InferenceStrategy:
    from manimux.runtime.dvac import DvacInferenceStrategy

    return DvacInferenceStrategy(config)


_STRATEGY_BUILTINS: dict[str, InferenceStrategyFactory] = {
    "manimux": DefaultChunkStrategy,
    "rtc": _rtc_strategy_factory,
    "act_temporal_ensemble": _act_temporal_ensemble_factory,
    "aac": _aac_strategy_factory,
    "paint": _paint_strategy_factory,
    "autohorizon": _autohorizon_strategy_factory,
    "dvac": _dvac_strategy_factory,
}


def build_inference_strategy(config: ManiMuxConfig) -> InferenceStrategy:
    factory = load_plugin(
        config.execution.runtime,
        group="manimux.inference_strategies",
        builtins=_STRATEGY_BUILTINS,
    )
    return factory(config)


def prepare_strategy_chunk(
    strategy: InferenceStrategy,
    *,
    chunk: ActionChunk,
    response: InferenceResponse,
    now_ns: int,
) -> ActionChunk:
    """Run an optional chunk transform without breaking older strategy plugins."""
    method = getattr(strategy, "prepare_chunk", None)
    if not callable(method):
        return chunk
    prepared = method(chunk=chunk, response=response, now_ns=now_ns)
    if not isinstance(prepared, ActionChunk):
        raise TypeError("inference strategy prepare_chunk must return ActionChunk")
    return prepared
