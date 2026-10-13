from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from manimux.plugins import load_plugin
from manimux.policy_adapter.base import PolicyAdapter
from manimux.runtime.safety import RuntimeState
from manimux.runtime.timeline import ActionTimeline, CommitResult
from manimux.types import (
    ActionChunk,
    ActionContext,
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
    multi_flight: bool = False
    # Shared process-decode forecast at submission time; zero for inline decoding.
    expected_decode_ns: int = 0


@dataclass(frozen=True, slots=True)
class InferenceSubmission:
    request: InferenceRequest
    event_fields: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CommitSettings:
    current_command: GroupVector
    blend_steps: int
    anchor_source: str


@runtime_checkable
class StrategyActionDecoder(Protocol):
    """Optional adapter-backed decode stage, shared by execution and warmup.

    Return one canonical full chunk; a candidate strategy may retain the other
    decoded candidates keyed by request_seq. This does not accept a plan or
    advance reference history. Reject/reset must discard pending candidates.
    """

    def decode_action(
        self, raw: object, context: ActionContext, *, adapter: PolicyAdapter,
    ) -> ActionChunk: ...


@runtime_checkable
class StrategyChunkSelector(Protocol):
    """Optional selection after decoding, immediately before preparation/commit.

    now_ns is the same timestamp used by prepare_chunk and Timeline.commit.
    Warmup may select a preview without accepting it. Only on_plan_accepted may
    advance accepted reference history; rejection/reset discard pending state.
    """

    def select_chunk(self, *, chunk: ActionChunk, now_ns: int) -> ActionChunk: ...


EnsembleScope = Literal["full_prediction", "execution_prefix"]


@runtime_checkable
class EnsembleSource(Protocol):
    """Declare which rows may contribute to temporal ensembling.

    full_prediction: fuse/store the full prediction, then prepare its execution
    prefix. execution_prefix: prepare/validate first, then fuse/store only those
    rows. prepare_chunk is called once in either case. Scope must stay fixed for
    the strategy's lifetime; changing configuration requires a new strategy.
    """

    @property
    def ensemble_scope(self) -> EnsembleScope: ...


class InferenceStrategy(Protocol):
    """Core lifecycle; decode/selection/fusion use the optional protocols above."""

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

    def decode_handoff(self, *, response: InferenceResponse) -> bool: ...

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
    """Shared Serial/Async chunk scheduling, isolated from the control loop."""

    def __init__(self, config: dict) -> None:
        self._config = config

    @property
    def discard_plans_while_paused(self) -> bool:
        return self._config["inference"]["inference_schedule"] == "multi_inflight"

    @property
    def name(self) -> str:
        return "serial" if self._config["inference"]["inference_schedule"] == "serial" else "async"

    @property
    def control_mode(self) -> str:
        return "managed"

    @property
    def required_sampling_modes(self) -> frozenset[str]:
        return frozenset({"default"})

    def reset(self) -> None:
        pass

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
        # RequestScheduler owns transport admission and ordinary refill timing.
        deadline_ns = now_ns + int(self._config["policy"]["timeout_s"] * 1_000_000_000)
        request = InferenceRequest(
            session_id=session_id,
            request_seq=request_seq,
            observation_time_ns=snapshot.state.monotonic_ns,
            deadline_ns=deadline_ns,
            observation=adapter.build_observation(snapshot),
            instruction=self._config["run"]["task"],
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
            blend_steps=self._config["inference"]["blend_policy_steps"],
            anchor_source="measured_state",
        )

    def decode_handoff(self, *, response: InferenceResponse) -> bool:
        """response: model output about to be decoded; True requests a waypoint handoff."""
        del response
        return self._config["inference"]["handoff"] == "waypoint"

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


InferenceStrategyFactory = Callable[[dict], InferenceStrategy]


_STRATEGY_BUILTINS: dict[str, InferenceStrategyFactory | str] = {
    "async": DefaultChunkStrategy,
    "serial": DefaultChunkStrategy,
    "rtc": "manimux.runtime.rtc.strategy:RtcInferenceStrategy",
    "train_time_rtc": "manimux.runtime.train_time_rtc:TrainTimeRtcInferenceStrategy",
    "act_temporal_ensemble": "manimux.runtime.temporal_ensemble:ACTTemporalEnsembleStrategy",
    "aac": "manimux.runtime.aac:AacInferenceStrategy",
    "bid_backward": "manimux.runtime.bid:BidBackwardStrategy",
    "paint": "manimux.runtime.paint:PaintInferenceStrategy",
    "autohorizon": "manimux.runtime.autohorizon:AutoHorizonInferenceStrategy",
}

INFERENCE_ALGORITHMS = frozenset(_STRATEGY_BUILTINS)


def build_inference_strategy(config: dict) -> InferenceStrategy:
    factory = load_plugin(
        config["inference"].get("strategy") or config["inference"]["algorithm"],
        group="manimux.inference_strategies",
        builtins=_STRATEGY_BUILTINS,
    )
    settings = config["inference"]
    ensemble = (settings.get("temporal_ensemble") or {}).get("enabled", False)
    serial_act = (
        settings["algorithm"] == "act_temporal_ensemble"
        and settings["inference_schedule"] == "serial"
    )
    if serial_act:
        from manimux.runtime.temporal_ensemble import SerialACTStrategy

        factory = SerialACTStrategy
    elif settings["algorithm"] == "aac" and (
        settings["inference_schedule"] != "serial" or ensemble
    ):
        from manimux.runtime.aac import CandidateAacStrategy

        factory = CandidateAacStrategy
    strategy = factory(config)
    if serial_act or (ensemble and settings["algorithm"] != "act_temporal_ensemble"):
        from manimux.runtime.temporal_ensemble import EnsembleStrategy

        strategy = EnsembleStrategy(strategy, config)
    return strategy


def build_warmup_submission(
    strategy: InferenceStrategy,
    config: dict,
    *,
    session_id: str,
    request_seq: int,
    now_ns: int,
    snapshot: ObservationSnapshot,
    adapter: PolicyAdapter,
    conditioned: bool = False,
) -> InferenceSubmission:
    """Build an inference-only request without advancing the formal scheduler."""
    method = getattr(strategy, "build_warmup_submission", None)
    if callable(method):
        return method(
            session_id=session_id,
            request_seq=request_seq,
            now_ns=now_ns,
            snapshot=snapshot,
            adapter=adapter,
            conditioned=conditioned,
        )
    return InferenceSubmission(
        request=InferenceRequest(
            session_id=session_id,
            request_seq=request_seq,
            observation_time_ns=snapshot.state.monotonic_ns,
            deadline_ns=now_ns + int(config["policy"]["timeout_s"] * 1_000_000_000),
            observation=adapter.build_observation(snapshot),
            instruction=config["run"]["task"],
        ),
        event_fields={"conditioned": False},
    )


def seed_strategy_warmup(
    strategy: InferenceStrategy,
    *,
    latency_ns: Sequence[int],
) -> None:
    """Pass successful E2E warmup durations to an optional strategy estimator."""
    method = getattr(strategy, "seed_warmup", None)
    if callable(method):
        method(latency_ns=latency_ns)


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
    return method(chunk=chunk, response=response, now_ns=now_ns)


def decode_strategy_action(
    strategy: InferenceStrategy, adapter: PolicyAdapter, raw: object, context: ActionContext,
) -> ActionChunk:
    """Let candidate selectors reuse adapter decoding; ordinary strategies decode once."""
    if isinstance(strategy, StrategyActionDecoder):
        return strategy.decode_action(raw, context, adapter=adapter)
    return adapter.decode_action(raw, context)
