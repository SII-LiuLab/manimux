"""ACT temporal ensembling adapted to ManiMux's asynchronous policy worker.

The aggregation formula follows the official ACT implementation at commit
742c753c0d4a5d87076c8f69e5628c79a8cc5488. ManiMux only parameterizes how
often a new chunk is requested; ``query_interval_policy_steps=1`` is ACT's original
temporal-aggregation cadence.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from manimux.policies.base import action_interval
from manimux.policy_adapter.base import PolicyAdapter
from manimux.runtime.candidates import align_prediction
from manimux.runtime.inference import (
    CommitSettings,
    DefaultChunkStrategy,
    InferenceSubmission,
    RequestState,
)
from manimux.runtime.safety import RuntimeState
from manimux.runtime.timeline import ActionTimeline, CommitResult
from manimux.types import (
    ActionChunk,
    FloatArray,
    GroupTrajectory,
    GroupVector,
    InferenceRequest,
    InferenceResponse,
    ObservationSnapshot,
)


@dataclass(frozen=True, slots=True)
class _StoredChunk:
    start_step: int
    end_step: int
    groups: GroupTrajectory
    request_seq: int


class ACTTemporalEnsembler:
    """Apply ACT's exponential weighting on absolute, overlapping timesteps."""

    def __init__(self, coefficient: float) -> None:
        if coefficient < 0:
            raise ValueError("temporal ensemble coefficient must be non-negative")
        self._coefficient = coefficient
        self.reset()

    def reset(self) -> None:
        self._origin_time_ns: int | None = None
        self._dt_ns: int | None = None
        self._chunks: list[_StoredChunk] = []
        self._timed_chunks: list[tuple[int, int, ActionChunk]] = []
        self.last_contributor_counts: tuple[int, ...] = ()

    def aggregate(self, chunk: ActionChunk) -> ActionChunk:
        start_step = self._start_step(chunk)
        stored = _StoredChunk(
            start_step=start_step,
            end_step=start_step + chunk.horizon_steps - 1,
            groups={name: values.copy() for name, values in chunk.groups.items()},
            request_seq=chunk.request_seq,
        )
        self._chunks.append(stored)
        self._chunks = [item for item in self._chunks if item.end_step >= start_step]

        output: dict[str, list[FloatArray]] = {name: [] for name in chunk.groups}
        contributor_counts: list[int] = []
        for offset in range(chunk.horizon_steps):
            target_step = start_step + offset
            contributors = [
                item for item in self._chunks if item.start_step <= target_step <= item.end_step
            ]
            contributor_counts.append(len(contributors))
            weights = self._weights(len(contributors))
            for name in output:
                predictions = np.stack(
                    [item.groups[name][target_step - item.start_step] for item in contributors]
                )
                output[name].append(np.sum(predictions * weights[:, None], axis=0))

        self.last_contributor_counts = tuple(contributor_counts)
        return ActionChunk(
            plan_id=chunk.plan_id,
            request_seq=chunk.request_seq,
            observation_time_ns=chunk.observation_time_ns,
            created_time_ns=chunk.created_time_ns,
            action_space=chunk.action_space,
            dt_ns=chunk.dt_ns,
            groups={name: np.stack(values) for name, values in output.items()},
        )

    def discard(self, request_seq: int) -> None:
        self._chunks = [item for item in self._chunks if item.request_seq != request_seq]
        self._timed_chunks = [
            item for item in self._timed_chunks if item[2].request_seq != request_seq
        ]

    def aggregate_aligned(self, chunk, *, origin_ns, first_step):
        """Compose strategies on exact action times, retaining chunk metadata."""
        times = origin_ns + np.arange(chunk.horizon_steps) * chunk.dt_ns
        self._timed_chunks = [
            item for item in self._timed_chunks
            if item[0] + (item[2].horizon_steps - 1) * item[2].dt_ns >= times[0]
        ]
        groups = {name: rows.copy() for name, rows in chunk.groups.items()}
        counts = np.ones(chunk.horizon_steps, dtype=int)
        contributions = []
        for origin, start, previous in self._timed_chunks:
            if previous.dt_ns != chunk.dt_ns:
                raise ValueError("temporal ensemble requires constant action dt")
            aligned = {}
            for name, rows in previous.groups.items():
                future_mask, aligned[name] = align_prediction(
                    rows, origin, times[first_step:], chunk.dt_ns, first_step=start,
                )
            mask = np.zeros(len(times), dtype=bool)
            mask[first_step:] = future_mask
            contributions.append((mask, aligned))
            counts += mask
        for step in range(first_step, chunk.horizon_steps):
            for name in groups:
                predictions = [
                    rows[name][np.count_nonzero(mask[:step])]
                    for mask, rows in contributions if mask[step]
                ] + [chunk.groups[name][step]]
                groups[name][step] = np.sum(
                    np.stack(predictions) * self._weights(len(predictions))[:, None], axis=0,
                )
        self._timed_chunks.append((origin_ns, first_step, replace(
            chunk, groups={name: rows.copy() for name, rows in chunk.groups.items()},
        )))
        self.last_contributor_counts = tuple(map(int, counts[first_step:]))
        return replace(chunk, groups=groups, metadata=dict(chunk.metadata))

    def _start_step(self, chunk: ActionChunk) -> int:
        if self._origin_time_ns is None:
            self._origin_time_ns = chunk.observation_time_ns
            self._dt_ns = chunk.dt_ns
            return 0
        if chunk.dt_ns != self._dt_ns:
            raise ValueError(
                "ACT temporal ensembling requires a constant action dt: "
                f"expected {self._dt_ns}, got {chunk.dt_ns}"
            )
        elapsed_ns = chunk.observation_time_ns - self._origin_time_ns
        if elapsed_ns < 0:
            raise ValueError("ACT temporal ensembling received out-of-order observation time")
        return int(round(elapsed_ns / self._dt_ns))

    def _weights(self, count: int) -> FloatArray:
        if count <= 0:
            raise ValueError("temporal ensemble requires at least one prediction")
        weights = np.exp(-self._coefficient * np.arange(count, dtype=np.float64))
        return np.asarray(weights / weights.sum(), dtype=np.float64)


class ACTTemporalEnsembleStrategy:
    """Run ACT aggregation without blocking ManiMux's control loop."""

    def __init__(self, config: dict) -> None:
        self._config = config
        settings = config["inference"]["temporal_ensemble"]
        self._query_interval_steps = settings["query_interval_policy_steps"]
        self._query_interval_ns = int(
            action_interval(config["policy"])
            * settings["query_interval_policy_steps"]
            * 1_000_000_000
        )
        self._ensembler = ACTTemporalEnsembler(settings["coefficient"])
        self._next_query_ns: int | None = None
        self._pending_ensemble_seq: int | None = None

    @property
    def name(self) -> str:
        return "act_temporal_ensemble"

    @property
    def control_mode(self) -> str:
        return "managed"

    @property
    def required_sampling_modes(self) -> frozenset[str]:
        return frozenset({"default"})

    def reset(self) -> None:
        self._ensembler.reset()
        self._next_query_ns = None
        self._pending_ensemble_seq = None

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
        del timeline, runtime_state
        if self._next_query_ns is not None and now_ns < self._next_query_ns:
            return None

        deadline_ns = now_ns + int(self._config["policy"]["timeout_s"] * 1_000_000_000)
        request = InferenceRequest(
            session_id=session_id,
            request_seq=request_seq,
            observation_time_ns=snapshot.state.monotonic_ns,
            deadline_ns=deadline_ns,
            observation=adapter.build_observation(snapshot),
            instruction=self._config["run"]["task"],
        )
        self._next_query_ns = now_ns + self._query_interval_ns
        return InferenceSubmission(
            request=request,
            event_fields={
                "query_interval_steps": self._query_interval_steps,
                "query_interval_ms": self._query_interval_ns / 1_000_000,
            },
        )

    def prepare_chunk(
        self,
        *,
        chunk: ActionChunk,
        response: InferenceResponse,
        now_ns: int,
    ) -> ActionChunk:
        del now_ns
        if self._config["inference"]["inference_schedule"] == "multi_inflight":
            self._pending_ensemble_seq = response.request_seq
        return self._ensembler.aggregate(chunk)

    def decode_handoff(self, *, response: InferenceResponse) -> bool:
        """response: model output about to be decoded; this strategy keeps its own trajectory."""
        del response
        return False

    def commit_settings(
        self,
        *,
        response: InferenceResponse,
        measured: GroupVector,
        last_command: GroupVector,
    ) -> CommitSettings:
        del response, measured
        return CommitSettings(
            current_command={name: values.copy() for name, values in last_command.items()},
            blend_steps=self._config["inference"]["blend_policy_steps"],
            anchor_source="act_temporal_ensemble",
        )

    def on_plan_accepted(
        self,
        *,
        chunk: ActionChunk,
        result: CommitResult,
        response: InferenceResponse,
        now_ns: int,
    ) -> dict[str, object]:
        del chunk, response, now_ns
        self._pending_ensemble_seq = None
        counts = self._ensembler.last_contributor_counts
        return {
            "trimmed_steps": result.trimmed_steps,
            "temporal_ensemble_min_contributors": min(counts, default=0),
            "temporal_ensemble_max_contributors": max(counts, default=0),
        }

    def on_response_rejected(self, response: InferenceResponse) -> None:
        if response.request_seq == self._pending_ensemble_seq:
            self._ensembler.discard(response.request_seq)
            self._pending_ensemble_seq = None

    def take_runtime_events(self, *, step: int) -> list[tuple[str, dict[str, object]]]:
        del step
        return []

    def on_tick(self, *, steps: int, loop_ms: float, control_dt_ns: int) -> None:
        del steps, loop_ms, control_dt_ns


def temporal_ensemble_parameters(**options) -> dict:
    """保留 ACT 时间融合的权重系数与查询步数。"""

    if "query_interval_steps" in options:
        raise ValueError("unsupported temporal ensemble field: query_interval_steps")
    values = {
        "enabled": False,
        "coefficient": 0.01,
        "query_interval_policy_steps": 1,
        **options,
    }
    return values


class SerialACTStrategy(DefaultChunkStrategy):
    """Serial request admission with a prefix cropped after full-horizon ACT fusion."""

    hold_last_step = True

    def prepare_chunk(self, *, chunk, response, now_ns):
        steps = self._config["inference"]["chunk_policy_steps"] or chunk.horizon_steps
        return replace(
            chunk, groups={name: rows[:steps].copy() for name, rows in chunk.groups.items()},
        )


class EnsembleStrategy:
    """Optional ACT fusion around one existing algorithm and its lifecycle."""

    discard_plans_while_paused = True

    def __init__(self, strategy, config):
        self._strategy = strategy
        self._config = config
        self._ensembler = ACTTemporalEnsembler(
            config["inference"]["temporal_ensemble"]["coefficient"],
        )

    def __getattr__(self, name):
        return getattr(self._strategy, name)

    def reset(self):
        self._strategy.reset()
        self._ensembler.reset()

    def decode_action(self, raw, context, *, adapter):
        from manimux.runtime.inference import decode_strategy_action

        chunk = decode_strategy_action(self._strategy, adapter, raw, context)
        if (
            chunk.action_space != "joint_position" or chunk.source_offset_steps
            or chunk.hold_from_step or chunk.handoff or chunk.runtime_trajectory
        ):
            raise ValueError("ACT composition requires complete absolute-joint source rows")
        return chunk

    def select_chunk(self, *, chunk, now_ns):
        select = getattr(self._strategy, "select_chunk", None)
        return select(chunk=chunk, now_ns=now_ns) if callable(select) else chunk

    def prepare_chunk(self, *, chunk, response, now_ns):
        from manimux.runtime.timeline import first_future_step

        # Fixed BID and serial ACT retain full predictions before prefix execution.
        # Adaptive horizon methods limit each prediction's ensemble contribution.
        bid = self.name == "bid_backward"
        adaptive_bid = bid and self._config["inference"]["bid"]["execution_horizon"] == "aac"
        full_prediction = (bid and not adaptive_bid) or self.name == "act_temporal_ensemble"
        if not full_prediction:
            chunk = self._strategy.prepare_chunk(chunk=chunk, response=response, now_ns=now_ns)
        origin = (
            now_ns if self._config["inference"]["action_start_mode"] == "first_step_when_ready"
            else chunk.observation_time_ns
        )
        start = first_future_step(origin, chunk.dt_ns, now_ns)
        if start >= chunk.horizon_steps:
            return chunk  # Timeline rejects it; it must not enter ensemble history.
        chunk = self._ensembler.aggregate_aligned(chunk, origin_ns=origin, first_step=start)
        if full_prediction:
            chunk = self._strategy.prepare_chunk(chunk=chunk, response=response, now_ns=now_ns)
        return chunk

    def on_plan_accepted(self, **kwargs):
        fields = self._strategy.on_plan_accepted(**kwargs)
        counts = self._ensembler.last_contributor_counts
        return {
            **fields,
            "temporal_ensemble_min_contributors": min(counts, default=0),
            "temporal_ensemble_max_contributors": max(counts, default=0),
        }

    def on_response_rejected(self, response):
        self._ensembler.discard(response.request_seq)
        self._strategy.on_response_rejected(response)
