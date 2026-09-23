"""ACT temporal ensembling adapted to ManiMux's asynchronous policy worker.

The aggregation formula follows the official ACT implementation at commit
742c753c0d4a5d87076c8f69e5628c79a8cc5488. ManiMux only parameterizes how
often a new chunk is requested; ``query_interval_policy_steps=1`` is ACT's original
temporal-aggregation cadence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from manimux.policies.base import action_interval
from manimux.policy_adapter.base import PolicyAdapter
from manimux.runtime.inference import (
    CommitSettings,
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
        self.last_contributor_counts: tuple[int, ...] = ()

    def aggregate(self, chunk: ActionChunk) -> ActionChunk:
        start_step = self._start_step(chunk)
        stored = _StoredChunk(
            start_step=start_step,
            end_step=start_step + chunk.horizon_steps - 1,
            groups={name: values.copy() for name, values in chunk.groups.items()},
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
        if request_state.in_flight:
            return None
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
        del response, now_ns
        return self._ensembler.aggregate(chunk)

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
            blend_steps=0,
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
        counts = self._ensembler.last_contributor_counts
        return {
            "trimmed_steps": result.trimmed_steps,
            "temporal_ensemble_min_contributors": min(counts, default=0),
            "temporal_ensemble_max_contributors": max(counts, default=0),
        }

    def on_response_rejected(self, response: InferenceResponse) -> None:
        del response

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
        "coefficient": 0.01,
        "query_interval_policy_steps": 1,
        **options,
    }
    return values
