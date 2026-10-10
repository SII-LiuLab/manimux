"""BID backward coherence on serial or time-aligned asynchronous action chunks."""

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np

from manimux.runtime.candidates import align_prediction, decode_candidates
from manimux.runtime.inference import InferenceSubmission
from manimux.runtime.safety import RuntimeState
from manimux.runtime.synchronous import SynchronousChunkStrategy
from manimux.runtime.timeline import first_future_step
from manimux.types import ActionChunk, InferenceRequest, copy_action_chunk


def select_backward(candidates, previous, *, replan_steps: int, rho: float):
    """Minimize sum(rho**t * L2(new[t] - previous[t + K])) over H-K rows."""
    rows = np.asarray(candidates, dtype=np.float64)
    if rows.ndim != 3 or min(rows.shape) <= 0 or not np.isfinite(rows).all():
        raise ValueError("BID candidates must be finite (N, H, D) arrays")
    if type(replan_steps) is not int or not 0 < replan_steps < rows.shape[1]:
        raise ValueError("BID requires 0 < replan_steps < candidate horizon")
    if not 0 < rho <= 1:
        raise ValueError("BID rho must be in (0, 1]")
    scores = np.zeros(len(rows), dtype=np.float64)
    if previous is not None:
        previous = np.asarray(previous, dtype=np.float64)
        if previous.shape != rows.shape[1:] or not np.isfinite(previous).all():
            raise ValueError("BID previous chunk must match the candidate horizon and dimensions")
        overlap = rows.shape[1] - replan_steps
        distances = np.linalg.norm(rows[:, :overlap] - previous[None, replan_steps:], axis=-1)
        scores = np.sum(distances * rho ** np.arange(overlap), axis=1)
    return int(np.argmin(scores)), scores


@dataclass(slots=True)
class BidInferenceRequest(InferenceRequest):
    bid_num_samples: int = 16
    aac_horizon: dict | None = None


class BidBackwardStrategy(SynchronousChunkStrategy):
    """Select decoded candidates against the last accepted full prediction."""

    name = "bid_backward"
    request_type = BidInferenceRequest
    label = "BID backward-only"
    discard_plans_while_paused = True
    hold_last_step = True

    def __init__(self, config):
        super().__init__(config)
        self._settings = config["inference"]["bid"]
        self._groups = tuple(config["robot"]["group_dims"])
        self.reset()

    @property
    def required_sampling_modes(self) -> frozenset[str]:
        # Reuse AAC's independent candidates, with optional AAC horizon selection.
        return frozenset({"aac"})

    def reset(self) -> None:
        self._previous = None
        self._previous_time_ns = 0
        self._previous_first_step = 0
        self._candidates: list[ActionChunk] = []
        self._pending: ActionChunk | None = None
        self._previous_execution_steps = self._settings["replan_policy_steps"]
        self._aac_metadata = None

    @property
    def _serial(self) -> bool:
        return self._config["inference"]["inference_schedule"] == "serial"

    def build_submission(self, **kwargs) -> InferenceSubmission | None:
        if kwargs["runtime_state"] != RuntimeState.RUNNING:
            return None
        # RequestScheduler owns serial completion, capacity, refill and rate admission.
        return self.build_warmup_submission(**{
            key: kwargs[key]
            for key in ("session_id", "request_seq", "now_ns", "snapshot", "adapter")
        })

    def request_options(self) -> dict:
        options = {"bid_num_samples": self._settings["num_samples"]}
        if self._settings["execution_horizon"] == "aac":
            options["aac_horizon"] = {
                **self._config["inference"]["aac"],
                "robot_config": str(self._config["robot"]["config"]),
            }
        return options

    def build_warmup_submission(
        self, *, session_id, request_seq, now_ns, snapshot, adapter, conditioned=False,
    ) -> InferenceSubmission:
        return self._make_submission(
            session_id=session_id, request_seq=request_seq,
            now_ns=now_ns, snapshot=snapshot, adapter=adapter,
        )

    def decode_action(self, raw, context, *, adapter) -> ActionChunk:
        candidates = raw.get("bid_candidates") if isinstance(raw, Mapping) else None
        chunks = decode_candidates(
            candidates, context, adapter=adapter, config=self._config,
            count=self._settings["num_samples"],
        )
        self._aac_metadata = raw.get("aac")
        if self._settings["execution_horizon"] == "aac":
            steps = (
                self._aac_metadata.get("execution_steps")
                if isinstance(self._aac_metadata, Mapping) else None
            )
            if type(steps) is not int or not 1 <= steps <= chunks[0].horizon_steps:
                raise ValueError("BID AAC horizon requires valid execution_steps")
        self._candidates = chunks
        # Selection waits until commit, after all candidate decoding has finished.
        return chunks[0]

    def select_chunk(self, *, chunk, now_ns) -> ActionChunk:
        if not self._candidates or self._candidates[0].request_seq != chunk.request_seq:
            raise ValueError("BID selection has no matching decoded candidates")
        rows = np.stack([self._rows(candidate) for candidate in self._candidates])
        rho = self._settings["rho"]
        if self._serial and self._previous_execution_steps < chunk.horizon_steps:
            index, scores = select_backward(
                rows, self._previous, replan_steps=self._previous_execution_steps,
                rho=rho,
            )
            overlap = 0 if self._previous is None else (
                chunk.horizon_steps - self._previous_execution_steps
            )
        elif self._serial:
            index, overlap, scores = 0, 0, np.zeros(len(rows))
        else:
            # Score only future rows, on the same clock used by ActionTimeline.commit.
            start = self._source_start(chunk, now_ns)
            origin = self._source_time(chunk, now_ns)
            times = origin + np.arange(start, chunk.horizon_steps) * chunk.dt_ns
            mask = np.zeros(len(times), dtype=bool)
            if self._previous is not None:
                mask, reference = align_prediction(
                    self._previous, self._previous_time_ns, times, chunk.dt_ns,
                    first_step=self._previous_first_step,
                )
            overlap = int(mask.sum())
            scores = np.zeros(len(rows))
            if overlap:
                distances = np.linalg.norm(rows[:, start:][:, mask] - reference, axis=-1)
                scores = np.sum(distances * rho ** np.arange(overlap), axis=1)
            index = int(np.argmin(scores))
        chosen = copy_action_chunk(self._candidates[index])
        if self._settings["execution_horizon"] == "aac":
            chosen.metadata["aac"] = dict(self._aac_metadata)
        for field in ("decode_ms", "observation_to_commit_ms"):
            if field in chunk.metadata:
                chosen.metadata[field] = chunk.metadata[field]
        chosen.metadata["bid"] = {
            "candidate_index": index, "num_samples": len(rows),
            "overlap_steps": overlap,
            "backward_costs": scores.tolist(), "rho": rho,
            "metric_space": "absolute_joint_position",
            "alignment": "fixed_prefix" if self._serial else "execution_time",
        }
        self._pending = copy_action_chunk(chosen)
        return chosen

    def _source_start(self, chunk, now_ns):
        if self._config["inference"]["action_start_mode"] == "first_step_when_ready":
            return 0
        return first_future_step(chunk.observation_time_ns, chunk.dt_ns, now_ns)

    def _source_time(self, chunk, now_ns):
        if self._config["inference"]["action_start_mode"] == "first_step_when_ready":
            return now_ns
        return chunk.observation_time_ns

    def _rows(self, chunk):
        return np.concatenate([chunk.groups[name] for name in self._groups], axis=1)

    def prepare_chunk(self, *, chunk, response, now_ns) -> ActionChunk:
        # Keep K future rows, not K rows counted from an already stale observation.
        steps = self._source_start(chunk, now_ns) + self._settings["replan_policy_steps"]
        if self._settings["execution_horizon"] == "aac":
            # AAC bounds source rows; latency must not extend the selected horizon.
            steps = chunk.metadata["aac"]["execution_steps"]
        return replace(
            chunk, groups={name: rows[:steps].copy() for name, rows in chunk.groups.items()},
        )

    def on_plan_accepted(self, *, chunk, result, response, now_ns) -> dict:
        if self._pending is None or self._pending.request_seq != response.request_seq:
            raise ValueError("BID accepted plan has no matching selected candidate")
        self._previous = self._rows(self._pending).copy()
        self._previous_time_ns = self._source_time(self._pending, now_ns)
        self._previous_first_step = result.time_trimmed_steps
        self._previous_execution_steps = chunk.horizon_steps
        self._pending = None
        self._candidates = []
        metadata = chunk.metadata["bid"]
        return {
            **super().on_plan_accepted(
                chunk=chunk, result=result, response=response, now_ns=now_ns,
            ),
            "bid_candidate_index": metadata["candidate_index"],
            "bid_overlap_steps": metadata["overlap_steps"],
            "bid_backward_cost": metadata["backward_costs"][metadata["candidate_index"]],
        }

    def on_response_rejected(self, response) -> None:
        if self._candidates and self._candidates[0].request_seq == response.request_seq:
            self._candidates = []
        if self._pending is not None and self._pending.request_seq == response.request_seq:
            self._pending = None


def bid_parameters(**options) -> dict:
    values = {
        "num_samples": 16, "rho": 0.9, "replan_policy_steps": 5,
        "execution_horizon": "fixed", **options,
    }
    if set(values) != {"num_samples", "rho", "replan_policy_steps", "execution_horizon"}:
        raise ValueError("unsupported BID parameter")
    if values["execution_horizon"] not in {"fixed", "aac"}:
        raise ValueError("BID execution_horizon must be fixed or aac")
    if type(values["num_samples"]) is not int or values["num_samples"] < 2:
        raise ValueError("BID num_samples must be an integer greater than one")
    if type(values["replan_policy_steps"]) is not int or values["replan_policy_steps"] < 1:
        raise ValueError("BID replan_policy_steps must be a positive integer")
    rho = values["rho"]
    if isinstance(rho, bool) or not isinstance(rho, int | float) or not 0 < rho <= 1:
        raise ValueError("BID rho must be in (0, 1]")
    return values
