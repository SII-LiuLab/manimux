from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from manimux.policies.aac import AacPreviousAction, select_candidate
from manimux.runtime.candidates import align_prediction, decode_candidates
from manimux.runtime.synchronous import SynchronousChunkStrategy
from manimux.runtime.timeline import CommitResult, first_future_step
from manimux.types import (
    ActionChunk,
    InferenceRequest,
    InferenceResponse,
    copy_action_chunk,
)


@dataclass(slots=True)
class AacInferenceRequest(InferenceRequest):
    aac_robot_config: str | None = None
    aac_num_samples: int = 20
    aac_motion_threshold: float = 3.0
    aac_ee_stats_path: str | None = None
    aac_chunk_id_selector: str = "0"
    aac_backward_beta: float = 0.99
    aac_defer_selection: bool = False


class AacInferenceStrategy(SynchronousChunkStrategy):
    """Official AAC synchronous cadence around an XPolicy multi-sample hook."""

    name = "aac"
    request_type = AacInferenceRequest
    label = "AAC"
    horizon_metadata = "aac"

    def request_options(self) -> dict:
        aac = self._config["inference"]["aac"]
        return {
            "aac_robot_config": str(self._config["robot"]["config"]),
            "aac_num_samples": aac["num_samples"],
            "aac_motion_threshold": aac["motion_threshold"],
            "aac_ee_stats_path": aac["ee_stats_path"],
            "aac_chunk_id_selector": aac["chunk_id_selector"],
            "aac_backward_beta": aac["backward_beta"],
        }

    def submission_fields(self) -> dict:
        aac = self._config["inference"]["aac"]
        return {
            key: aac[key]
            for key in ("num_samples", "motion_threshold", "ee_stats_path", "chunk_id_selector")
        }

    def on_plan_accepted(
        self,
        *,
        chunk: ActionChunk,
        result: CommitResult,
        response: InferenceResponse,
        now_ns: int,
    ) -> dict[str, object]:
        fields = super().on_plan_accepted(
            chunk=chunk, result=result, response=response, now_ns=now_ns
        )
        raw = response.raw_action
        metadata = raw.get("aac") if isinstance(raw, Mapping) else None
        if isinstance(metadata, Mapping):
            for key in ("chunk_id", "entropy_elbow", "motion_floor"):
                value = metadata.get(key)
                if isinstance(value, int | float) and not isinstance(value, bool):
                    fields[key] = value
        return fields


def aac_parameters(**options) -> dict:
    """保留自适应分块的采样数量、运动阈值与选择规则。"""

    values = {
        "num_samples": 20,
        "motion_threshold": 3.0,
        "ee_stats_path": None,
        "chunk_id_selector": "0",
        "backward_beta": 0.99,
        **options,
    }
    return values


class CandidateAacStrategy(AacInferenceStrategy):
    """Select at commit so rejected asynchronous replies cannot advance history."""

    discard_plans_while_paused = True

    def __init__(self, config):
        super().__init__(config)
        self.reset()

    def reset(self):
        self._previous = None
        self._previous_time_ns = 0
        self._previous_first_step = 0
        self._pending = None

    def request_options(self):
        return {**super().request_options(), "aac_defer_selection": True}

    def decode_action(self, raw, context, *, adapter):
        if not isinstance(raw, Mapping):
            raise ValueError("AAC requires candidate response metadata")
        settings = self._config["inference"]["aac"]
        chunks = decode_candidates(
            raw.get("aac_candidates"), context, adapter=adapter, config=self._config,
            count=settings["num_samples"],
        )
        features = np.asarray(raw.get("aac_features"), dtype=float)
        if (
            features.shape != (len(chunks), chunks[0].horizon_steps, 2, 7)
            or not np.isfinite(features).all()
        ):
            raise ValueError("AAC requires normalized N x H x 2 x 7 scoring features")
        metadata = raw.get("aac")
        steps = metadata.get("execution_steps") if isinstance(metadata, Mapping) else None
        if type(steps) is not int or not 1 <= steps <= chunks[0].horizon_steps:
            raise ValueError("AAC requires valid execution_steps")
        self._pending = (chunks, features.copy(), dict(metadata))
        return chunks[0]

    def select_chunk(self, *, chunk, now_ns):
        if self._pending is None or self._pending[0][0].request_seq != chunk.request_seq:
            raise ValueError("AAC selection has no matching candidates")
        chunks, features, metadata = self._pending
        settings = self._config["inference"]["aac"]
        method = settings["chunk_id_selector"]
        beta = settings["backward_beta"]
        steps = metadata["execution_steps"]
        if method != "backward" or self._config["inference"]["inference_schedule"] == "serial":
            index = select_candidate(
                features, method=method, chunk_size=steps, previous=self._previous, beta=beta,
            )
        else:
            index = 0
            origin = self._origin(chunk, now_ns)
            start = first_future_step(origin, chunk.dt_ns, now_ns)
            if self._previous is not None and start < steps:
                times = origin + np.arange(start, steps) * chunk.dt_ns
                mask, reference = align_prediction(
                    self._previous.ee_features.transpose(1, 0, 2, 3),
                    self._previous_time_ns, times, chunk.dt_ns,
                    first_step=self._previous_first_step,
                )
                if mask.any():
                    distances = np.linalg.norm(
                        features[:, start:steps][:, mask] - reference.transpose(1, 0, 2, 3),
                        axis=-1,
                    ).mean(axis=2)
                    weights = beta ** np.arange(mask.sum())
                    index = int(np.argmin(distances @ (weights / weights.sum())))
        chosen = copy_action_chunk(chunks[index])
        chosen.metadata.update(chunk.metadata)
        chosen.metadata["aac"] = {**metadata, "chunk_id": index}
        return chosen

    def _origin(self, chunk, now_ns):
        return (
            now_ns if self._config["inference"]["action_start_mode"] == "first_step_when_ready"
            else chunk.observation_time_ns
        )

    def on_plan_accepted(self, *, chunk, result, response, now_ns):
        if self._pending is None or self._pending[0][0].request_seq != response.request_seq:
            raise ValueError("AAC accepted plan has no matching candidates")
        self._previous = AacPreviousAction(
            ee_features=self._pending[1], chunk_size=chunk.horizon_steps,
        )
        self._previous_time_ns = self._origin(chunk, now_ns)
        self._previous_first_step = result.time_trimmed_steps
        self._pending = None
        return {
            **super().on_plan_accepted(
                chunk=chunk, result=result, response=response, now_ns=now_ns,
            ),
            **{
                key: chunk.metadata["aac"][key]
                for key in ("chunk_id", "entropy_elbow", "motion_floor")
            },
        }

    def on_response_rejected(self, response):
        if self._pending is not None and self._pending[0][0].request_seq == response.request_seq:
            self._pending = None
