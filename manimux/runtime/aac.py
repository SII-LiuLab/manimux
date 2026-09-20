from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from manimux.runtime.synchronous import SynchronousChunkStrategy
from manimux.runtime.timeline import CommitResult
from manimux.types import (
    ActionChunk,
    InferenceRequest,
    InferenceResponse,
)


@dataclass(slots=True)
class AacInferenceRequest(InferenceRequest):
    aac_num_samples: int = 20
    aac_motion_threshold: float = 3.0
    aac_ee_stats_path: str | None = None
    aac_chunk_id_selector: str = "0"
    aac_backward_beta: float = 0.99


class AacInferenceStrategy(SynchronousChunkStrategy):
    """Official AAC synchronous cadence around an XPolicy multi-sample hook."""

    name = "aac"
    request_type = AacInferenceRequest
    label = "AAC"

    def request_options(self) -> dict:
        aac = self._config["inference"]["aac"]
        return {
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
