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
class DvacInferenceRequest(InferenceRequest):
    dvac: bool = True
    dvac_tail_steps: int = 5
    dvac_alpha: float = 2.0
    dvac_rolling_window_size: int = 5
    dvac_min_execution_steps: int = 1
    dvac_max_execution_steps: int = 50


class DvacInferenceStrategy(SynchronousChunkStrategy):
    """Paper-synchronous execution around an XPolicy denoising-variance hook."""

    name = "dvac"
    request_type = DvacInferenceRequest
    label = "DVAC"
    horizon_metadata = "dvac"

    def request_options(self) -> dict:
        settings = self._config["inference"]["dvac"]
        return {
            "dvac_tail_steps": settings["tail_policy_steps"],
            "dvac_alpha": settings["alpha"],
            "dvac_rolling_window_size": settings["rolling_window_size"],
            "dvac_min_execution_steps": settings["min_execution_policy_steps"],
            "dvac_max_execution_steps": settings["max_execution_policy_steps"]
            or self._config["policy"]["horizon_policy_steps"],
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
        metadata = raw.get("dvac") if isinstance(raw, Mapping) else None
        if isinstance(metadata, Mapping):
            for key in (
                "threshold",
                "rolling_mean",
                "rolling_std",
                "total_variance",
                "first_threshold_crossing",
                "rolling_states",
                "cold_start",
                "tail_steps",
                "action_dim",
                "method",
                "source",
            ):
                value = metadata.get(key)
                if value is None or isinstance(value, int | float | str | bool):
                    fields[key] = value
        return fields


def dvac_parameters(**options) -> dict:
    """保留 DVAC 方差窗口与自适应执行步数范围。"""

    unsupported = {"tail_steps", "min_execution_steps", "max_execution_steps"}.intersection(
        options
    )
    if unsupported:
        raise ValueError(f"unsupported DVAC fields: {sorted(unsupported)}")
    values = {
        "tail_policy_steps": 5,
        "alpha": 2.0,
        "rolling_window_size": 5,
        "min_execution_policy_steps": 1,
        "max_execution_policy_steps": None,
        **options,
    }
    return values
