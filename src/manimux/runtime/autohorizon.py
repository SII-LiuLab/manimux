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
class AutoHorizonInferenceRequest(InferenceRequest):
    autohorizon: bool = True


class AutoHorizonInferenceStrategy(SynchronousChunkStrategy):
    """Official synchronous execution cadence around an XPolicy attention hook."""

    name = "autohorizon"
    request_type = AutoHorizonInferenceRequest
    label = "AutoHorizon"
    horizon_metadata = "autohorizon"

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
        metadata = raw.get("autohorizon") if isinstance(raw, Mapping) else None
        if isinstance(metadata, Mapping):
            for key in (
                "attention_step",
                "forward_horizon",
                "backward_horizon",
                "join_row",
                "method",
                "framework",
                "upstream_commit",
            ):
                value = metadata.get(key)
                if isinstance(value, int | float | str) and not isinstance(value, bool):
                    fields[key] = value
        return fields
