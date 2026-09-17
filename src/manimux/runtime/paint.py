from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from manimux.policies.base import PolicyAdapter, action_interval
from manimux.runtime.inference import (
    CommitSettings,
    InferenceSubmission,
    RequestState,
)
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


@dataclass(slots=True)
class PaintInferenceRequest(InferenceRequest):
    """PAINT request carrying the paper's executed prefix and delay ``d``."""

    paint_action_prefix: NDArray[np.float64] | None = None
    paint_delay_steps: int | None = None
    paint_execution_steps: int | None = None


class PaintInferenceStrategy:
    """Algorithm 1 scheduling around an XPolicy flow-sampler hook."""

    def __init__(self, config: dict) -> None:
        self._config = config
        paint = config["execution"]["paint"]
        self._group_order = tuple(config["robot"]["group_dims"])
        self._execution_steps = int(paint["execution_steps"])
        self._initial_delay_steps = int(paint["initial_delay_steps"])
        self._delay_buffer_size = int(paint["delay_buffer_size"])
        self._active_rows: np.ndarray | None
        self._active_offset: int
        self._delay_forecast: deque[int]
        self._request_started_ns: dict[int, int]
        self._request_forecast: dict[int, int]
        self._request_execution: dict[int, int]
        self._conditioned_requests: set[int]
        self._infeasible_pending: tuple[int, int, int] | None
        self._latency: list[tuple[int, int, float, float]]
        self._loop_ms: list[float]
        self._last_report_step: int
        self.reset()

    @property
    def name(self) -> str:
        return "paint"

    @property
    def control_mode(self) -> str:
        return "paint"

    @property
    def required_sampling_modes(self) -> frozenset[str]:
        return frozenset({"paint"})

    def reset(self) -> None:
        self._active_rows = None
        self._active_offset = 0
        self._delay_forecast = deque(
            [self._initial_delay_steps],
            maxlen=self._delay_buffer_size,
        )
        self._request_started_ns = {}
        self._request_forecast = {}
        self._request_execution = {}
        self._conditioned_requests = set()
        self._infeasible_pending = None
        self._latency = []
        self._loop_ms = []
        self._last_report_step = 0

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
        if request_state.in_flight or runtime_state != RuntimeState.RUNNING:
            return None

        deadline_ns = now_ns + int(self._config["policy"]["timeout_s"] * 1_000_000_000)
        request_fields = {
            "session_id": session_id,
            "request_seq": request_seq,
            "observation_time_ns": snapshot.state.monotonic_ns,
            "deadline_ns": deadline_ns,
            "observation": adapter.build_observation(snapshot),
            "instruction": self._config["run"]["task"],
        }
        active = self._active_rows
        if active is None:
            self._request_started_ns[request_seq] = now_ns
            self._request_forecast[request_seq] = 0
            self._request_execution[request_seq] = 0
            return InferenceSubmission(
                request=InferenceRequest(**request_fields),
                event_fields={"conditioned": False, "execution_steps": 0, "delay_steps": 0},
            )

        horizon = len(active)
        executed = min(self._active_offset + timeline.cursor(now_ns), horizon)
        if executed < self._execution_steps:
            return None
        delay = max(self._delay_forecast)
        if delay > executed or executed + delay > horizon:
            self._infeasible_pending = (executed, delay, horizon)
            return None

        prefix = active[executed : executed + delay].astype(np.float64, copy=True)
        request = PaintInferenceRequest(
            **request_fields,
            paint_action_prefix=prefix,
            paint_delay_steps=delay,
            paint_execution_steps=executed,
        )
        self._request_started_ns[request_seq] = now_ns
        self._request_forecast[request_seq] = delay
        self._request_execution[request_seq] = executed
        self._conditioned_requests.add(request_seq)
        return InferenceSubmission(
            request=request,
            event_fields={
                "conditioned": True,
                "execution_steps": executed,
                "delay_steps": delay,
            },
        )

    def commit_settings(
        self,
        *,
        response: InferenceResponse,
        measured: GroupVector,
        last_command: GroupVector,
    ) -> CommitSettings:
        conditioned = response.request_seq in self._conditioned_requests
        return CommitSettings(
            current_command=copy_group_vector(last_command if conditioned else measured),
            blend_steps=0,
            anchor_source="last_command" if conditioned else "measured_state",
        )

    def _actual_trimmed_steps(self, chunk: ActionChunk, now_ns: int) -> int:
        commit_time_ns = now_ns + int(self._config["execution"]["commit_lead_s"] * 1_000_000_000)
        age_ns = max(0, commit_time_ns - chunk.observation_time_ns)
        return int(age_ns // chunk.dt_ns)

    def prepare_chunk(
        self,
        *,
        chunk: ActionChunk,
        response: InferenceResponse,
        now_ns: int,
    ) -> ActionChunk:
        forecast = self._request_forecast.get(response.request_seq, 0)
        if forecast == 0:
            return chunk
        actual = self._actual_trimmed_steps(chunk, now_ns)
        if actual > forecast:
            self._delay_forecast.append(actual)
            raise ValueError(
                "PAINT response advanced beyond its anchored prefix: "
                f"actual d={actual}, conditioned d={forecast}"
            )
        return chunk

    def on_plan_accepted(
        self,
        *,
        chunk: ActionChunk,
        result: CommitResult,
        response: InferenceResponse,
        now_ns: int,
    ) -> dict[str, object]:
        self._active_rows = np.concatenate(
            [chunk.groups[name] for name in self._group_order],
            axis=1,
        )
        self._active_offset = result.trimmed_steps
        started_ns = self._request_started_ns.pop(response.request_seq, now_ns)
        forecast = self._request_forecast.pop(response.request_seq, 0)
        executed = self._request_execution.pop(response.request_seq, 0)
        actual = self._actual_trimmed_steps(chunk, now_ns)
        if forecast > 0:
            self._delay_forecast.append(actual)
        self._latency.append(
            (
                actual,
                forecast,
                response.inference_ms,
                (now_ns - started_ns) / 1e6,
            )
        )
        self._conditioned_requests.discard(response.request_seq)
        fields: dict[str, object] = {
            "execution_steps": executed,
            "measured_delay": actual,
            "forecast_delay": forecast,
            "server_ms": round(response.inference_ms, 1),
            "trimmed_steps": result.trimmed_steps,
        }
        raw = response.raw_action
        metadata = raw.get("paint") if isinstance(raw, Mapping) else None
        if isinstance(metadata, Mapping):
            for key in ("num_steps", "model_evaluations", "inversion"):
                value = metadata.get(key)
                if isinstance(value, int | float | str) and not isinstance(value, bool):
                    fields[key] = value
        return fields

    def on_response_rejected(self, response: InferenceResponse) -> None:
        started_ns = self._request_started_ns.pop(response.request_seq, None)
        forecast = self._request_forecast.pop(response.request_seq, 0)
        self._request_execution.pop(response.request_seq, None)
        self._conditioned_requests.discard(response.request_seq)
        if started_ns is not None and forecast > 0:
            elapsed_ns = max(0, response.finished_time_ns - started_ns)
            dt_ns = int(action_interval(self._config["policy"]) * 1_000_000_000)
            self._delay_forecast.append(int(np.ceil(elapsed_ns / dt_ns)))

    def take_runtime_events(self, *, step: int) -> list[tuple[str, dict[str, object]]]:
        pending = self._infeasible_pending
        if pending is None:
            return []
        self._infeasible_pending = None
        executed, delay, horizon = pending
        return [
            (
                "paint_delay_infeasible",
                {
                    "step": step,
                    "execution_steps": executed,
                    "delay_steps": delay,
                    "horizon_steps": horizon,
                },
            )
        ]

    def on_tick(self, *, steps: int, loop_ms: float, control_dt_ns: int) -> None:
        self._loop_ms.append(loop_ms)
        report_every = max(1, int(round(2e9 / control_dt_ns)))
        if steps - self._last_report_step < report_every:
            return
        self._last_report_step = steps
        recent = self._latency[-5:]
        if recent:
            delay = sum(item[0] for item in recent) / len(recent)
            forecast = sum(item[1] for item in recent) / len(recent)
            server = sum(item[2] for item in recent) / len(recent)
            trip = sum(item[3] for item in recent) / len(recent)
            inference = (
                f"| infer server {server:5.0f}ms trip {trip:5.0f}ms "
                f"d={delay:4.1f}/{forecast:4.1f}步"
            )
        else:
            inference = "| infer --"
        print(
            f"[paint] step {steps:5d} | loop "
            f"{sum(self._loop_ms) / len(self._loop_ms):5.1f}/"
            f"{control_dt_ns / 1e6:.0f}ms max {max(self._loop_ms):5.1f} {inference}",
            flush=True,
        )
        self._loop_ms.clear()


def paint_parameters(**options) -> dict:
    """保留 PAINT 执行前缀与延迟估计的默认步数。"""

    values = {
        "execution_steps": 10,
        "initial_delay_steps": 4,
        "delay_buffer_size": 10,
        **options,
    }
    return values
