from __future__ import annotations

import logging
from collections import deque

import numpy as np

from manimux.policy_adapter.base import PolicyAdapter
from manimux.runtime.inference import (
    CommitSettings,
    InferenceSubmission,
    RequestState,
)
from manimux.runtime.rtc.mask import inpainting_condition
from manimux.runtime.rtc.request import RtcInferenceRequest
from manimux.runtime.safety import RuntimeState
from manimux.runtime.timeline import ActionTimeline, CommitResult
from manimux.types import (
    ActionChunk,
    GroupVector,
    InferenceResponse,
    ObservationSnapshot,
    copy_group_vector,
)

logger = logging.getLogger(__name__)


class RtcInferenceStrategy:
    """Pi-guided RTC request scheduling and chunk conditioning."""

    discard_plans_while_paused = True

    def __init__(self, config: dict) -> None:
        self._config = config
        rtc = config["inference"]["rtc"]
        self._group_order = tuple(config["robot"]["group_dims"])
        self._rtc_beta = float(rtc["beta"])
        self._min_execute_steps = rtc["min_execute_steps"]
        self._initial_delay_steps = int(rtc["initial_delay_steps"])
        self._delay_buffer_size = int(rtc["delay_buffer_size"])
        self._delay_forecast: deque[int]
        self._active_rows: np.ndarray | None
        self._active_offset: int
        self._conditioned_requests: set[int]
        self._request_started_ns: dict[int, int]
        self._request_observation_ns: dict[int, int]
        self._request_forecast: dict[int, int]
        self._infeasible_reported: bool
        self._infeasible_pending: tuple[int, int] | None
        self._latency: list[tuple[int, float, float]]
        self._loop_ms: list[float]
        self._last_report_step: int
        self.reset()

    @property
    def name(self) -> str:
        return "rtc"

    @property
    def control_mode(self) -> str:
        return "rtc"

    @property
    def required_sampling_modes(self) -> frozenset[str]:
        return frozenset({"rtc"})

    def reset(self) -> None:
        self._delay_forecast = deque([self._initial_delay_steps], maxlen=self._delay_buffer_size)
        self._active_rows = None
        self._active_offset = 0
        self._conditioned_requests = set()
        self._request_started_ns = {}
        self._request_observation_ns = {}
        self._request_forecast = {}
        self._infeasible_reported = False
        self._infeasible_pending = None
        self._latency = []
        self._loop_ms = []
        self._last_report_step = 0

    def execution_horizon(self, horizon: int, delay: int) -> int:
        floor = self._min_execute_steps or max(1, horizon // 2)
        return int(min(max(floor, delay), horizon - delay))

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

        active = self._active_rows
        condition = None
        weights = None
        forecast_used = 0
        executed = 0
        ready = active is None
        if active is not None:
            horizon = len(active)
            executed = min(self._active_offset + timeline.cursor(now_ns), horizon)
            delay = max(self._delay_forecast)
            if 2 * delay > horizon and not self._infeasible_reported:
                self._infeasible_reported = True
                self._infeasible_pending = (delay, horizon)
            if executed >= self.execution_horizon(horizon, delay):
                ready = True
                if delay <= executed <= horizon - delay:
                    condition, weights = inpainting_condition(
                        active,
                        executed_steps=executed,
                        delay_steps=delay,
                    )
                    forecast_used = delay
        if not ready:
            return None

        deadline_ns = now_ns + int(self._config["policy"]["timeout_s"] * 1_000_000_000)
        request = RtcInferenceRequest(
            session_id=session_id,
            request_seq=request_seq,
            observation_time_ns=snapshot.state.monotonic_ns,
            deadline_ns=deadline_ns,
            observation=adapter.build_observation(snapshot),
            instruction=self._config["run"]["task"],
            action_condition=None if condition is None else condition.astype(np.float64),
            condition_weights=None if weights is None else weights.astype(np.float64),
            rtc_beta=self._rtc_beta,
        )
        if condition is not None:
            self._conditioned_requests.add(request_seq)
        self._request_started_ns[request_seq] = now_ns
        self._request_observation_ns[request_seq] = snapshot.state.monotonic_ns
        self._request_forecast[request_seq] = forecast_used
        return InferenceSubmission(
            request=request,
            event_fields={
                "executed_steps": executed,
                "forecast_delay": forecast_used,
                "conditioned": condition is not None,
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
        blend_steps = 0 if conditioned else self._config["inference"]["blend_steps"]
        logger.info(
            "rtc_commit_settings seq=%d conditioned=%s blend_steps=%d",
            response.request_seq,
            conditioned,
            blend_steps,
        )
        return CommitSettings(
            current_command=copy_group_vector(last_command),
            blend_steps=blend_steps,
            anchor_source="last_command",
        )

    def clear_condition(self, request_seq: int) -> None:
        """A history adapter can discover that no committed overlap remains."""
        self._conditioned_requests.discard(request_seq)
        self._request_forecast[request_seq] = 0

    def prepare_chunk(
        self,
        *,
        chunk: ActionChunk,
        response: InferenceResponse,
        now_ns: int,
    ) -> ActionChunk:
        del response, now_ns
        source_horizon = chunk.source_offset_steps + chunk.horizon_steps
        if source_horizon != self._config["policy"]["horizon_steps"]:
            raise ValueError("RTC decoded suffix must retain the configured source horizon")
        if chunk.hold_from_step:
            raise ValueError("RTC requires a complete joint plan for both arms")
        return chunk

    def on_plan_accepted(
        self,
        *,
        chunk: ActionChunk,
        result: CommitResult,
        response: InferenceResponse,
        now_ns: int,
    ) -> dict[str, object]:
        rows = np.concatenate([chunk.groups[name] for name in self._group_order], axis=1)
        # Restore source indices only. The discarded prefix is never executable
        # or conditioned: executed always starts beyond these zero-weight rows.
        self._active_rows = np.pad(rows, ((chunk.source_offset_steps, 0), (0, 0)))
        self._active_offset = chunk.source_offset_steps + result.trimmed_steps
        started_ns = self._request_started_ns.pop(response.request_seq, now_ns)
        observation_ns = self._request_observation_ns.pop(response.request_seq, started_ns)
        forecast_used = self._request_forecast.pop(response.request_seq, 0)
        executable_ns = now_ns + round(self._config["inference"]["commit_lead_s"] * 1e9)
        # Include observation age, request preparation, transport, model, both
        # decoder processes and commit lead; never round a partial step down.
        delay_ns = max(0, executable_ns - min(started_ns, observation_ns))
        measured_delay = (delay_ns + chunk.dt_ns - 1) // chunk.dt_ns
        self._delay_forecast.append(measured_delay)
        self._latency.append(
            (
                measured_delay,
                response.inference_ms,
                (now_ns - started_ns) / 1e6,
            )
        )
        self._conditioned_requests.discard(response.request_seq)
        return {
            "measured_delay": measured_delay,
            "forecast_delay": forecast_used,
            "server_ms": round(response.inference_ms, 1),
            "trimmed_steps": result.trimmed_steps,
            "rtc_source_horizon": len(self._active_rows),
            "rtc_executed_steps_at_commit": self._active_offset,
            "request_to_commit_ms": (now_ns - started_ns) / 1e6,
            "rtc_delay_ms": delay_ns / 1e6,
        }

    def on_response_rejected(self, response: InferenceResponse) -> None:
        self._conditioned_requests.discard(response.request_seq)
        self._request_started_ns.pop(response.request_seq, None)
        self._request_observation_ns.pop(response.request_seq, None)
        self._request_forecast.pop(response.request_seq, None)

    def take_runtime_events(self, *, step: int) -> list[tuple[str, dict[str, object]]]:
        pending = self._infeasible_pending
        if pending is None:
            return []
        self._infeasible_pending = None
        delay, horizon = pending
        return [
            (
                "rtc_delay_infeasible",
                {
                    "step": step,
                    "delay": delay,
                    "horizon": horizon,
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
            server = sum(item[1] for item in recent) / len(recent)
            trip = sum(item[2] for item in recent) / len(recent)
            horizon = 0 if self._active_rows is None else len(self._active_rows)
            margin = horizon - max(self._delay_forecast) - delay
            inference = (
                f"| infer server {server:5.0f}ms trip {trip:5.0f}ms "
                f"d={delay:4.1f} margin={margin:+5.1f}步"
            )
        else:
            inference = "| infer --"
        print(
            f"[rtc] step {steps:5d} | loop {sum(self._loop_ms) / len(self._loop_ms):5.1f}/"
            f"{control_dt_ns / 1e6:.0f}ms max {max(self._loop_ms):5.1f} {inference}",
            flush=True,
        )
        self._loop_ms.clear()


def rtc_parameters(**options) -> dict:
    """RTC 只改变推理与分块时序；执行器限位仍由公共 runtime 管理。"""

    values = {
        "min_execute_steps": None,
        "initial_delay_steps": 4,
        "delay_buffer_size": 10,
        "beta": 5.0,
        **options,
    }
    return values
