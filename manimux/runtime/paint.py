from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from manimux.policies.base import action_interval
from manimux.policy_adapter.base import PolicyAdapter
from manimux.runtime.candidates import sample_reference
from manimux.runtime.inference import (
    CommitSettings,
    EnsembleScope,
    InferenceSubmission,
    InferenceTimingError,
    RequestState,
)
from manimux.runtime.safety import RuntimeState
from manimux.runtime.timeline import ActionTimeline, CommitResult, first_future_step
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

    ensemble_scope: EnsembleScope = "execution_prefix"
    # Make the terminal source row executable for its full policy interval.
    # EdgeRuntime then retains the last command while inference is pending.
    hold_last_step = True
    # Compile both sampler branches, then measure each without its first call.
    warmup_required_calls = {"ordinary": 1, "conditioned": 2}

    def __init__(self, config: dict) -> None:
        self._config = config
        paint = config["inference"]["paint"]
        self._group_order = tuple(config["robot"]["group_dims"])
        self._execution_steps = int(paint["execution_policy_steps"])
        self._initial_delay_steps = int(paint["initial_delay_policy_steps"])
        self._delay_buffer_size = int(paint["delay_buffer_size"])
        self._warmup_delay_steps: tuple[int, ...] = ()
        # A buffered single-inflight reply can outlive its conditioning plan too.
        self._check_reference = (
            config["inference"]["inference_schedule"] == "multi_inflight"
            or config["inference"].get("min_plan_hold_policy_steps", 0) > 0
        )
        self._time_aligned = (
            self._check_reference
            or (config["inference"].get("temporal_ensemble") or {}).get("enabled", False)
        )
        self._active_rows: np.ndarray | None
        self._active_offset: int
        self._delay_forecast: deque[int]
        self._request_started_ns: dict[int, int]
        self._request_forecast: dict[int, int]
        self._request_execution: dict[int, int]
        self._conditioned_requests: set[int]
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
        self._active_request_seq = -1
        self._request_reference_seq: dict[int, int] = {}
        self._active_rows = None
        self._active_offset = 0
        self._delay_forecast = deque(
            self._warmup_delay_steps or (self._initial_delay_steps,),
            maxlen=self._delay_buffer_size,
        )
        self._request_started_ns = {}
        self._request_forecast = {}
        self._request_execution = {}
        self._conditioned_requests = set()

    def seed_warmup(self, *, latency_ns: Sequence[int]) -> None:
        """Retain steady E2E samples across reset, with the configured initial floor."""
        dt_ns = int(action_interval(self._config["policy"]) * 1_000_000_000)
        recent = [int(value) for value in latency_ns][-self._delay_buffer_size :]
        if any(value < 0 for value in recent):
            raise ValueError("warmup latency_ns must be nonnegative")
        self._warmup_delay_steps = tuple(
            max(self._initial_delay_steps, (value + dt_ns - 1) // dt_ns)
            for value in recent
        )
        self._delay_forecast = deque(
            self._warmup_delay_steps or (self._initial_delay_steps,),
            maxlen=self._delay_buffer_size,
        )

        self._require_future_output(self._config["policy"]["horizon_policy_steps"])

    def _require_future_output(self, horizon: int) -> None:
        delay = max(self._delay_forecast)
        if delay >= horizon:
            dt = action_interval(self._config["policy"])
            raise InferenceTimingError(
                f"PAINT timing infeasible: delay_steps={delay}, horizon_steps={horizon}; "
                f"forecast_delay={delay * dt:.3f}s consumes the entire NEW chunk. "
                "Holding the old terminal target is supported, but the new chunk "
                "must retain future actions after latency trimming.",
                algorithm="paint",
            )

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
        if runtime_state != RuntimeState.RUNNING:
            return None

        if request_state.multi_flight:
            timeout = int(self._config["policy"]["timeout_s"] * 1e9)
            for seq, started in list(self._request_started_ns.items()):
                if now_ns - started > timeout:
                    self._forget_request(seq)

        deadline_ns = now_ns + int(self._config["policy"]["timeout_s"] * 1_000_000_000)
        request_fields = {
            "session_id": session_id,
            "request_seq": request_seq,
            "observation_time_ns": snapshot.state.monotonic_ns,
            "deadline_ns": deadline_ns,
            "observation": adapter.build_observation(snapshot),
            "instruction": self._config["run"]["task_prompt"],
        }
        active = self._active_rows
        if active is None:
            self._require_future_output(self._config["policy"]["horizon_policy_steps"])
            if self._check_reference:
                self._request_reference_seq[request_seq] = timeline.accepted_request_seq
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
        self._require_future_output(self._config["policy"]["horizon_policy_steps"])
        # The old plan may run out while the model is working. Continue its
        # terminal target (including gripper), matching the runtime's hold.
        indices = np.minimum(np.arange(executed, executed + delay), horizon - 1)
        prefix = active[indices].astype(np.float64, copy=True)
        if self._time_aligned and delay:
            valid, prefix = sample_reference(
                timeline, origin_ns=snapshot.state.monotonic_ns,
                dt_ns=int(action_interval(self._config["policy"]) * 1e9),
                steps=delay, group_order=self._group_order,
                terminal_hold=True,
            )
            if not valid.all():
                if self._request_started_ns:
                    return None
                raise InferenceTimingError(
                    "PAINT timing infeasible: the timeline no longer contains "
                    "a complete time-aligned conditioning prefix.", algorithm="paint",
                )
        request = PaintInferenceRequest(
            **request_fields,
            paint_action_prefix=prefix,
            paint_delay_steps=delay,
            paint_execution_steps=executed,
            action_condition_space=timeline.action_space,
        )
        self._request_started_ns[request_seq] = now_ns
        self._request_forecast[request_seq] = delay
        self._request_execution[request_seq] = executed
        self._conditioned_requests.add(request_seq)
        if self._check_reference:
            self._request_reference_seq[request_seq] = timeline.accepted_request_seq
        return InferenceSubmission(
            request=request,
            event_fields={
                "conditioned": True,
                "execution_steps": executed,
                "delay_steps": delay,
                "forecast_terminal_hold_policy_steps": max(0, executed + delay - horizon),
            },
        )

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
        conditioned = response.request_seq in self._conditioned_requests
        return CommitSettings(
            current_command=copy_group_vector(last_command if conditioned else measured),
            blend_steps=self._config["inference"]["blend_policy_steps"],
            anchor_source="last_command" if conditioned else "measured_state",
        )

    def _actual_trimmed_steps(self, chunk: ActionChunk, now_ns: int) -> int:
        age_ns = max(0, now_ns - chunk.observation_time_ns)
        if self._time_aligned:
            # New combinations use Timeline's first-future-row convention.
            return first_future_step(chunk.observation_time_ns, chunk.dt_ns, now_ns)
        return int(age_ns // chunk.dt_ns)

    def prepare_chunk(
        self,
        *,
        chunk: ActionChunk,
        response: InferenceResponse,
        now_ns: int,
    ) -> ActionChunk:
        if self._check_reference and (
            self._request_reference_seq.get(response.request_seq) != self._active_request_seq
        ):
            raise ValueError("paint_condition_plan_replaced")
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
        previous_horizon = len(self._active_rows) if self._active_rows is not None else 0
        self._active_request_seq = response.request_seq
        self._request_reference_seq.pop(response.request_seq, None)
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
        self._conditioned_requests.discard(response.request_seq)
        fields: dict[str, object] = {
            "paint_wait_policy": "terminal_hold_v1",
            "execution_steps": executed,
            "measured_delay": actual,
            "forecast_delay": forecast,
            "server_ms": round(response.inference_ms, 1),
            "request_to_commit_ms": (now_ns - started_ns) / 1e6,
            "trimmed_steps": result.trimmed_steps,
            "forecast_terminal_hold_policy_steps": max(0, executed + forecast - previous_horizon) if forecast else 0,
            "elapsed_terminal_hold_policy_steps": max(0, executed + actual - previous_horizon) if forecast else 0,
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
        self._request_reference_seq.pop(response.request_seq, None)
        started_ns = self._request_started_ns.pop(response.request_seq, None)
        forecast = self._request_forecast.pop(response.request_seq, 0)
        self._request_execution.pop(response.request_seq, None)
        self._conditioned_requests.discard(response.request_seq)
        if started_ns is not None and forecast > 0:
            elapsed_ns = max(0, response.finished_time_ns - started_ns)
            dt_ns = int(action_interval(self._config["policy"]) * 1_000_000_000)
            self._delay_forecast.append(int(np.ceil(elapsed_ns / dt_ns)))

    def _forget_request(self, seq):
        self._request_reference_seq.pop(seq, None)
        self._request_started_ns.pop(seq, None)
        self._request_forecast.pop(seq, None)
        self._request_execution.pop(seq, None)
        self._conditioned_requests.discard(seq)

    def take_runtime_events(self, *, step: int) -> list[tuple[str, dict[str, object]]]:
        return []

    def on_tick(self, *, steps: int, loop_ms: float, control_dt_ns: int) -> None:
        del steps, loop_ms, control_dt_ns


def paint_parameters(**options) -> dict:
    """Preserve default PAINT execution-prefix and delay-estimation step counts."""
    from manimux.config_validation import merge_parameters

    unsupported = {"execution_steps", "initial_delay_steps"}.intersection(options)
    if unsupported:
        raise ValueError(f"unsupported PAINT fields: {sorted(unsupported)}")
    values = merge_parameters({
        "execution_policy_steps": 10,
        "initial_delay_policy_steps": 4,
        "delay_buffer_size": 10,
    }, options, 'inference.paint', extra=())
    return values
