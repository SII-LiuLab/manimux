"""RTC scheduling with a trained hard prefix, without inference-time guidance."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields

import numpy as np

from manimux.policies.base import action_interval
from manimux.runtime.inference import InferenceSubmission
from manimux.runtime.rtc.strategy import RtcInferenceStrategy
from manimux.types import InferenceRequest


@dataclass(slots=True)
class TrainTimeRtcRequest(InferenceRequest):
    """A contiguous prefix in robot units, converted by the policy adapter.

    The horizon-sized condition and binary weights reuse the existing coordinate
    conversion contract. Only the first prefix_steps rows constrain sampling.
    """

    action_condition: np.ndarray | None = None
    condition_weights: np.ndarray | None = None
    prefix_steps: int = 0


class TrainTimeRtcInferenceStrategy(RtcInferenceStrategy):
    """RTC scheduling with a fixed prefix or a separately forecast decode budget."""

    def __init__(self, config: dict) -> None:
        self._delay_estimator = config["inference"]["train_time_rtc"]["delay_estimator"]
        self._fixed_prefix_steps = config["inference"]["train_time_rtc"]["fixed_prefix_steps"]
        identity = (config["policy"].get("expected_backend") or {}).get("model", {})
        self._action_offset_ns = round(
            config["policy"]
            .get("adapter", {})
            .get("execution_offset_s", identity.get("first_action_offset_s", 0.0))
            * 1e9
        )
        delegate = deepcopy(config)
        # The parent owns scheduling and accepted-plan feedback, not this sampler.
        delegate["inference"]["rtc"] = {
            **config["inference"]["train_time_rtc"],
            "beta": 5.0,
        }
        if self._delay_estimator == "fixed":
            delegate["inference"]["rtc"]["initial_delay_policy_steps"] = self._fixed_prefix_steps
        super().__init__(delegate)

    def reset(self) -> None:
        super().reset()
        self._last_model_latency_ns: int | None = None
        self._prefix_forecasts: dict[int, dict[str, object]] = {}

    def seed_warmup(self, *, latency_ns) -> None:
        # An aggregate E2E sample cannot seed a model-only predictor. Process
        # decoding supplies its own shared forecast through RequestState.
        if self._delay_estimator == "end_to_end_max":
            super().seed_warmup(latency_ns=latency_ns)

    def _forecast_prefix(self, *, now_ns, snapshot, request_state):
        age_ns = max(0, now_ns - snapshot.state.monotonic_ns)
        decode_ns = request_state.expected_decode_ns
        margin_ns = round(self._config["inference"]["handoff_margin_s"] * 1e9)
        total_ns = None
        if self._last_model_latency_ns is None:
            count = self._initial_delay_steps or 0
        else:
            # Unlike a waypoint selected after inference, a hard prefix must be
            # sent before inference. Predict the unknown model trip with the last
            # completed trip, while only decoding retains a rolling maximum.
            total_ns = age_ns + self._last_model_latency_ns + decode_ns + margin_ns
            dt_ns = round(action_interval(self._config["policy"]) * 1e9)
            from_row0_ns = max(0, total_ns - self._action_offset_ns)
            count = (from_row0_ns + dt_ns - 1) // dt_ns
        return count, {
            "prefix_delay_estimator": self._delay_estimator,
            "prefix_observation_age_ms": age_ns / 1e6,
            "prefix_model_forecast_ms": (
                None if self._last_model_latency_ns is None else self._last_model_latency_ns / 1e6
            ),
            "prefix_decode_forecast_ms": decode_ns / 1e6,
            "prefix_handoff_margin_ms": margin_ns / 1e6,
            "prefix_action_offset_ms": self._action_offset_ns / 1e6,
            "prefix_forecast_ms": None if total_ns is None else total_ns / 1e6,
            "predicted_prefix_steps": count,
        }

    @property
    def name(self) -> str:
        return "train_time_rtc"

    @property
    def control_mode(self) -> str:
        return "train_time_rtc"

    @property
    def required_sampling_modes(self) -> frozenset[str]:
        return frozenset({"default", "train_time_rtc"})

    def _hard_prefix(self, submission: InferenceSubmission) -> InferenceSubmission:
        source = submission.request
        # The parent's soft mask is only an intermediate scheduling result.
        # No soft weights or guidance parameters leave this strategy.
        count = (
            0
            if source.condition_weights is None
            else int(np.count_nonzero(source.condition_weights == 1.0))
        )
        condition = weights = None
        if count:
            condition = np.zeros_like(source.action_condition)
            condition[:count] = source.action_condition[:count]
            weights = np.zeros(len(condition), dtype=np.float64)
            weights[:count] = 1.0
        elif source.request_seq in self._request_forecast:
            self.clear_condition(source.request_seq)
        request = TrainTimeRtcRequest(
            **{item.name: getattr(source, item.name) for item in fields(InferenceRequest)},
            action_condition=condition,
            condition_weights=weights,
            prefix_steps=count,
        )
        return InferenceSubmission(
            request=request,
            event_fields={
                **submission.event_fields,
                "prefix_steps": count,
                "conditioned": count > 0,
            },
        )

    def build_submission(self, **kwargs) -> InferenceSubmission | None:
        forecast = {"prefix_delay_estimator": self._delay_estimator}
        if self._delay_estimator == "fixed":
            self._delay_forecast.clear()
            self._delay_forecast.append(self._fixed_prefix_steps)
            forecast["configured_prefix_steps"] = self._fixed_prefix_steps
        elif self._delay_estimator == "decode_max":
            count, forecast = self._forecast_prefix(
                now_ns=kwargs["now_ns"],
                snapshot=kwargs["snapshot"],
                request_state=kwargs["request_state"],
            )
            # The parent owns feasibility and scheduling. Its E2E history must
            # not participate in this estimator, even after an expensive first call.
            self._delay_forecast.clear()
            self._delay_forecast.append(count)
        submission = super().build_submission(**kwargs)
        if submission is None:
            return None
        submission = self._hard_prefix(submission)
        submission.event_fields.update(forecast)
        self._prefix_forecasts[submission.request.request_seq] = forecast
        return submission

    def on_plan_accepted(self, *, chunk, result, response, now_ns):
        started_ns = self._request_started_ns.get(response.request_seq)
        forecast = self._prefix_forecasts.pop(response.request_seq, {})
        previous_forecast = tuple(self._delay_forecast)
        event = super().on_plan_accepted(
            chunk=chunk,
            result=result,
            response=response,
            now_ns=now_ns,
        )
        if self._delay_estimator != "end_to_end_max":
            if started_ns is not None:
                # Worker completion excludes IK; includes request preparation,
                # queueing, transport and model execution since submission.
                self._last_model_latency_ns = max(0, response.finished_time_ns - started_ns)
            self._delay_forecast.clear()
            self._delay_forecast.extend(previous_forecast)
            from_row0_ns = max(0, now_ns - response.observation_time_ns - self._action_offset_ns)
            event["measured_prefix_steps"] = (from_row0_ns + chunk.dt_ns - 1) // chunk.dt_ns
            event["request_to_model_ms"] = (
                None if self._last_model_latency_ns is None else self._last_model_latency_ns / 1e6
            )
        event.update(forecast)
        return event

    def on_response_rejected(self, response) -> None:
        self._prefix_forecasts.pop(response.request_seq, None)
        super().on_response_rejected(response)

    @property
    def warmup_required_calls(self):
        # A zero fixed prefix deliberately has no conditioned branch to compile.
        if self._delay_estimator == "fixed" and self._fixed_prefix_steps == 0:
            return {"ordinary": 2}
        return super().warmup_required_calls


def train_time_rtc_parameters(**options) -> dict:
    values = {
        "delay_estimator": "decode_max",
        "fixed_prefix_steps": 5,
        "min_execute_policy_steps": None,
        "initial_delay_policy_steps": 5,
        "delay_buffer_size": 10,
    }
    unknown = set(options) - set(values)
    if unknown:
        raise ValueError(f"unsupported train-time RTC fields: {sorted(unknown)}")
    values.update(options)
    if values["delay_estimator"] not in {"fixed", "decode_max", "end_to_end_max"}:
        raise ValueError(
            "train_time_rtc.delay_estimator must be fixed, decode_max or end_to_end_max"
        )
    for key, minimum in (
        ("fixed_prefix_steps", 0),
        ("delay_buffer_size", 1),
        ("initial_delay_policy_steps", 0),
        ("min_execute_policy_steps", 1),
    ):
        value = values[key]
        if value is None and key not in {"delay_buffer_size", "fixed_prefix_steps"}:
            continue
        if type(value) is not int or value < minimum:
            raise ValueError(f"train_time_rtc.{key} must be an integer >= {minimum}")
    return values
