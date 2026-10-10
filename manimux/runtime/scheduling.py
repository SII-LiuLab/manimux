"""Request admission, independent of action sampling and chunk fusion."""

from __future__ import annotations

import math

from manimux.runtime.safety import RuntimeState

_CHUNK_STRATEGIES = {"manimux", "async", "serial", "bid_backward"}
_SINGLE = {"rtc", "paint", "act_temporal_ensemble"}
_ADAPTIVE = {"aac", "autohorizon"}
_STREAMING = {
    "manimux", "async", "rtc", "act_temporal_ensemble", "bid_backward",
    "paint", "aac", "autohorizon",
}


def resolve_request_schedule(values: dict) -> None:
    """Resolve old algorithm-owned schedules without changing their cadence."""
    algorithm = values["algorithm"]
    schedule = values["inference_schedule"]
    # Historically 'deadline' was an ignored default for specialized strategies.
    if schedule == "deadline":
        if algorithm in _SINGLE:
            values["inference_schedule"] = "single_inflight"
        elif algorithm in _ADAPTIVE:
            values["inference_schedule"] = "serial"
    if values["request_trigger"] is None:
        values["request_trigger"] = (
            "refill" if algorithm in _CHUNK_STRATEGIES | _ADAPTIVE
            and values["inference_schedule"] != "serial" else "algorithm"
        )


def validate_request_schedule(values: dict, *, provided=frozenset()) -> None:
    algorithm = values["algorithm"]
    schedule = values["inference_schedule"]
    trigger = values["request_trigger"]
    if schedule not in {"deadline", "single_inflight", "multi_inflight", "serial"}:
        raise ValueError(f"unknown inference_schedule: {schedule!r}")
    if trigger not in {"refill", "continuous", "algorithm"}:
        raise ValueError(f"unknown inference.request_trigger: {trigger!r}")
    if values["strategy"] is not None and (
        schedule == "multi_inflight" or trigger == "continuous"
    ):
        raise ValueError("custom strategies do not support streaming or continuous requests")
    if schedule == "multi_inflight" and algorithm not in _STREAMING:
        raise ValueError(f"multi_inflight is not supported by {algorithm}")
    serial_act = algorithm == "act_temporal_ensemble" and schedule == "serial"
    if algorithm in _SINGLE and not serial_act and schedule not in {
        "single_inflight", "multi_inflight",
    }:
        raise ValueError(f"{algorithm} requires single_inflight or a supported multi_inflight")
    if schedule == "serial" and not serial_act and algorithm not in _CHUNK_STRATEGIES | _ADAPTIVE:
        raise ValueError(f"serial scheduling is not supported by {algorithm}")
    if algorithm == "serial" and schedule != "serial":
        raise ValueError("inference.algorithm=serial requires inference_schedule=serial")
    ordinary_async = algorithm in _CHUNK_STRATEGIES | _ADAPTIVE and schedule != "serial"
    if ordinary_async and trigger not in {"refill", "continuous"}:
        raise ValueError("ordinary asynchronous requests require refill or continuous trigger")
    if not ordinary_async and trigger != "algorithm":
        raise ValueError(f"{algorithm} with {schedule} requires request_trigger=algorithm")
    if trigger != "refill" and "refill_threshold_s" in provided:
        raise ValueError("refill_threshold_s is only used by request_trigger=refill")
    threshold = values["refill_threshold_s"]
    if isinstance(threshold, bool) or not isinstance(threshold, int | float) or not (
        math.isfinite(threshold) and threshold >= 0
    ):
        raise ValueError("refill_threshold_s must be finite and non-negative")
    rate = values["observation_hz"]
    if rate is None:
        if schedule == "multi_inflight":
            raise ValueError("multi_inflight requires an explicit observation_hz")
    elif isinstance(rate, bool) or not isinstance(rate, int | float) or not (
        math.isfinite(rate) and 0 < rate <= 1e9
    ):
        raise ValueError("observation_hz must be finite, positive and at most 1e9")


class RequestScheduler:
    """Admit requests; the selected strategy still decides algorithm readiness.

    This object never prepares actions or advances algorithm state. Call
    on_submitted only after the worker accepts the submission into its queue.
    Warmup deliberately bypasses it and uses its own reset-fenced lifecycle.
    """

    def __init__(self, settings: dict) -> None:
        self.mode = settings["inference_schedule"]
        self.trigger = settings["request_trigger"]
        self._refill_ns = round(settings["refill_threshold_s"] * 1e9)
        rate = settings["observation_hz"]
        self._interval_ns = 0 if rate is None else max(1, round(1e9 / rate))
        self.reset()

    @property
    def multi_inflight(self) -> bool:
        return self.mode == "multi_inflight"

    def reset(self) -> None:
        self._next_request_ns = 0

    def ready(self, *, now_ns, timeline, request_state, runtime_state) -> bool:
        if now_ns < self._next_request_ns:
            return False
        if self.mode in {"serial", "multi_inflight"} and runtime_state != RuntimeState.RUNNING:
            return False
        if self.mode in {"serial", "single_inflight"} and request_state.in_flight:
            return False
        if self.mode == "deadline" and not (
            request_state.last_submitted_seq < 0 or now_ns > request_state.last_deadline_ns
        ):
            return False
        if self.mode == "serial" and timeline.remaining_ns(now_ns) > 0:
            return False
        return not (
            self.trigger == "refill" and timeline.remaining_ns(now_ns) >= self._refill_ns
        )

    def on_submitted(self, now_ns: int) -> None:
        self._next_request_ns = now_ns + self._interval_ns

    def build_submission(self, strategy, **kwargs):
        # History collection must run on every tick, including ticks where the
        # transport is busy. Sampling it only on admitted requests loses frames.
        observe = getattr(strategy, "observe_snapshot", None)
        if callable(observe):
            observe(kwargs["snapshot"])
        if not self.ready(**{
            key: kwargs[key]
            for key in ("now_ns", "timeline", "request_state", "runtime_state")
        }):
            return None
        return strategy.build_submission(**kwargs)
