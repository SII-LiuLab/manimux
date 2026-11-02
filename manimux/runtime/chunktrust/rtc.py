"""RTC scheduling around ChunkTrust's selected execution budget.

The selector and soft-overlap mask retain ChunkTrust semantics. Delay estimation
and target playback use ManiMux's fixed policy grid, not the source TOPP protocol.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import Any

import numpy as np

from manimux.runtime.chunktrust.strategy import ChunkTrustAhsStrategy, RequestMark
from manimux.runtime.inference import CommitSettings, InferenceSubmission, InferenceTimingError
from manimux.runtime.rtc.request import RtcInferenceRequest
from manimux.runtime.safety import RuntimeState
from manimux.types import GroupVector, InferenceResponse, copy_group_vector


def soft_overlap_mask(frozen: int, overlap: int, horizon: int) -> np.ndarray:
    """ChunkTrust's frozen-budget mask, from its pinned RTC common.py.

    Unlike the ordinary RTC mask, the selected budget can be shorter than H/2.
    Only the selected remaining budget is frozen, while the full tail supplies
    soft overlap. Model padding must be masked by the declared backend profile.
    """
    if not 0 <= frozen <= overlap <= horizon:
        raise ValueError("ChunkTrust RTC mask requires 0 <= frozen <= overlap <= H")
    index = np.arange(horizon)
    weight = np.clip((overlap - index) / (overlap - frozen + 1), 0.0, 1.0)
    return np.where(
        index < frozen, 1.0,
        np.where(index < overlap, weight * np.expm1(weight) / np.expm1(1.0), 0.0),
    ).astype(np.float32)


class ChunkTrustAhsRtcStrategy(ChunkTrustAhsStrategy):
    """One in-flight RTC request, full prediction reference, bounded AHS execution."""

    name = "chunktrust_ahs_rtc"
    label = "ChunkTrust AHS + RTC"
    is_rtc = True
    allow_retimed_suffix = True
    warmup_required_calls = {"ordinary": 1, "conditioned": 2}

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        rtc = config["inference"]["rtc"]
        self._beta = float(rtc["beta"])
        self._initial_delay = rtc["initial_delay_policy_steps"]
        self._delay_size = int(rtc["delay_buffer_size"])
        self._warmup_delays: tuple[int, ...] = ()
        self._calibrating = False
        self._reset_delay()

    @property
    def required_sampling_modes(self) -> frozenset[str]:
        return frozenset({"default", "rtc"})

    def _reset_delay(self) -> None:
        initial = (
            self._warmup_delays or (0,)
            if self._initial_delay is None else (self._initial_delay,)
        )
        self._delay = deque(initial, maxlen=self._delay_size)

    def reset(self) -> None:
        super().reset()
        self._calibrating = False
        self._reset_delay()

    def begin_warmup(self) -> None:
        """The warmup coordinator owns branch-specific cold-call exclusion."""
        self._calibrating = True

    def on_warmup_response(self, *, latency_ns: int, first_call: bool) -> None:
        if not self._calibrating:
            raise RuntimeError("ChunkTrust warmup latency requires an isolated preview strategy")
        if not first_call:
            self._delay.append((latency_ns + self._dt_ns - 1) // self._dt_ns)

    def seed_warmup(self, *, latency_ns: Sequence[int]) -> None:
        if self._initial_delay is not None:
            return
        if any(type(value) is not int or value < 0 for value in latency_ns):
            raise ValueError("ChunkTrust warmup latency must contain nonnegative integer ns")
        self._warmup_delays = tuple(
            (value + self._dt_ns - 1) // self._dt_ns for value in latency_ns
        )[-self._delay_size:]
        self._reset_delay()

    def build_submission(self, **arguments: Any) -> InferenceSubmission | None:
        self._require_profile()
        if arguments["runtime_state"] != RuntimeState.RUNNING:
            return None
        active = self._active
        condition = weights = None
        forecast = max(self._delay)
        frozen = cursor = 0
        if active is not None:
            if forecast > min(min(self.options.candidates), self._horizon - max(
                self.options.candidates
            )):
                raise InferenceTimingError(
                    "ChunkTrust RTC delay cannot preserve the configured candidate suffix",
                    algorithm=self.name,
                )
            cursor = active.consumed
            if cursor < max(0, active.steps - forecast):
                return None
            if self._full_actions is None:
                raise RuntimeError("ChunkTrust RTC has no full source prediction")
            source_cursor = active.source_offset + cursor
            remaining = self._full_actions[source_cursor:]
            frozen = active.steps - cursor
            condition = np.zeros_like(self._full_actions, dtype=np.float64)
            condition[:len(remaining)] = remaining
            weights = soft_overlap_mask(frozen, len(remaining), self._horizon).astype(np.float64)
        snapshot = arguments["snapshot"]
        now_ns = arguments["now_ns"]
        request = RtcInferenceRequest(
            session_id=arguments["session_id"], request_seq=arguments["request_seq"],
            observation_time_ns=snapshot.state.monotonic_ns,
            deadline_ns=now_ns + int(self._config["policy"]["timeout_s"] * 1e9),
            observation=arguments["adapter"].build_observation(snapshot),
            instruction=self._config["run"]["task_prompt"],
            action_condition=condition, condition_weights=weights, rtc_beta=self._beta,
        )
        submission = InferenceSubmission(request, {
            "conditioned": condition is not None, "forecast_delay": forecast,
            "frozen_budget": frozen, "consumed_policy_rows": cursor,
            "executed_steps": 0 if active is None else active.source_offset + cursor,
        })
        self._remember(submission, now_ns, forecast)
        return submission

    def _offset(self, mark: RequestMark) -> int:
        current_plan = None if self._active is None else self._active.plan_id
        if current_plan != mark.reference_plan:
            raise ValueError("ChunkTrust RTC response refers to a replaced plan")
        return self._history_revision - mark.submitted_rows

    def commit_settings(
        self, *, response: InferenceResponse, measured: GroupVector, last_command: GroupVector,
    ) -> CommitSettings:
        del response, measured
        return CommitSettings(copy_group_vector(last_command), 0, "last_command")

    def _after_accept(self, mark: RequestMark, now_ns: int, fields: dict[str, object]) -> None:
        elapsed = max(0, now_ns - min(mark.started_ns, mark.observation_ns))
        delay = (elapsed + self._dt_ns - 1) // self._dt_ns
        if not self._calibrating:
            self._delay.append(delay)
        fields.update(
            forecast_delay=mark.forecast, measured_delay=delay,
            request_to_commit_ms=(now_ns - mark.started_ns) / 1e6,
            observation_to_commit_ms=elapsed / 1e6,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            **super().state_dict(), "delay_policy_steps": list(self._delay),
            "warmup_delay_policy_steps": list(self._warmup_delays),
        }
