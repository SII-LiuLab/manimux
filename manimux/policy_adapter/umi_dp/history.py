"""Measured observation history decorating existing inference strategies.

build_submission already runs once per control tick. This plugin only gates
history readiness and supplies observations; scheduling stays in the delegate.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from manimux.policies.base import action_interval
from manimux.runtime import validate_inference_parameters, validate_runtime_parameters
from manimux.runtime.inference import build_inference_strategy
from manimux.types import ObservationSnapshot, RobotState, copy_group_vector


@dataclass(slots=True)
class WindowSnapshot(ObservationSnapshot):
    previous: ObservationSnapshot | None = None


class MeasuredHistory:
    def __init__(
        self,
        *,
        camera_names,
        period_s=0.1,
        tolerance_s=0.04,
        state_tolerance_s=0.02,
        camera_skew_s=0.04,
        max_age_s=0.15,
    ):
        self.names = tuple(camera_names)
        self.period_ns = int(period_s * 1e9)
        self.tolerance_ns = int(tolerance_s * 1e9)
        self.state_tolerance_ns = int(state_tolerance_s * 1e9)
        self.skew_ns = int(camera_skew_s * 1e9)
        self.max_age_ns = int(max_age_s * 1e9)
        if not 0 < self.tolerance_ns < self.period_ns:
            raise ValueError("History tolerance must be positive and smaller than its period")
        if min(self.state_tolerance_ns, self.skew_ns, self.max_age_ns) <= 0:
            raise ValueError("History timing bounds must be positive")
        self.reset()

    def reset(self):
        self.states = deque(maxlen=2048)
        self.samples = deque(maxlen=128)
        self.last_sequences = None

    def observe(self, snapshot):
        state = snapshot.state
        if self.states and state.monotonic_ns < self.states[-1].monotonic_ns:
            self.reset()
        if not self.states or state.monotonic_ns != self.states[-1].monotonic_ns:
            self.states.append(
                RobotState(copy_group_vector(state.groups), state.monotonic_ns, state.sequence)
            )
        if any(name not in snapshot.frames for name in self.names):
            return
        frames = {name: snapshot.frames[name] for name in self.names}
        sequences = tuple(frames[name].sequence for name in self.names)
        if self.last_sequences is not None and any(
            a == b for a, b in zip(sequences, self.last_sequences, strict=True)
        ):
            return
        captures = [frame.capture_monotonic_ns for frame in frames.values()]
        if max(captures) - min(captures) > self.skew_ns:
            return
        capture = sum(captures) // len(captures)
        measured = min(self.states, key=lambda item: abs(item.monotonic_ns - capture))
        if any(
            abs(measured.monotonic_ns - timestamp) > self.state_tolerance_ns
            for timestamp in captures
        ):
            return
        if self.samples and measured.monotonic_ns <= self.samples[-1].state.monotonic_ns:
            return
        # SensorFrame data are owned by the immutable received PUB payload.
        self.samples.append(ObservationSnapshot(measured, frames))
        self.last_sequences = sequences

    def window(self, now_ns):
        if len(self.samples) < 2:
            return None
        newest = self.samples[-1]
        if now_ns - newest.state.monotonic_ns > self.max_age_ns:
            return None
        target = newest.state.monotonic_ns - self.period_ns
        previous = min(
            list(self.samples)[:-1], key=lambda sample: abs(sample.state.monotonic_ns - target)
        )
        if abs(previous.state.monotonic_ns - target) > self.tolerance_ns:
            return None
        # Enforce the same interval for each physical camera, not receipt polls.
        if any(
            abs(
                newest.frames[name].capture_monotonic_ns
                - previous.frames[name].capture_monotonic_ns
                - self.period_ns
            )
            > self.tolerance_ns
            for name in self.names
        ):
            return None
        frames = dict(newest.frames)
        frames.update({name + "_prev": frame for name, frame in previous.frames.items()})
        return WindowSnapshot(newest.state, frames, previous)


def align_rtc_condition(request, timeline, now_ns, *, offset_ns, dt_ns, group_order, horizon):
    """Sample the actual committed trajectory at the new model's target times.

    This handles first_action_offset != action_dt and removes weights beyond the
    committed trajectory, including after timeline trim. Pose FK happens later.
    """
    if getattr(request, "action_condition", None) is None:
        return
    active = timeline.active_horizon()
    if active is None:
        raise ValueError("RTC condition has no committed trajectory")
    old = np.concatenate([active.groups[name] for name in group_order], axis=1)
    target = request.observation_time_ns + offset_ns + np.arange(horizon) * dt_ns
    old_times = active.start_time_ns + np.arange(len(old)) * active.dt_ns
    weight_times = (
        active.start_time_ns
        + (timeline.cursor(now_ns) + np.arange(len(request.condition_weights))) * dt_ns
    )
    rows = np.column_stack(
        [np.interp(target, old_times, old[:, column]) for column in range(old.shape[1])]
    )
    weights = np.interp(target, weight_times, request.condition_weights, left=0.0, right=0.0)
    weights[(target < old_times[0]) | (target > old_times[-1])] = 0
    request.action_condition = rows
    request.condition_weights = weights


class HistoryStrategy:
    # Read by EdgeRuntime: while the viewer holds the rollout paused, submit no
    # inference and drop plans. A plan committed during the pause keeps its
    # wall-clock start, so Start would make the arms chase it mid-way from an
    # observation taken before Start.
    discard_plans_while_paused = True

    def __init__(self, config):
        options = config["policy"]["adapter"]
        expected = config["policy"].get("expected_backend")
        identity = {} if expected is None else expected.get("model", {})
        for key in ("observation_period_s", "first_action_offset_s"):
            if not np.isfinite(identity.get(key, np.nan)) or identity[key] <= 0:
                raise ValueError(f"Bind UMI checkpoint {key} before constructing history")
        history = config["inference"]["history"]
        delegate_name = config["inference"]["algorithm"]
        if delegate_name not in {"manimux", "rtc"}:
            raise ValueError("UMI history supports the existing manimux and rtc strategies")
        delegate_config = deepcopy(config)
        delegate_config["inference"]["strategy"] = None
        # 配置已补齐默认值；只复查切换 delegate 后的时间与动作约束。
        validate_inference_parameters(delegate_config["inference"], delegate_config["executor"])
        validate_runtime_parameters(delegate_config)
        self.delegate = build_inference_strategy(delegate_config)
        camera_map = options["camera_map"]
        names = [value for key, value in camera_map.items() if not key.endswith("_prev")]
        self.history = MeasuredHistory(
            camera_names=names,
            period_s=float(identity["observation_period_s"]),
            tolerance_s=float(history["period_tolerance_s"]),
            state_tolerance_s=float(history["state_tolerance_s"]),
            camera_skew_s=float(history["camera_skew_s"]),
        )
        self.offset_ns = round(float(identity["first_action_offset_s"]) * 1e9)
        self.dt_ns = round(action_interval(config["policy"]) * 1e9)
        self.group_order = tuple(config["robot"]["group_dims"])
        self.horizon = config["policy"]["horizon_policy_steps"]

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def reset(self):
        self.history.reset()
        self.delegate.reset()

    def build_submission(self, **kwargs):
        self.history.observe(kwargs["snapshot"])
        window = self.history.window(kwargs["now_ns"])
        if window is None:
            return None
        submission = self.delegate.build_submission(**{**kwargs, "snapshot": window})
        if submission is not None:
            align_rtc_condition(
                submission.request,
                kwargs["timeline"],
                kwargs["now_ns"],
                offset_ns=self.offset_ns,
                dt_ns=self.dt_ns,
                group_order=self.group_order,
                horizon=self.horizon,
            )
            weights = getattr(submission.request, "condition_weights", None)
            if weights is not None and not np.any(weights > 0):
                submission.request.action_condition = None
                submission.request.condition_weights = None
                self.delegate.clear_condition(submission.request.request_seq)
                submission.event_fields.update(
                    conditioned=False, forecast_delay=0, condition_reason="no_committed_overlap"
                )
        return submission


def build_strategy(config):
    return HistoryStrategy(config)
