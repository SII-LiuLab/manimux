"""Time-addressed episode memory for InternW0-Delta.

The model conditions on three frames: the episode anchor (first observation after Start),
the frame ``lag_policy_steps`` policy steps before the request, and the current frame.
Frames are looked up by capture time, so serial, async and multi-inflight scheduling all
see the same history; request count and cadence do not matter.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from manimux.types import ObservationSnapshot, RobotState, copy_group_vector


@dataclass(slots=True)
class MemorySnapshot(ObservationSnapshot):
    anchor: ObservationSnapshot | None = None
    recent: ObservationSnapshot | None = None
    recent_is_real: bool = False
    episode_step: int = 0


def _capture_ns(frames, names) -> int:
    return sum(frames[name].capture_monotonic_ns for name in names) // len(names)


class EpisodeMemory:
    def __init__(
        self,
        camera_names,
        *,
        step_s: float,
        lag_steps: int = 32,
        frame_tolerance_s: float = 0.05,
        state_tolerance_s: float = 0.02,
        camera_skew_s: float = 0.1,
    ):
        self.names = tuple(camera_names)
        self.step_ns = round(step_s * 1e9)
        self.lag_steps = int(lag_steps)
        self.frame_tolerance_ns = round(frame_tolerance_s * 1e9)
        self.state_tolerance_ns = round(state_tolerance_s * 1e9)
        self.skew_ns = round(camera_skew_s * 1e9)
        if min(self.step_ns, self.lag_steps, self.frame_tolerance_ns, self.state_tolerance_ns) <= 0:
            raise ValueError("InternW0 memory step, lag and tolerances must be positive")
        self.reset()

    def reset(self):
        self.anchor = None  # (capture_ns, frames, RobotState); never trimmed
        self.frames = deque()  # (capture_ns, frames, sequences)
        self.states = deque()  # RobotState
        self.last_sequences = None

    def _usable(self, snapshot) -> bool:
        if any(name not in snapshot.frames for name in self.names):
            return False
        captures = [snapshot.frames[name].capture_monotonic_ns for name in self.names]
        return max(captures) - min(captures) <= self.skew_ns

    def observe(self, snapshot):
        state = snapshot.state
        if self.states and state.monotonic_ns < self.states[-1].monotonic_ns:
            self.reset()
        if not self.states or state.monotonic_ns != self.states[-1].monotonic_ns:
            self.states.append(
                RobotState(copy_group_vector(state.groups), state.monotonic_ns, state.sequence)
            )
        if not self._usable(snapshot):
            return
        frames = {name: snapshot.frames[name] for name in self.names}
        sequences = tuple(frames[name].sequence for name in self.names)
        if sequences == self.last_sequences:
            return
        capture = _capture_ns(frames, self.names)
        if self.anchor is None:
            anchor_state = self._state_at(capture)
            if anchor_state is None:
                return  # wait for a frame whose measured state is available
            self.anchor = (capture, frames, anchor_state)
        self.last_sequences = sequences
        self.frames.append((capture, frames, sequences))
        horizon = capture - (self.lag_steps * self.step_ns + 2 * self.frame_tolerance_ns)
        while len(self.frames) > 2 and self.frames[0][0] < horizon:
            self.frames.popleft()
        while len(self.states) > 2 and self.states[0].monotonic_ns < horizon - self.state_tolerance_ns:
            self.states.popleft()

    def _state_at(self, capture_ns: int):
        if not self.states:
            return None
        state = min(self.states, key=lambda item: abs(item.monotonic_ns - capture_ns))
        if abs(state.monotonic_ns - capture_ns) > self.state_tolerance_ns:
            return None
        return state

    def snapshot(self, current: ObservationSnapshot):
        """Attach anchor/recent history to ``current``; None while history is unavailable."""
        if self.anchor is None or not self._usable(current):
            return None
        now = _capture_ns(current.frames, self.names)
        anchor_ns, anchor_frames, anchor_state = self.anchor
        step = max(0, round((now - anchor_ns) / self.step_ns))
        anchor = ObservationSnapshot(anchor_state, anchor_frames)
        if step < self.lag_steps:
            return MemorySnapshot(current.state, current.frames, anchor, anchor, False, step)
        target = now - self.lag_steps * self.step_ns
        capture, frames, _ = min(self.frames, key=lambda item: abs(item[0] - target))
        state = self._state_at(capture)
        if abs(capture - target) > self.frame_tolerance_ns or state is None:
            return None
        recent = ObservationSnapshot(state, frames)
        return MemorySnapshot(current.state, current.frames, anchor, recent, True, step)


class MemoryStrategy:
    """Feed anchor/recent history to an unchanged scheduling strategy."""

    # While paused no inference is scheduled and observe_snapshot is not called, so memory
    # starts at the first running tick; the pause transition resets it.
    discard_plans_while_paused = True

    def __init__(self, config):
        from manimux.runtime.inference import build_inference_strategy

        algorithm = config["inference"]["algorithm"]
        if algorithm not in {"serial", "async"}:
            raise ValueError(
                f"InternW0 memory supports serial and async only, not {algorithm!r}: the "
                "policy has no sampler hooks for RTC, PAINT, BID or AutoHorizon"
            )
        options = config["policy"]["adapter"].get("memory", {})
        delegate = deepcopy(config)
        delegate["inference"]["strategy"] = None
        self.delegate = build_inference_strategy(delegate)
        self.memory = EpisodeMemory(
            config["policy"]["adapter"]["physical_cameras"],
            step_s=config["policy"]["action_dt_s"],
            lag_steps=options.get("lag_policy_steps", 32),
            frame_tolerance_s=options.get("frame_tolerance_s", 0.05),
            state_tolerance_s=options.get("state_tolerance_s", 0.02),
            camera_skew_s=options.get("camera_skew_s", 0.1),
        )

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def reset(self):
        self.memory.reset()
        self.delegate.reset()

    def observe_snapshot(self, snapshot):
        self.memory.observe(snapshot)

    def build_submission(self, **kwargs):
        window = self.memory.snapshot(kwargs["snapshot"])
        if window is None:
            return None
        return self.delegate.build_submission(**{**kwargs, "snapshot": window})


def build_strategy(config):
    return MemoryStrategy(config)


__all__ = ["EpisodeMemory", "MemorySnapshot", "MemoryStrategy", "build_strategy"]
