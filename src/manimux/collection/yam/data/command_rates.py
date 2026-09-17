"""Successful application-to-SDK submissions, measured over elapsed wall time."""

from __future__ import annotations

import time
from collections import deque
from threading import Lock

import numpy as np


class CommandRates:
    """Count completed dual-arm submissions, not loop iterations or CAN frames.

    A new source sequence counts once even when the executor sends it repeatedly.
    Skipped source sequences do not count, and equal joint values in independently
    sampled targets still count. Call ``record`` only after send_command succeeds.
    The small statistics lock never protects device I/O; percentile calculation is
    performed by the status reader after releasing it, outside the control thread.
    """

    def __init__(self, window_s: float = 2.0, *, clock=time.monotonic_ns):
        if not np.isfinite(window_s) or window_s <= 0:
            raise ValueError("rate window must be finite and positive")
        self.window_ns = round(window_s * 1e9)
        if self.window_ns < 1:
            raise ValueError("rate window must be at least one nanosecond")
        self._clock = clock
        self._lock = Lock()
        self._events = deque()
        self._last_sequence = None
        self._last_submit_ns = None
        self._last_fresh_ns = None

    def record(self, sequence: int) -> None:
        with self._lock:
            now = self._clock()
            fresh = sequence != self._last_sequence
            self._last_sequence = sequence
            submit_gap = None if self._last_submit_ns is None else now - self._last_submit_ns
            fresh_gap = (
                now - self._last_fresh_ns
                if fresh and self._last_fresh_ns is not None else None
            )
            self._last_submit_ns = now
            if fresh:
                self._last_fresh_ns = now
            self._events.append((now, fresh, submit_gap, fresh_gap))
            cutoff = now - self.window_ns
            # Store intervals at completion so long gaps survive expiration of
            # their starting events. Idle time stays in the rate denominator.
            while self._events and self._events[0][0] <= cutoff:
                self._events.popleft()

    def snapshot(self) -> dict:
        with self._lock:
            now = self._clock()
            events = tuple(self._events)
            last_submit_ns = self._last_submit_ns
        cutoff = now - self.window_ns
        current = [event for event in events if cutoff < event[0] <= now]
        fresh_count = sum(event[1] for event in current)
        window_s = self.window_ns / 1e9

        def intervals(index):
            spans = [event[index] / 1e6 for event in current if event[index] is not None]
            return {
                "count": len(spans),
                "p95": float(np.percentile(spans, 95)) if spans else None,
                "max": max(spans) if spans else None,
            }

        return {
            "window_s": window_s,
            "submitted_count": len(current),
            "fresh_target_count": fresh_count,
            "repeated_count": len(current) - fresh_count,
            "submitted_hz": len(current) / window_s,
            "fresh_target_hz": fresh_count / window_s,
            "last_submit_age_ms": (
                (now - last_submit_ns) / 1e6 if last_submit_ns is not None else None
            ),
            "submit_interval_ms": intervals(2),
            "fresh_target_interval_ms": intervals(3),
        }
