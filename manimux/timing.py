"""Passive timing spans shared by runtime instrumentation and robot wrappers.

Only the calling thread's active diagnostic row is updated. No device access,
file I/O, lock replacement, or background SDK instrumentation happens here.
"""

from __future__ import annotations

import json
import time
from collections import deque
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

active_timing: ContextVar[dict | None] = ContextVar("manimux_timing", default=None)
_prefix: ContextVar[str] = ContextVar("manimux_timing_prefix", default="")


def timed(name: str):
    """Measure a synchronous call only when this thread has an active cycle."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            if active_timing.get() is None:
                return function(*args, **kwargs)
            with stage(name):
                return function(*args, **kwargs)
        return wrapped
    return decorate


@contextmanager
def timed_lock(lock, name: str):
    if active_timing.get() is None:
        with lock:
            yield
        return
    # Register release before closing the timing span, including if recording
    # the span or the protected body raises. Preserve the lock's context protocol.
    with ExitStack() as stack:
        with stage(name):
            stack.enter_context(lock)
        yield


@contextmanager
def timing_scope(name: str):
    if active_timing.get() is None:
        yield
        return
    token = _prefix.set(_prefix.get() + name + ".")
    try:
        yield
    finally:
        _prefix.reset(token)


@contextmanager
def stage(name: str, **attributes):
    row = active_timing.get()
    if row is None:
        yield
        return
    name = _prefix.get() + name
    start = time.monotonic_ns()
    cpu_start = time.thread_time_ns()
    span = {"offset_ns": start - row["cycle_monotonic_ns"], **attributes}
    try:
        yield
    except BaseException as exc:
        span["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        span["cpu_ns"] = time.thread_time_ns() - cpu_start
        span["duration_ns"] = time.monotonic_ns() - start
        previous = row["stages"].get(name)
        if previous is None:
            row["stages"][name] = span
        else:
            # Alignment can submit many commands inside one operation. Keep every
            # call's timestamp instead of silently overwriting the earlier ones.
            if "spans" not in previous:
                previous["spans"] = [dict(previous)]
            previous["spans"].append(span)
            previous["duration_ns"] += span["duration_ns"]
            previous["cpu_ns"] += span["cpu_ns"]
            if "error" in span:
                previous["error"] = span["error"]


class LoopTiming:
    """Bounded, caller-thread diagnostics; persistence happens only in write().

    Sampling always uses host monotonic/thread CPU clocks, independently of an
    injected scheduler clock. This measures host calls, not motor/CAN arrival.
    """

    def __init__(
        self, *, enabled=False, max_cycles=20000, period_ns, clock_source="SystemClock",
    ) -> None:
        if type(max_cycles) is not int or max_cycles <= 0:
            raise ValueError("max_cycles must be a positive integer")
        if type(period_ns) is not int or period_ns <= 0:
            raise ValueError("period_ns must be a positive integer")
        self.enabled = enabled
        self.max_cycles = max_cycles
        self.period_ns = period_ns
        self.clock_source = clock_source
        self._rows: deque[dict] = deque(maxlen=max_cycles)
        self._row: dict | None = None
        self._token = None
        self._cycles = 0
        self._last_phase: str | None = None
        self._segment_id = -1

    def begin(self, scheduled_start_ns: int, phase: str, step: int) -> None:
        if not self.enabled:
            return
        self.end(completed=False)
        self._row = {
            "cycle_index": self._cycles,
            "cycle_monotonic_ns": time.monotonic_ns(),
            "cycle_thread_cpu_ns": time.thread_time_ns(),
            "scheduled_start_ns": scheduled_start_ns,
            "phase": phase,
            "step": step,
            "stages": {},
        }
        self._token = active_timing.set(self._row)

    def set_phase(self, phase: str) -> None:
        if self._row is not None:
            self._row["phase"] = phase

    def mark_work_end(self) -> None:
        row = self._row
        if row is None or "work_end_ns" in row:
            return
        row["work_end_ns"] = time.monotonic_ns()
        row["work_ns"] = row["work_end_ns"] - row["cycle_monotonic_ns"]
        row["work_cpu_ns"] = time.thread_time_ns() - row["cycle_thread_cpu_ns"]
        row["deadline_lag_ns"] = (
            max(0, row["work_end_ns"] - row["scheduled_start_ns"] - self.period_ns)
            if self.clock_source == "SystemClock" else None
        )

    def sleep_until(self, clock, target_ns: int) -> None:
        row = self._row
        if row is None:
            clock.sleep_until_ns(target_ns)
            return
        self.mark_work_end()
        row["sleep_target_ns"] = target_ns
        row["sleep_start_ns"] = time.monotonic_ns()
        try:
            clock.sleep_until_ns(target_ns)
        finally:
            row["sleep_end_ns"] = time.monotonic_ns()
            row["sleep_ns"] = row["sleep_end_ns"] - row["sleep_start_ns"]
            # A virtual/custom scheduler may use a different epoch. Never
            # subtract its target from an unrelated host-clock timestamp.
            row["sleep_overshoot_ns"] = (
                max(0, row["sleep_end_ns"] - target_ns)
                if self.clock_source == "SystemClock" else None
            )

    def end(self, completed: bool = True) -> None:
        row = self._row
        if row is None:
            return
        try:
            self.mark_work_end()
            row["cycle_end_ns"] = time.monotonic_ns()
            row["cycle_cpu_ns"] = time.thread_time_ns() - row["cycle_thread_cpu_ns"]
            row["completed"] = completed
            if row["phase"] != self._last_phase:
                self._segment_id += 1
            row["segment_id"] = self._segment_id
            self._last_phase = row["phase"]
            self._rows.append(row)
            self._cycles += 1
        finally:
            active_timing.reset(self._token)
            self._token = None
            self._row = None

    def write(self, directory: Path | str) -> dict | None:
        """Finalize any interrupted cycle and write the two episode artifacts."""
        if not self.enabled:
            return None
        self.end(completed=False)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema": "manimux-control-timing-v1",
            "period_ns": self.period_ns,
            "clock_source": self.clock_source,
            "measurement_clock": "time.monotonic_ns",
            "cpu_clock": "time.thread_time_ns",
            "real_monotonic_timestamps": True,
            "schedule_clock_comparable": self.clock_source == "SystemClock",
            "hardware_rate_verified": False,
            "max_cycles": self.max_cycles,
            "cycles_observed": self._cycles,
            "rows_retained": len(self._rows),
            "rows_dropped": self._cycles - len(self._rows),
            "truncated": self._cycles > len(self._rows),
        }
        rows = list(self._rows)
        summary = summarize_control_timing(rows, metadata)
        for name, payload in (
            ("control_timing.jsonl", rows),
            ("control_timing_summary.json", summary),
        ):
            temporary = directory / f".{name}.tmp"
            with temporary.open("w", encoding="utf-8") as handle:
                if name.endswith("jsonl"):
                    for row in payload:
                        handle.write(json.dumps(row, allow_nan=False) + "\n")
                else:
                    json.dump(payload, handle, indent=2, allow_nan=False)
                    handle.write("\n")
            temporary.replace(directory / name)
        return summary


def _duration_stats(values: list[int]) -> dict:
    """Offline-only statistics; interpolate percentiles like numpy's default."""
    if not values:
        return {"count": 0}
    ordered = sorted(values)

    def percentile(fraction):
        index = (len(ordered) - 1) * fraction
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        return (ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)) / 1e6

    return {
        "count": len(values),
        "mean_ms": sum(values) / len(values) / 1e6,
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "max_ms": ordered[-1] / 1e6,
        "over_10ms_count": sum(value > 10_000_000 for value in values),
        "over_10ms_pct": 100 * sum(value > 10_000_000 for value in values) / len(values),
    }


def _spans(stage_data: dict) -> list[dict]:
    return stage_data.get("spans", [stage_data])


def _summarize_segment(rows: list[dict], period_ns: int) -> dict:
    complete = [row for row in rows if row["completed"]]
    periods = []
    send_intervals = []
    previous_send = None
    previous_index = None
    for previous, current in zip(rows, rows[1:], strict=False):
        if (
            previous["completed"] and current["completed"]
            and current["cycle_index"] == previous["cycle_index"] + 1
        ):
            periods.append(current["cycle_monotonic_ns"] - previous["cycle_monotonic_ns"])
    for row in rows:
        if (
            not row["completed"]
            or (previous_index is not None and row["cycle_index"] != previous_index + 1)
        ):
            previous_send = None
        previous_index = row["cycle_index"]
        if not row["completed"]:
            continue
        send = row["stages"].get("robot_send_command")
        if send is not None:
            for span in _spans(send):
                stamp = row["cycle_monotonic_ns"] + span["offset_ns"]
                if previous_send is not None:
                    send_intervals.append(stamp - previous_send)
                previous_send = stamp
    stage_values: dict[str, dict[str, list[int]]] = {}
    for row in complete:
        for name, data in row["stages"].items():
            values = stage_values.setdefault(name, {"wall": [], "cpu": []})
            for span in _spans(data):
                values["wall"].append(span["duration_ns"])
                values["cpu"].append(span["cpu_ns"])
    periods_stats = _duration_stats(periods)
    periods_stats["over_target_count"] = sum(value > period_ns for value in periods)
    work = _duration_stats([row["work_ns"] for row in complete])
    work["over_target_count"] = sum(row["work_ns"] > period_ns for row in complete)
    work["over_target_pct"] = (
        100 * work["over_target_count"] / len(complete) if complete else None
    )
    return {
        "segment_id": rows[0]["segment_id"],
        "phase": rows[0]["phase"],
        "first_cycle_index": rows[0]["cycle_index"],
        "last_cycle_index": rows[-1]["cycle_index"],
        "rows": len(rows),
        "completed_rows": len(complete),
        "incomplete_rows": len(rows) - len(complete),
        "actual_hz": len(periods) * 1e9 / sum(periods) if periods and sum(periods) > 0 else None,
        "period": periods_stats,
        "work": work,
        "work_cpu": _duration_stats([row["work_cpu_ns"] for row in complete]),
        "cycle_cpu": _duration_stats([row["cycle_cpu_ns"] for row in complete]),
        "sleep": _duration_stats([row["sleep_ns"] for row in complete if "sleep_ns" in row]),
        "sleep_overshoot": _duration_stats([
            row["sleep_overshoot_ns"] for row in complete
            if row.get("sleep_overshoot_ns") is not None
        ]),
        "send_interval": _duration_stats(send_intervals),
        "deadline_lag": _duration_stats([
            row["deadline_lag_ns"] for row in complete
            if row.get("deadline_lag_ns") is not None
        ]),
        "send_actual_hz": (
            len(send_intervals) * 1e9 / sum(send_intervals)
            if send_intervals and sum(send_intervals) > 0 else None
        ),
        "stages": {
            name: {kind: _duration_stats(values) for kind, values in fields.items()}
            for name, fields in sorted(stage_values.items())
        },
    }


def summarize_control_timing(rows: list[dict], metadata: dict) -> dict:
    """Summarize only recorded host spans, never infer timing from robot ticks."""
    segments: list[list[dict]] = []
    for row in rows:
        if (
            not segments
            or row["segment_id"] != segments[-1][-1]["segment_id"]
            or row["phase"] != segments[-1][-1]["phase"]
        ):
            segments.append([])
        segments[-1].append(row)
    return {
        "metadata": dict(metadata),
        "scope": {
            "rates": "Host loop start-to-start, within a continuous phase only; not motor Hz.",
            "paused": "Paused and warmup segments are separate from running segments.",
            "spans": "Nested stage durations overlap and must not be added together.",
            "cpu": "Caller-thread CPU only; wall minus CPU is not proof of lock contention.",
            "send": "robot_send_command call starts, not CAN arrival or actuator completion.",
            "coverage": (
                "Statistics describe retained completed rows; dropped rows are not estimated."
            ),
            "clock": (
                "Synthetic scheduling and real host timing do not establish hardware performance."
            ),
        },
        "segments": [_summarize_segment(segment, metadata["period_ns"]) for segment in segments],
    }
