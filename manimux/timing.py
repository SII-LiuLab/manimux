"""Passive timing spans shared by runtime instrumentation and robot wrappers.

Only the calling thread's active diagnostic row is updated. No device access,
file I/O, lock replacement, or background SDK instrumentation happens here.
"""

from __future__ import annotations

import time
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar

active_timing: ContextVar[dict | None] = ContextVar("manimux_timing", default=None)
_prefix: ContextVar[str] = ContextVar("manimux_timing_prefix", default="")


@contextmanager
def timed_lock(lock, name: str):
    # Register release before closing the timing span, including if recording
    # the span or the protected body raises. Preserve the lock's context protocol.
    with ExitStack() as stack:
        with stage(name):
            stack.enter_context(lock)
        yield


@contextmanager
def timing_scope(name: str):
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
