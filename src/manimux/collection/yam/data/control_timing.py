"""Thread-local teleop stage timings; no device reads or control-thread file I/O."""

from __future__ import annotations

import copy
import json
import logging
import time
import uuid
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from threading import Lock, Thread

import numpy as np

from manimux.timing import active_timing as _active
from manimux.timing import stage as stage
from manimux.timing import timing_scope as arm_scope

__all__ = ["ControlTimings", "arm_scope", "recording_snapshot", "stage", "summarize"]


def recording_snapshot() -> dict | None:
    """Copy before the recorder releases its lock, so freeze cannot split a row.

    The saved work duration ends here (after buffering the sample); live work
    duration additionally includes returning from recorder.tick and the loop.
    """
    row = _active.get()
    if row is None:
        return None
    return {
        **row,
        "work_ns": time.monotonic_ns() - row["cycle_monotonic_ns"],
        "work_cpu_ns": time.thread_time_ns() - row["cycle_thread_cpu_ns"],
        "stages": copy.deepcopy(row["stages"]),
    }


def summarize(rows: list[dict]) -> dict:
    values: dict[str, list[float]] = {}
    cpu_values: dict[str, list[float]] = {}
    for row in rows:
        for key in ("work_ns", "period_ns"):
            if row.get(key) is not None:
                values.setdefault(key.removesuffix("_ns"), []).append(row[key] / 1e6)
        if "work_cpu_ns" in row:
            cpu_values.setdefault("work", []).append(row["work_cpu_ns"] / 1e6)
        for name, span in row["stages"].items():
            values.setdefault(name, []).append(span["duration_ns"] / 1e6)
            if "cpu_ns" in span:
                cpu_values.setdefault(name, []).append(span["cpu_ns"] / 1e6)
        pacing = row.get("previous_pacing") or {}
        for key in ("requested_sleep_ns", "actual_sleep_ns", "sleep_overshoot_ns"):
            if key in pacing:
                values.setdefault("previous." + key.removesuffix("_ns"), []).append(
                    pacing[key] / 1e6
                )
    return {
        "samples": len(rows),
        "stages_ms": {
            name: {
                "count": len(v), "last": v[-1], "mean": float(np.mean(v)),
                "p95": float(np.percentile(v, 95)), "max": max(v),
                "cpu_mean": float(np.mean(cpu_values[name])) if name in cpu_values else None,
            }
            for name, v in values.items()
        },
    }


class ControlTimings:
    def __init__(self, window: int = 200):
        self._rows = deque(maxlen=window)
        self._lock = Lock()
        self._previous_start = None
        self._previous_pacing = None
        self._output_root = None
        self._metadata = {}
        self._pending = []
        self._save_reason = None
        self._saves = []

    def configure_output(self, root: str | Path, metadata: dict) -> None:
        """Enable complete teleop logs independently of episode recording."""
        self._output_root = Path(root)
        self._metadata = metadata

    def request_save(self, reason: str) -> None:
        # A button callback can run inside the current cycle. Defer detaching
        # the buffer until that cycle has finished, including its final command.
        with self._lock:
            self._save_reason = reason

    def flush(self, reason: str) -> None:
        with self._lock:
            self._save_reason = None
            if self._output_root is None or not self._pending:
                return
            rows, self._pending = self._pending, []
            path = self._output_root / (time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
            result = {"path": str(path), "status": "saving", "rows": len(rows), "reason": reason}
            worker = Thread(target=self._save, args=(path, rows, result), name="teleop-diagnostics")
            self._saves.append((worker, result))
            worker.start()

    def _save(self, path, rows, result):
        try:
            path.mkdir(parents=True)
            pending = path / "control-timing.jsonl.part"
            with pending.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
            pending.rename(path / "control-timing.jsonl")
            (path / "metadata.json").write_text(json.dumps({
                "schema_version": 2, **self._metadata, "reason": result["reason"],
                "rows": len(rows), "scope": "application/SDK timings, not CAN reception; "
                "alignment and pause are separate operations; nested stages overlap; "
                "wall minus thread CPU includes both waiting and descheduling; "
                "previous_pacing refers to its explicitly named preceding cycle",
            }, indent=2, allow_nan=False), encoding="utf-8")
            (path / "write_complete.flag").touch()
            with self._lock:
                result["status"] = "saved"
            print(f"[yam-abc] teleop diagnostics saved: {path}", flush=True)
        except Exception as exc:
            with self._lock:
                result.update(status="error", error=f"{type(exc).__name__}: {exc}")
            logging.exception("failed to save teleop diagnostics to %s", path)

    def wait_for_saves(self):
        with self._lock:
            workers = [thread for thread, _ in self._saves]
        for thread in workers:
            thread.join()

    def output_status(self):
        with self._lock:
            return {"buffered_rows": len(self._pending),
                    "latest": dict(self._saves[-1][1]) if self._saves else None}

    def record_pacing(self, *, work_s, requested_s, actual_ns, total_s, actual_hz):
        self._previous_pacing = {
            "cycle_monotonic_ns": self._previous_start,
            "pre_sleep_work_ns": round(work_s * 1e9),
            "requested_sleep_ns": round(requested_s * 1e9),
            "actual_sleep_ns": actual_ns,
            "sleep_overshoot_ns": max(0, actual_ns - round(requested_s * 1e9)),
            "loop_period_ns": round(total_s * 1e9),
            "display_actual_hz": actual_hz,
        }

    @contextmanager
    def cycle(self, budget_s: float, *, kind="cycle", sync_enabled=False):
        start = time.monotonic_ns()
        row = {
            "schema_version": 2, "kind": kind, "sync_enabled": sync_enabled,
            "cycle_monotonic_ns": start, "cycle_unix_ns": time.time_ns(),
            "cycle_thread_cpu_ns": time.thread_time_ns(),
            "period_ns": (start - self._previous_start
                          if kind == "cycle" and self._previous_start is not None else None),
            "budget_ns": round(budget_s * 1e9), "stages": {},
        }
        if kind == "cycle":
            self._previous_start = start
            row["previous_pacing"], self._previous_pacing = self._previous_pacing, None
        token = _active.set(row)
        try:
            yield row
        except BaseException as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            row["work_cpu_ns"] = time.thread_time_ns() - row["cycle_thread_cpu_ns"]
            row["work_ns"] = time.monotonic_ns() - start
            _active.reset(token)
            with self._lock:
                if kind == "cycle":
                    self._rows.append(row)
                if self._output_root is not None and (
                    kind != "cycle" or sync_enabled or "error" in row
                    or "backend.follower_sdk_submit" in row["stages"] or self._save_reason
                ):
                    self._pending.append(row)
                save_reason = self._save_reason if kind == "cycle" else None
            if save_reason:
                self.flush(save_reason)

    def snapshot(self) -> dict:
        with self._lock:
            rows = list(self._rows)
        return {**summarize(rows), "latest": rows[-1] if rows else None}
