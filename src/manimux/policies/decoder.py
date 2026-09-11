"""Bounded, process-isolated embodiment decoding with optional independent partitions."""

from __future__ import annotations

import copy
import multiprocessing as mp
import queue
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from manimux.config import PolicyConfig, RobotConfig
from manimux.policies import build_policy_adapter
from manimux.policies.base import decode_policy_action
from manimux.types import ActionChunk, ActionContext, InferenceResponse


@dataclass
class DecodeResult:
    response: InferenceResponse
    chunk: ActionChunk | None
    error: str | None = None


def _decode_main(requests, results, startup, robot_data, policy_data, partition):
    try:
        robot = RobotConfig.model_validate(robot_data)
        policy = PolicyConfig.model_validate(policy_data)
        adapter = build_policy_adapter(robot, policy)
        adapter.validate(robot, policy)
        if not getattr(adapter, "supports_context_only_decode", False):
            raise ValueError("adapter does not support context-only decoding")
        warmup = getattr(adapter, "warmup_decode", None)
        if callable(warmup):
            warmup(partition)
        startup.put((partition, None))
    except Exception as exc:
        startup.put((partition, f"{type(exc).__name__}:{exc}"))
        return
    while True:
        job = requests.get()
        if job is None:
            return
        raw, context, deadline_ns = job
        chunk, error = None, None
        started = time.monotonic_ns()
        try:
            if started > deadline_ns:
                raise TimeoutError("decode deadline exceeded before start")
            if partition is None:
                chunk = decode_policy_action(adapter, raw, context)
            else:
                chunk = adapter.decode_action_partition(raw, context, partition)
            if time.monotonic_ns() > deadline_ns:
                raise TimeoutError("decode deadline exceeded")
        except Exception as exc:
            chunk = None
            error = f"decode_error:{type(exc).__name__}:{exc}"
        elapsed_ms = (time.monotonic_ns() - started) / 1e6
        results.put((context.request_seq, partition, chunk, error, elapsed_ms))


class ActionDecoderClient:
    """One in-flight chunk; optionally substitute a hold for late partitions.

    Each spawned process constructs an independent adapter/kinematics instance.
    Only raw actions and an immutable snapshot of decode context cross the queue;
    camera frames, robot drivers and control sockets never enter these processes.
    """

    def __init__(self, robot: RobotConfig, policy: PolicyConfig, adapter: Any):
        if not getattr(adapter, "supports_context_only_decode", False):
            raise ValueError("process decoding requires a context-only adapter")
        self._partitions = tuple(getattr(adapter, "decode_partitions", ())) or (None,)
        self._adapter = adapter
        self._context: ActionContext | None = None
        self._worker_jobs: dict[str | None, int] = {}
        if len(set(self._partitions)) != len(self._partitions):
            raise ValueError("decode partitions must be unique")
        ctx = mp.get_context("spawn")
        self._results = ctx.Queue(maxsize=len(self._partitions))
        self._startup = ctx.Queue(maxsize=len(self._partitions))
        self._requests = [ctx.Queue(maxsize=1) for _ in self._partitions]
        self._processes = [
            ctx.Process(
                target=_decode_main,
                args=(
                    requests,
                    self._results,
                    self._startup,
                    robot.model_dump(mode="python"),
                    policy.model_dump(mode="python"),
                    partition,
                ),
                name=f"manimux-decode-{partition or 'action'}",
                daemon=True,
            )
            for partition, requests in zip(self._partitions, self._requests, strict=True)
        ]
        self._startup_timeout_s = policy.startup_timeout_s
        self._started = False
        self._pending: InferenceResponse | None = None
        self._pieces: dict[str | None, tuple] = {}
        self._submitted_ns = 0
        self._deadline_ns = 0

    @property
    def busy(self) -> bool:
        return self._pending is not None

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        try:
            for process in self._processes:
                process.start()
            deadline = time.monotonic() + self._startup_timeout_s
            for _ in self._partitions:
                _, error = self._startup.get(timeout=max(0.001, deadline - time.monotonic()))
                if error:
                    raise RuntimeError(f"action decoder startup failed: {error}")
        except BaseException:
            self.close()
            raise

    def submit(self, response: InferenceResponse, context: ActionContext, deadline_ns: int) -> None:
        if self.busy:
            raise RuntimeError("action decoder already has an in-flight chunk")
        if not self._started or not all(p.is_alive() for p in self._processes):
            raise RuntimeError("action decoder is not running")
        # Queue feeder threads serialize later: don't retain mutable sensor/state arrays.
        job = copy.deepcopy((response.raw_action, context, deadline_ns))
        self._pending = response
        self._submitted_ns = time.monotonic_ns()
        self._deadline_ns = deadline_ns
        self._pieces = {}
        self._context = context
        if context.independent_groups:
            if context.decode_budget_ms is None:
                raise ValueError("independent decoding requires decode_budget_ms")
            # Leave 40 ms for queue scheduling/serialization beyond the IK budget.
            self._deadline_ns = min(deadline_ns, self._submitted_ns
                                    + int((context.decode_budget_ms + 40) * 1e6))
        for partition, requests in zip(self._partitions, self._requests, strict=True):
            if partition in self._worker_jobs:
                if not context.independent_groups:
                    raise RuntimeError("previous decoder job is still running")
                self._pieces[partition] = (self._hold(partition, "worker_busy"), None, 0.0)
                continue
            self._worker_jobs[partition] = response.request_seq
            requests.put_nowait(job)

    def _hold(self, partition, reason):
        assert self._pending is not None and self._context is not None
        return self._adapter.decode_hold_partition(
            self._pending.raw_action, self._context, partition, reason,
        )

    def poll(self) -> DecodeResult | None:
        if self._started and not all(p.is_alive() for p in self._processes):
            raise RuntimeError("action decoder process stopped")
        while True:
            try:
                seq, partition, chunk, error, elapsed_ms = self._results.get_nowait()
            except queue.Empty:
                break
            if partition not in self._partitions or self._worker_jobs.get(partition) != seq:
                raise RuntimeError("action decoder returned an unexpected job")
            del self._worker_jobs[partition]
            if self._pending is None or seq != self._pending.request_seq:
                # A timed-out partition may finish after another plan was submitted.
                continue
            if (error and error.startswith("decode_error:TimeoutError:")
                    and self._context.independent_groups):
                chunk, error = self._hold(partition, "worker_timeout"), None
            self._pieces[partition] = (chunk, error, elapsed_ms)
        if self._pending is None:
            return None
        if len(self._pieces) != len(self._partitions):
            if time.monotonic_ns() > self._deadline_ns:
                if not self._context.independent_groups:
                    raise TimeoutError("action decoder exceeded the request deadline")
                for partition in self._partitions:
                    if partition not in self._pieces:
                        self._pieces[partition] = (
                            self._hold(partition, "worker_timeout"), None,
                            (time.monotonic_ns() - self._submitted_ns) / 1e6,
                        )
            else:
                return None
        response = self._pending
        self._pending = None
        errors = [p[1] for p in self._pieces.values() if p[1] is not None]
        if errors:
            return DecodeResult(response, None, ";".join(errors))
        ordered = [self._pieces[p] for p in self._partitions]
        merged = ordered[0][0]
        if not isinstance(merged, ActionChunk):
            raise TypeError("action decoder returned an invalid chunk")
        fields = (
            "request_seq",
            "observation_time_ns",
            "created_time_ns",
            "action_space",
            "dt_ns",
            "source_offset_steps",
            "horizon_steps",
        )
        for other, _, _ in ordered[1:]:
            if not isinstance(other, ActionChunk) or any(
                getattr(merged, f) != getattr(other, f) for f in fields
            ):
                raise ValueError("action decode partitions have mismatched contracts")
            if merged.groups.keys() & other.groups.keys():
                raise ValueError("action decode partitions overlap")
            merged.groups.update(other.groups)
            merged.hold_from_step.update(other.hold_from_step)
            for key, value in other.metadata.items():
                if key not in merged.metadata:
                    merged.metadata[key] = value
                elif isinstance(value, dict) and isinstance(merged.metadata[key], dict):
                    merged.metadata[key].update(value)
                elif merged.metadata[key] != value:
                    raise ValueError(f"action decode partition metadata conflicts: {key}")
        merged.metadata.update(
            decode_ms=max(p[2] for p in ordered),
            decode_stage_ms=(time.monotonic_ns() - self._submitted_ns) / 1e6,
            decode_partition_ms={str(k): v[2] for k, v in self._pieces.items()},
            decode_mode="process",
            hold_from_step=dict(merged.hold_from_step),
        )
        return DecodeResult(response, merged)

    def close(self) -> None:
        if not self._started:
            return
        for requests in self._requests:
            with suppress(OSError, ValueError, queue.Full):
                requests.put_nowait(None)
        for process in self._processes:
            if process.pid is None:
                continue
            process.join(timeout=0.5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
        for q in [*self._requests, self._results, self._startup]:
            q.cancel_join_thread()
            q.close()
        self._pending = None
        self._started = False
