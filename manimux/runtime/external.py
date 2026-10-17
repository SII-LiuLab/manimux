"""Bounded external actions consumed by the existing hardware control loop."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field

import numpy as np

from manimux.policies.capabilities import PolicyCapabilities
from manimux.types import (
    ActionChunk,
    GroupTrajectory,
    GroupVector,
    ObservationSnapshot,
    RobotCommand,
)


@dataclass
class ExternalBatch:
    groups: GroupTrajectory
    future: Future[ExternalBatch] = field(default_factory=Future)
    executed: int = 0
    feedback: list[ObservationSnapshot] = field(default_factory=list)
    sent: bool = False


class ExternalActionSource:
    """One outstanding batch, with completion acknowledged after a control period.

    Stop signals never wait for the dispatch lock. Only the runtime thread writes
    hardware; callers may submit, observe, arm, or request a latched stop.
    """

    capabilities = PolicyCapabilities(backend_metadata={"server": "external_actions"})

    def __init__(self) -> None:
        self.condition = threading.Condition(threading.RLock())
        self.stopped = threading.Event()
        self.stopped.set()
        self.shutdown = threading.Event()
        self.snapshot: ObservationSnapshot | None = None
        self.last_command: dict[str, np.ndarray] | None = None
        self.batch: ExternalBatch | None = None
        self.failure: BaseException | None = None
        self.executed = 0
        self._seq = 0
        self._stop_applied = True  # No commands have been dispatched at startup.
        self._stop_lock = threading.Lock()
        self.before_tick: Callable[[], None] = lambda: None

    def arm(self) -> None:
        with self.condition:
            if self.failure or self.shutdown.is_set() or self.batch is not None:
                raise RuntimeError("external runtime is unavailable or busy")
            with self._stop_lock:
                if self.stopped.is_set() and not self._stop_applied:
                    raise RuntimeError("external stop is awaiting hardware acknowledgement")
                self.stopped.clear()
                self._stop_applied = False

    def request_stop(self) -> None:
        with self._stop_lock:
            if not self.stopped.is_set():
                self._stop_applied = False
            self.stopped.set()

    def submit(self, groups: GroupTrajectory) -> ExternalBatch:
        chunk = ActionChunk("validate", 0, 0, 0, "joint_position", 1, groups)
        with self.condition:
            if self.stopped.is_set() or self.failure or self.shutdown.is_set():
                raise RuntimeError("external runtime is stopped or unavailable")
            if self.batch is not None:
                raise RuntimeError("external runtime is busy")
            self.batch = ExternalBatch({k: v.copy() for k, v in chunk.groups.items()})
            return self.batch

    def observe_tick(self, snapshot: ObservationSnapshot, last_command: GroupVector) -> None:
        self.before_tick()
        with self.condition:
            self.snapshot = snapshot
            self.last_command = {k: v.copy() for k, v in last_command.items()}
            batch = self.batch
            if batch is not None:
                if batch.sent:
                    batch.feedback.append(snapshot)
                    batch.sent = False
                count = next(iter(batch.groups.values())).shape[0]
                if (self.stopped.is_set() and self._stop_applied) or (
                    not self.stopped.is_set() and batch.executed == count
                ):
                    batch.future.set_result(batch)
                    self.batch = None
            self.condition.notify_all()

    def stop_needed(self) -> bool:
        with self._stop_lock:
            return self.stopped.is_set() and not self._stop_applied

    def acknowledge_stop(self) -> None:
        """Called by the control thread only after hardware hold and history reset."""
        with self._stop_lock:
            self._stop_applied = True

    def next_chunk(self, now_ns: int, dt_ns: int) -> ActionChunk | None:
        with self.condition:
            batch = self.batch
            if batch is None or self.stopped.is_set():
                return None
            self._seq += 1
            return ActionChunk(
                f"external-{uuid.uuid4().hex}",
                self._seq,
                now_ns,
                now_ns,
                "joint_position",
                dt_ns,
                {k: v[batch.executed : batch.executed + 1].copy() for k, v in batch.groups.items()},
            )

    def dispatch(self, send: Callable[[RobotCommand], None], command: RobotCommand) -> bool:
        with self.condition:
            if self.stopped.is_set():
                return False
            send(command)
            if self.batch is not None and command.plan_id is not None:
                self.batch.executed += 1
                self.batch.sent = True
                self.executed += 1
            return True

    def close(self) -> None:
        self.stopped.set()
        self.shutdown.set()
        with self.condition:
            if self.batch is not None and not self.batch.future.done():
                self.batch.future.set_exception(
                    self.failure or RuntimeError("external runtime closed")
                )
                self.batch = None
            self.condition.notify_all()
