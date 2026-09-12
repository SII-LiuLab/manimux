from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from manimux.types import (
    ActionChunk,
    ActionHorizon,
    GroupTrajectory,
    GroupVector,
    copy_group_vector,
)


@dataclass(frozen=True, slots=True)
class CommitResult:
    accepted: bool
    reason: str
    trimmed_steps: int = 0


@dataclass(slots=True)
class _ActivePlan:
    plan_id: str
    request_seq: int
    start_time_ns: int
    dt_ns: int
    groups: GroupTrajectory
    hold_from_step: dict[str, int] = field(default_factory=dict)
    unblended_groups: GroupTrajectory | None = None
    observation_time_ns: int | None = None
    hold_last_step: bool = False

    @property
    def horizon_steps(self) -> int:
        return int(next(iter(self.groups.values())).shape[0])

    @property
    def end_time_ns(self) -> int:
        intervals = self.horizon_steps if self.hold_last_step else self.horizon_steps - 1
        return self.start_time_ns + intervals * self.dt_ns


class ActionTimeline:
    """Single-plan, time-indexed, atomically replaced action reference."""

    def __init__(
        self,
        group_dims: dict[str, int],
        *,
        max_source_steps: int | None = None,
        start_on_commit: bool = False,
    ) -> None:
        if max_source_steps is not None and max_source_steps < 2:
            raise ValueError("max_source_steps must be at least two")
        self._max_source_steps = max_source_steps
        self._start_on_commit = start_on_commit
        self._group_dims = dict(group_dims)
        self._active: _ActivePlan | None = None
        self._accepted_request_seq = -1

    @property
    def active_plan_id(self) -> str | None:
        return None if self._active is None else self._active.plan_id

    @property
    def accepted_request_seq(self) -> int:
        return self._accepted_request_seq

    def remaining_ns(self, now_ns: int) -> int:
        if self._active is None:
            return 0
        return max(0, self._active.end_time_ns - now_ns)

    def cursor(self, now_ns: int) -> int:
        """Return the current index in the committed plan for observers."""
        active = self._active
        if active is None or now_ns <= active.start_time_ns:
            return 0
        position = (now_ns - active.start_time_ns) // active.dt_ns
        return min(max(0, int(position)), active.horizon_steps)

    def active_horizon(self) -> ActionHorizon | None:
        """Copy the exact trimmed/blended horizon committed for execution."""
        active = self._active
        if active is None:
            return None
        return ActionHorizon(
            start_time_ns=active.start_time_ns,
            dt_ns=active.dt_ns,
            plan_id=active.plan_id,
            groups={name: values.copy() for name, values in active.groups.items()},
            observation_time_ns=active.observation_time_ns,
        )

    def commit(
        self,
        chunk: ActionChunk,
        *,
        now_ns: int,
        commit_lead_ns: int,
        max_plan_age_ns: int,
        current_command: GroupVector,
        blend_steps: int,
    ) -> CommitResult:
        if chunk.request_seq <= self._accepted_request_seq:
            return CommitResult(False, "stale_request_seq")
        if now_ns - chunk.observation_time_ns > max_plan_age_ns:
            return CommitResult(False, "plan_too_old")
        if self._start_on_commit and chunk.source_offset_steps:
            return CommitResult(False, "serial_requires_untrimmed_chunk")
        if set(chunk.groups) != set(self._group_dims):
            return CommitResult(False, "group_mismatch")
        if set(current_command) != set(self._group_dims):
            return CommitResult(False, "current_command_group_mismatch")
        for name, dim in self._group_dims.items():
            if chunk.groups[name].shape[1] != dim or current_command[name].shape != (dim,):
                return CommitResult(False, f"dimension_mismatch:{name}")

        start_time_ns = now_ns + commit_lead_ns
        age_at_commit_ns = max(0, start_time_ns - chunk.observation_time_ns)
        source_cursor = int(age_at_commit_ns // chunk.dt_ns)
        trimmed_steps = (
            0 if self._start_on_commit else max(0, source_cursor - chunk.source_offset_steps)
        )
        end = chunk.horizon_steps
        if self._max_source_steps is not None:
            end = min(end, self._max_source_steps - chunk.source_offset_steps)
        if trimmed_steps >= end:
            return CommitResult(False, "no_future_horizon")

        groups = {name: values[trimmed_steps:end].copy() for name, values in chunk.groups.items()}
        unblended_groups = {name: values.copy() for name, values in groups.items()}
        hold_from_step = {
            name: max(0, step - trimmed_steps)
            for name, step in chunk.hold_from_step.items() if step < end
        }
        actual_blend_steps = min(blend_steps, next(iter(groups.values())).shape[0])
        if actual_blend_steps:
            current = copy_group_vector(current_command)
            for name, values in groups.items():
                for index in range(actual_blend_steps):
                    alpha = (index + 1) / actual_blend_steps
                    values[index] = (1.0 - alpha) * current[name] + alpha * values[index]

        new_plan = _ActivePlan(
            plan_id=chunk.plan_id,
            request_seq=chunk.request_seq,
            start_time_ns=start_time_ns,
            dt_ns=chunk.dt_ns,
            groups=groups,
            hold_from_step=hold_from_step,
            unblended_groups=unblended_groups,
            observation_time_ns=chunk.observation_time_ns,
            hold_last_step=self._start_on_commit,
        )
        self._active = new_plan
        self._accepted_request_seq = chunk.request_seq
        return CommitResult(True, "accepted", trimmed_steps)

    def sample(self, time_ns: int) -> GroupVector | None:
        active = self._active
        if active is None or time_ns < active.start_time_ns or time_ns > active.end_time_ns:
            return None
        if active.hold_last_step and time_ns >= active.end_time_ns:
            return None
        position = (time_ns - active.start_time_ns) / active.dt_ns
        lower = min(int(np.floor(position)), active.horizon_steps - 1)
        upper = min(lower + 1, active.horizon_steps - 1)
        alpha = min(position - lower, 1.0)
        return {
            name: (1.0 - alpha) * values[lower] + alpha * values[upper]
            for name, values in active.groups.items()
        }

    def reference_horizon(
        self,
        *,
        now_ns: int,
        dt_ns: int,
        horizon_steps: int,
    ) -> ActionHorizon | None:
        active = self._active
        if active is None:
            return None
        samples: dict[str, list[np.ndarray]] = {name: [] for name in self._group_dims}
        last: GroupVector | None = None
        for step in range(horizon_steps):
            sample = self.sample(now_ns + step * dt_ns)
            if sample is None:
                if last is None:
                    return None
                sample = last
            last = sample
            for name, value in sample.items():
                samples[name].append(value)
        return ActionHorizon(
            start_time_ns=now_ns,
            dt_ns=dt_ns,
            plan_id=active.plan_id,
            groups={name: np.stack(values) for name, values in samples.items()},
            tracking_groups=self._tracking_sample(now_ns),
            observation_time_ns=active.observation_time_ns,
            # Stop before interpolation/feedforward could enter an invalid row.
            hold_groups=tuple(
                name for name, first_invalid in active.hold_from_step.items()
                if now_ns + max(0, horizon_steps - 1) * dt_ns
                >= active.start_time_ns + max(0, first_invalid - 1) * active.dt_ns
            ),
        )

    def _tracking_sample(self, now_ns: int) -> GroupVector | None:
        active = self._active
        if active is None or active.unblended_groups is None:
            return None
        position = np.clip((now_ns - active.start_time_ns) / active.dt_ns,
                           0, active.horizon_steps - 1)
        lo = int(np.floor(position))
        hi = min(lo + 1, active.horizon_steps - 1)
        alpha = position - lo
        return {name: (1 - alpha) * values[lo] + alpha * values[hi]
                for name, values in active.unblended_groups.items()}
