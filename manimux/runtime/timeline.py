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
    timeline_latency_ns: int = 0


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


def _sample_plan(plan: _ActivePlan, time_ns: int) -> GroupVector | None:
    if not plan.start_time_ns <= time_ns <= plan.end_time_ns:
        return None
    if plan.hold_last_step and time_ns >= plan.end_time_ns:
        return None
    position = (time_ns - plan.start_time_ns) / plan.dt_ns
    lower = min(int(np.floor(position)), plan.horizon_steps - 1)
    upper = min(lower + 1, plan.horizon_steps - 1)
    alpha = min(position - lower, 1.0)
    return {
        name: (1.0 - alpha) * values[lower] + alpha * values[upper]
        for name, values in plan.groups.items()
    }


class ActionTimeline:
    """Time-indexed action reference; a commit_lead window still plays the outgoing plan."""

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
        self._runtime_active: _ActivePlan | None = None
        # Kept only to cover the commit_lead window before _active starts.
        self._outgoing: _ActivePlan | None = None
        self._runtime_outgoing: _ActivePlan | None = None
        self._accepted_request_seq = -1

    @property
    def active_plan_id(self) -> str | None:
        return None if self._active is None else self._active.plan_id

    @property
    def accepted_request_seq(self) -> int:
        return self._accepted_request_seq

    def remaining_ns(self, now_ns: int) -> int:
        if self._runtime_active is None:
            return 0
        return max(0, self._runtime_active.end_time_ns - now_ns)

    def cursor(self, now_ns: int) -> int:
        """Return the current index in the committed plan for observers."""
        active = self._active
        if active is None or now_ns <= active.start_time_ns:
            return 0
        position = (now_ns - active.start_time_ns) // active.dt_ns
        return min(max(0, int(position)), active.horizon_steps)

    def active_horizon(self) -> ActionHorizon | None:
        """Copy the trimmed/blended model-source horizon for observers and RTC."""
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
        """Validate, time-align, trim, and activate a new action chunk."""
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
        runtime = chunk.runtime_trajectory
        if runtime is not None:
            if set(runtime.groups) != set(self._group_dims):
                return CommitResult(False, "runtime_group_mismatch")
            for name, dim in self._group_dims.items():
                if runtime.groups[name].shape[1] != dim:
                    return CommitResult(False, f"runtime_dimension_mismatch:{name}")

        # Earliest wall-clock time at which the committed plan may start.
        earliest_ns = now_ns + commit_lead_ns
        # Elapsed source-trajectory time when execution starts.
        age_at_commit_ns = max(0, earliest_ns - chunk.observation_time_ns)
        # First source row whose timestamp is not earlier than earliest_ns.
        # An exact row boundary keeps that row; a start between rows discards
        # the earlier row instead of retiming an already-expired target.
        source_cursor = int(
            (age_at_commit_ns + chunk.dt_ns - 1) // chunk.dt_ns
            if age_at_commit_ns
            else 0
        )
        # Rows to remove from this chunk, excluding rows already removed upstream.
        trimmed_steps = (
            0 if self._start_on_commit else max(0, source_cursor - chunk.source_offset_steps)
        )
        end = chunk.horizon_steps
        if self._max_source_steps is not None:
            end = min(end, self._max_source_steps - chunk.source_offset_steps)
        if trimmed_steps >= end:
            return CommitResult(False, "no_future_horizon")

        # Anchor row 0 at its own source timestamp. earliest_ns is wherever the
        # decode happened to finish, so re-pinning row 0 to it shifts the whole
        # plan off the source grid by up to one dt -- a phase error the executor
        # cannot distinguish from the policy changing its mind. Anchoring here
        # delays the start by less than one dt, which _outgoing covers. Serial
        # scheduling means "start on commit", so it keeps the wall-clock anchor.
        first_source_step = chunk.source_offset_steps + trimmed_steps
        start_time_ns = (
            earliest_ns
            if self._start_on_commit
            else chunk.observation_time_ns + first_source_step * chunk.dt_ns
        )

        groups = {name: values[trimmed_steps:end].copy() for name, values in chunk.groups.items()}
        unblended_groups = {name: values.copy() for name, values in groups.items()}
        hold_from_step = {
            name: max(0, step - trimmed_steps)
            for name, step in chunk.hold_from_step.items()
            if step < end
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

        runtime_plan = new_plan
        if runtime is not None:
            source_end_time_ns = (
                start_time_ns + (end - trimmed_steps - 1) * chunk.dt_ns
                if self._start_on_commit
                else chunk.observation_time_ns
                + (chunk.source_offset_steps + end - 1) * chunk.dt_ns
            )
            if self._start_on_commit:
                runtime_start_index = 0
                runtime_end_index = runtime.horizon_steps
                runtime_start_time_ns = earliest_ns
            else:
                runtime_age_ns = max(0, earliest_ns - runtime.start_time_ns)
                runtime_start_index = int(
                    (runtime_age_ns + runtime.dt_ns - 1) // runtime.dt_ns
                    if runtime_age_ns
                    else 0
                )
                runtime_end_index = min(
                    runtime.horizon_steps,
                    int((source_end_time_ns - runtime.start_time_ns) // runtime.dt_ns) + 1,
                )
                runtime_start_time_ns = (
                    runtime.start_time_ns + runtime_start_index * runtime.dt_ns
                )
            if runtime_start_index >= runtime_end_index:
                return CommitResult(False, "no_future_runtime_horizon")

            runtime_groups = {
                name: values[runtime_start_index:runtime_end_index].copy()
                for name, values in runtime.groups.items()
            }
            unblended_runtime_groups = {
                name: values.copy() for name, values in runtime_groups.items()
            }
            if actual_blend_steps:
                blend_duration_ns = actual_blend_steps * chunk.dt_ns
                current = copy_group_vector(current_command)
                for name, values in runtime_groups.items():
                    for index in range(len(values)):
                        row_time_ns = runtime_start_time_ns + index * runtime.dt_ns
                        alpha = np.clip(
                            (row_time_ns - start_time_ns + chunk.dt_ns) / blend_duration_ns,
                            0.0,
                            1.0,
                        )
                        if alpha >= 1.0:
                            break
                        values[index] = (1.0 - alpha) * current[name] + alpha * values[index]

            runtime_hold_from_step = {}
            for name, step in chunk.hold_from_step.items():
                invalid_time_ns = (
                    start_time_ns + max(0, step - trimmed_steps) * chunk.dt_ns
                    if self._start_on_commit
                    else chunk.observation_time_ns
                    + (chunk.source_offset_steps + step) * chunk.dt_ns
                )
                relative_ns = invalid_time_ns - runtime_start_time_ns
                first_invalid = max(
                    0,
                    int((relative_ns + runtime.dt_ns - 1) // runtime.dt_ns),
                )
                if first_invalid < len(next(iter(runtime_groups.values()))):
                    runtime_hold_from_step[name] = first_invalid

            runtime_plan = _ActivePlan(
                plan_id=chunk.plan_id,
                request_seq=chunk.request_seq,
                start_time_ns=runtime_start_time_ns,
                dt_ns=runtime.dt_ns,
                groups=runtime_groups,
                hold_from_step=runtime_hold_from_step,
                unblended_groups=unblended_runtime_groups,
                observation_time_ns=chunk.observation_time_ns,
                hold_last_step=self._start_on_commit,
            )
        self._outgoing = self._active
        self._runtime_outgoing = self._runtime_active
        self._active = new_plan
        self._runtime_active = runtime_plan
        self._accepted_request_seq = chunk.request_seq
        return CommitResult(
            True,
            "accepted",
            trimmed_steps=trimmed_steps,
            timeline_latency_ns=age_at_commit_ns,
        )

    def _plan_at(self, time_ns: int) -> _ActivePlan | None:
        """The plan that owns ``time_ns``: the outgoing one until _active starts."""
        active = self._active
        if active is not None and time_ns < active.start_time_ns:
            # With commit_lead > 0 a committed plan starts slightly in the future.
            # Keep executing the outgoing plan across that window; dropping to a
            # measured-state hold would yank the command back by the tracking error.
            return self._outgoing
        return active

    def _runtime_plan_at(self, time_ns: int) -> _ActivePlan | None:
        """Runtime plan owning ``time_ns``; it may use denser adapter samples."""
        active = self._runtime_active
        if active is not None and time_ns < active.start_time_ns:
            return self._runtime_outgoing
        return active

    def sample(self, time_ns: int) -> GroupVector | None:
        """Sample the model/source trajectory used by observers and RTC."""
        plan = self._plan_at(time_ns)
        return None if plan is None else _sample_plan(plan, time_ns)

    def _sample_runtime(self, time_ns: int) -> GroupVector | None:
        plan = self._runtime_plan_at(time_ns)
        return None if plan is None else _sample_plan(plan, time_ns)

    def reference_horizon(
        self,
        *,
        now_ns: int,
        dt_ns: int,
        horizon_steps: int,
    ) -> ActionHorizon | None:
        # Report the plan that owns now_ns: across a commit_lead window the
        # executor is still tracking the outgoing plan, not the committed one.
        active = self._runtime_plan_at(now_ns)
        if active is None:
            return None
        samples: dict[str, list[np.ndarray]] = {name: [] for name in self._group_dims}
        last: GroupVector | None = None
        for step in range(horizon_steps):
            sample = self._sample_runtime(now_ns + step * dt_ns)
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
                name
                for name, first_invalid in active.hold_from_step.items()
                if now_ns + max(0, horizon_steps - 1) * dt_ns
                >= active.start_time_ns + max(0, first_invalid - 1) * active.dt_ns
            ),
        )

    def _tracking_sample(self, now_ns: int) -> GroupVector | None:
        active = self._runtime_plan_at(now_ns)
        if active is None or active.unblended_groups is None:
            return None
        position = np.clip(
            (now_ns - active.start_time_ns) / active.dt_ns, 0, active.horizon_steps - 1
        )
        lo = int(np.floor(position))
        hi = min(lo + 1, active.horizon_steps - 1)
        alpha = position - lo
        return {
            name: (1 - alpha) * values[lo] + alpha * values[hi]
            for name, values in active.unblended_groups.items()
        }
