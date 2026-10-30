from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from manimux.policy_adapter.pose_output import POSE_SPACE, interpolate_pose
from manimux.types import (
    ACTION_START_MODES,
    ActionChunk,
    ActionHorizon,
    GroupTrajectory,
    GroupVector,
    copy_group_vector,
)


def first_future_step(observation_time_ns: int, dt_ns: int, now_ns: int) -> int:
    """First source row on or after the handoff time, rounding up on the row grid."""
    age_ns = max(0, now_ns - observation_time_ns)
    return (age_ns + dt_ns - 1) // dt_ns


@dataclass(frozen=True, slots=True)
class CommitResult:
    accepted: bool
    reason: str
    trimmed_steps: int = 0  # Rows removed from this chunk at commit.
    timeline_latency_ns: int = 0
    time_trimmed_steps: int = 0
    # Intentional post-decode prefix crop, included in trimmed_steps.
    handoff_skipped_steps: int = 0


@dataclass(slots=True)
class _CommandBlend:
    start_ns: int
    end_ns: int
    position: GroupVector
    velocity: GroupVector
    gripper_indices: dict[str, int]


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
    retain_terminal_target: bool = False

    command_blend: _CommandBlend | None = None
    action_space: str = "joint_position"

    @property
    def activation_time_ns(self) -> int:
        return self.start_time_ns if self.command_blend is None else self.command_blend.start_ns

    @property
    def horizon_steps(self) -> int:
        return int(next(iter(self.groups.values())).shape[0])

    @property
    def end_time_ns(self) -> int:
        intervals = self.horizon_steps if self.hold_last_step else self.horizon_steps - 1
        return self.start_time_ns + intervals * self.dt_ns


def _sample_plan(plan: _ActivePlan, time_ns: int) -> GroupVector | None:
    if plan.retain_terminal_target and time_ns >= plan.end_time_ns:
        return {name: values[-1].copy() for name, values in plan.groups.items()}
    blend = plan.command_blend
    if blend is not None and blend.start_ns <= time_ns < blend.end_ns:
        duration = (blend.end_ns - blend.start_ns) / 1e9
        u = (time_ns - blend.start_ns) / (blend.end_ns - blend.start_ns)
        end_index = int(round((blend.end_ns - plan.start_time_ns) / plan.dt_ns))
        result = {}
        for name, rows in plan.groups.items():
            target = rows[end_index]
            end_velocity = (
                (rows[end_index + 1] - target) / (plan.dt_ns / 1e9)
                if end_index + 1 < len(rows) else np.zeros_like(target)
            )
            value = ((2*u**3 - 3*u**2 + 1) * blend.position[name]
                     + (u**3 - 2*u**2 + u) * duration * blend.velocity[name]
                     + (-2*u**3 + 3*u**2) * target
                     + (u**3 - u**2) * duration * end_velocity)
            if name in blend.gripper_indices:
                index = blend.gripper_indices[name]
                # Apertures use a monotone transition, without cubic overshoot.
                weight = u*u*(3 - 2*u)
                value[index] = (1-weight)*blend.position[name][index] + weight*target[index]
            result[name] = value
        return result
    if not plan.activation_time_ns <= time_ns <= plan.end_time_ns:
        return None
    if plan.hold_last_step and time_ns >= plan.end_time_ns:
        return None
    position = (time_ns - plan.start_time_ns) / plan.dt_ns
    lower = min(int(np.floor(position)), plan.horizon_steps - 1)
    upper = min(lower + 1, plan.horizon_steps - 1)
    alpha = min(position - lower, 1.0)
    return {
        name: (interpolate_pose(values[lower], values[upper], alpha)
               if plan.action_space == POSE_SPACE else
               (1.0 - alpha) * values[lower] + alpha * values[upper])
        for name, values in plan.groups.items()
    }


class ActionTimeline:
    """Time-indexed action reference; the outgoing plan plays until a committed plan starts."""

    def __init__(
        self,
        group_dims: dict[str, int],
        *,
        action_space: str = "joint_position",
        max_source_steps: int | None = None,
        action_start_mode: str = "drop_infer_latency",
        hold_last_step: bool = False,
        allow_retimed_suffix: bool = False,
        blend_mode: str = "linear",
        gripper_indices: dict[str, int] | None = None,
    ) -> None:
        if max_source_steps is not None and max_source_steps < 2:
            raise ValueError("max_source_steps must be at least two")
        if action_start_mode not in ACTION_START_MODES:
            raise ValueError(f"action_start_mode must be one of {sorted(ACTION_START_MODES)}")
        if blend_mode not in {"linear", "command_cubic"}:
            raise ValueError("unknown timeline blend_mode")
        if action_space not in {"joint_position", POSE_SPACE}:
            raise ValueError("unknown timeline action space")
        if action_space == POSE_SPACE and blend_mode != "linear":
            raise ValueError("pose timelines require linear/SLERP blending")
        self.action_space = action_space
        self._blend_mode = blend_mode
        self._gripper_indices = dict(gripper_indices or {})
        self._max_source_steps = max_source_steps
        # first_step_when_ready starts the untrimmed chunk at commit; drop_infer_latency
        # keeps the source row grid, so a plan may start up to one row after the commit.
        self._starts_at_commit = action_start_mode == "first_step_when_ready"
        if allow_retimed_suffix and not (self._starts_at_commit and hold_last_step):
            raise ValueError("retimed suffixes require commit-time playback with a final interval")
        self._allow_retimed_suffix = allow_retimed_suffix
        self._hold_last_step = hold_last_step
        self._group_dims = dict(group_dims)
        self._active: _ActivePlan | None = None
        self._runtime_active: _ActivePlan | None = None
        # Retain the executing owner and pending transitions before _active starts.
        # A later response may arrive before an already committed plan takes over.
        self._outgoing: list[_ActivePlan] = []
        self._runtime_outgoing: list[_ActivePlan] = []
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

    def replacement_ready(self, now_ns: int, minimum_policy_steps: int) -> bool:
        """Protect policy intervals from activation, capped by available execution time.

        Count from activation (including a command blend), not observation time.
        Dense adapter trajectories still use the model's policy-step duration.
        """
        active, runtime = self._active, self._runtime_active
        if not minimum_policy_steps or active is None or runtime is None:
            return True
        earliest = min(
            runtime.activation_time_ns + minimum_policy_steps * active.dt_ns,
            runtime.end_time_ns,
        )
        return now_ns >= earliest

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
        groups = {name: values.copy() for name, values in active.groups.items()}
        if active.command_blend is not None:
            samples = [_sample_plan(active, active.start_time_ns + i * active.dt_ns)
                       for i in range(active.horizon_steps)]
            groups = {name: np.stack([sample[name] for sample in samples]) for name in groups}
        return ActionHorizon(
            start_time_ns=active.start_time_ns,
            dt_ns=active.dt_ns,
            plan_id=active.plan_id,
            groups=groups,
            observation_time_ns=active.observation_time_ns,
            action_space=self.action_space,
        )

    def commit(
        self,
        chunk: ActionChunk,
        *,
        now_ns: int,
        max_plan_age_ns: int | None,
        current_command: GroupVector,
        blend_steps: int,
        handoff_skip_steps: int = 0,
        command_velocity: GroupVector | None = None,
        retain_terminal_target: bool = False,
    ) -> CommitResult:
        """Validate, time-align, trim, and activate a new action chunk."""
        if type(handoff_skip_steps) is not int:
            raise ValueError("handoff_skip_steps must be an integer")
        if chunk.action_space != self.action_space:
            return CommitResult(False, "action_space_mismatch")
        if self.action_space == POSE_SPACE and (chunk.runtime_trajectory is not None or chunk.handoff is not None):
            return CommitResult(False, "joint_handoff_on_pose_timeline")
        if chunk.request_seq <= self._accepted_request_seq:
            return CommitResult(False, "stale_request_seq")
        if max_plan_age_ns is not None and now_ns - chunk.observation_time_ns > max_plan_age_ns:
            return CommitResult(False, "plan_too_old")
        if self._hold_last_step and chunk.source_offset_steps and not self._allow_retimed_suffix:
            return CommitResult(False, "serial_requires_untrimmed_chunk")
        if set(chunk.groups) != set(self._group_dims):
            return CommitResult(False, "group_mismatch")
        if set(current_command) != set(self._group_dims):
            return CommitResult(False, "current_command_group_mismatch")
        for name, dim in self._group_dims.items():
            if chunk.groups[name].shape[1] != dim or current_command[name].shape != (dim,):
                return CommitResult(False, f"dimension_mismatch:{name}")
        runtime = chunk.runtime_trajectory
        if retain_terminal_target and (
            chunk.action_space != "joint_position" or runtime is not None
            or chunk.hold_from_step or chunk.handoff is not None
        ):
            return CommitResult(False, "terminal_target_requires_complete_joint_path")
        if runtime is not None:
            if set(runtime.groups) != set(self._group_dims):
                return CommitResult(False, "runtime_group_mismatch")
            for name, dim in self._group_dims.items():
                if runtime.groups[name].shape[1] != dim:
                    return CommitResult(False, f"runtime_dimension_mismatch:{name}")

        # Earliest wall-clock time at which the committed plan may start.
        earliest_ns = now_ns
        if chunk.handoff is not None:
            # The handoff time is on the source row grid that drop_infer_latency keeps.
            if self._starts_at_commit:
                raise ValueError(
                    "waypoint handoff chunks require action_start_mode=drop_infer_latency"
                )
            # A waypoint handoff already replaced the blend before embodiment decoding.
            if blend_steps:
                raise ValueError("waypoint handoff chunks must not be blended again")
            # Validate the uncut decode seed and clock. A requested final prefix
            # crop may remove that continuous seed; it must not replan the lead-in.
            reason = self._check_handoff(chunk, earliest_ns)
            if reason is not None:
                return CommitResult(False, reason)
        # Elapsed source-trajectory time when execution starts.
        age_at_commit_ns = max(0, earliest_ns - chunk.observation_time_ns)
        # First source row whose timestamp is not earlier than earliest_ns.
        # An exact row boundary keeps that row; a start between rows discards
        # the earlier row instead of retiming an already-expired target.
        source_cursor = first_future_step(chunk.observation_time_ns, chunk.dt_ns, earliest_ns)
        # Rows to remove from this chunk, excluding rows already removed upstream.
        time_trimmed_steps = (
            0 if self._starts_at_commit else max(0, source_cursor - chunk.source_offset_steps)
        )
        clock_trimmed_steps = time_trimmed_steps
        # Skip source actions without moving the time at which the new plan starts.
        # The first plan has no outgoing chunk to hand off from.
        handoff_skipped_steps = handoff_skip_steps if self._active is not None else 0
        # All adapters finish decoding before the execution-only crop is applied.
        handoff_skipped_steps = max(-time_trimmed_steps, handoff_skipped_steps)
        row_skip_steps = handoff_skipped_steps
        trimmed_steps = time_trimmed_steps + row_skip_steps
        end = chunk.horizon_steps
        if self._max_source_steps is not None:
            end = min(end, self._max_source_steps - chunk.source_offset_steps)
        if trimmed_steps >= end:
            return CommitResult(False, "no_future_horizon")

        # Time trim determines the handoff instant. The intentional skip changes
        # only which source row occupies that instant, not the handoff clock.
        first_time_step = chunk.source_offset_steps + clock_trimmed_steps
        start_time_ns = (
            earliest_ns
            if self._starts_at_commit
            else chunk.observation_time_ns + first_time_step * chunk.dt_ns
        )

        groups = {name: values[trimmed_steps:end].copy() for name, values in chunk.groups.items()}
        unblended_groups = {name: values.copy() for name, values in groups.items()}
        hold_from_step = {
            name: max(0, step - trimmed_steps)
            for name, step in chunk.hold_from_step.items()
            if step < end
        }
        actual_blend_steps = min(blend_steps, next(iter(groups.values())).shape[0])
        if actual_blend_steps and self._blend_mode == "linear":
            current = copy_group_vector(current_command)
            for name, values in groups.items():
                for index in range(actual_blend_steps):
                    alpha = (index + 1) / actual_blend_steps
                    values[index] = (interpolate_pose(current[name], values[index], alpha)
                                     if self.action_space == POSE_SPACE else
                                     (1.0 - alpha) * current[name] + alpha * values[index])

        new_plan = _ActivePlan(
            plan_id=chunk.plan_id,
            request_seq=chunk.request_seq,
            start_time_ns=start_time_ns,
            dt_ns=chunk.dt_ns,
            groups=groups,
            hold_from_step=hold_from_step,
            unblended_groups=unblended_groups,
            observation_time_ns=chunk.observation_time_ns,
            hold_last_step=self._hold_last_step,
            retain_terminal_target=retain_terminal_target,
            action_space=self.action_space,
        )

        if self._blend_mode == "command_cubic" and blend_steps:
            if runtime is not None or chunk.handoff is not None or self._hold_last_step:
                raise ValueError("command_cubic requires an ordinary continuous joint timeline")
            if actual_blend_steps < 2:
                return CommitResult(False, "insufficient_blend_horizon")
            velocity = command_velocity or {
                name: np.zeros_like(q) for name, q in current_command.items()
            }
            if set(velocity) != set(self._group_dims) or any(
                velocity[name].shape != current_command[name].shape
                or not np.isfinite(velocity[name]).all() for name in self._group_dims
            ):
                raise ValueError("invalid command velocity")
            new_plan.command_blend = _CommandBlend(
                now_ns, start_time_ns + (actual_blend_steps - 1) * chunk.dt_ns,
                copy_group_vector(current_command), copy_group_vector(velocity),
                self._gripper_indices,
            )

        runtime_plan = new_plan
        if runtime is not None:
            source_end_time_ns = (
                start_time_ns + (end - trimmed_steps - 1) * chunk.dt_ns
                if self._starts_at_commit
                else chunk.observation_time_ns
                + (chunk.source_offset_steps + end - 1) * chunk.dt_ns
            )
            if self._starts_at_commit:
                runtime_time_index = 0
                # Cap dense rows on their original clock before retiming to commit.
                # Inference/decode latency must not consume a serial prefix.
                source_prefix_end_ns = (
                    chunk.observation_time_ns
                    + (chunk.source_offset_steps + end - 1) * chunk.dt_ns
                )
                runtime_end_index = min(
                    runtime.horizon_steps,
                    (source_prefix_end_ns - runtime.start_time_ns) // runtime.dt_ns + 1,
                )
                runtime_start_time_ns = earliest_ns
            else:
                runtime_age_ns = max(0, earliest_ns - runtime.start_time_ns)
                runtime_time_index = int(
                    (runtime_age_ns + runtime.dt_ns - 1) // runtime.dt_ns
                    if runtime_age_ns
                    else 0
                )
                runtime_end_index = min(
                    runtime.horizon_steps,
                    int((source_end_time_ns - runtime.start_time_ns) // runtime.dt_ns) + 1,
                )
                runtime_start_time_ns = (
                    runtime.start_time_ns + runtime_time_index * runtime.dt_ns
                )
            runtime_skip_steps = round(row_skip_steps * chunk.dt_ns / runtime.dt_ns)
            runtime_start_index = runtime_time_index + runtime_skip_steps
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
                    if self._starts_at_commit
                    else chunk.observation_time_ns
                    + (chunk.source_offset_steps + step - row_skip_steps) * chunk.dt_ns
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
                hold_last_step=self._hold_last_step,
                action_space=self.action_space,
            )
        self._outgoing = self._retain_outgoing(self._outgoing, self._active, new_plan, now_ns)
        self._runtime_outgoing = self._retain_outgoing(
            self._runtime_outgoing, self._runtime_active, runtime_plan, now_ns
        )
        self._active = new_plan
        self._runtime_active = runtime_plan
        self._accepted_request_seq = chunk.request_seq
        return CommitResult(
            True,
            "accepted",
            trimmed_steps=trimmed_steps,
            timeline_latency_ns=age_at_commit_ns,
            time_trimmed_steps=time_trimmed_steps,
            handoff_skipped_steps=handoff_skipped_steps,
        )

    def handoff_reference(self, time_ns: int, window_ns: int) -> ActionHorizon | None:
        """Copy runtime rows covering a handoff; time_ns: handoff time, window_ns: span after it."""
        plan = self._runtime_plan_at(time_ns)
        if (
            plan is None
            or plan.hold_last_step
            or time_ns < plan.start_time_ns
            or time_ns + window_ns > plan.end_time_ns
        ):
            return None
        lower = (time_ns - plan.start_time_ns) // plan.dt_ns
        upper = min(
            plan.horizon_steps,
            -(-(time_ns + window_ns - plan.start_time_ns) // plan.dt_ns) + 1,
        )
        return ActionHorizon(
            start_time_ns=plan.start_time_ns + lower * plan.dt_ns,
            dt_ns=plan.dt_ns,
            plan_id=plan.plan_id,
            groups={name: values[lower:upper].copy() for name, values in plan.groups.items()},
            observation_time_ns=plan.observation_time_ns,
        )

    def _check_handoff(self, chunk: ActionChunk, earliest_ns: int) -> str | None:
        """Reject a handoff that no longer continues the command; earliest_ns: first start time."""
        handoff = chunk.handoff
        runtime = chunk.runtime_trajectory
        if runtime is None or runtime.start_time_ns != handoff.time_ns:
            return "handoff_runtime_misaligned"
        plan = self._runtime_plan_at(handoff.time_ns)
        if plan is None or plan.plan_id != handoff.plan_id:
            return "handoff_plan_changed"
        if earliest_ns > handoff.time_ns:
            return "handoff_missed"
        reference = _sample_plan(plan, handoff.time_ns)
        if reference is None or any(
            not np.allclose(reference[name], handoff.reference[name], rtol=0.0, atol=1e-9)
            or not np.array_equal(runtime.groups[name][0], handoff.reference[name])
            for name in self._group_dims
        ):
            return "handoff_reference_mismatch"
        return None

    @staticmethod
    def _retain_outgoing(
        outgoing: list[_ActivePlan],
        active: _ActivePlan | None,
        incoming: _ActivePlan,
        now_ns: int,
    ) -> list[_ActivePlan]:
        # Acceptance does not mean execution: keep the current owner and every
        # pending transition before incoming starts. Incoming supersedes later
        # transitions, even when its start precedes the last accepted plan.
        prior = list(outgoing)
        if active is not None:
            prior.append(active)
        prior = [plan for plan in prior if plan.activation_time_ns < incoming.activation_time_ns]
        # Keep even an expired owner: falling back to an older, longer plan
        # would resurrect a trajectory that a started plan already superseded.
        started = [index for index, plan in enumerate(prior) if plan.activation_time_ns <= now_ns]
        return prior[started[-1]:] if started else prior

    def _plan_at(self, time_ns: int) -> _ActivePlan | None:
        """The plan that owns ``time_ns``: the outgoing one until _active starts."""
        active = self._active
        if active is not None and time_ns < active.activation_time_ns:
            # A grid-anchored or waypoint plan starts slightly in the future. Keep
            # executing the outgoing plan across that window; dropping to a
            # measured-state hold would yank the command back by the tracking error.
            return next(
                (plan for plan in reversed(self._outgoing) if plan.activation_time_ns <= time_ns),
                None,
            )
        return active

    def _runtime_plan_at(self, time_ns: int) -> _ActivePlan | None:
        """Runtime plan owning ``time_ns``; it may use denser adapter samples."""
        active = self._runtime_active
        if active is not None and time_ns < active.activation_time_ns:
            return next(
                (
                    plan for plan in reversed(self._runtime_outgoing)
                    if plan.activation_time_ns <= time_ns
                ),
                None,
            )
        return active

    def sample(self, time_ns: int, *, terminal_hold: bool = False) -> GroupVector | None:
        """Sample source targets; optional terminal hold is for conditioning only.

        Execution coverage is unchanged. Do not revive an earlier plan or fill
        time before the owning plan activates.
        """
        plan = self._plan_at(time_ns)
        if (terminal_hold and plan is not None
                and time_ns >= max(plan.activation_time_ns, plan.end_time_ns)):
            return {name: values[-1].copy() for name, values in plan.groups.items()}
        return None if plan is None else _sample_plan(plan, time_ns)

    def _sample_runtime(self, time_ns: int) -> GroupVector | None:
        plan = self._runtime_plan_at(time_ns)
        return None if plan is None else _sample_plan(plan, time_ns)

    def sample_runtime(self, time_ns: int) -> GroupVector | None:
        """Sample the executed reference grid, returning None outside its coverage."""
        return self._sample_runtime(time_ns)

    def reference_horizon(
        self,
        *,
        now_ns: int,
        dt_ns: int,
        horizon_steps: int,
    ) -> ActionHorizon | None:
        # Report the plan that owns now_ns: before a committed plan starts the
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
            action_space=self.action_space,
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
