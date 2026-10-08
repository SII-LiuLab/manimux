"""EE-space waypoint handoff between action chunks, applied before dense IK.

Adapted from ManiUniCon's PoseTrajectoryInterpolator.schedule_waypoint as used by
its robot loop (https://github.com/Universal-Control/ManiUniCon, maniunicon/utils/
pose_trajectory_interpolator.py and maniunicon/core/robot.py): the new chunk starts
from the commanded pose at the handoff, every row is scheduled in order, and each
segment is speed limited. A too-fast segment reaches only part of the way by its
row time and the next segment continues from there, so rows keep their times.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict, PositiveFloat
from scipy.spatial.transform import Rotation

from manimux.types import ActionChunk, ActionContext, ActionHorizon, AppliedHandoff, GroupVector

# (4x4 pose, gripper) target for one group at one time.
PoseTarget = tuple[np.ndarray, float]


class WaypointHandoffConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_pos_speed_m_s: PositiveFloat
    max_rot_speed_rad_s: PositiveFloat


@dataclass(frozen=True, slots=True)
class HandoffPlan:
    plan_id: str  # Outgoing runtime plan the handoff starts from.
    time_ns: int  # Handoff time t_s', on the new runtime grid.
    join_row: int  # First source time slot after the handoff time; earlier slots are dropped.
    start_state: GroupVector  # Outgoing command at time_ns, the dense IK seed.
    lead_in: dict[str, list[PoseTarget]]  # Per group, one target per runtime_dt up to join_row.
    targets: dict[str, list[PoseTarget]]  # Per group, speed-limited rows after join_row.
    limited_rows: int  # Rows (all groups) whose target was not reached by its row time.
    skipped_steps: int  # Source rows skipped: slot k holds row k + skipped_steps.


class WaypointHandoff:
    """Plans an EE lead-in from the outgoing command to a new chunk; adapters opt in."""

    def __init__(
        self, config: WaypointHandoffConfig, *, source_dt_ns: int, runtime_dt_ns: int
    ) -> None:
        """config: EE speed limits; source_dt_ns: model row step; runtime_dt_ns: dense IK step."""
        if not 0 < runtime_dt_ns <= source_dt_ns:
            raise ValueError("waypoint handoff requires 0 < runtime_dt_ns <= source_dt_ns")
        self.config = config
        self.source_dt_ns = source_dt_ns
        self.runtime_dt_ns = runtime_dt_ns

    @classmethod
    def from_options(
        cls, options: Mapping, *, source_dt_ns: int, runtime_dt_ns: int
    ) -> WaypointHandoff | None:
        """options: policy.adapter; returns None unless handoff_waypoint is configured."""
        values = options.get("handoff_waypoint")
        if values is None:
            return None
        return cls(
            WaypointHandoffConfig.model_validate(values),
            source_dt_ns=source_dt_ns,
            runtime_dt_ns=runtime_dt_ns,
        )

    @staticmethod
    def pose_distance(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        """a, b: 4x4 poses; returns (translation m, rotation rad)."""
        rotation = Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude()
        return float(np.linalg.norm(b[:3, 3] - a[:3, 3])), float(rotation)

    @staticmethod
    def interpolate(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
        """a, b: 4x4 poses; alpha in [0, 1]; linear translation and slerp rotation."""
        pose = np.eye(4)
        pose[:3, 3] = (1.0 - alpha) * a[:3, 3] + alpha * b[:3, 3]
        delta = Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).as_rotvec()
        pose[:3, :3] = a[:3, :3] @ Rotation.from_rotvec(alpha * delta).as_matrix()
        return pose

    def limit_step(
        self, start: np.ndarray, target: np.ndarray, duration_ns: int
    ) -> tuple[np.ndarray, bool]:
        """start, target: 4x4 poses; duration_ns: segment time; returns the pose reached within
        the speed limits and whether the limit cut the segment short."""
        translation, rotation = self.pose_distance(start, target)
        need_s = max(
            translation / self.config.max_pos_speed_m_s,
            rotation / self.config.max_rot_speed_rad_s,
        )
        if need_s * 1e9 <= duration_ns:
            return target, False
        return self.interpolate(start, target, duration_ns / (need_s * 1e9)), True

    @staticmethod
    def sample_reference(reference: ActionHorizon, time_ns: int) -> GroupVector:
        """reference: copied runtime rows; time_ns: sample time; interpolates like the timeline."""
        position = (time_ns - reference.start_time_ns) / reference.dt_ns
        last = reference.horizon_steps - 1
        if not 0 <= position <= last:
            raise ValueError("waypoint handoff time lies outside the outgoing reference")
        lower = min(int(np.floor(position)), last)
        upper = min(lower + 1, last)
        alpha = min(position - lower, 1.0)
        return {
            name: (1.0 - alpha) * values[lower] + alpha * values[upper]
            for name, values in reference.groups.items()
        }

    def plan(
        self,
        context: ActionContext,
        *,
        origin_ns: int,
        targets: Mapping[str, Sequence[PoseTarget]],
        fk: Mapping[str, Callable[[np.ndarray], PoseTarget]],
    ) -> HandoffPlan:
        """context: carries handoff_reference, execution_time_ns and handoff_skip_steps;
        origin_ns: time of source row 0; targets: per group model rows as (pose, gripper); fk:
        per group, state row -> (pose, gripper). All groups share one join row so decode
        partitions agree. A skip moves later rows up without moving the handoff time."""
        reference = context.handoff_reference
        earliest_ns = context.execution_time_ns
        skip = context.handoff_skip_steps
        if reference is None or earliest_ns is None:
            raise ValueError("waypoint handoff requires a reference and an execution time")
        if type(skip) is not int or skip < 0:
            raise ValueError("waypoint handoff skip must be a non-negative integer")
        # Time slots on the source grid; slot k holds row k + skip, so the tail loses skip slots.
        horizon = len(next(iter(targets.values()))) - skip
        for row in range(horizon):
            row_ns = origin_ns + row * self.source_dt_ns
            steps = (row_ns - earliest_ns) // self.runtime_dt_ns
            if steps >= 1:
                break
        else:
            raise ValueError("waypoint handoff has no chunk row after the handoff time")
        # Align the handoff to the new runtime grid, which ends exactly at the join row.
        time_ns = row_ns - steps * self.runtime_dt_ns
        state = self.sample_reference(reference, time_ns)
        lead_in, limited_targets, limited_rows = {}, {}, 0
        for group in targets:
            start_pose, start_grip = fk[group](state[group])
            pose, previous_ns, rows = start_pose, time_ns, []
            for index in range(row, horizon):
                target, grip = targets[group][index + skip]
                index_ns = origin_ns + index * self.source_dt_ns
                pose, limited = self.limit_step(pose, target, index_ns - previous_ns)
                limited_rows += limited
                rows.append((pose, grip))
                previous_ns = index_ns
            join_pose, join_grip = rows[0]
            lead_in[group] = [
                (
                    self.interpolate(start_pose, join_pose, index / steps),
                    start_grip + (join_grip - start_grip) * index / steps,
                )
                for index in range(1, steps + 1)
            ]
            limited_targets[group] = rows[1:]
        return HandoffPlan(
            reference.plan_id, time_ns, row, state, lead_in, limited_targets, limited_rows, skip
        )

    def finish(self, chunk: ActionChunk, plan: HandoffPlan) -> ActionChunk:
        """chunk: decoded rows join_row.. with the lead-in prepended to its runtime trajectory;
        plan: the plan used; starts the runtime path at the handoff command and marks the chunk."""
        runtime = chunk.runtime_trajectory
        if runtime is None or runtime.dt_ns != self.runtime_dt_ns:
            raise ValueError("waypoint handoff requires the adapter's dense runtime trajectory")
        # Row 0 is the outgoing command itself, so the switch at time_ns is continuous.
        runtime.groups = {
            name: np.vstack([plan.start_state[name], values])
            for name, values in runtime.groups.items()
        }
        runtime.start_time_ns = plan.time_ns
        chunk.source_offset_steps = plan.join_row
        chunk.handoff = AppliedHandoff(
            plan.plan_id, plan.time_ns, plan.start_state, plan.skipped_steps
        )
        chunk.metadata.update(
            handoff_mode="waypoint",
            handoff_time_ns=plan.time_ns,
            handoff_join_row=plan.join_row,
            handoff_lead_in_steps=len(next(iter(plan.lead_in.values()))),
            handoff_limited_rows=plan.limited_rows,
        )
        return chunk
