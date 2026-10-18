"""Compose reference conversion and joint execution without owning the timeline."""

import time
from dataclasses import dataclass, replace

from manimux.runtime.ik import StepIKResolver, ik_parameters
from manimux.runtime.pose_limits import PoseSpeedLimiter
from manimux.timing import stage
from manimux.types import ActionHorizon, RobotCommand, RobotState, copy_group_vector


@dataclass(slots=True)
class ExecutionResult:
    reference: ActionHorizon
    command: RobotCommand
    ik_status: dict | None = None


class ExecutionPipeline:
    """One active-tick entry point; scheduling, safety and hardware remain outside.

    The same joint executor is reused by runtime lifecycle/hold handling. Pose
    limiting and IK are optional stages, constructed once, never per tick.
    """

    def __init__(self, executor, *, kinematics, ik_options, control_dt_ns, motion_limits=None):
        self.executor = executor
        options = ik_parameters(**(ik_options or {}))
        self.resolver = None
        self.pose_limiter = None
        if options["mode"] == "step":
            self.resolver = StepIKResolver(
                kinematics, budget_ms=options["budget_ms"] or control_dt_ns / 2e6,
                backend=options["backend"], diff_ik=options["diff_ik"],
                motion_limits=motion_limits, control_dt_ns=control_dt_ns,
            )
            if options["pose_speed_limit"] is not None:
                self.pose_limiter = PoseSpeedLimiter(kinematics, options["pose_speed_limit"])

    @property
    def horizon_steps(self):
        return 1 if self.resolver is not None else self.executor.horizon_steps

    def warmup(self, measured):
        if self.resolver is not None:
            self.resolver.warmup(measured)

    def reset(self):
        """Drop solver warm state on pause, timeline gaps and episode boundaries."""
        if self.resolver is not None:
            self.resolver.reset()

    def reference_from_joints(self, groups):
        """Convert the Timeline handoff anchor into its configured reference space."""
        return self.resolver.poses_from_joints(groups) if self.resolver is not None else groups

    def step(self, now_ns, state, reference, *, last_command):
        ik_status = None
        if self.resolver is not None:
            # Preserve one shared budget for pose limiting and both arm solves.
            with stage("step_ik"):
                started = time.monotonic_ns()
                deadline = started + self.resolver.budget_ns
                limited_groups = {}
                target = reference
                if self.pose_limiter is not None:
                    target, limited_groups = self.pose_limiter.apply(reference, last_command)
                # Differential IK integrates one command step. Start from the
                # last dispatched command, including executor shaping, so servo
                # lag does not discard progress. Analytic IK uses measured seeds.
                seed = last_command if self.resolver.backend == "diff" else state.groups
                resolved = self.resolver.resolve(target, seed, deadline_ns=deadline)
                ik_status = {
                    **self.resolver.last_result,
                    "elapsed_ms": (time.monotonic_ns() - started) / 1e6,
                    "pose_speed_limited": limited_groups,
                }
            if resolved is None:
                # Atomic hold belongs to execution, not to the mathematical IK solver.
                held = copy_group_vector(last_command)
                with stage("executor_step"):
                    self.executor.reset(RobotState(held, now_ns, state.sequence))
                    command = RobotCommand(copy_group_vector(held), now_ns, reference.plan_id)
                reference = replace(
                    reference,
                    groups={g: q[None, :] for g, q in held.items()},
                    action_space="joint_position",
                    tracking_groups=None,
                    hold_groups=(),
                )
                return ExecutionResult(reference, command, ik_status)
            reference = resolved
        if reference.action_space != "joint_position":
            raise ValueError("joint executor requires a joint_position reference")
        with stage("executor_step"):
            command = self.executor.step(now_ns, state, reference)
        return ExecutionResult(reference, command, ik_status)
