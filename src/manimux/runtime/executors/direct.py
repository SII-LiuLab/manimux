from __future__ import annotations

import numpy as np

from manimux.config import MotionLimitsConfig
from manimux.runtime.executors.limits import ScalarLimits, limit_step, limit_velocity
from manimux.types import ActionHorizon, GroupVector, RobotCommand, RobotState, copy_group_vector


class DirectExecutor:
    """Forward the reference, applying only explicitly shared motion limits."""

    def __init__(
        self, motion_limits: MotionLimitsConfig | None = None, control_dt_s: float = 0.01
    ) -> None:
        self._motion = motion_limits
        self._dt_s = control_dt_s
        self._previous: GroupVector | None = None
        self._previous_velocity: GroupVector | None = None

    @property
    def horizon_steps(self) -> int:
        return 2

    def reset(self, state: RobotState) -> None:
        self._previous = copy_group_vector(state.groups)
        self._previous_velocity = {
            name: np.zeros_like(value) for name, value in state.groups.items()
        }

    def step(
        self,
        now_ns: int,
        state: RobotState,
        reference: ActionHorizon,
    ) -> RobotCommand:
        output = {name: values[0].copy() for name, values in reference.groups.items()}
        if self._motion is not None:
            if self._previous is None or self._previous_velocity is None:
                self.reset(state)
            assert self._previous is not None and self._previous_velocity is not None
            output, velocities = limit_step(
                output, self._previous, self._previous_velocity, dt_s=self._dt_s,
                limits=ScalarLimits(
                    self._motion.arm.max_velocity, self._motion.arm.max_acceleration, None
                ),
            )
            gripper = self._motion.gripper
            for name, index in gripper.group_indices.items():
                desired = reference.groups[name][0, index]
                previous = self._previous[name][index]
                velocity = float(limit_velocity(
                    np.asarray((desired - previous) / self._dt_s),
                    np.asarray(self._previous_velocity[name][index]), self._dt_s,
                    gripper.max_velocity, gripper.max_acceleration,
                    gripper.max_closing_velocity,
                ))
                output[name][index] = (
                    desired if (
                        gripper.max_velocity is None and gripper.max_acceleration is None
                        and (gripper.max_closing_velocity is None or desired >= previous)
                    )
                    else previous + velocity * self._dt_s
                )
                velocities[name][index] = velocity
            self._previous = copy_group_vector(output)
            self._previous_velocity = velocities
        return RobotCommand(
            groups=output,
            monotonic_ns=now_ns,
            plan_id=reference.plan_id,
        )
