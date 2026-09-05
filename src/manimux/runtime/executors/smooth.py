from __future__ import annotations

import math

import numpy as np

from manimux.config import SmoothConfig
from manimux.runtime.executors.limits import ScalarLimits, limit_step
from manimux.types import (
    ActionHorizon,
    GroupVector,
    RobotCommand,
    RobotState,
    copy_group_vector,
)


class SmoothExecutor:
    def __init__(self, config: SmoothConfig, control_dt_s: float) -> None:
        self._dt_s = control_dt_s
        rc = 1.0 / (2.0 * math.pi * config.cutoff_hz)
        self._alpha = control_dt_s / (rc + control_dt_s)
        self._limits = ScalarLimits(
            max_velocity=config.max_velocity,
            max_acceleration=config.max_acceleration,
            position_limit_abs=config.position_limit_abs,
        )
        self._gripper = config.gripper
        self._previous: GroupVector | None = None
        self._previous_velocity: GroupVector | None = None
        self._gripper_closed: dict[str, bool] = {}
        self._gripper_closed_since_ns: dict[str, int | None] = {}
        self._gripper_open_candidate_ns: dict[str, int | None] = {}

    @property
    def horizon_steps(self) -> int:
        return 2

    def reset(self, state: RobotState) -> None:
        self._previous = copy_group_vector(state.groups)
        self._previous_velocity = {
            name: np.zeros_like(value) for name, value in state.groups.items()
        }
        self._gripper_closed = {}
        self._gripper_closed_since_ns = {}
        self._gripper_open_candidate_ns = {}
        if self._gripper is None:
            return
        for name, index in self._gripper.group_indices.items():
            if name not in state.groups:
                raise ValueError(f"smooth gripper group {name!r} is absent from robot state")
            if index >= len(state.groups[name]):
                raise ValueError(
                    f"smooth gripper index {index} is outside group {name!r} "
                    f"with dimension {len(state.groups[name])}"
                )
            closed = bool(state.groups[name][index] <= self._gripper.close_threshold)
            self._gripper_closed[name] = closed
            self._gripper_closed_since_ns[name] = state.monotonic_ns if closed else None
            self._gripper_open_candidate_ns[name] = None

    def _shape_grippers(
        self,
        now_ns: int,
        reference: ActionHorizon,
        output: GroupVector,
        velocities: GroupVector,
    ) -> None:
        if self._gripper is None:
            return
        assert self._previous is not None
        assert self._previous_velocity is not None
        min_closed_ns = int(self._gripper.min_closed_s * 1_000_000_000)
        open_confirm_ns = int(self._gripper.open_confirm_s * 1_000_000_000)
        for name, index in self._gripper.group_indices.items():
            desired = float(reference.groups[name][0, index])
            closed = self._gripper_closed[name]
            if self._gripper.mode == "continuous":
                goal = float(
                    np.clip(
                        desired,
                        self._gripper.closed_value,
                        self._gripper.open_value,
                    )
                )
            else:
                if not closed and desired <= self._gripper.close_threshold:
                    closed = True
                    self._gripper_closed[name] = True
                    self._gripper_closed_since_ns[name] = now_ns
                    self._gripper_open_candidate_ns[name] = None
                elif closed:
                    closed_since = self._gripper_closed_since_ns[name]
                    hold_elapsed = (
                        closed_since is not None
                        and now_ns - closed_since >= min_closed_ns
                    )
                    if hold_elapsed and desired >= self._gripper.open_threshold:
                        candidate = self._gripper_open_candidate_ns[name]
                        if candidate is None:
                            self._gripper_open_candidate_ns[name] = now_ns
                        elif now_ns - candidate >= open_confirm_ns:
                            closed = False
                            self._gripper_closed[name] = False
                            self._gripper_closed_since_ns[name] = None
                            self._gripper_open_candidate_ns[name] = None
                    else:
                        self._gripper_open_candidate_ns[name] = None

                goal = (
                    self._gripper.closed_value if closed else self._gripper.open_value
                )
            previous = float(self._previous[name][index])
            previous_velocity = float(self._previous_velocity[name][index])
            velocity = float(
                np.clip(
                    (goal - previous) / self._dt_s,
                    -self._gripper.max_velocity,
                    self._gripper.max_velocity,
                )
            )
            velocity = float(
                np.clip(
                    velocity,
                    previous_velocity - self._gripper.max_acceleration * self._dt_s,
                    previous_velocity + self._gripper.max_acceleration * self._dt_s,
                )
            )
            command = previous + velocity * self._dt_s
            output[name][index] = float(
                np.clip(
                    command,
                    self._gripper.closed_value,
                    self._gripper.open_value,
                )
            )
            velocities[name][index] = velocity

    def step(
        self,
        now_ns: int,
        state: RobotState,
        reference: ActionHorizon,
    ) -> RobotCommand:
        if self._previous is None or self._previous_velocity is None:
            self.reset(state)
        assert self._previous is not None
        assert self._previous_velocity is not None
        target = {
            name: self._previous[name] + self._alpha * (values[0] - self._previous[name])
            for name, values in reference.groups.items()
        }
        output, velocities = limit_step(
            target,
            self._previous,
            self._previous_velocity,
            dt_s=self._dt_s,
            limits=self._limits,
        )
        self._shape_grippers(now_ns, reference, output, velocities)
        self._previous = copy_group_vector(output)
        self._previous_velocity = copy_group_vector(velocities)
        return RobotCommand(groups=output, monotonic_ns=now_ns, plan_id=reference.plan_id)
