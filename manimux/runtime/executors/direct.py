from __future__ import annotations

import math

import numpy as np

from manimux.runtime.executors.limits import ScalarLimits, limit_step, limit_velocity
from manimux.types import ActionHorizon, GroupVector, RobotCommand, RobotState, copy_group_vector


class DirectExecutor:
    """Forward references with normalized gripper bounds and explicit motion limits."""

    def __init__(
        self, motion_limits: dict | None = None, control_dt_s: float = 0.01,
        *, gripper_indices: dict[str, int] | None = None,
    ) -> None:
        self._motion = motion_limits
        self._dt_s = control_dt_s
        motion_indices = None if motion_limits is None else motion_limits["gripper"]["group_indices"]
        if (
            gripper_indices is not None
            and motion_indices is not None
            and gripper_indices != motion_indices
        ):
            raise ValueError("Direct gripper indices conflict with motion limits")
        indices = motion_indices if gripper_indices is None else gripper_indices
        self._gripper_indices = dict(indices or {})
        self.gripper_clip_diagnostics: dict[str, dict[str, float | int]] = {}
        self._previous: GroupVector | None = None
        self._previous_velocity: GroupVector | None = None

    @property
    def horizon_steps(self) -> int:
        return 2

    def set_control_period(self, control_dt_s: float) -> None:
        """Change timing while retaining command and velocity history."""
        if not math.isfinite(control_dt_s) or control_dt_s <= 0:
            raise ValueError("control_dt_s must be finite and positive")
        self._dt_s = control_dt_s

    def reset(self, state: RobotState) -> None:
        self.gripper_clip_diagnostics = {}
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
        self.gripper_clip_diagnostics = {}
        raw_apertures = {}
        for name, index in self._gripper_indices.items():
            raw = float(output[name][index])
            if not math.isfinite(raw):
                raise ValueError(f"{name}: Direct gripper opening must be finite")
            raw_apertures[name] = raw
            output[name][index] = np.clip(raw, 0.0, 1.0)
        if self._motion is not None:
            if self._previous is None or self._previous_velocity is None:
                self.reset(state)
            output, velocities = limit_step(
                output,
                self._previous,
                self._previous_velocity,
                dt_s=self._dt_s,
                limits=ScalarLimits(
                    self._motion["arm"]["max_velocity"],
                    self._motion["arm"]["max_acceleration"],
                    None,
                    self._motion["arm"]["mode"],
                    self._motion["arm"]["max_step_dt_s"],
                ),
                gripper_indices=self._motion["gripper"]["group_indices"],
            )
            gripper = self._motion["gripper"]
            for name, index in gripper["group_indices"].items():
                desired = float(np.clip(raw_apertures[name], 0.0, 1.0))
                previous = self._previous[name][index]
                velocity = float(
                    limit_velocity(
                        np.asarray((desired - previous) / self._dt_s),
                        np.asarray(self._previous_velocity[name][index]),
                        self._dt_s,
                        gripper["max_velocity"],
                        gripper["max_acceleration"],
                        gripper["max_closing_velocity"],
                    )
                )
                output[name][index] = (
                    desired
                    if (
                        gripper["max_velocity"] is None
                        and gripper["max_acceleration"] is None
                        and (gripper["max_closing_velocity"] is None or desired >= previous)
                    )
                    else previous + velocity * self._dt_s
                )
                velocities[name][index] = velocity
        # Rate-limited motion can itself overshoot an endpoint while decelerating.
        # Bound the final command too, and retain the actual emitted increment.
        for name, index in self._gripper_indices.items():
            raw = raw_apertures[name]
            target = float(np.clip(raw, 0.0, 1.0))
            before = float(output[name][index])
            bounded = float(np.clip(before, 0.0, 1.0))
            output[name][index] = bounded
            if self._motion is not None:
                velocities[name][index] = (bounded - self._previous[name][index]) / self._dt_s
            if raw != target or before != bounded:
                self.gripper_clip_diagnostics[name] = {
                    "coordinate_index": index,
                    "raw_aperture": raw,
                    "clipped_aperture": target,
                    "unclipped_command_aperture": before,
                    "command_aperture": bounded,
                    "reference_clip_delta": target - raw,
                    "command_clip_delta": bounded - before,
                }
        if self._motion is not None:
            self._previous = copy_group_vector(output)
            self._previous_velocity = velocities
        return RobotCommand(
            groups=output,
            monotonic_ns=now_ns,
            plan_id=reference.plan_id,
        )
