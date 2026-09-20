from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from manimux.types import GroupVector


@dataclass(frozen=True, slots=True)
class ScalarLimits:
    max_velocity: float | None
    max_acceleration: float | None
    position_limit_abs: float | None
    mode: Literal["per_joint", "isotropic"] = "per_joint"
    max_step_dt_s: float | None = None

    def velocity_bound(self, dt_s: float) -> float | None:
        """Cap the time redeemable as a position step, keeping nominal tick timing."""
        if self.max_velocity is None or self.max_step_dt_s is None:
            return self.max_velocity
        return self.max_velocity * min(dt_s, self.max_step_dt_s) / dt_s


def _scale_vector(values: np.ndarray, bound: float, independent_index: int | None) -> np.ndarray:
    """One scale per arm; an embedded gripper never determines that scale."""
    arm = values if independent_index is None else np.delete(values, independent_index)
    peak = float(np.max(np.abs(arm), initial=0.0))
    result = values.copy() if peak <= bound else values * (bound / peak)
    if independent_index is not None:
        result[independent_index] = np.clip(values[independent_index], -bound, bound)
    return result


def limit_velocity(
    desired: np.ndarray,
    previous: np.ndarray,
    dt_s: float,
    max_velocity: float | None,
    max_acceleration: float | None,
    max_closing_velocity: float | None = None,
    *,
    mode: Literal["per_joint", "isotropic"] = "per_joint",
    independent_index: int | None = None,
) -> np.ndarray:
    """Limit velocity, then its change; each isotropic stage scales one vector.

    With acceleration enabled the second stage preserves the direction of the
    velocity *change*, not necessarily the requested position increment.
    """
    velocity = np.asarray(desired, dtype=np.float64).copy()
    if max_closing_velocity is not None:
        velocity = np.maximum(velocity, -max_closing_velocity)
    if max_velocity is not None:
        velocity = (
            _scale_vector(velocity, max_velocity, independent_index)
            if mode == "isotropic"
            else np.clip(velocity, -max_velocity, max_velocity)
        )
    if max_acceleration is not None:
        delta = max_acceleration * dt_s
        velocity = (
            previous + _scale_vector(velocity - previous, delta, independent_index)
            if mode == "isotropic"
            else np.clip(velocity, previous - delta, previous + delta)
        )
    return velocity


def decelerate_velocity(
    velocity: np.ndarray,
    max_acceleration: float | None,
    dt_s: float,
    *,
    mode: Literal["per_joint", "isotropic"] = "per_joint",
    independent_index: int | None = None,
) -> np.ndarray:
    if max_acceleration is None:
        return np.zeros_like(velocity)
    if mode == "isotropic":
        return limit_velocity(
            np.zeros_like(velocity),
            velocity,
            dt_s,
            None,
            max_acceleration,
            mode=mode,
            independent_index=independent_index,
        )
    return np.sign(velocity) * np.maximum(np.abs(velocity) - max_acceleration * dt_s, 0.0)


def braking_velocity(distance: np.ndarray, acceleration: float, dt_s: float) -> np.ndarray:
    """Conservative speed that leaves room for this tick and a complete stop.

    Solve distance = v * dt + v**2 / (2*a). The extra tick avoids the
    discrete-time overshoot of the continuous sqrt(2*a*distance) rule.
    """
    distance = np.maximum(distance, 0.0)
    step = acceleration * dt_s
    # Rationalized form avoids cancellation very close to the target.
    return 2 * acceleration * distance / (np.sqrt(step * step + 2 * acceleration * distance) + step)


def tracking_step(
    target: GroupVector,
    target_velocity: GroupVector,
    previous: GroupVector,
    previous_velocity: GroupVector,
    *,
    dt_s: float,
    position_gain: float,
    limits: ScalarLimits,
    gripper_indices: dict[str, int] | None = None,
) -> tuple[GroupVector, GroupVector]:
    """Track a moving reference with feedforward and distance-aware braking.

    Preserve command velocity across replans. A new target inside the existing
    stopping distance can still be crossed: acceleration limits take precedence
    over teleporting the command onto it. Fixed reachable targets converge
    without the legacy repeated overshoot. This is not jerk-limited planning.
    """
    output, velocities = {}, {}
    acceleration = limits.max_acceleration
    max_velocity = limits.velocity_bound(dt_s)
    bound = limits.position_limit_abs
    assert bound is not None
    for name, desired in target.items():
        independent_index = (gripper_indices or {}).get(name)
        error = np.clip(desired, -bound, bound) - previous[name]
        correction = position_gain * error
        if acceleration is not None:
            correction = np.sign(error) * np.minimum(
                np.abs(correction), braking_velocity(np.abs(error), acceleration, dt_s)
            )
        velocity = (
            limit_velocity(
                target_velocity[name],
                previous_velocity[name],
                dt_s,
                max_velocity,
                None,
                mode=limits.mode,
                independent_index=independent_index,
            )
            + correction
        )
        # Also start braking before the absolute position limits.
        if acceleration is None:
            lower = (-bound - previous[name]) / dt_s
            upper = (bound - previous[name]) / dt_s
        else:
            lower = -braking_velocity(previous[name] + bound, acceleration, dt_s)
            upper = braking_velocity(bound - previous[name], acceleration, dt_s)
        if max_velocity is not None and limits.mode == "per_joint":
            lower = np.maximum(lower, -max_velocity)
            upper = np.minimum(upper, max_velocity)
        velocity = np.clip(velocity, lower, upper)
        velocity = limit_velocity(
            velocity,
            previous_velocity[name],
            dt_s,
            max_velocity if limits.mode == "isotropic" else None,
            acceleration,
            mode=limits.mode,
            independent_index=independent_index,
        )
        output[name] = previous[name] + velocity * dt_s
        velocities[name] = velocity
    return output, velocities


def limit_step(
    target: GroupVector,
    previous: GroupVector,
    previous_velocity: GroupVector,
    *,
    dt_s: float,
    limits: ScalarLimits,
    gripper_indices: dict[str, int] | None = None,
) -> tuple[GroupVector, GroupVector]:
    output: GroupVector = {}
    velocities: GroupVector = {}
    for name, desired in target.items():
        velocity = limit_velocity(
            (desired - previous[name]) / dt_s,
            previous_velocity[name],
            dt_s,
            limits.velocity_bound(dt_s),
            limits.max_acceleration,
            mode=limits.mode,
            independent_index=(gripper_indices or {}).get(name),
        )
        command = (
            desired.copy()
            if limits.max_velocity is None and limits.max_acceleration is None
            else previous[name] + velocity * dt_s
        )
        if limits.position_limit_abs is not None:
            command = np.clip(command, -limits.position_limit_abs, limits.position_limit_abs)
        output[name] = command
        velocities[name] = velocity
    return output, velocities


def arm_motion_parameters(**options) -> dict:
    """补齐本模块的默认参数；返回独立字典，不创建配置对象。"""

    values = {
        "max_velocity": None,
        "max_acceleration": None,
        "mode": "per_joint",
        "max_step_dt_s": None,
        **options,
    }
    # 非有限或非正的速率会破坏执行限幅，不能交给控制循环处理。
    if any(
        values[key] is not None and (not math.isfinite(values[key]) or values[key] <= 0)
        for key in ["max_velocity", "max_acceleration", "max_step_dt_s"]
    ):
        raise ValueError("motion rates must be finite and positive")
    if values["mode"] not in {"per_joint", "isotropic"}:
        raise ValueError("unknown arm motion mode")
    return values


def gripper_motion_parameters(**options) -> dict:
    """补齐本模块的默认参数；返回独立字典，不创建配置对象。"""

    values = {
        "max_velocity": None,
        "max_acceleration": None,
        "max_closing_velocity": None,
        **options,
    }
    # 非有限或非正的速率会破坏执行限幅，不能交给控制循环处理。
    if any(
        values[key] is not None and (not math.isfinite(values[key]) or values[key] <= 0)
        for key in ["max_velocity", "max_acceleration", "max_closing_velocity"]
    ):
        raise ValueError("motion rates must be finite and positive")
    return values


def motion_limits_parameters(**options) -> dict:
    """补齐本模块的默认参数；返回独立字典，不创建配置对象。"""

    values = {
        **options,
    }
    values["arm"] = arm_motion_parameters(**values["arm"])
    values["gripper"] = gripper_motion_parameters(**values["gripper"])
    return values
