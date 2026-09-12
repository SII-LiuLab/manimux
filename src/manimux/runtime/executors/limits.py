from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from manimux.types import GroupVector


@dataclass(frozen=True, slots=True)
class ScalarLimits:
    max_velocity: float | None
    max_acceleration: float | None
    position_limit_abs: float | None


def limit_velocity(
    desired: np.ndarray,
    previous: np.ndarray,
    dt_s: float,
    max_velocity: float | None,
    max_acceleration: float | None,
    max_closing_velocity: float | None = None,
) -> np.ndarray:
    velocity = np.asarray(desired, dtype=np.float64).copy()
    if max_closing_velocity is not None:
        velocity = np.maximum(velocity, -max_closing_velocity)
    if max_velocity is not None:
        velocity = np.clip(velocity, -max_velocity, max_velocity)
    if max_acceleration is not None:
        delta = max_acceleration * dt_s
        velocity = np.clip(velocity, previous - delta, previous + delta)
    return velocity


def decelerate_velocity(
    velocity: np.ndarray, max_acceleration: float | None, dt_s: float
) -> np.ndarray:
    if max_acceleration is None:
        return np.zeros_like(velocity)
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
) -> tuple[GroupVector, GroupVector]:
    """Track a moving reference with feedforward and distance-aware braking.

    Preserve command velocity across replans. A new target inside the existing
    stopping distance can still be crossed: acceleration limits take precedence
    over teleporting the command onto it. Fixed reachable targets converge
    without the legacy repeated overshoot. This is not jerk-limited planning.
    """
    output, velocities = {}, {}
    acceleration = limits.max_acceleration
    max_velocity = limits.max_velocity
    bound = limits.position_limit_abs
    assert bound is not None
    for name, desired in target.items():
        error = np.clip(desired, -bound, bound) - previous[name]
        correction = position_gain * error
        if acceleration is not None:
            correction = np.sign(error) * np.minimum(
                np.abs(correction), braking_velocity(np.abs(error), acceleration, dt_s)
            )
        velocity = limit_velocity(
            target_velocity[name], previous_velocity[name], dt_s, max_velocity, None
        ) + correction
        # Also start braking before the absolute position limits.
        if acceleration is None:
            lower = (-bound - previous[name]) / dt_s
            upper = (bound - previous[name]) / dt_s
        else:
            lower = -braking_velocity(previous[name] + bound, acceleration, dt_s)
            upper = braking_velocity(bound - previous[name], acceleration, dt_s)
        if max_velocity is not None:
            lower = np.maximum(lower, -max_velocity)
            upper = np.minimum(upper, max_velocity)
        velocity = np.clip(velocity, lower, upper)
        velocity = limit_velocity(
            velocity, previous_velocity[name], dt_s, None, acceleration
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
) -> tuple[GroupVector, GroupVector]:
    output: GroupVector = {}
    velocities: GroupVector = {}
    for name, desired in target.items():
        velocity = limit_velocity(
            (desired - previous[name]) / dt_s,
            previous_velocity[name],
            dt_s,
            limits.max_velocity,
            limits.max_acceleration,
        )
        command = (
            desired.copy() if limits.max_velocity is None and limits.max_acceleration is None
            else previous[name] + velocity * dt_s
        )
        if limits.position_limit_abs is not None:
            command = np.clip(command, -limits.position_limit_abs, limits.position_limit_abs)
        output[name] = command
        velocities[name] = velocity
    return output, velocities
