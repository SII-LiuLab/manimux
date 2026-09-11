from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from manimux.types import GroupVector


@dataclass(frozen=True, slots=True)
class ScalarLimits:
    max_velocity: float
    max_acceleration: float
    position_limit_abs: float


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
    a, vmax, bound = limits.max_acceleration, limits.max_velocity, limits.position_limit_abs
    for name, desired in target.items():
        error = np.clip(desired, -bound, bound) - previous[name]
        correction = np.sign(error) * np.minimum(
            position_gain * np.abs(error), braking_velocity(np.abs(error), a, dt_s)
        )
        velocity = np.clip(target_velocity[name], -vmax, vmax) + correction
        # Also start braking before the absolute position limits.
        lower = -np.minimum(vmax, braking_velocity(previous[name] + bound, a, dt_s))
        upper = np.minimum(vmax, braking_velocity(bound - previous[name], a, dt_s))
        velocity = np.clip(velocity, lower, upper)
        velocity = np.clip(
            velocity, previous_velocity[name] - a * dt_s, previous_velocity[name] + a * dt_s
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
        velocity = np.clip(
            (desired - previous[name]) / dt_s,
            -limits.max_velocity,
            limits.max_velocity,
        )
        velocity = np.clip(
            velocity,
            previous_velocity[name] - limits.max_acceleration * dt_s,
            previous_velocity[name] + limits.max_acceleration * dt_s,
        )
        command = previous[name] + velocity * dt_s
        output[name] = np.clip(
            command,
            -limits.position_limit_abs,
            limits.position_limit_abs,
        )
        velocities[name] = velocity
    return output, velocities
