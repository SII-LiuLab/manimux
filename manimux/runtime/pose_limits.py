"""Optional execution-time TCP target limiting, independent of IK solving."""

from dataclasses import replace

import numpy as np

from manimux.kinematics.poses import matrix_pose
from manimux.policy_adapter.pose_output import POSE_SPACE, interpolate_pose


def pose_speed_limit_parameters(options):
    if options is None:
        return None
    if not isinstance(options, dict):
        raise ValueError("execution.ik.pose_speed_limit must be null or a mapping")
    fields = {"max_linear_m_s", "max_angular_rad_s"}
    if set(options) - fields:
        raise ValueError("unknown execution.ik.pose_speed_limit fields")
    values = {key: options.get(key) for key in sorted(fields)}
    if all(value is None for value in values.values()):
        raise ValueError("pose_speed_limit needs at least one speed; use null to disable")
    for key, value in values.items():
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"pose_speed_limit.{key} must be positive and finite")
    return values


def limit_pose_target(previous, target, dt_s, limits):
    """Limit the next TCP target from FK(last sent command), not the IK seed.

    One fraction preserves the requested translation/rotation path. Aperture has
    its own executor limits and is not slowed by a TCP speed limit.
    """
    distance = float(np.linalg.norm(target[:3] - previous[:3]))
    dot = float(np.clip(abs(np.dot(previous[3:7], target[3:7])), 0, 1))
    angle = 2 * np.arccos(dot)
    fraction = 1.0
    for delta, speed in (
        (distance, limits["max_linear_m_s"]),
        (angle, limits["max_angular_rad_s"]),
    ):
        if speed is not None and delta > 1e-12:
            fraction = min(fraction, speed * dt_s / delta)
    limited = interpolate_pose(previous, target, fraction)
    limited[7] = target[7]
    return limited, fraction


class PoseSpeedLimiter:
    """Limit one sampled pose per group, with no trajectory clock or hidden cursor."""

    def __init__(self, kinematics, limits):
        self.models = kinematics.models
        self.limits = pose_speed_limit_parameters(limits)
        if self.limits is None:
            raise ValueError("do not construct a pose limiter when limits are disabled")

    def apply(self, reference, last_command):
        if reference.action_space != POSE_SPACE or set(reference.groups) != set(last_command):
            raise ValueError("pose limiting requires matching pose and command groups")
        groups, fractions = {}, {}
        for name, rows in reference.groups.items():
            q = last_command[name]
            previous = np.r_[matrix_pose(self.models[name].fk(q)), q[-1]]
            target, fraction = limit_pose_target(
                previous, rows[0], reference.dt_ns / 1e9, self.limits
            )
            groups[name] = target[None, :]
            if fraction < 1:
                fractions[name] = fraction
        return replace(reference, groups=groups, tracking_groups=None), fractions
