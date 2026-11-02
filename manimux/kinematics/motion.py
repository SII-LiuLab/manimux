"""Offline bounded point-to-point planning over the existing TCP model contract.

The output is a joint-space path, not a Cartesian straight line or a collision
certificate. No agent protocol, hardware session or control thread is owned here.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from manimux.kinematics.base import ManipulatorKinematicsBase, rigid_transform


def pose_error(target, actual):
    """Return TCP position error in metres and geodesic orientation error in radians."""
    target = rigid_transform(target, "target")
    actual = rigid_transform(actual, "actual")
    return (
        float(np.linalg.norm(actual[:3, 3] - target[:3, 3])),
        float(Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).magnitude()),
    )


@dataclass(frozen=True)
class PoseMotionLimits:
    joint_speed_rad_s: float = 0.03
    linear_speed_m_s: float = 0.004
    angular_speed_rad_s: float = 0.04
    minimum_motion_s: float = 1.0
    max_translation_m: float = 0.03
    max_rotation_rad: float = 0.2

    def __post_init__(self):
        values = list(vars(self).values())
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("motion limits must be positive and finite")


class JointPathPlanner:
    """Sample a cosine joint path to a bounded TCP goal for one revolute arm.

    Tool coordinates are fixed explicitly by the caller through the model's IK
    contract. TCP checks sample workspace bounds; they do not check obstacles.
    """

    def __init__(
        self, kinematics: ManipulatorKinematicsBase, *, control_dt_s,
        workspace_min_m, workspace_max_m, limits: PoseMotionLimits, max_waypoints=2000,
    ):
        if not np.isfinite(control_dt_s) or control_dt_s <= 0:
            raise ValueError("control_dt_s must be positive and finite")
        if not isinstance(max_waypoints, int) or max_waypoints < 2:
            raise ValueError("max_waypoints must be an integer of at least two")
        bounds = np.array([workspace_min_m, workspace_max_m], dtype=float)
        if (
            bounds.shape != (2, 3) or not np.isfinite(bounds).all()
            or np.any(bounds[0] >= bounds[1])
        ):
            raise ValueError("workspace bounds must be finite ordered XYZ pairs")
        if any(c.unit == "m" for c in kinematics.coordinates):
            raise NotImplementedError("joint path planner currently supports revolute arms")
        self.joint_indices = [i for i, c in enumerate(kinematics.coordinates) if c.unit == "rad"]
        if not self.joint_indices:
            raise ValueError("joint path planner requires revolute coordinates")
        self.kinematics, self.limits = kinematics, limits
        self.dt_s, self.max_waypoints = float(control_dt_s), max_waypoints
        self.lower, self.upper = bounds

    def validate(self, path, start):
        path = np.asarray(path, dtype=float)
        start = np.asarray(start, dtype=float)
        if (
            path.ndim != 2 or path.shape[1] != self.kinematics.num_coordinates
            or not 1 <= len(path) <= self.max_waypoints
            or start.shape != (self.kinematics.num_coordinates,)
            or not np.isfinite(path).all() or not np.isfinite(start).all()
        ):
            raise ValueError("invalid joint path shape, values or waypoint count")
        previous = start
        for q in path:
            pose = self.kinematics.fk(q)
            if np.any(pose[:3, 3] < self.lower) or np.any(pose[:3, 3] > self.upper):
                raise ValueError("TCP path exceeds the configured workspace")
            delta = np.abs(q[self.joint_indices] - previous[self.joint_indices])
            if np.max(delta) > self.limits.joint_speed_rad_s * self.dt_s + 1e-8:
                raise ValueError("path exceeds the configured joint reference speed")
            previous = q

    def plan(self, target, seed, *, fixed_coordinates):
        seed = np.asarray(seed, dtype=float)
        target = rigid_transform(target, "target")
        distance, angle = pose_error(target, self.kinematics.fk(seed))
        limits = self.limits
        if distance > limits.max_translation_m + 1e-8 or angle > limits.max_rotation_rad + 1e-8:
            raise ValueError("primitive exceeds configured translation/rotation bounds")
        solution = self.kinematics.ik(target, seed, fixed_coordinates=fixed_coordinates)
        if not solution.converged:
            raise ValueError(f"IK failed: {solution.reason}")
        duration = max(
            limits.minimum_motion_s,
            1.6 * distance / limits.linear_speed_m_s,
            1.6 * angle / limits.angular_speed_rad_s,
            1.6 * np.max(np.abs(solution.joints - seed)[self.joint_indices])
            / limits.joint_speed_rad_s,
        )
        count = int(np.ceil(duration / self.dt_s)) + 1
        if count > self.max_waypoints:
            raise ValueError("primitive exceeds max_batch_actions")
        fraction = (1 - np.cos(np.linspace(0, np.pi, count))) / 2
        path = seed + fraction[:, None] * (solution.joints - seed)
        self.validate(path, seed)
        return path
