"""Flexiv teleoperation differential IK, adapted to offline SI-unit geometry.

Derived from SII-LiuLab/tianji-control, algos/flexiv_diff_ik.py at
4d20258bdf8d53d66798da07691d7c5bbcb4c023. Uses ManiMux's calibrated URDF and
base/flange transforms, with no SDK or external checkout dependency.
"""

import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

from manimux.kinematics.base import rigid_transform
from manimux.kinematics.differential import DifferentialIKResult


class FlexivDiffIK:
    """One velocity-bounded increment from the previous command, not exact IK."""

    def __init__(
        self,
        model,
        *,
        max_joint_velocity_rad_s,
        max_linear_m_s,
        max_angular_rad_s,
        max_step_dt_s,
        tracking_limit_rad,
        slowdown_zone_rad,
        avoidance_zone_rad,
        nullspace_rate_rad_s,
        nullspace_enabled,
        joint_margin_rad,
        rotation_weight_m,
        damping,
    ):
        positive = {
            "max_joint_velocity_rad_s": max_joint_velocity_rad_s,
            "max_linear_m_s": max_linear_m_s,
            "max_angular_rad_s": max_angular_rad_s,
            "max_step_dt_s": max_step_dt_s,
            "tracking_limit_rad": tracking_limit_rad,
            "slowdown_zone_rad": slowdown_zone_rad,
            "avoidance_zone_rad": avoidance_zone_rad,
            "rotation_weight_m": rotation_weight_m,
            "damping": damping,
        }
        for name, value in positive.items():
            if (
                isinstance(value, bool)
                or not np.isscalar(value)
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"differential IK {name} must be finite and positive")
            setattr(self, name, float(value))
        for name, value in {
            "joint_margin_rad": joint_margin_rad,
            "nullspace_rate_rad_s": nullspace_rate_rad_s,
        }.items():
            if (
                isinstance(value, bool)
                or not np.isscalar(value)
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"differential IK {name} must be finite and nonnegative")
            setattr(self, name, float(value))
        if type(nullspace_enabled) is not bool:
            raise ValueError("differential IK nullspace_enabled must be boolean")
        self.nullspace_enabled = nullspace_enabled
        self.model = model
        self.lower = model.bounds[:, 0] + joint_margin_rad
        self.upper = model.bounds[:, 1] - joint_margin_rad
        if np.any(self.lower >= self.upper):
            raise ValueError("differential IK joint margin leaves no valid range")
        self.weights = np.array([1, 1, 1, *([rotation_weight_m] * 3)])
        self.direction = None

    def reset(self):
        self.direction = None

    def step(self, target, previous, measured, *, dt_s, bias_rate_rad_s=0.0):
        target = rigid_transform(target, "differential IK target_flange")
        q = self.model._joints(previous)
        measured = self.model._joints(measured)
        if not np.isfinite(dt_s) or dt_s < 0 or not np.isfinite(bias_rate_rad_s):
            raise ValueError("invalid differential IK timestep/bias")
        if np.any(q < self.lower) or np.any(q > self.upper):
            return DifferentialIKResult(False, reason="seed_outside_model_bounds")
        dt = min(dt_s, self.max_step_dt_s)
        if dt == 0:
            return DifferentialIKResult(True, q, position_bounds=np.column_stack((q, q)))
        here, jacobian = self.model.fk_flange_jacobian(q)
        error = np.r_[
            target[:3, 3] - here[:3, 3],
            Rotation.from_matrix(target[:3, :3] @ here[:3, :3].T).as_rotvec(),
        ]
        twist = error / dt
        for section, speed in (
            (slice(0, 3), self.max_linear_m_s),
            (slice(3, 6), self.max_angular_rad_s),
        ):
            norm = np.linalg.norm(twist[section])
            if norm > speed:
                twist[section] *= speed / norm

        # Keep measurement distinct: a lagging joint must not gain outward drive.
        low_distance = np.minimum(q, measured) - self.lower
        high_distance = self.upper - np.maximum(q, measured)
        lower = -self.max_joint_velocity_rad_s * np.clip(
            low_distance / self.slowdown_zone_rad, 0, 1
        )
        upper = self.max_joint_velocity_rad_s * np.clip(
            high_distance / self.slowdown_zone_rad, 0, 1
        )
        lower = np.maximum(
            lower, np.maximum(self.lower - q, measured - self.tracking_limit_rad - q) / dt
        )
        upper = np.minimum(
            upper, np.minimum(self.upper - q, measured + self.tracking_limit_rad - q) / dt
        )
        if np.any(lower >= upper):
            return DifferentialIKResult(False, reason="tracking")

        preferred = np.zeros(7)
        moving = np.linalg.norm(error[:3]) > 1e-9 or np.linalg.norm(error[3:]) > 1e-8
        if moving or abs(bias_rate_rad_s) > 1e-9:
            _, singular, vh = np.linalg.svd(jacobian, full_matrices=True)
            direction = vh[-1]
            if self.direction is not None and np.dot(direction, self.direction) < 0:
                direction = -direction
            if self.direction is None and direction[np.argmax(np.abs(direction))] < 0:
                direction = -direction
            self.direction = direction.copy()
            if self.nullspace_enabled:
                push = np.clip(1 - (q - self.lower) / self.avoidance_zone_rad, 0, 1) ** 2
                push -= np.clip(1 - (self.upper - q) / self.avoidance_zone_rad, 0, 1) ** 2
                preferred = (np.eye(7) - np.linalg.pinv(jacobian, rcond=1e-4) @ jacobian) @ (
                    self.nullspace_rate_rad_s * push
                )
            if singular[-1] >= 1e-4:
                preferred += direction / max(np.max(np.abs(direction)), 1e-12) * bias_rate_rad_s

        matrix = np.vstack((self.weights[:, None] * jacobian, self.damping * np.eye(7)))
        rhs = np.r_[self.weights * twist, self.damping * preferred]
        result = lsq_linear(
            matrix, rhs, bounds=(lower, upper), method="bvls", tol=1e-9, max_iter=30
        )
        if not result.success or not np.isfinite(result.x).all():
            return DifferentialIKResult(False, reason="solver")
        candidate = np.clip(q + result.x * dt, self.lower, self.upper)
        limited = bool(np.any(result.active_mask))
        near_limit = np.any(np.minimum(low_distance, high_distance) < self.slowdown_zone_rad)
        reason = ("soft_joint_limit" if near_limit else "joint_rate_limit") if limited else ""
        bounds = np.column_stack((q + lower * dt, q + upper * dt))
        return DifferentialIKResult(True, candidate, limited, reason, bounds)
