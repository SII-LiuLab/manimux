"""Damped-least-squares (DLS) chunk IK for delta-pose policies on YAM: bounded, motion-regularized.

Each output step starts from the previous accepted joints and minimizes

    |pose error|^2 + joint_motion_cost^2 * |q - q_previous|^2

inside a box of max_joint_step_rad per joint (iterated damped least squares). The box
keeps the solution on one branch. Well-conditioned motion tracks to ~1-2 mm; near a
singularity (target near the base yaw axis at lift-off, wrist at j5 ~ +-pi/2) a large
joint swing that buys little pose is not taken. A residual within the max errors is
accepted as an approximation; a larger one is an IK failure for
execution.ik.failure_policy.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class DLSIKConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    max_joint_step_rad: float = Field(gt=0)
    max_position_error_m: float = Field(gt=0)
    max_rotation_error_rad: float = Field(gt=0)
    # Zero disables the regularizer (pure closest pose inside the box).
    joint_motion_cost: float = Field(ge=0)


class DLSIK:
    def __init__(self, model, config: DLSIKConfig):
        if not hasattr(model, "ik_nearby"):
            raise ValueError(f"{type(model).__name__} does not support DLS IK")
        self.model = model
        self.config = config

    def solve(self, target, joints, gripper):
        """(accepted, joints, residual); residual is set only for an accepted approximation."""
        config = self.config
        converged, configuration, info = self.model.ik_nearby(
            target,
            np.r_[joints, gripper],
            fixed_coordinates={self.model.coordinates[-1].name: gripper},
            max_joint_step=config.max_joint_step_rad,
            joint_motion_cost=config.joint_motion_cost or None,
        )
        solved = configuration[: len(joints)]
        if converged:
            return True, solved, None
        residual = {
            "position_error_m": float(info["position_error_m"]),
            "rotation_error_rad": float(info["rotation_error_rad"]),
        }
        if (
            residual["position_error_m"] <= config.max_position_error_m
            and residual["rotation_error_rad"] <= config.max_rotation_error_rad
        ):
            return True, solved, residual
        return False, np.asarray(joints).copy(), None
