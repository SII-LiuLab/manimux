"""Offline nominal Rizon4s flange FK and bounded, seed-initialized numerical IK.

The model is not a per-device calibration or a collision/path planner. Poses use
the physical base_link and flange, never a station/world or active RDK TCP frame.
"""

import threading

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from yourdfpy import URDF

from manimux.kinematics.base import FlangeKinematicsBase, IKResult, rigid_transform


class FlexivKinematics(FlangeKinematicsBase):
    num_arm_joints = 7

    def __init__(self, path, *, ik_max_joint_step_rad=None, ik_joint_motion_cost=0.0):
        if ik_max_joint_step_rad is not None and (
            isinstance(ik_max_joint_step_rad, bool)
            or not np.isfinite(ik_max_joint_step_rad)
            or ik_max_joint_step_rad <= 0
        ):
            raise ValueError("ik_max_joint_step_rad must be finite and positive")
        if (
            isinstance(ik_joint_motion_cost, bool)
            or not np.isfinite(ik_joint_motion_cost)
            or ik_joint_motion_cost < 0
        ):
            raise ValueError("ik_joint_motion_cost must be finite and nonnegative")
        if ik_joint_motion_cost and ik_max_joint_step_rad is None:
            raise ValueError("regularized Flexiv IK requires an explicit joint step bound")
        self.max_joint_step = ik_max_joint_step_rad
        self.motion_cost = ik_joint_motion_cost
        self._urdf = URDF.load(str(path), load_meshes=False)
        if self._urdf.actuated_joint_names != [f"joint{i}" for i in range(1, 8)]:
            raise ValueError("Flexiv URDF must contain joint1..joint7 in RDK order")
        self.bounds = np.array(
            [[joint.limit.lower, joint.limit.upper] for joint in self._urdf.actuated_joints]
        )
        if not np.isfinite(self.bounds).all() or np.any(self.bounds[:, 0] >= self.bounds[:, 1]):
            raise ValueError("Flexiv URDF requires finite joint limits")
        self._urdf.get_transform("flange", "base_link")
        self._lock = threading.RLock()

    @staticmethod
    def _joints(value):
        q = np.asarray(value, dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError("Flexiv joints must contain seven finite radians")
        return q

    def fk_flange(self, joints):
        q = self._joints(joints)
        with self._lock:
            self._urdf.update_cfg(q)
            return self._urdf.get_transform("flange", "base_link").copy()

    def ik_flange(self, target_flange, seed_joints):
        target = rigid_transform(target_flange, "target_flange")
        seed = self._joints(seed_joints)
        if self.max_joint_step is not None and np.any(
            (seed < self.bounds[:, 0]) | (seed > self.bounds[:, 1])
        ):
            return IKResult(False, reason="seed_outside_model_bounds")
        seed = np.clip(seed, self.bounds[:, 0], self.bounds[:, 1])
        lower, upper = self.bounds[:, 0].copy(), self.bounds[:, 1].copy()
        if self.max_joint_step is not None:
            lower = np.maximum(lower, seed - self.max_joint_step)
            upper = np.minimum(upper, seed + self.max_joint_step)

        def residual(q):
            pose = self.fk_flange(q)
            error = np.r_[
                pose[:3, 3] - target[:3, 3],
                Rotation.from_matrix(target[:3, :3].T @ pose[:3, :3]).as_rotvec(),
            ]
            return np.r_[error, self.motion_cost * (q - seed)] if self.motion_cost else error

        result = least_squares(
            residual,
            seed,
            bounds=(lower, upper),
            max_nfev=200,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        error = residual(result.x)[:6]
        if np.linalg.norm(error[:3]) <= 1e-4 and np.linalg.norm(error[3:]) <= 1e-4:
            return IKResult(True, joints=result.x)
        return IKResult(
            False,
            reason=(
                "no_solution_within_joint_step_bounds"
                if self.max_joint_step is not None
                else "no_solution_within_model_bounds"
            ),
        )
