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

    def __init__(self, path):
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
        seed = np.clip(self._joints(seed_joints), self.bounds[:, 0], self.bounds[:, 1])

        def residual(q):
            pose = self.fk_flange(q)
            return np.r_[
                pose[:3, 3] - target[:3, 3],
                Rotation.from_matrix(target[:3, :3].T @ pose[:3, :3]).as_rotvec(),
            ]

        result = least_squares(
            residual,
            seed,
            bounds=(self.bounds[:, 0], self.bounds[:, 1]),
            max_nfev=200,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        error = residual(result.x)
        if np.linalg.norm(error[:3]) <= 1e-4 and np.linalg.norm(error[3:]) <= 1e-4:
            return IKResult(True, joints=result.x)
        return IKResult(False, reason="no_solution_within_model_bounds")
