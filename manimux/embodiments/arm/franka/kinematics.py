"""Offline seven-joint FR3 flange geometry; no control SDK imports."""

import threading
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from yourdfpy import URDF

from manimux.kinematics.base import FlangeKinematicsBase, IKResult, rigid_transform

MODEL_PATH = Path(__file__).parent / "assets" / "fr3v2_1.urdf"


class FrankaKinematics(FlangeKinematicsBase):
    """Nominal FR3 v2.1 geometry, radians/metres, bounded local IK.

    The redundant solution stays near its seed; this solver does not check
    collisions. A configured URDF may supply the robot's calibrated geometry.
    """

    num_arm_joints = 7

    def __init__(
        self,
        *,
        urdf_path=None,
        position_tolerance_m=1e-4,
        rotation_tolerance_rad=1e-4,
        max_iterations=200,
    ):
        self.path = Path(urdf_path or MODEL_PATH)
        self._urdf = URDF.load(str(self.path), load_meshes=False)
        if self._urdf.actuated_joint_names != [f"joint{i}" for i in range(1, 8)]:
            raise ValueError("Franka URDF must contain joint1..joint7 in command order")
        self.bounds = np.array(
            [[j.limit.lower, j.limit.upper] for j in self._urdf.actuated_joints], dtype=float
        )
        if not np.isfinite(self.bounds).all() or np.any(self.bounds[:, 0] >= self.bounds[:, 1]):
            raise ValueError("Franka model requires finite ordered joint bounds")
        if (
            not np.isfinite([position_tolerance_m, rotation_tolerance_rad]).all()
            or min(position_tolerance_m, rotation_tolerance_rad) <= 0
            or not isinstance(max_iterations, int)
            or max_iterations < 1
        ):
            raise ValueError("IK tolerances and iteration budget must be positive")
        self.position_tolerance = position_tolerance_m
        self.rotation_tolerance = rotation_tolerance_rad
        self.max_iterations = max_iterations
        self._lock = threading.RLock()
        self._urdf.get_transform("link8", "link0")

    @staticmethod
    def _q(joints):
        q = np.asarray(joints, dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError("Franka requires seven finite joint angles in radians")
        return q

    def fk_flange(self, joints):
        q = self._q(joints)
        with self._lock:
            self._urdf.update_cfg(q)
            return self._urdf.get_transform("link8", "link0").copy()

    def ik_flange(self, target_flange, seed_joints):
        target = rigid_transform(target_flange, "target_flange")
        seed = self._q(seed_joints)
        lower, upper = self.bounds.T
        if np.any(seed < lower) or np.any(seed > upper):
            raise ValueError("Franka IK seed is outside model joint bounds")

        def residual(q):
            pose = self.fk_flange(q)
            return np.r_[
                pose[:3, 3] - target[:3, 3],
                Rotation.from_matrix(target[:3, :3].T @ pose[:3, :3]).as_rotvec(),
            ]

        # Each FK update/read is atomic and returns a copy. Holding its lock for
        # the whole optimizer would block runtime feedback/guards during planning.
        result = least_squares(
            residual,
            seed,
            bounds=(lower, upper),
            max_nfev=self.max_iterations,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        error = residual(result.x)
        if (
            np.linalg.norm(error[:3]) <= self.position_tolerance
            and np.linalg.norm(error[3:]) <= self.rotation_tolerance
        ):
            return IKResult(True, joints=result.x)
        return IKResult(False, reason="no_solution_within_model_bounds")
