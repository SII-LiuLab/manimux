"""Offline URDF kinematics for six joints and an integrated scalar gripper."""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from yourdfpy import URDF

from .base import (
    FloatArray,
    IKResult,
    KinematicCoordinate,
    ManipulatorKinematicsBase,
    rigid_transform,
)

_POSITION_TOLERANCE_M = 1e-4
_ROTATION_TOLERANCE_RAD = 1e-4


class URDFManipulatorKinematics(ManipulatorKinematicsBase):
    """Explicit TCP and visual gripper mapping; bounded IK checks its final pose."""

    coordinates = tuple(KinematicCoordinate(f"joint{i}", "rad") for i in range(1, 7)) + (
        KinematicCoordinate("gripper", "normalized"),
    )

    def __init__(
        self,
        path: Path,
        *,
        base_frame: str,
        tcp_frame: str,
        joint_names: Sequence[str],
        gripper_joints: Sequence[str] = (),
        gripper_closed: ArrayLike = (),
        gripper_open: ArrayLike = (),
        joint_limits: ArrayLike | None = None,
    ) -> None:
        self._base_frame = base_frame
        self._tcp_frame = tcp_frame
        self._urdf = URDF.load(str(path), load_meshes=False)
        self._gripper_closed = np.asarray(gripper_closed, dtype=float)
        self._gripper_open = np.asarray(gripper_open, dtype=float)
        if len(joint_names) != 6 or self._urdf.actuated_joint_names != [
            *joint_names,
            *gripper_joints,
        ]:
            raise ValueError("URDF must declare the selected six joints and visual gripper order")
        if (
            self._gripper_closed.shape != (len(gripper_joints),)
            or self._gripper_open.shape != self._gripper_closed.shape
            or not np.isfinite(self._gripper_closed).all()
            or not np.isfinite(self._gripper_open).all()
        ):
            raise ValueError("invalid visual gripper endpoints")
        bounds = np.array([[j.limit.lower, j.limit.upper] for j in self._urdf.actuated_joints[:6]])
        if joint_limits is not None:
            bounds = np.asarray(joint_limits, dtype=float)
        if (
            bounds.shape != (6, 2)
            or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] >= bounds[:, 1])
        ):
            raise ValueError("joint_limits must contain six finite lower/upper pairs")
        self._lower = np.r_[bounds[:, 0], 0.0]
        self._upper = np.r_[bounds[:, 1], 1.0]
        self._urdf.get_transform(tcp_frame, base_frame)
        self._lock = threading.RLock()

    @property
    def base_frame(self) -> str:
        return self._base_frame

    @property
    def tcp_frame(self) -> str:
        return self._tcp_frame

    @staticmethod
    def _configuration(value: ArrayLike) -> FloatArray:
        q = np.asarray(value, dtype=np.float64)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError("configuration must contain six radians and a gripper opening")
        if not 0.0 <= q[6] <= 1.0:
            raise ValueError("gripper opening must be in [0, 1]")
        return q

    def visual_configuration(self, configuration: ArrayLike) -> FloatArray:
        q = self._configuration(configuration)
        return np.r_[
            q[:6], self._gripper_closed + q[6] * (self._gripper_open - self._gripper_closed)
        ]

    def fk(self, configuration: FloatArray) -> FloatArray:
        q = self.visual_configuration(configuration)
        with self._lock:
            self._urdf.update_cfg(q)
            return self._urdf.get_transform(self.tcp_frame, self.base_frame).copy()

    def ik(
        self,
        target_tcp: FloatArray,
        seed_configuration: FloatArray,
        *,
        fixed_coordinates: Mapping[str, float],
    ) -> IKResult:
        target = rigid_transform(target_tcp, "target_tcp")
        q = self._configuration(seed_configuration).copy()
        names = [coordinate.name for coordinate in self.coordinates]
        if set(fixed_coordinates) - set(names):
            raise ValueError("unknown fixed coordinate")
        for name, value in fixed_coordinates.items():
            index = names.index(name)
            if not np.isfinite(value) or not self._lower[index] <= value <= self._upper[index]:
                raise ValueError(f"fixed coordinate {name} is outside model bounds")
            q[index] = value
        free = [i for i, name in enumerate(names) if name not in fixed_coordinates]
        q[free] = np.clip(q[free], self._lower[free], self._upper[free])

        def residual(values: FloatArray) -> FloatArray:
            q[free] = values
            pose = self.fk(q)
            return np.r_[
                pose[:3, 3] - target[:3, 3],
                Rotation.from_matrix(target[:3, :3].T @ pose[:3, :3]).as_rotvec(),
            ]

        with self._lock:
            if free:
                result = least_squares(
                    residual,
                    q[free],
                    bounds=(self._lower[free], self._upper[free]),
                    max_nfev=200,
                    ftol=1e-10,
                    xtol=1e-10,
                    gtol=1e-10,
                )
                q[free] = result.x
            error = residual(q[free])
        if (
            np.linalg.norm(error[:3]) <= _POSITION_TOLERANCE_M
            and np.linalg.norm(error[3:]) <= _ROTATION_TOLERANCE_RAD
        ):
            return IKResult(True, joints=q)
        return IKResult(False, reason="no_solution_within_model_bounds")
