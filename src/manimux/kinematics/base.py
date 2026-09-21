"""Common contract for one mechanical arm's offline kinematics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from scipy.spatial.transform import Rotation

FloatArray = NDArray[np.float64]


def transform_from_xyz_rpy(
    xyz: Sequence[float] = (0.0, 0.0, 0.0),
    rpy: Sequence[float] = (0.0, 0.0, 0.0),
    *,
    name: str = "frame",
) -> FloatArray:
    """Convert finite metre/XYZ-radian values to a rigid transform."""
    translation = np.asarray(xyz, dtype=np.float64)
    angles = np.asarray(rpy, dtype=np.float64)
    if (
        translation.shape != (3,)
        or angles.shape != (3,)
        or not np.isfinite(np.r_[translation, angles]).all()
    ):
        raise ValueError(f"{name} xyz/rpy must each contain three finite values")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_euler("xyz", angles).as_matrix()
    transform[:3, 3] = translation
    return transform


_IDENTITY3 = np.eye(3)


def rigid_transform(value: FloatArray, name: str) -> FloatArray:
    """Copy and validate a right-handed homogeneous transform.

    The same judgement as np.allclose/np.isclose at the tolerances this has
    always used (atol only, rtol=0), written out: every differential IK
    substep validates a target through here, and the numpy route costs about
    24 us against 3 us for the identical decisions.
    """
    pose = np.array(value, dtype=np.float64, copy=True)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    rotation = pose[:3, :3]
    bottom = pose[3]
    (m00, m01, m02), (m10, m11, m12), (m20, m21, m22) = rotation.tolist()
    determinant = (
        m00 * (m11 * m22 - m12 * m21)
        - m01 * (m10 * m22 - m12 * m20)
        + m02 * (m10 * m21 - m11 * m20)
    )
    if (
        abs(bottom[0]) > 1e-9
        or abs(bottom[1]) > 1e-9
        or abs(bottom[2]) > 1e-9
        or abs(bottom[3] - 1.0) > 1e-9
        or not np.all(np.abs(rotation.T @ rotation - _IDENTITY3) <= 1e-6)
        or abs(determinant - 1.0) > 1e-6
    ):
        raise ValueError(f"{name} must be a rigid homogeneous transform")
    return pose


class Frame(BaseModel):
    """Immutable xyz/rpy transform used by component assembly YAML."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def matrix(self) -> FloatArray:
        return transform_from_xyz_rpy(self.xyz, self.rpy, name="frame")


@dataclass(frozen=True, slots=True)
class IKResult:
    """Accepted joint output or an explicit geometric/solver rejection.

    Analytic implementations normally report a converged target solution;
    differential implementations report an accepted bounded step. Both use
    the same result contract. Backend-specific data belongs in diagnostics.
    """

    converged: bool
    joints: FloatArray | None = None
    reason: str = ""
    diagnostics: dict[str, object] = field(default_factory=dict)
    target_reached: bool = True

    def __post_init__(self) -> None:
        if self.converged:
            if self.joints is None:
                raise ValueError("successful IK requires joint positions")
            joints = np.array(self.joints, dtype=np.float64, copy=True)
            if joints.ndim != 1 or not joints.size or not np.isfinite(joints).all():
                raise ValueError("IK joint positions must be a non-empty finite vector")
            joints.setflags(write=False)
            object.__setattr__(self, "joints", joints)
        elif self.joints is not None or not self.reason.strip():
            raise ValueError("failed IK requires a reason and no joint solution")
        if not isinstance(self.target_reached, bool):
            raise TypeError("IK target_reached must be a boolean")
        if not self.converged:
            object.__setattr__(self, "target_reached", False)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    @property
    def ok(self) -> bool:
        """Neutral success spelling shared by analytic and differential solvers."""
        return self.converged


@dataclass(frozen=True, slots=True)
class KinematicCoordinate:
    """One independent geometric coordinate in a stable configuration layout."""

    name: str
    unit: Literal["rad", "m", "normalized"]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("kinematic coordinate name must not be empty")
        if self.unit not in {"rad", "m", "normalized"}:
            raise ValueError(f"unsupported kinematic coordinate unit: {self.unit!r}")


class ArmKinematicsBase(ABC):
    """Offline kinematics from one arm base to its mechanical flange.

    Concrete analytic and differential solvers share this interface. Poses are
    right-handed arm-base-to-flange matrices in metres; joint vectors exclude
    mounted tools. duration_s carries timing when a solver needs it.
    """

    @property
    @abstractmethod
    def num_joints(self) -> int:
        raise NotImplementedError

    @property
    def max_step_duration_s(self) -> float | None:
        """Optional integration-step cap; None means no backend cap."""
        return None

    @abstractmethod
    def fk(self, joints: FloatArray) -> FloatArray:
        """Return the arm-base-to-flange transform."""
        raise NotImplementedError

    @abstractmethod
    def ik(
        self,
        target: FloatArray,
        seed: FloatArray,
        *,
        duration_s: float | None = None,
    ) -> IKResult:
        """Return an accepted arm-joint solution or bounded solver step."""
        raise NotImplementedError

    def reset(self) -> None:
        """Discard optional stateful solver state; analytic solvers are no-ops."""
        return None
