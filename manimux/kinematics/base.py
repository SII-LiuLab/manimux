"""Contracts for complete manipulator and optional flange-only kinematics.

``ManipulatorKinematicsBase`` describes a configured TCP using either an
integrated model or a composition of arm and tool geometry. It does not require
``FlangeKinematicsBase``, which is an optional interface for flange-only solvers.
``ArmKinematics`` retains the existing configured end-effector pose convention
for current policies and drivers. These interfaces are not aliases: their pose
and configuration meanings must be adapted explicitly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def rigid_transform(value: FloatArray, name: str) -> FloatArray:
    """Copy a finite rigid transform at the configuration/pose input boundary."""
    pose = np.array(value, dtype=np.float64, copy=True)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    rotation = pose[:3, :3]
    if (
        not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError(f"{name} must be a rigid homogeneous transform")
    return pose


@dataclass(frozen=True, slots=True)
class IKResult:
    """Outcome of an IK solve, never an instruction to move hardware.

    On success, ``joints`` is a finite vector in the solver's declared order
    satisfying the solver's configured pose tolerances and joint constraints.
    A flange solver returns arm joints only; a manipulator solver returns all
    declared independent coordinates, including fixed end-effector coordinates.
    On failure, ``joints`` is ``None`` and ``reason`` explains the rejection
    (for example, ``no_solution``, ``joint_limit`` or ``iteration_limit``).
    Rejected candidates and fallback seeds are not exposed as solutions.
    """

    converged: bool
    joints: FloatArray | None = None
    reason: str = ""

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


@dataclass(frozen=True, slots=True)
class KinematicCoordinate:
    """One independent input/output coordinate of a manipulator model.

    Use ``rad`` for revolute joints and ``m`` for prismatic joints. ``normalized``
    denotes a calibrated stroke fraction in [0, 1]; an implementation must
    document its direction and mapping (for example, 0 closed and 1 open).
    Model-internal mimic joints are derived rather than separate coordinates.
    These are geometric coordinates, not effort or actuator-enable commands.
    """

    name: str
    unit: Literal["rad", "m", "normalized"]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("kinematic coordinate name must not be empty")
        if self.unit not in {"rad", "m", "normalized"}:
            raise ValueError(f"unsupported kinematic coordinate unit: {self.unit!r}")


class ManipulatorKinematicsBase(ABC):
    """Offline TCP FK/IK for an integrated or composed arm-and-tool model.

    No decomposition into a flange solver and a separate tool is required.
    All poses are right-handed 4x4 transforms ``T_base_tcp`` acting on column
    vectors, with translation in metres and a 3x3 rotation matrix. The reference
    base and selected TCP are explicitly named below and stable for this instance.
    Installation transforms and tool offsets are already included in its FK/IK;
    callers must not apply them again.

    Configuration vectors are finite arrays of shape ``(num_coordinates,)`` in
    ``coordinates`` order. They describe arm AND relevant end-effector geometry,
    regardless of whether the hardware has one SDK or several. Implementations
    own model-specific mappings and document limits and acceptance tolerances.
    They must not connect hardware, read live state, or issue commands.
    """

    @property
    @abstractmethod
    def coordinates(self) -> tuple[KinematicCoordinate, ...]:
        """Non-empty, stable coordinate layout with unique names and declared units."""
        raise NotImplementedError

    @property
    def num_coordinates(self) -> int:
        """Dimension of FK input, IK seed and successful IK result."""
        return len(self.coordinates)

    @property
    @abstractmethod
    def base_frame(self) -> str:
        """Non-empty name of the reference frame for all input/output poses."""
        raise NotImplementedError

    @property
    @abstractmethod
    def tcp_frame(self) -> str:
        """Non-empty name of the configured TCP, including its tool convention."""
        raise NotImplementedError

    @abstractmethod
    def fk(self, configuration: FloatArray) -> FloatArray:
        """Return ``T_base_tcp`` from a complete geometric configuration.

        For measured TCP poses, callers supply measured arm and tool coordinates.
        A model with a fixed TCP may be independent of gripper opening, but must
        still accept and validate its declared configuration layout.
        """
        raise NotImplementedError

    @abstractmethod
    def ik(
        self,
        target_tcp: FloatArray,
        seed_configuration: FloatArray,
        *,
        fixed_coordinates: Mapping[str, float],
    ) -> IKResult:
        """Solve a TCP target with explicit fixed-coordinate constraints.

        ``fixed_coordinates`` maps coordinate names to required target values in
        their declared units, e.g. ``{"gripper": 0.5}``. These override the seed
        at those coordinates and must be respected during solving and final FK
        validation, not patched into the solution afterward. Unlisted coordinates
        may be solved; an explicit empty mapping permits all coordinates to vary.
        A solver unable to honor the requested free/fixed pattern must raise
        ``NotImplementedError`` rather than silently freeze or free a coordinate.

        The seed initializes the solve/branch selection; it is not itself a set
        of constraints. A successful result contains ALL coordinates, including
        fixed ones, in the declared order. It must satisfy the TCP target, fixed
        values and joint limits to documented tolerances. Ordinary rejection
        returns ``IKResult(False, reason=...)`` without a candidate solution.

        Invalid shapes, transforms, non-finite values or unknown coordinate names
        raise ``ValueError``. Backend/dependency failures propagate as exceptions.
        No hardware motion or continuous/collision-free path is implied by success.
        """
        raise NotImplementedError


class FlangeKinematicsBase(ABC):
    """Optional offline FK/IK for a mechanical arm, excluding mounted tools.

    Integrated manipulator models need not implement this interface.

    Poses are right-handed 4x4 homogeneous transforms ``T_base_flange`` acting
    on column vectors: ``p_base = T_base_flange @ p_flange``. Translation is in
    metres; the upper-left 3x3 block is a rotation matrix. The base is the arm's
    own base frame, not the station/world frame.

    Joint vectors have shape ``(num_arm_joints,)`` in the implementation's
    documented order: radians for revolute joints, metres for prismatic joints.
    Gripper values are excluded. Implementations document limits and solver
    tolerances, validate inputs, and never connect or command robot hardware.
    """

    @property
    @abstractmethod
    def num_arm_joints(self) -> int:
        """Number of arm joints, excluding the end effector."""
        raise NotImplementedError

    @abstractmethod
    def fk_flange(self, joints: FloatArray) -> FloatArray:
        """Return ``T_base_flange`` for the given arm joint positions.

        Include the arm's mechanical flange geometry, but no mounting adapter,
        gripper or TCP offset. Invalid input raises ``ValueError``.
        """
        raise NotImplementedError

    @abstractmethod
    def ik_flange(self, target_flange: FloatArray, seed_joints: FloatArray) -> IKResult:
        """Solve a flange target in the arm base frame from an explicit seed.

        Use the seed to initialize the solve or select a solution branch; this
        does not guarantee a unique or continuous solution. Accept a solution
        only after checking configured pose tolerances and joint constraints.
        Ordinary solve rejection returns a failed ``IKResult``. Invalid input
        raises ``ValueError``; unavailable backends or SDK errors raise exceptions
        rather than being disguised as geometric non-convergence.
        """
        raise NotImplementedError


class ArmKinematics(Protocol):
    """Existing configured end-effector FK/IK interface, kept for compatibility.

    Depending on configuration, the pose may be a grasp site, TCP or bare flange.
    Do not treat ``fk``/``ik`` as flange operations without checking that contract.
    """

    @property
    def num_arm_joints(self) -> int: ...

    def fk(self, joints: FloatArray, gripper: float) -> FloatArray:
        """Joint positions -> 4x4 end-effector transform."""
        ...

    def ik(
        self,
        target_pose: FloatArray,
        init_joints: FloatArray,
        gripper: float,
    ) -> tuple[bool, FloatArray]:
        """4x4 end-effector target -> (converged, joint positions).

        ``init_joints`` seeds the solver; passing the current measured joints
        keeps the solution on the branch the arm is already on.
        """
        ...
