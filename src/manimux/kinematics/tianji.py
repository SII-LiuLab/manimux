"""Tianji Marvin M6 arm kinematics.

Forward kinematics is the controller's own modified-DH chain, copied from the
vendor's ``CommonConfig/ccs_m6_40.MvKDCfg``. It reproduces the SDK's
``Marvin_Kine.fk()`` to numerical precision without loading the closed-source
library, so the viewer and pose-space adapters can use it in any environment.
Both arms share one DH table and one set of joint limits.

ManiMux-facing units follow the rest of the runtime: joints in radians, poses in
metres. The vendor SDK works in degrees and millimetres; conversion happens here
and nowhere else.

The tool frame comes from a mounted end effector (see
:mod:`manimux.kinematics.end_effector`) or, for a bare tool offset, from the
controller's ``xyzabc`` form (mm, degrees, extrinsic x-y-z Euler angles). With
neither, :meth:`fk` returns the flange.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from manimux.kinematics.base import FloatArray
from manimux.kinematics.end_effector import EndEffector, load_end_effector

# [alpha_deg, a_mm, d_mm, theta0_deg]; rows 0-6 are joints 1-7 (the joint angle
# is added to theta0), row 7 is the static flange offset.
DH_TABLE_M6_40: tuple[tuple[float, float, float, float], ...] = (
    (0.0, 0.0, 174.5, 0.0),
    (90.0, 0.0, 0.0, 0.0),
    (-90.0, 0.0, 287.0, 0.0),
    (90.0, 18.0, 0.0, 180.0),
    (90.0, 18.0, 314.0, 180.0),
    (90.0, 0.0, 0.0, 90.0),
    (90.0, 0.0, 0.0, 90.0),
    (90.0, 0.0, 95.0, 90.0),
)

# Controller position limits (PNVA table), degrees, joints 1-7.
JOINT_LIMITS_DEG: tuple[tuple[float, float], ...] = (
    (-170.0, 170.0),
    (-120.0, 120.0),
    (-170.0, 170.0),
    (-145.0, 60.0),
    (-170.0, 170.0),
    (-60.0, 60.0),
    (-90.0, 90.0),
)

NUM_ARM_JOINTS = 7


def xyzabc_to_matrix(xyzabc: Sequence[float]) -> FloatArray:
    """SDK ``xyzabc`` (mm, degrees) -> 4x4 transform in metres."""

    values = np.asarray(xyzabc, dtype=np.float64).reshape(-1)
    if values.size != 6 or not np.isfinite(values).all():
        raise ValueError(f"xyzabc must be 6 finite values, got {values.tolist()}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_euler("xyz", values[3:], degrees=True).as_matrix()
    transform[:3, 3] = values[:3] * 1e-3
    return transform


def _link_transform(alpha_deg: float, a_mm: float, d_mm: float, theta_deg: float) -> FloatArray:
    """Modified-DH link: Rx(alpha) Tx(a) Rz(theta) Tz(d), translation in metres."""

    alpha, theta = math.radians(alpha_deg), math.radians(theta_deg)
    ca, sa, ct, st = math.cos(alpha), math.sin(alpha), math.cos(theta), math.sin(theta)
    a, d = a_mm * 1e-3, d_mm * 1e-3
    return np.array(
        [
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -sa * d],
            [st * sa, ct * sa, ca, ca * d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


class TianjiKinematics:
    """Forward kinematics for one 7-DoF Marvin M6 arm, in its own base frame."""

    def __init__(
        self,
        tool_xyzabc: Sequence[float] | None = None,
        end_effector: EndEffector | str | None = None,
        dh_table: Sequence[Sequence[float]] = DH_TABLE_M6_40,
    ) -> None:
        if tool_xyzabc is not None and end_effector is not None:
            raise ValueError("pass either tool_xyzabc or end_effector, not both")
        rows = [tuple(float(value) for value in row) for row in dh_table]
        if len(rows) != NUM_ARM_JOINTS + 1 or any(len(row) != 4 for row in rows):
            raise ValueError("dh_table must have 8 rows of [alpha_deg, a_mm, d_mm, theta0_deg]")
        self._dh = rows
        self._flange_offset = _link_transform(*rows[NUM_ARM_JOINTS])
        if isinstance(end_effector, str):
            end_effector = None if end_effector == "none" else load_end_effector(end_effector)
        self.end_effector = end_effector
        self._tool: FloatArray | None = None
        if end_effector is not None:
            self._tool = end_effector.tool_transform()
        elif tool_xyzabc is not None:
            self._tool = xyzabc_to_matrix(tool_xyzabc)
        self._lower = np.radians([low for low, _ in JOINT_LIMITS_DEG])
        self._upper = np.radians([high for _, high in JOINT_LIMITS_DEG])

    @property
    def num_arm_joints(self) -> int:
        return NUM_ARM_JOINTS

    @property
    def state_dim(self) -> int:
        inputs = 0 if self.end_effector is None else self.end_effector.inputs
        return NUM_ARM_JOINTS + inputs

    @property
    def tool_transform(self) -> FloatArray | None:
        return None if self._tool is None else self._tool.copy()

    def flange(self, joints: FloatArray) -> FloatArray:
        """Joint positions (rad) -> 4x4 flange transform (m)."""

        q = np.asarray(joints, dtype=np.float64).reshape(-1)
        if q.size != NUM_ARM_JOINTS or not np.isfinite(q).all():
            raise ValueError(f"expected {NUM_ARM_JOINTS} finite joint angles, got {q.tolist()}")
        q_deg = np.degrees(q)
        transform = np.eye(4, dtype=np.float64)
        for (alpha, a, d, theta0), angle in zip(self._dh[:NUM_ARM_JOINTS], q_deg, strict=True):
            transform = transform @ _link_transform(alpha, a, d, theta0 + angle)
        return transform @ self._flange_offset

    def fk(self, joints: FloatArray, gripper: float) -> FloatArray:
        """Joint positions (rad) -> tool (or flange) transform. The jaws do not move the TCP."""

        del gripper
        flange = self.flange(joints)
        return flange if self._tool is None else flange @ self._tool

    def pose(self, configuration: FloatArray) -> FloatArray:
        """7 joints, optionally followed by end-effector inputs -> end-effector transform."""

        values = np.asarray(configuration, dtype=np.float64).reshape(-1)
        if values.size < NUM_ARM_JOINTS:
            raise ValueError(f"Tianji configuration needs 7 joints, got {values.size} values")
        return self.fk(values[:NUM_ARM_JOINTS], 1.0)

    def ik(
        self,
        target_pose: FloatArray,
        init_joints: FloatArray,
        gripper: float,
    ) -> tuple[bool, FloatArray]:
        raise NotImplementedError(
            "Tianji IK wraps the vendor's analytic solver and ships with the tianji_dual driver"
        )

    def joint_position_limits(self) -> tuple[FloatArray, FloatArray]:
        return self._lower.copy(), self._upper.copy()

    def clip_arm_joints(self, joints: FloatArray) -> FloatArray:
        q = np.asarray(joints, dtype=np.float64).reshape(-1)
        if q.size != NUM_ARM_JOINTS:
            raise ValueError(f"expected {NUM_ARM_JOINTS} joint angles, got {q.size}")
        return np.clip(q, self._lower, self._upper)

    def joint_limit_margins(self, joints: FloatArray) -> FloatArray:
        q = np.asarray(joints, dtype=np.float64).reshape(-1)
        if q.size != NUM_ARM_JOINTS:
            raise ValueError(f"expected {NUM_ARM_JOINTS} joint angles, got {q.size}")
        return np.minimum(q - self._lower, self._upper - q)
