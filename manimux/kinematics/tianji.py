"""Legacy Tianji entry point; new robots use arm solvers and shared composed kinematics.

Keep legacy end_effector/tool_xyzabc parameters and the TCP interface for callers
that have not migrated. TianjiArmKinematics supplies the official IK algorithm and joint constraints.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence

import numpy as np

from manimux.embodiments.arm.tianji.kinematics import (
    BD67_REAL as BD67_REAL,
)
from manimux.embodiments.arm.tianji.kinematics import (
    DH_TABLE_M6_40,
    IK_MAX_STEP_DEG,
    IK_POS_TOL_MM,
    IK_ROT_TOL_DEG,
    LIMIT_MARGIN_DEG,
    NUM_ARM_JOINTS,
    TianjiArmKinematics,
    xyzabc_to_matrix,
)
from manimux.embodiments.arm.tianji.kinematics import (
    JOINT_LIMITS_DEG as JOINT_LIMITS_DEG,
)
from manimux.embodiments.arm.tianji.kinematics import (
    j67_ok as j67_ok,
)
from manimux.kinematics.base import FloatArray
from manimux.kinematics.end_effector import EndEffector, load_end_effector


class TianjiKinematics(TianjiArmKinematics):
    """Legacy TCP API compatibility; new robots do not use this class to assemble tools."""

    def __init__(
        self,
        tool_xyzabc: Sequence[float] | None = None,
        end_effector: EndEffector | str | None = None,
        dh_table: Sequence[Sequence[float]] = DH_TABLE_M6_40,
        *,
        arm: str = "left",
        sdk_root: str | None = None,
        joint_limits_deg: Mapping[int | str, Sequence[float]] | None = None,
        pos_tol_mm: float = IK_POS_TOL_MM,
        rot_tol_deg: float = IK_ROT_TOL_DEG,
        max_step_deg: float = IK_MAX_STEP_DEG,
        limit_margin_deg: float = LIMIT_MARGIN_DEG,
    ) -> None:
        if tool_xyzabc is not None and end_effector is not None:
            raise ValueError("pass either tool_xyzabc or end_effector, not both")
        super().__init__(
            dh_table=dh_table,
            arm=arm,
            sdk_root=sdk_root,
            joint_limits_deg=joint_limits_deg,
            pos_tol_mm=pos_tol_mm,
            rot_tol_deg=rot_tol_deg,
            max_step_deg=max_step_deg,
            limit_margin_deg=limit_margin_deg,
        )
        # Support legacy names and objects only; RobotModel assembles components in new configurations.
        if isinstance(end_effector, str):
            end_effector = None if end_effector == "none" else load_end_effector(end_effector)
        self.end_effector = end_effector
        # _tool is T_flange_tcp in meters; gripper opening does not change the legacy fixed TCP.
        self._tool: FloatArray | None = None
        if end_effector is not None:
            self._tool = end_effector.tool_transform()
        elif tool_xyzabc is not None:
            self._tool = xyzabc_to_matrix(tool_xyzabc)
        self._tool_inverse = None if self._tool is None else np.linalg.inv(self._tool)

    @property
    def state_dim(self) -> int:
        inputs = 0 if self.end_effector is None else self.end_effector.inputs
        return NUM_ARM_JOINTS + inputs

    @property
    def tool_transform(self) -> FloatArray | None:
        return None if self._tool is None else self._tool.copy()

    def fk(self, joints: FloatArray, gripper: float) -> FloatArray:
        """Joint positions (rad) -> tool (or flange) transform. The jaws do not move the TCP."""

        del gripper
        flange = self.flange(joints)
        # T_base_tcp = T_base_flange @ T_flange_tcp; apply the tool offset once.
        return flange if self._tool is None else flange @ self._tool

    def pose(self, configuration: FloatArray) -> FloatArray:
        """7 joints, optionally followed by end-effector inputs -> end-effector transform."""

        values = np.asarray(configuration, dtype=np.float64).reshape(-1)
        if values.size < NUM_ARM_JOINTS:
            raise ValueError(f"Tianji configuration needs 7 joints, got {values.size} values")
        return self.fk(values[:NUM_ARM_JOINTS], 1.0)

    def pose_error(
        self, target_pose: FloatArray, joints: FloatArray, gripper: float
    ) -> tuple[float, float]:
        """Translational (m) and rotational (rad) FK residuals for IK diagnostics."""

        target = np.asarray(target_pose, dtype=np.float64)
        if target.shape != (4, 4) or not np.isfinite(target).all():
            raise ValueError(f"target_pose must be a finite 4x4 transform, got {target.shape}")
        actual = self.fk(joints, gripper)
        position_error_m = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
        relative = actual[:3, :3].T @ target[:3, :3]
        cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
        return position_error_m, float(np.arccos(cosine))

    def ik(
        self,
        target_pose: FloatArray,
        init_joints: FloatArray,
        gripper: float,
    ) -> tuple[bool, FloatArray]:
        """Tool target (m, arm base frame) -> (valid, joints in rad).

        ``init_joints`` is the reference: the SDK picks the solution nearest it
        and a solution further than ``max_step_deg`` on any joint is rejected
        unless another SDK solution is close enough. On rejection the clipped
        seed is returned.
        """

        del gripper
        converged, joints, _ = self._solve(target_pose, init_joints)
        return converged, joints

    def ik_bounded(
        self,
        target_pose: FloatArray,
        init_joints: FloatArray,
        gripper: float,
        *,
        deadline_ns: int,
    ) -> tuple[bool, FloatArray, dict[str, object]]:
        """``ik`` with diagnostics; the closed-form solve is one bounded call."""

        if time.monotonic_ns() >= deadline_ns:
            seed = np.asarray(init_joints, dtype=np.float64).reshape(-1)[:NUM_ARM_JOINTS]
            return False, self.clip_arm_joints(seed), {"reason": "budget_exceeded", "iterations": 0}
        converged, joints, info = self._solve(target_pose, init_joints)
        position_error, rotation_error = self.pose_error(target_pose, joints, gripper)
        info.update(
            iterations=1,
            position_error_m=position_error,
            rotation_error_rad=rotation_error,
            joint_limit_margin_rad=float(np.min(self.joint_limit_margins(joints))),
        )
        return converged, joints, info

    def _solve(
        self, target_pose: FloatArray, init_joints: FloatArray
    ) -> tuple[bool, FloatArray, dict[str, object]]:
        # Legacy input is a TCP target: T_base_flange = T_base_tcp @ inv(T_flange_tcp).
        # Remove only the tool offset; keep the arm base frame without RoboGUI scene transforms.
        target = np.asarray(target_pose, dtype=np.float64)
        flange = target if self._tool_inverse is None else target @ self._tool_inverse
        return super()._solve(flange, init_joints)
