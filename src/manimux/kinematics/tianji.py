"""Tianji Marvin M6 arm kinematics.

Forward kinematics is the controller's own modified-DH chain, copied from the
vendor's ``CommonConfig/ccs_m6_40.MvKDCfg``. It reproduces the SDK's
``Marvin_Kine.fk()`` to numerical precision without loading the closed-source
library, so the viewer and pose-space adapters can use it in any environment.
Both arms share one DH table and one set of joint limits.

Inverse kinematics is tianji-control's ``ArmIK.solve``: the vendor's closed-form
``ik`` + ``ik_nsp`` seeded by the reference joints, then validated by FK
back-substitution, joint-limit margin, J6/J7 interference and a branch-jump
check against the seed, falling back to the most continuous of the SDK's other
solutions. The vendor library is loaded on the first IK call.

ManiMux-facing units follow the rest of the runtime: joints in radians, poses in
metres. The vendor SDK works in degrees and millimetres; conversion happens here
and nowhere else.

The tool frame comes from a mounted end effector (see
:mod:`manimux.kinematics.end_effector`) or, for a bare tool offset, from the
controller's ``xyzabc`` form (mm, degrees, extrinsic x-y-z Euler angles). With
neither, :meth:`fk` returns the flange.
"""

from __future__ import annotations

import contextlib
import io
import math
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

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

# IK validation defaults, tianji-control config.py.
IK_POS_TOL_MM = 0.01
IK_ROT_TOL_DEG = 0.2
IK_MAX_STEP_DEG = 1.8  # branch-jump detector against the seed, not a velocity limit
LIMIT_MARGIN_DEG = 5.0

# J6/J7 self-interference table measured on this hardware (ik_solver.BD67_REAL).
BD67_REAL = (
    (0.0, -1.025, 110.5),
    (0.0, 1.025, 110.5),
    (0.0, -1.025, -110.5),
    (0.0, 1.025, -110.5),
)

NUM_ARM_JOINTS = 7
ARM_TYPES = {"left": 0, "right": 1}


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


def j67_ok(joints_deg: Sequence[float] | FloatArray) -> bool:
    """J6/J7 self-interference check (``ArmIK.j67_ok``)."""

    j6, j7 = float(joints_deg[5]), float(joints_deg[6])
    row = 0 if (j6 >= 0 and j7 >= 0) else 1 if (j6 < 0 and j7 >= 0) else 2 if j6 < 0 else 3
    a0, a1, a2 = BD67_REAL[row]
    limit = a0 * j6 * j6 + a1 * j6 + a2
    return j7 <= limit if j7 >= 0 else j7 >= limit


class TianjiKinematics:
    """Kinematics for one 7-DoF Marvin M6 arm, in its own base frame."""

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
        if arm not in ARM_TYPES:
            raise ValueError(f"arm must be one of {sorted(ARM_TYPES)}, got {arm!r}")
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
        self._tool_inverse = None if self._tool is None else np.linalg.inv(self._tool)

        lower = np.array([low for low, _ in JOINT_LIMITS_DEG], dtype=np.float64)
        upper = np.array([high for _, high in JOINT_LIMITS_DEG], dtype=np.float64)
        # merge_limit_override: keep whichever limit is stricter.
        for joint, bounds in (joint_limits_deg or {}).items():
            index = int(joint) - 1
            if not 0 <= index < NUM_ARM_JOINTS or len(bounds) != 2 or bounds[0] >= bounds[1]:
                raise ValueError(f"joint_limits_deg.{joint} must be [lo, hi] for joints 1-7")
            lower[index] = max(lower[index], float(bounds[0]))
            upper[index] = min(upper[index], float(bounds[1]))
        self._lower_deg, self._upper_deg = lower, upper
        self._lower, self._upper = np.radians(lower), np.radians(upper)

        self._arm_type = ARM_TYPES[arm]
        self._sdk_root = sdk_root
        self._pos_tol_mm = float(pos_tol_mm)
        self._rot_tol_deg = float(rot_tol_deg)
        self._max_step_deg = float(max_step_deg)
        self._limit_margin_deg = float(limit_margin_deg)
        self._sdk: tuple[Any, Any] | None = None
        self._lock = threading.Lock()

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

    # ---------- forward ----------

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

    # ---------- inverse ----------

    def _load_sdk(self) -> tuple[Any, Any]:
        if self._sdk is None:
            from manimux.robots.tianji import sdk

            fx_kine = sdk.load_marvin_kine(self._sdk_root)
            with contextlib.redirect_stdout(io.StringIO()):
                kine = fx_kine.Marvin_Kine()
            kine.log_switch(0)
            index = self._arm_type
            config = kine.load_config(
                arm_type=index, config_path=str(sdk.kine_config(self._sdk_root))
            )
            if not config:
                raise RuntimeError("Marvin kinematics load_config failed")
            if not kine.initial_kine(
                robot_type=config["TYPE"][index],
                dh=config["DH"][index],
                pnva=config["PNVA"][index],
                j67=[list(row) for row in BD67_REAL],
            ):
                raise RuntimeError("Marvin kinematics initial_kine failed")
            self._sdk = (fx_kine, kine)
        return self._sdk

    def _validate(
        self, kine: Any, q: list[float], target_xyzabc: list[float], reference: list[float]
    ) -> tuple[str, float]:
        back = kine.mat4x4_to_xyzabc(pose_mat=kine.fk(joints=q))
        position_error = math.dist(back[:3], target_xyzabc[:3])
        rotation_error = max(abs((back[i] - target_xyzabc[i] + 180) % 360 - 180) for i in (3, 4, 5))
        if position_error > self._pos_tol_mm or rotation_error > self._rot_tol_deg:
            return "fk_mismatch", 0.0
        margins = np.minimum(self._upper_deg - q, q - self._lower_deg)
        if float(np.min(margins)) < self._limit_margin_deg:
            return "joint_limit", 0.0
        if not j67_ok(q):
            return "j67_interference", 0.0
        step = max(abs(a - b) for a, b in zip(q, reference, strict=True))
        if step > self._max_step_deg:
            return "branch_jump", step
        return "converged", step

    def _solve(
        self, target_pose: FloatArray, init_joints: FloatArray
    ) -> tuple[bool, FloatArray, dict[str, object]]:
        target = np.asarray(target_pose, dtype=np.float64)
        if target.shape != (4, 4) or not np.isfinite(target).all():
            raise ValueError(f"target_pose must be a finite 4x4 transform, got {target.shape}")
        seed = np.asarray(init_joints, dtype=np.float64).reshape(-1)[:NUM_ARM_JOINTS]
        if seed.size != NUM_ARM_JOINTS or not np.isfinite(seed).all():
            raise ValueError("init_joints must contain 7 finite joint angles")
        flange = target if self._tool_inverse is None else target @ self._tool_inverse
        flange_mm = flange.copy()
        flange_mm[:3, 3] *= 1e3
        reference = np.degrees(seed).tolist()
        fallback = self.clip_arm_joints(seed)
        with self._lock:
            fx_kine, kine = self._load_sdk()
            target_list = flange_mm.tolist()
            target_xyzabc = kine.mat4x4_to_xyzabc(pose_mat=target_list)
            params = fx_kine.FX_InvKineSolvePara()
            params.set_input_ik_target_tcp(kine.mat4x4_to_mat1x16(target_list))
            params.set_input_ik_ref_joint(reference)
            params.set_input_ik_zsp_type(0)
            if not kine.ik(structure_data=params):
                return False, fallback, {"reason": "ik_failed"}
            # With zsp_type 0 the null-space solve still refines the ik() result.
            params.set_input_zsp_angle(0.0)
            params.set_dgr1(0.05)
            params.set_dgr2(0.05)
            if not kine.ik_nsp(sturcture_data=params):
                return False, fallback, {"reason": "ik_nsp_failed"}
            solution = list(params.m_Output_RetJoint.to_list())
            reason, step = self._validate(kine, solution, target_xyzabc, reference)
            switched = False
            if reason == "branch_jump":
                rows = fx_kine.convert_to_8x8_matrix(params.m_OutPut_AllJoint.to_list())
                count = int(params.m_OutPut_Result_Num)
                best_step = math.inf
                for row in rows[: max(0, min(count, len(rows)))]:
                    candidate = list(row[:NUM_ARM_JOINTS])
                    distance = max(abs(a - b) for a, b in zip(candidate, reference, strict=True))
                    if distance >= best_step:
                        continue
                    candidate_reason, _ = self._validate(kine, candidate, target_xyzabc, reference)
                    if candidate_reason == "converged":
                        solution, reason, step, best_step, switched = (
                            candidate,
                            candidate_reason,
                            distance,
                            distance,
                            True,
                        )
        info: dict[str, object] = {"reason": reason, "max_step_deg": step}
        if switched:
            info["branch_switched"] = True
        if reason != "converged":
            return False, fallback, info
        return True, np.radians(solution), info

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

    # ---------- limits ----------

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
