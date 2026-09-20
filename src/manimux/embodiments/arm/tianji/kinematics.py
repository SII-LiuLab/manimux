"""Tianji 手臂运动学：仅处理各臂基座到法兰的变换。

TianjiSDKKinematics 提供官方 FK/IK 接口。TianjiArmKinematics 保存旧版
ik + ik_nsp 求解流程、关节限位和分支选择；TianjiDifferentialKinematics
基于同一 DH 法兰雅可比提供有界微分 IK。
末端装配与 TCP 偏移由公共 ComposedManipulatorKinematics 处理。
输入关节为弧度、位姿为米，SDK 内部的度/毫米换算在本文件完成。
"""

from __future__ import annotations

import contextlib
import io
import math
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy import sparse
from scipy.spatial.transform import Rotation

from manimux.kinematics.base import ArmKinematicsBase, FloatArray, IKResult

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


class TianjiArmKinematics(ArmKinematicsBase):
    """单臂法兰求解及关节约束；不读取末端配置，不保存 TCP 偏移。"""

    def __init__(
        self,
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
        if arm not in ARM_TYPES:
            raise ValueError(f"arm must be one of {sorted(ARM_TYPES)}, got {arm!r}")
        rows = [tuple(float(value) for value in row) for row in dh_table]
        if len(rows) != NUM_ARM_JOINTS + 1 or any(len(row) != 4 for row in rows):
            raise ValueError("dh_table must have 8 rows of [alpha_deg, a_mm, d_mm, theta0_deg]")
        self._dh = rows
        self._flange_offset = _link_transform(*rows[NUM_ARM_JOINTS])
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
    def num_joints(self) -> int:
        return NUM_ARM_JOINTS

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

    def fk(self, joints: FloatArray) -> FloatArray:
        """Expose the DH flange model through the shared arm contract."""
        return self.flange(joints)

    def flange_and_jacobian(self, joints: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Flange pose (m) and spatial Jacobian (m/rad, rad/rad), without the SDK.

        Modified DH uses each joint's *post-link* z axis. The static flange
        offset contributes to the lever arm; the mounted tool does not.
        """
        q = np.asarray(joints, dtype=np.float64).reshape(-1)
        if q.size != NUM_ARM_JOINTS or not np.isfinite(q).all():
            raise ValueError("expected seven finite radian joint angles")
        transform = np.eye(4)
        frames = []
        for (alpha, a, d, theta0), angle in zip(self._dh[:7], np.degrees(q), strict=True):
            transform = transform @ _link_transform(alpha, a, d, theta0 + angle)
            frames.append(transform)
        flange = transform @ self._flange_offset
        jacobian = np.empty((6, 7))
        for index, frame in enumerate(frames):
            z = frame[:3, 2]
            rx, ry, rz = flange[:3, 3] - frame[:3, 3]
            jacobian[:3, index] = (
                z[1] * rz - z[2] * ry,
                z[2] * rx - z[0] * rz,
                z[0] * ry - z[1] * rx,
            )
            jacobian[3:, index] = z
        return flange, jacobian

    @property
    def limit_margin_deg(self) -> float:
        """Configured joint-limit buffer, shared by analytic and differential IK."""
        return self._limit_margin_deg

    # ---------- inverse ----------

    def _load_sdk(self) -> tuple[Any, Any]:
        if self._sdk is None:
            from importlib import import_module
            from pathlib import Path

            fx_kine = import_module("manimux.embodiments.arm.tianji.sdk.marvin.fx_kine")
            config_path = (
                Path(self._sdk_root) / "ccs_m6_40.MvKDCfg"
                if self._sdk_root is not None
                else Path(fx_kine.__file__).parent / "ccs_m6_40.MvKDCfg"
            )
            with contextlib.redirect_stdout(io.StringIO()):
                kine = fx_kine.Marvin_Kine()
            kine.log_switch(0)
            index = self._arm_type
            config = kine.load_config(arm_type=index, config_path=str(config_path))
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
        # 保留旧版验解次序和度量：FK 回代、限位余量、J6/J7 干涉、种子跳变。
        # 旋转误差仍取环绕后的 XYZ 欧拉角最大差，不改成旋转向量误差。
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
        # 输入已经是该臂基座下的法兰目标，组合层已去除末端偏移。
        # 此处只做 SDK 单位换算：米 -> 毫米、弧度 -> 度。
        flange_mm = target.copy()
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
            # 最近解跳变时，按旧逻辑从 SDK 其它分支中选满足约束且步长最小的解。
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
        target_flange: FloatArray,
        seed_joints: FloatArray,
        *,
        duration_s: float | None = None,
    ) -> IKResult:
        """Expose the Tianji solver through the shared arm contract."""
        del duration_s
        ok, joints, info = self._solve(target_flange, seed_joints)
        return IKResult(True, joints) if ok else IKResult(False, reason=info["reason"])

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


# 整机通过下方统一的 fk / ik 接口调用官方 SDK。
# 上方保留旧版 IK 数值算法及微分 IK 所需的 DH 雅可比，不负责末端装配。
DEFAULT_CONFIG = Path(__file__).parent / "sdk" / "marvin" / "ccs_m6_40.MvKDCfg"
_SDK_LOCK = threading.Lock()


def _joints(values: FloatArray, label: str) -> FloatArray:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.shape != (7,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must contain seven finite joint angles in radians")
    return array


def _pose(values: FloatArray, label: str) -> FloatArray:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.shape != (4, 4) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite 4x4 transform")
    rotation = array[:3, :3]
    if (
        not np.allclose(array[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0)
    ):
        raise ValueError(f"{label} must be a rigid, right-handed homogeneous transform")
    return array


class TianjiSDKKinematics(TianjiArmKinematics):
    """Official flange FK plus the existing Tianji IK algorithm, unchanged.

    Poses are in this arm's base, metres; joints are J1..J7, radians. The
    inherited _solve keeps ik + ik_nsp, FK back-substitution, installation joint
    limits/margin, measured J6/J7 interference, and seed/alternate-branch checks.
    It receives a bare flange target: Robot applies the end-effector offset once.
    """

    def __init__(
        self,
        arm="left",
        *,
        config_path=DEFAULT_CONFIG,
        joint_limits_deg=None,
        position_tolerance_m=IK_POS_TOL_MM / 1000,
        orientation_tolerance_rad=IK_ROT_TOL_DEG * np.pi / 180,
        max_step_deg=IK_MAX_STEP_DEG,
        limit_margin_deg=LIMIT_MARGIN_DEG,
    ):
        # Reuse the original solver rather than reimplement its numerical checks.
        # Its rotation tolerance is the maximum wrapped XYZ Euler difference.
        super().__init__(
            arm=arm,
            joint_limits_deg=joint_limits_deg,
            pos_tol_mm=position_tolerance_m * 1000,
            rot_tol_deg=np.degrees(orientation_tolerance_rad),
            max_step_deg=max_step_deg,
            limit_margin_deg=limit_margin_deg,
        )
        self.arm = arm
        self.config_path = Path(config_path).expanduser().resolve()
        self._lock = _SDK_LOCK
        from .sdk.marvin import fx_kine

        self._binding = fx_kine
        with self._lock:
            self._kine = fx_kine.Marvin_Kine()
            self._kine.log_switch(0)
            config = self._kine.load_config(self._arm_type, str(self.config_path))
            if not config:
                raise RuntimeError(f"Marvin could not load {self.config_path}")
            self._robot_type = config["TYPE"][self._arm_type]
            self._native_dh = config["DH"][self._arm_type]
            self._pnva = config["PNVA"][self._arm_type]
            self._activate()

    def _activate(self):
        # Native model/tool slots are process-global. Restore them inside the
        # shared lock before every call, including independent decoding models.
        if not self._kine.initial_kine(
            self._robot_type, self._native_dh, self._pnva, [list(row) for row in BD67_REAL]
        ):
            raise RuntimeError("Marvin kinematics initialization failed")
        if not self._kine.remove_tool_kine():
            raise RuntimeError("Marvin could not clear the flange model's tool offset")

    def _load_sdk(self):
        # Called by inherited _solve while holding self._lock.
        self._activate()
        return self._binding, self._kine

    def fk(self, joints: FloatArray) -> FloatArray:
        values = _joints(joints, "joints")
        with self._lock:
            self._activate()
            result = self._kine.fk(np.degrees(values).tolist())
            if result is False or result is None:
                raise RuntimeError("Marvin FK failed")
            pose = _pose(result, "Marvin FK result")
            pose[:3, 3] *= 1e-3  # SDK millimetres -> ManiMux metres.
            return pose

    def _validate(self, kine, q, target_xyzabc, reference):
        # Invalid native output must never become a command. All numerical
        # acceptance criteria below are inherited from the original solver.
        if not np.isfinite(q).all():
            raise RuntimeError("Marvin IK returned invalid joint positions")
        return super()._validate(kine, q, target_xyzabc, reference)

    def ik(
        self,
        target_flange: FloatArray,
        seed_joints: FloatArray,
        *,
        duration_s: float | None = None,
    ) -> IKResult:
        del duration_s
        target = _pose(target_flange, "target_flange")
        seed = _joints(seed_joints, "seed_joints")
        ok, joints, info = self._solve(target, seed)
        # The old helper also returns a seed on failure; never expose it as a solution.
        return IKResult(True, joints) if ok else IKResult(False, reason=info["reason"])


# Differential IK, ported from tianji-control's algos/diff_ik.py, uses the
# same public arm contract as the analytic solvers. Public poses/joints use
# metres/radians; its QP retains millimetres/degrees to match the reference.
# It stays here because both implementations share Tianji's DH model, limits,
# interference geometry and flange conventions.
class DifferentialIKConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    # Bound from the effective shared executor profile; never a second rate constant.
    max_velocity_rad_s: float = Field(gt=0)
    dt_max_s: float = Field(gt=0)
    w_pos: float = Field(default=1.0, gt=0)
    w_rot: float = Field(default=1.0, gt=0)
    lam: float = Field(default=0.001, ge=0)
    limit_margin_deg: float | None = Field(default=None, gt=0)
    check_j67: bool = True
    j67_margin_deg: float | None = Field(default=None, gt=0)
    mu_nullspace: float = Field(default=1000.0, ge=0)
    nullspace_activation_deg: float | None = Field(default=25.0, gt=0)
    nullspace_weights: list[float] | None = None
    max_lag_mm: float = Field(default=5.0, gt=0)
    max_lag_deg: float | None = Field(default=None, gt=0)
    # abort turns a solved step whose target residual exceeds max_lag_* into a
    # tracking_lag failure; report keeps the bounded step and only flags it, as
    # CalibWrist real_run's diff_lag_policy. Backend failures reject in both.
    lag_policy: Literal["abort", "report"] = "abort"

    @model_validator(mode="after")
    def validate_weights(self):
        if self.nullspace_weights is not None and (
            len(self.nullspace_weights) != 7
            or any(not math.isfinite(value) or value < 0 for value in self.nullspace_weights)
        ):
            raise ValueError("nullspace_weights must contain seven finite nonnegative values")
        return self


def _upper_csc(matrix):
    rows, cols = np.triu_indices(7)
    return sparse.csc_matrix((matrix[rows, cols], (rows, cols)), shape=(7, 7))


def _constraint_matrix(c6, c7):
    return sparse.csc_matrix(
        ([1.0] * 7 + [c6, c7], (list(range(7)) + [7, 7], list(range(7)) + [5, 6])),
        shape=(8, 7),
    )


def rotation_vector(matrix):
    """3x3 rotation -> rotation vector (rad): scipy's Shepperd quaternion route in plain math.

    A per-step scipy Rotation costs about 20 us; this agrees with it to ~1e-13 deg.
    """
    (m00, m01, m02), (m10, m11, m12), (m20, m21, m22) = np.asarray(matrix, dtype=float).tolist()
    trace = m00 + m11 + m22
    largest = max(trace, m00, m11, m22)
    if largest == trace:
        w, x, y, z = 1 + trace, m21 - m12, m02 - m20, m10 - m01
    elif largest == m00:
        x, y, z, w = 1 + 2 * m00 - trace, m01 + m10, m02 + m20, m21 - m12
    elif largest == m11:
        y, x, z, w = 1 + 2 * m11 - trace, m01 + m10, m12 + m21, m02 - m20
    else:
        z, x, y, w = 1 + 2 * m22 - trace, m02 + m20, m12 + m21, m10 - m01
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if w < 0:
        norm = -norm
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    angle = 2 * math.atan2(math.sqrt(x * x + y * y + z * z), w)
    if angle > 1e-3:
        scale = angle / math.sin(angle / 2)
    else:
        scale = 2 + angle * angle / 12 + 7 * angle**4 / 2880
    return np.array([scale * x, scale * y, scale * z])


def rotation_matrix(vector):
    """Rotation vector (rad) -> 3x3 rotation, through the quaternion as scipy does."""
    x, y, z = (float(value) for value in vector)
    angle = math.sqrt(x * x + y * y + z * z)
    if angle > 1e-3:
        scale = math.sin(angle / 2) / angle
    else:
        scale = 0.5 - angle * angle / 48 + angle**4 / 3840
    w, x, y, z = math.cos(angle / 2), scale * x, scale * y, scale * z
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def euler_xyz_deg(matrix):
    """Extrinsic xyz Euler angles (deg), as scipy's ``as_euler("xyz")``.

    Near gimbal lock scipy picks a particular split between the first and third
    angle; defer to it there so the residual keeps its original definition.
    """
    m = np.asarray(matrix, dtype=float)
    if 1.0 - abs(m[2, 0]) < 1e-6:
        return Rotation.from_matrix(m).as_euler("xyz", degrees=True)
    return np.array(
        [
            math.degrees(math.atan2(m[2, 1], m[2, 2])),
            math.degrees(math.asin(-m[2, 0])),
            math.degrees(math.atan2(m[1, 0], m[0, 0])),
        ]
    )


def _interference_row(joints, margin, dt):
    j6, j7 = joints[5:7]
    quadrant = 0 if j6 >= 0 and j7 >= 0 else 1 if j6 < 0 <= j7 else 2 if j6 < 0 else 3
    a0, a1, a2 = BD67_REAL[quadrant]
    boundary = a0 * j6 * j6 + a1 * j6 + a2
    slope = 2.0 * a0 * j6 + a1
    if j7 >= 0:
        gap, c6, c7 = j7 - boundary, -slope, 1.0
    else:
        gap, c6, c7 = boundary - j7, slope, -1.0
    return c6, c7, (-margin - gap) / dt


class TianjiDifferentialKinematics(ArmKinematicsBase):
    """One solver per arm. Calls are sequential; ``reset`` drops all QP state.

    An entire speculative chunk starts with reset, so OSQP primal/dual state
    from a rejected chunk or previous engagement cannot affect the next one.
    Within a chunk the reference warm-start/update sequence is preserved.
    """

    def __init__(self, kinematics: TianjiArmKinematics, config: DifferentialIKConfig):
        try:
            import osqp
        except ImportError as exc:
            raise ImportError("Differential IK needs pip install -e '.[tianji-diff-ik]'") from exc
        self._kinematics = kinematics
        self._osqp = osqp
        self.config = config
        lower, upper = kinematics.joint_position_limits()
        self.lower, self.upper = np.degrees(lower), np.degrees(upper)
        self.margin = (
            kinematics.limit_margin_deg
            if config.limit_margin_deg is None
            else config.limit_margin_deg
        )
        if self.margin < kinematics.limit_margin_deg:
            raise ValueError("Differential IK cannot reduce the embodiment joint-limit margin")
        if np.any(self.lower + self.margin >= self.upper - self.margin):
            raise ValueError("Differential IK margin leaves no joint range")
        self.j67_margin = self.margin if config.j67_margin_deg is None else config.j67_margin_deg
        self.vmax = math.degrees(config.max_velocity_rad_s)
        self.weights = [1.0] * 7 if config.nullspace_weights is None else config.nullspace_weights
        self.W = np.diag([config.w_pos] * 3 + [config.w_rot] * 3)
        self._p_rows = np.concatenate([np.arange(j + 1) for j in range(7)])
        self._p_cols = np.concatenate([np.full(j + 1, j) for j in range(7)])
        probe = np.arange(49, dtype=float).reshape(7, 7)
        if not np.array_equal(_upper_csc(probe).data, probe[self._p_rows, self._p_cols]):
            raise RuntimeError("Unexpected sparse upper-triangle layout")
        marker = _constraint_matrix(-2.0, -3.0)
        self._a_data = marker.data.copy()
        self._a_c6 = int(np.flatnonzero(marker.data == -2.0)[0])
        self._a_c7 = int(np.flatnonzero(marker.data == -3.0)[0])
        self.reset()

    @property
    def kinematics(self) -> TianjiArmKinematics:
        return self._kinematics

    @property
    def num_joints(self) -> int:
        return self.kinematics.num_joints

    @property
    def max_step_duration_s(self) -> float:
        return self.config.dt_max_s

    def fk(self, joints: FloatArray) -> FloatArray:
        return self.kinematics.fk(joints)

    # np.allclose(R.T @ R, I, atol=1e-7) with its default rtol, as a precomputed bound.
    _ORTHONORMAL_TOL = 1e-7 + 1e-5 * np.eye(3)

    def reset(self) -> None:
        self._problem = None
        self._previous_velocity = np.zeros(7)
        # Flange pose and Jacobian at the joints the last step produced. A chunk
        # seeds its next step with exactly those joints, so they are reused once.
        self._next_frame = None

    def _cost(self, joints):
        band = self.config.nullspace_activation_deg
        if band is None:
            mid, span = (self.lower + self.upper) / 2, self.upper - self.lower
            return (
                sum(
                    weight * ((value - center) / width) ** 2
                    for value, center, width, weight in zip(
                        joints, mid, span, self.weights, strict=True
                    )
                )
                / 14.0
            )
        margins = np.minimum(self.upper - joints, joints - self.lower)
        return (
            sum(
                weight * ((band - margin) / band) ** 2
                for margin, weight in zip(margins, self.weights, strict=True)
                if margin < band
            )
            / 14.0
        )

    def _nullspace_gradient(self, joints):
        band = self.config.nullspace_activation_deg
        if (
            band is not None
            and np.min(np.minimum(self.upper - joints, joints - self.lower)) >= band
        ):
            return np.zeros(7)
        gradient = np.zeros(7)
        for index in range(7):
            plus, minus = joints.copy(), joints.copy()
            plus[index] += 1e-4
            minus[index] -= 1e-4
            gradient[index] = (self._cost(plus) - self._cost(minus)) / 2e-4
        return gradient

    def ik(
        self,
        target_flange: FloatArray,
        previous_joints: FloatArray,
        *,
        duration_s: float | None = None,
    ) -> IKResult:
        start = time.perf_counter()

        def failure(reason, **detail):
            return IKResult(
                False,
                reason=reason,
                diagnostics={
                    "solve_time_ms": (time.perf_counter() - start) * 1000,
                    "detail": detail,
                },
                target_reached=False,
            )

        previous = np.asarray(previous_joints, dtype=float)
        target = np.asarray(target_flange, dtype=float)
        if (
            previous.shape != (7,)
            or not np.isfinite(previous).all()
            or target.shape != (4, 4)
            or not np.isfinite(target).all()
            or duration_s is None
            or not math.isfinite(duration_s)
            or duration_s <= 0
        ):
            return failure("invalid_input")
        # np.allclose/np.isclose with their default rtol=1e-5, written out: the same
        # accept/reject decisions without their per-call overhead.
        bottom, rotation = target[3], target[:3, :3]
        (m00, m01, m02), (m10, m11, m12), (m20, m21, m22) = rotation.tolist()
        determinant = (
            m00 * (m11 * m22 - m12 * m21)
            - m01 * (m10 * m22 - m12 * m20)
            + m02 * (m10 * m21 - m11 * m20)
        )
        if not (
            abs(bottom[0]) <= 1e-8
            and abs(bottom[1]) <= 1e-8
            and abs(bottom[2]) <= 1e-8
            and abs(bottom[3] - 1.0) <= 1e-8 + 1e-5
        ) or not (
            np.all(np.abs(rotation.T @ rotation - np.eye(3)) <= self._ORTHONORMAL_TOL)
            and abs(determinant - 1.0) <= 1e-7 + 1e-5
        ):
            return failure("invalid_input", note="target must be a rigid transform")
        dt = min(max(duration_s, 1e-6), self.config.dt_max_s)
        joints = np.degrees(previous)
        cached = self._next_frame
        if cached is not None and np.array_equal(cached[0], previous):
            current, jacobian = cached[1], cached[2]
        else:
            current, jacobian = self.kinematics.flange_and_jacobian(previous)
        jacobian = jacobian.copy()
        jacobian[:3] *= 1000 * math.pi / 180  # m/rad -> mm/degree
        rotation_error = np.degrees(rotation_vector(target[:3, :3] @ current[:3, :3].T))
        desired = np.empty(6)
        desired[:3] = (target[:3, 3] - current[:3, 3]) * 1000
        desired[3:] = rotation_error
        desired /= dt
        quadratic = 2 * (jacobian.T @ self.W @ jacobian + self.config.lam * np.eye(7))
        linear = -2 * (jacobian.T @ self.W @ desired)
        if self.config.mu_nullspace > 0:
            linear += self.config.mu_nullspace * dt * self._nullspace_gradient(joints)
        lower = np.maximum(-self.vmax, (self.lower + self.margin - joints) / dt)
        upper = np.minimum(self.vmax, (self.upper - self.margin - joints) / dt)
        bad = np.flatnonzero(lower > upper)
        if bad.size:
            # OSQP update(l>u) otherwise silently retains the previous problem.
            return failure("qp_infeasible", joint=int(bad[0]), note="empty qdot box")
        if self.config.check_j67:
            c6, c7, bound = _interference_row(joints, self.j67_margin, dt)
            self._a_data[self._a_c6], self._a_data[self._a_c7] = c6, c7
            full_lower, full_upper = np.empty(8), np.empty(8)
            full_lower[:7], full_lower[7] = lower, -np.inf
            full_upper[:7], full_upper[7] = upper, bound
        else:
            full_lower, full_upper = lower, upper
        if self._problem is None:
            self._problem = self._osqp.OSQP()
            constraints = (
                _constraint_matrix(c6, c7) if self.config.check_j67 else sparse.eye(7, format="csc")
            )
            self._problem.setup(
                P=_upper_csc(quadratic),
                q=linear,
                A=constraints,
                l=full_lower,
                u=full_upper,
                verbose=False,
                polishing=False,
                warm_starting=True,
            )
        else:
            updates = {"Ax": self._a_data} if self.config.check_j67 else {}
            self._problem.update(
                Px=quadratic[self._p_rows, self._p_cols],
                q=linear,
                l=full_lower,
                u=full_upper,
                **updates,
            )
        self._problem.warm_start(x=self._previous_velocity)
        solved = self._problem.solve(raise_error=False)
        if solved.info.status_val not in (1, 2):
            return failure("qp_infeasible", osqp_status=solved.info.status)
        if solved.x is None or not np.isfinite(solved.x).all():
            return failure("fk_mismatch", note="non-finite QP velocity")
        velocity = np.clip(solved.x, lower, upper)
        self._previous_velocity = velocity.copy()
        next_deg = joints + velocity * dt
        next_rad = np.radians(next_deg)
        # flange_and_jacobian's pose is bit-identical to flange(); keep its Jacobian
        # for the next step, which starts from exactly these joints.
        actual, next_jacobian = self.kinematics.flange_and_jacobian(next_rad)
        self._next_frame = (next_rad, actual, next_jacobian)
        pos_error = float(np.linalg.norm((actual[:3, 3] - target[:3, 3]) * 1000))
        actual_euler = euler_xyz_deg(actual[:3, :3])
        target_euler = euler_xyz_deg(target[:3, :3])
        rot_error = float(np.max(np.abs((actual_euler - target_euler + 180) % 360 - 180)))
        margin = float(np.min(np.minimum(self.upper - next_deg, next_deg - self.lower)))
        lag_exceeded = pos_error > self.config.max_lag_mm or (
            self.config.max_lag_deg is not None and rot_error > self.config.max_lag_deg
        )
        reason = "ok"
        if margin < self.margin:
            reason = "joint_limit"
        elif self.config.check_j67 and not j67_ok(next_deg):
            reason = "j67_interference"
        elif lag_exceeded and self.config.lag_policy == "abort":
            reason = "tracking_lag"
        diagnostics = {
            "qdot_rad_s": np.radians(velocity),
            "pos_err_mm": pos_error,
            "rot_err_deg": rot_error,
            "max_step_deg": float(np.max(np.abs(next_deg - joints))),
            "min_margin_deg": margin,
            "solve_time_ms": (time.perf_counter() - start) * 1000,
            "detail": {},
            "lag_exceeded": lag_exceeded,
        }
        if reason != "ok":
            return IKResult(
                False,
                reason=reason,
                diagnostics=diagnostics,
                target_reached=False,
            )
        return IKResult(
            True,
            joints=next_rad,
            reason=reason,
            diagnostics=diagnostics,
            target_reached=False,
        )
