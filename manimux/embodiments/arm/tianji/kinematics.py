"""Tianji 手臂运动学：仅处理各臂基座到法兰的变换。

TianjiSDKKinematics 提供官方 FK/IK 接口。TianjiArmKinematics 保存旧版
ik + ik_nsp 求解流程、关节限位和分支选择；DH 法兰雅可比供可选微分 IK 使用。
末端装配与 TCP 偏移由公共 ComposedManipulatorKinematics 处理。
输入关节为弧度、位姿为米，SDK 内部的度/毫米换算在本文件完成。
"""

from __future__ import annotations

import contextlib
import io
import math
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from manimux.kinematics.base import FlangeKinematicsBase, FloatArray, IKResult

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


class TianjiArmKinematics:
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
    def num_arm_joints(self) -> int:
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


# 整机通过下方统一的 fk_flange / ik_flange 接口调用官方 SDK。
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


class TianjiSDKKinematics(TianjiArmKinematics, FlangeKinematicsBase):
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

    def fk_flange(self, joints: FloatArray) -> FloatArray:
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

    def ik_flange(self, target_flange: FloatArray, seed_joints: FloatArray) -> IKResult:
        target = _pose(target_flange, "target_flange")
        seed = _joints(seed_joints, "seed_joints")
        ok, joints, info = self._solve(target, seed)
        # The old helper also returns a seed on failure; never expose it as a solution.
        return IKResult(True, joints) if ok else IKResult(False, reason=info["reason"])
