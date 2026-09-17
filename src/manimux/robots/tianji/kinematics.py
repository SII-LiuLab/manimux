"""Official Marvin FK/IK adapted to the flange-only ManiMux contract.

No hardware connection, tool transform, legacy driver or legacy solver is used.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Literal

import numpy as np

from manimux.kinematics.base import FlangeKinematicsBase, FloatArray, IKResult

DEFAULT_CONFIG = Path(__file__).parent / "vendor" / "marvin" / "ccs_m6_40.MvKDCfg"

# libKine stores models in global slots indexed by robot_tag, not by Python
# object. Serialize model activation AND calculation across all our instances.
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


class TianjiSDKKinematics(FlangeKinematicsBase):
    """Seven-joint Marvin arm, using the SDK's standard seeded analytical IK.

    ``left`` selects SDK arm 0 (A), ``right`` selects arm 1 (B). Public joint
    vectors are J1..J7 in radians; poses are T_base_flange with translation in
    metres, in that arm's own base frame. The vendor uses degrees and millimetres.

    The supplied M6 4.0 table provides DH, position limits and J6/J7 interference
    parameters. There are no station-specific limit overrides or step clamps.
    IK uses ZSP type 0 and the SDK-selected solution, without invoking ik_nsp or
    selecting alternate branches. Success additionally requires valid SDK flags,
    position limits and FK residuals within the configured tolerances.

    Construction loads only the kinematics library. Every operation restores its
    model and removes any SDK tool offset under a shared lock, so instances with
    different configurations cannot silently replace one another's model. Direct
    concurrent access to libKine outside this wrapper is not synchronized.
    """

    def __init__(
        self,
        arm: Literal["left", "right"] = "left",
        *,
        config_path: Path | str = DEFAULT_CONFIG,
        position_tolerance_m: float = 1e-5,
        orientation_tolerance_rad: float = 1e-3,
    ) -> None:
        if arm not in {"left", "right"}:
            raise ValueError("arm must be 'left' or 'right'")
        self.arm = arm
        self._index = 0 if arm == "left" else 1
        self.config_path = Path(config_path).expanduser().resolve()
        if not self.config_path.is_file():
            raise FileNotFoundError(self.config_path)
        for name, value in (
            ("position_tolerance_m", position_tolerance_m),
            ("orientation_tolerance_rad", orientation_tolerance_rad),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        self._position_tolerance = float(position_tolerance_m)
        self._orientation_tolerance = float(orientation_tolerance_rad)

        # Importing the package does not load a native library or control SDK.
        from .vendor.marvin import fx_kine

        self._sdk = fx_kine
        with _SDK_LOCK:
            self._kine = fx_kine.Marvin_Kine()
            self._kine.log_switch(0)
            config = self._kine.load_config(self._index, str(self.config_path))
            if config is None:
                raise RuntimeError(f"Marvin could not load kinematics config: {self.config_path}")
            self._robot_type = config["TYPE"][self._index]
            self._dh = config["DH"][self._index]
            self._pnva = config["PNVA"][self._index]
            self._j67 = config["BD"][self._index]
            limits = np.asarray(self._pnva, dtype=np.float64)
            if (
                limits.shape != (7, 4)
                or not np.isfinite(limits).all()
                or np.any(limits[:, 1] >= limits[:, 0])
            ):
                raise RuntimeError("Marvin config contains invalid joint limits")
            self._lower = np.radians(limits[:, 1])
            self._upper = np.radians(limits[:, 0])
            self._activate()

    @property
    def num_arm_joints(self) -> int:
        return 7

    def joint_position_limits(self) -> tuple[FloatArray, FloatArray]:
        """SDK-table position limits in radians, returned as independent arrays."""
        return self._lower.copy(), self._upper.copy()

    def _activate(self) -> None:
        """Restore the instance model in its shared SDK slot; caller holds lock."""
        if not self._kine.initial_kine(self._robot_type, self._dh, self._pnva, self._j67):
            raise RuntimeError("Marvin kinematics initialization failed")
        if not self._kine.remove_tool_kine():
            raise RuntimeError("Marvin could not clear the tool offset for flange kinematics")

    def _fk(self, joints: FloatArray) -> FloatArray:
        """Calculate with the active model; caller holds lock."""
        result = self._kine.fk(np.degrees(joints).tolist())
        if result is False or result is None:
            raise RuntimeError("Marvin FK failed")
        try:
            pose = _pose(result, "Marvin FK result")
        except ValueError as exc:
            raise RuntimeError("Marvin FK returned an invalid transform") from exc
        pose[:3, 3] *= 1e-3
        return pose

    def fk_flange(self, joints: FloatArray) -> FloatArray:
        """Return the bare flange pose in metres for J1..J7 in radians."""
        values = _joints(joints, "joints")
        with _SDK_LOCK:
            self._activate()
            return self._fk(values)

    def ik_flange(self, target_flange: FloatArray, seed_joints: FloatArray) -> IKResult:
        """Solve a flange target; no fallback joints are returned on rejection.

        Reasons: ``unreachable``, ``singular``, ``joint_limit``, ``no_solution``
        or ``fk_mismatch``. Backend initialization/FK errors raise RuntimeError.
        Invalid input raises ValueError. Success does not imply a collision-free
        path, tracking feasibility or a bounded step from the seed.
        """
        target = _pose(target_flange, "target_flange")
        seed = _joints(seed_joints, "seed_joints")
        target_mm = target.copy()
        target_mm[:3, 3] *= 1e3
        with _SDK_LOCK:
            self._activate()
            params = self._sdk.FX_InvKineSolvePara()
            params.set_input_ik_target_tcp(target_mm.ravel().tolist())
            params.set_input_ik_ref_joint(np.degrees(seed).tolist())
            params.set_input_ik_zsp_type(0)
            solved = self._kine.ik(params)
            if params.m_Output_IsOutRange:
                return IKResult(False, reason="unreachable")
            if any(params.m_Output_IsDeg):
                return IKResult(False, reason="singular")
            if params.m_Output_IsJntExd or any(params.m_Output_JntExdTags):
                return IKResult(False, reason="joint_limit")
            if not solved:
                return IKResult(False, reason="no_solution")
            solution = np.radians(params.m_Output_RetJoint.to_list())
            if solution.shape != (7,) or not np.isfinite(solution).all():
                raise RuntimeError("Marvin IK returned invalid joint positions")
            if np.any(solution < self._lower - 1e-10) or np.any(solution > self._upper + 1e-10):
                return IKResult(False, reason="joint_limit")
            actual = self._fk(solution)
            position_error = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
            relative = actual[:3, :3].T @ target[:3, :3]
            angle_error = float(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
            if (
                position_error > self._position_tolerance
                or angle_error > self._orientation_tolerance
            ):
                return IKResult(False, reason="fk_mismatch")
            return IKResult(True, solution)
