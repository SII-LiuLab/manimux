"""Compose flange kinematics and tool geometry without accessing hardware."""

from collections.abc import Mapping

import numpy as np

from manimux.kinematics.base import (
    FlangeKinematicsBase,
    FloatArray,
    IKResult,
    KinematicCoordinate,
    ManipulatorKinematicsBase,
    rigid_transform,
)
from manimux.kinematics.tool import ToolGeometryBase


class ComposedManipulatorKinematics(ManipulatorKinematicsBase):
    """An arm plus a mounted tool, ordered as [arm coordinates, tool coordinates].

    ``mount`` is T_flange_tool_base, in metres. All poses stay in the arm
    base frame; scene placement is deliberately excluded. Arm coordinate
    metadata must match the flange solver's native order and radian/metre units.
    The assembly owner must use this same layout and mount for robot state and
    commands. Tool calibration and limits belong to the supplied geometry model.

    IK supports free arm joints and explicitly fixed tool coordinates only.
    Arm limits are enforced by the flange solver. Successful solutions are also
    checked at the TCP with Euclidean position tolerance (metres) and rotation
    angle tolerance (radians). Backend exceptions propagate unchanged.
    """

    def __init__(
        self,
        arm: FlangeKinematicsBase,
        tool: ToolGeometryBase,
        *,
        arm_coordinates: tuple[KinematicCoordinate, ...],
        mount: FloatArray,
        base_frame: str,
        position_tolerance: float | None = 1e-5,
        rotation_tolerance: float = 1e-3,
    ) -> None:
        arm_layout = tuple(arm_coordinates)
        tool_layout = tuple(tool.coordinates)
        layout = arm_layout + tool_layout
        if not arm_layout or len(arm_layout) != arm.num_arm_joints:
            raise ValueError("arm coordinates must match the flange solver dimension")
        if any(item.unit not in {"rad", "m"} for item in arm_layout):
            raise ValueError("arm coordinates must use radians or metres")
        if len({item.name for item in layout}) != len(layout):
            raise ValueError("arm and tool coordinate names must be unique")
        for tolerance in (position_tolerance, rotation_tolerance):
            if tolerance is not None and (not np.isfinite(tolerance) or tolerance <= 0):
                raise ValueError("pose tolerances must be finite and positive")
        self._arm = arm
        self._tool = tool
        self._arm_size = len(arm_layout)
        self._coordinates = layout
        self._mount = rigid_transform(mount, "mount")
        self._base_frame = base_frame
        self._tcp_frame = tool.tcp_frame
        self._position_tolerance = position_tolerance
        self._rotation_tolerance = rotation_tolerance
        self._coordinate_indices = {item.name: index for index, item in enumerate(layout)}
        self._tool_names = frozenset(item.name for item in tool_layout)
        self._offset_inverse_cache: tuple[bytes, FloatArray] | None = None

    @property
    def coordinates(self) -> tuple[KinematicCoordinate, ...]:
        return self._coordinates

    @property
    def base_frame(self) -> str:
        return self._base_frame

    @property
    def tcp_frame(self) -> str:
        return self._tcp_frame

    def _configuration(self, value: FloatArray) -> FloatArray:
        state = np.array(value, dtype=np.float64, copy=True)
        if state.shape != (self.num_coordinates,) or not np.isfinite(state).all():
            raise ValueError(
                f"configuration must be a finite vector of length {self.num_coordinates}"
            )
        for coordinate, entry in zip(self.coordinates, state, strict=True):
            if coordinate.unit == "normalized" and not 0 <= entry <= 1:
                raise ValueError(f"{coordinate.name!r} must be in [0, 1]")
        return state

    def _offset(self, tool_state: FloatArray) -> FloatArray:
        # 法兰 -> 末端安装基座 -> TCP。安装变换和末端自身几何各应用一次。
        return self._mount @ self._tool.tcp_transform(tool_state.copy())

    def fk(self, configuration: FloatArray) -> FloatArray:
        state = self._configuration(configuration)
        offset = self._offset(state[self._arm_size :])
        flange = self._arm.fk_flange(state[: self._arm_size].copy())
        # T_arm_base_tcp = T_arm_base_flange @ T_flange_tcp。
        # 各臂在 Viewer 中的场景位置不参与控制 FK。
        return flange @ offset

    @property
    def arm(self) -> FlangeKinematicsBase:
        """Official arm solver; useful for its optional differential IK support."""
        return self._arm

    def _offset_inverse(self, tool_state: FloatArray) -> FloatArray:
        """Return the inverse tool offset, reusing it while tool state is unchanged."""
        key = tool_state.tobytes()
        cached = self._offset_inverse_cache
        if cached is None or cached[0] != key:
            cached = (key, np.linalg.inv(self._offset(tool_state)))
            self._offset_inverse_cache = cached
        return cached[1]

    def flange_target(self, target_tcp: FloatArray, tool_state: FloatArray) -> FloatArray:
        """Remove only the mounted tool offset, staying in the same arm base."""
        # T_arm_base_flange = T_arm_base_tcp @ inv(T_flange_tcp)。
        # 转换后的目标交给 arm 官方 IK，避免 SDK 与组合层重复补偿工具偏移。
        return rigid_transform(target_tcp, "target_tcp") @ self._offset_inverse(tool_state)

    def ik(
        self,
        target_tcp: FloatArray,
        seed_configuration: FloatArray,
        *,
        fixed_coordinates: Mapping[str, float],
    ) -> IKResult:
        target = rigid_transform(target_tcp, "target_tcp")
        seed = self._configuration(seed_configuration)
        for name, value in fixed_coordinates.items():
            if name not in self._coordinate_indices:
                raise KeyError(f"unknown fixed coordinate {name!r}")
            scalar = np.asarray(value, dtype=np.float64)
            if scalar.shape != () or not np.isfinite(scalar):
                raise ValueError(f"fixed coordinate {name!r} must be a finite scalar")
            seed[self._coordinate_indices[name]] = float(scalar)
        seed = self._configuration(seed)
        if self._tool_names != fixed_coordinates.keys():
            raise NotImplementedError(
                "IK requires all tool coordinates fixed and all arm joints free"
            )
        tool_state = seed[self._arm_size :].copy()
        # The target is already this call's validated copy. Avoid repeating
        # that work for each differential-IK substep.
        target_flange = target @ self._offset_inverse(tool_state)
        result = self._arm.ik_flange(target_flange, seed[: self._arm_size].copy())
        if not result.converged:
            return result
        if result.joints is None or result.joints.shape != (self._arm_size,):
            raise ValueError("flange solver returned an invalid joint dimension")
        solution = np.concatenate((result.joints, tool_state))
        # A solver that already validates its original acceptance criteria must
        # not acquire stricter TCP checks merely because a tool is composed here.
        if self._position_tolerance is None:
            return IKResult(True, joints=solution)
        actual = self.fk(solution)
        position_error = np.linalg.norm(actual[:3, 3] - target[:3, 3])
        relative_rotation = target[:3, :3].T @ actual[:3, :3]
        rotation_error = np.arccos(np.clip((np.trace(relative_rotation) - 1) / 2, -1, 1))
        if position_error > self._position_tolerance or rotation_error > self._rotation_tolerance:
            return IKResult(False, reason="tcp_pose_tolerance")
        return IKResult(True, joints=solution)
