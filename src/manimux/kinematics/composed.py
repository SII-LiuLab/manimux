"""Compose flange kinematics and tool geometry without accessing hardware."""

from collections.abc import Mapping

import numpy as np

from manimux.kinematics.base import (
    FlangeKinematicsBase,
    FloatArray,
    IKResult,
    KinematicCoordinate,
    ManipulatorKinematicsBase,
)
from manimux.kinematics.tool import ToolGeometryBase


def _pose(value: FloatArray, name: str) -> FloatArray:
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


class ComposedManipulatorKinematics(ManipulatorKinematicsBase):
    """An arm plus a mounted tool, ordered as [arm coordinates, tool coordinates].

    ``mount`` is T_flange_tool_base, in metres. ``base_frame`` names the arm's
    existing base; naming it does not apply a world transform. Arm coordinate
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
        position_tolerance: float = 1e-5,
        rotation_tolerance: float = 1e-3,
    ) -> None:
        arm_layout = tuple(arm_coordinates)
        tool_layout = tuple(tool.coordinates)
        layout = arm_layout + tool_layout
        if not arm_layout or len(arm_layout) != arm.num_arm_joints:
            raise ValueError("arm coordinates must match the flange solver dimension")
        if any(not isinstance(item, KinematicCoordinate) for item in layout):
            raise ValueError("coordinates must be KinematicCoordinate instances")
        if any(item.unit not in {"rad", "m"} for item in arm_layout):
            raise ValueError("arm coordinates must use radians or metres")
        if len({item.name for item in layout}) != len(layout):
            raise ValueError("arm and tool coordinate names must be unique")
        for frame in (base_frame, tool.base_frame, tool.tcp_frame):
            if not isinstance(frame, str) or not frame.strip():
                raise ValueError("frame names must be non-empty strings")
        for tolerance in (position_tolerance, rotation_tolerance):
            if not np.isfinite(tolerance) or tolerance <= 0:
                raise ValueError("pose tolerances must be finite and positive")
        self._arm = arm
        self._tool = tool
        self._arm_size = len(arm_layout)
        self._coordinates = layout
        self._mount = _pose(mount, "mount")
        self._base_frame = base_frame
        self._tcp_frame = tool.tcp_frame
        self._position_tolerance = position_tolerance
        self._rotation_tolerance = rotation_tolerance

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
        return self._mount @ _pose(self._tool.tcp_transform(tool_state.copy()), "tool transform")

    def fk(self, configuration: FloatArray) -> FloatArray:
        state = self._configuration(configuration)
        offset = self._offset(state[self._arm_size :])
        flange = _pose(self._arm.fk_flange(state[: self._arm_size].copy()), "flange transform")
        return flange @ offset

    def ik(
        self,
        target_tcp: FloatArray,
        seed_configuration: FloatArray,
        *,
        fixed_coordinates: Mapping[str, float],
    ) -> IKResult:
        target = _pose(target_tcp, "target_tcp")
        seed = self._configuration(seed_configuration)
        indices = {coordinate.name: index for index, coordinate in enumerate(self.coordinates)}
        unknown = set(fixed_coordinates) - indices.keys()
        if unknown:
            raise ValueError(f"unknown fixed coordinates: {unknown}")
        for name, value in fixed_coordinates.items():
            scalar = np.asarray(value, dtype=np.float64)
            if scalar.shape != () or not np.isfinite(scalar):
                raise ValueError(f"fixed coordinate {name!r} must be a finite scalar")
            seed[indices[name]] = float(scalar)
        seed = self._configuration(seed)
        tool_names = {item.name for item in self.coordinates[self._arm_size :]}
        if set(fixed_coordinates) != tool_names:
            raise NotImplementedError(
                "IK requires all tool coordinates fixed and all arm joints free"
            )
        tool_state = seed[self._arm_size :].copy()
        offset = self._offset(tool_state)
        target_flange = target @ np.linalg.inv(offset)
        result = self._arm.ik_flange(target_flange, seed[: self._arm_size].copy())
        if not result.converged:
            return result
        if result.joints is None or result.joints.shape != (self._arm_size,):
            raise ValueError("flange solver returned an invalid joint dimension")
        solution = np.concatenate((result.joints, tool_state))
        actual = self.fk(solution)
        position_error = np.linalg.norm(actual[:3, 3] - target[:3, 3])
        relative_rotation = target[:3, :3].T @ actual[:3, :3]
        rotation_error = np.arccos(np.clip((np.trace(relative_rotation) - 1) / 2, -1, 1))
        if position_error > self._position_tolerance or rotation_error > self._rotation_tolerance:
            return IKResult(False, reason="tcp_pose_tolerance")
        return IKResult(True, joints=solution)
