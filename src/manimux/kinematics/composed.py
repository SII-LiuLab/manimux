"""Compose arm, mounted-tool and grouped robot kinematics offline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from manimux.kinematics.base import (
    ArmKinematicsBase,
    FloatArray,
    IKResult,
    KinematicCoordinate,
    rigid_transform,
)


class ToolGeometryBase(ABC):
    """Geometry from a tool base to its selected TCP, excluding the mount."""

    @property
    @abstractmethod
    def coordinates(self) -> tuple[KinematicCoordinate, ...]:
        raise NotImplementedError

    @property
    def num_coordinates(self) -> int:
        return len(self.coordinates)

    @property
    @abstractmethod
    def base_frame(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def tcp_frame(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def tcp_transform(self, tool_state: FloatArray | None = None) -> FloatArray:
        """Return the tool-base-to-TCP transform."""
        raise NotImplementedError


class FixedToolGeometry(ToolGeometryBase):
    """A fixed TCP transform that may retain tool-state coordinates."""

    def __init__(
        self,
        transform: FloatArray,
        *,
        coordinates: tuple[KinematicCoordinate, ...] = (),
        base_frame: str = "tool_base",
        tcp_frame: str = "tcp",
    ) -> None:
        self._transform = rigid_transform(transform, "tool transform")
        self._coordinates = tuple(coordinates)
        if len({item.name for item in self._coordinates}) != len(self._coordinates):
            raise ValueError("tool coordinate names must be unique")
        self._base_frame = base_frame
        self._tcp_frame = tcp_frame

    @property
    def coordinates(self) -> tuple[KinematicCoordinate, ...]:
        return self._coordinates

    @property
    def base_frame(self) -> str:
        return self._base_frame

    @property
    def tcp_frame(self) -> str:
        return self._tcp_frame

    def tcp_transform(self, tool_state: FloatArray | None = None) -> FloatArray:
        if tool_state is None:
            if self.num_coordinates:
                raise ValueError("tool_state is required for declared tool coordinates")
            state = np.empty(0, dtype=np.float64)
        else:
            state = np.asarray(tool_state, dtype=np.float64)
        if state.shape != (self.num_coordinates,) or not np.isfinite(state).all():
            raise ValueError(f"tool_state must be a finite vector of length {self.num_coordinates}")
        for coordinate, value in zip(self.coordinates, state, strict=True):
            if coordinate.unit == "normalized" and not 0 <= value <= 1:
                raise ValueError(f"normalized coordinate {coordinate.name!r} must be in [0, 1]")
        return self._transform.copy()


class ManipulatorKinematicsBase(ABC):
    """Robot/policy-facing TCP kinematics for one configured arm group."""

    @property
    @abstractmethod
    def coordinates(self) -> tuple[KinematicCoordinate, ...]:
        raise NotImplementedError

    @property
    def num_coordinates(self) -> int:
        return len(self.coordinates)

    @property
    @abstractmethod
    def base_frame(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def tcp_frame(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def fk(self, configuration: FloatArray) -> FloatArray:
        raise NotImplementedError

    @abstractmethod
    def ik(
        self,
        target_tcp: FloatArray,
        seed_configuration: FloatArray,
        *,
        fixed_coordinates: Mapping[str, float],
        duration_s: float | None = None,
    ) -> IKResult:
        raise NotImplementedError

    def reset(self) -> None:
        """Reset optional state carried by the selected arm solver."""
        return None


class ComposedManipulatorKinematics(ManipulatorKinematicsBase):
    """An arm plus a flange mount and tool, ordered arm coordinates first."""

    def __init__(
        self,
        arm: ArmKinematicsBase,
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
        if not arm_layout or len(arm_layout) != arm.num_joints:
            raise ValueError("arm coordinates must match the arm solver dimension")
        if any(item.unit not in {"rad", "m"} for item in arm_layout):
            raise ValueError("arm coordinates must use radians or metres")
        if len({item.name for item in layout}) != len(layout):
            raise ValueError("arm and tool coordinate names must be unique")
        for tolerance in (position_tolerance, rotation_tolerance):
            if tolerance is not None and (not np.isfinite(tolerance) or tolerance <= 0):
                raise ValueError("pose tolerances must be finite and positive")
        self._arm = arm
        self._tool = tool
        self._arm_coordinates = arm_layout
        self._arm_size = len(arm_layout)
        self._coordinates = layout
        self._mount = rigid_transform(mount, "mount")
        self._base_frame = base_frame
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
        return self._tool.tcp_frame

    @property
    def arm(self) -> ArmKinematicsBase:
        return self._arm

    def with_arm(self, arm: ArmKinematicsBase) -> ComposedManipulatorKinematics:
        """Reuse the exact mount/tool contract with another arm IK implementation."""
        return type(self)(
            arm,
            self._tool,
            arm_coordinates=self._arm_coordinates,
            mount=self._mount,
            base_frame=self._base_frame,
            position_tolerance=self._position_tolerance,
            rotation_tolerance=self._rotation_tolerance,
        )

    def reset(self) -> None:
        self._arm.reset()

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
        return self._mount @ self._tool.tcp_transform(tool_state.copy())

    def fk(self, configuration: FloatArray) -> FloatArray:
        state = self._configuration(configuration)
        flange = self._arm.fk(state[: self._arm_size].copy())
        return flange @ self._offset(state[self._arm_size :])

    def flange_target(self, target_tcp: FloatArray, tool_state: FloatArray) -> FloatArray:
        """Remove the mounted-tool offset while staying in the arm base frame."""
        return rigid_transform(target_tcp, "target_tcp") @ np.linalg.inv(
            self._offset(tool_state)
        )

    def ik(
        self,
        target_tcp: FloatArray,
        seed_configuration: FloatArray,
        *,
        fixed_coordinates: Mapping[str, float],
        duration_s: float | None = None,
    ) -> IKResult:
        target = rigid_transform(target_tcp, "target_tcp")
        seed = self._configuration(seed_configuration)
        indices = {coordinate.name: index for index, coordinate in enumerate(self.coordinates)}
        for name, value in fixed_coordinates.items():
            if name not in indices:
                raise KeyError(f"unknown fixed coordinate {name!r}")
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
        result = self._arm.ik(
            self.flange_target(target, tool_state),
            seed[: self._arm_size].copy(),
            duration_s=duration_s,
        )
        if not result.ok:
            return result
        if result.joints is None or result.joints.shape != (self._arm_size,):
            raise ValueError("arm solver returned an invalid joint dimension")
        solution = np.concatenate((result.joints, tool_state))
        # Rate-based IK returns an accepted bounded step, rather than claiming
        # that one call reached the target. The result contract states which
        # semantic applies, so composition stays independent of its class.
        if self._position_tolerance is not None and result.target_reached:
            actual = self.fk(solution)
            position_error = np.linalg.norm(actual[:3, 3] - target[:3, 3])
            relative_rotation = target[:3, :3].T @ actual[:3, :3]
            rotation_error = np.arccos(
                np.clip((np.trace(relative_rotation) - 1) / 2, -1, 1)
            )
            if (
                position_error > self._position_tolerance
                or rotation_error > self._rotation_tolerance
            ):
                return IKResult(False, reason="tcp_pose_tolerance", diagnostics=result.diagnostics)
        return IKResult(
            True,
            joints=solution,
            diagnostics=result.diagnostics,
            target_reached=result.target_reached,
        )


class RobotKinematics:
    """Independent grouped TCP kinematics matching RobotState group names."""

    def __init__(self, models: Mapping[str, ManipulatorKinematicsBase]) -> None:
        if not models or any(not isinstance(name, str) or not name.strip() for name in models):
            raise ValueError("kinematic groups must have non-empty names")
        self._models = MappingProxyType(dict(models))

    @property
    def models(self) -> Mapping[str, ManipulatorKinematicsBase]:
        return self._models

    def _groups(self, values: Mapping) -> None:
        if not values or set(values) - self.models.keys():
            raise ValueError("expected a non-empty subset of configured groups")

    def fk(self, configuration: Mapping[str, FloatArray]) -> dict[str, FloatArray]:
        self._groups(configuration)
        return {name: self.models[name].fk(q) for name, q in configuration.items()}

    def ik(
        self,
        targets: Mapping[str, FloatArray],
        seed: Mapping[str, FloatArray],
        *,
        fixed_coordinates: Mapping[str, Mapping[str, float]],
        duration_s: float | None = None,
    ) -> dict[str, IKResult]:
        self._groups(targets)
        if set(seed) != set(targets) or set(fixed_coordinates) != set(targets):
            raise ValueError(
                "targets, seeds and fixed-coordinate mappings must have identical groups"
            )
        return {
            name: self.models[name].ik(
                target,
                seed[name],
                fixed_coordinates=fixed_coordinates[name],
                duration_s=duration_s,
            )
            for name, target in targets.items()
        }
