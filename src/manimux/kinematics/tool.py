"""Offline tool geometry, independent of hardware and flange mounting."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from manimux.kinematics.base import FloatArray, KinematicCoordinate, rigid_transform


class ToolGeometryBase(ABC):
    """Geometry from the tool's own base to one selected TCP.

    Transforms are right-handed homogeneous 4x4 matrices acting on column
    vectors, with translations in metres. The flange-to-tool installation
    transform belongs to the composition, not to this model. Implementations
    must not connect hardware or read live state.

    Coordinates describe independent geometric inputs, excluding mimic joints.
    A changing TCP can depend on these inputs; a fixed TCP can still declare a
    gripper opening to preserve the complete manipulator configuration layout.
    Normalized inputs use [0, 1]; concrete tools must document their direction
    and calibration. Limits for radian/metre inputs are model-specific.
    """

    @property
    @abstractmethod
    def coordinates(self) -> tuple[KinematicCoordinate, ...]:
        """Stable layout with unique names; empty for a tool with no inputs."""
        raise NotImplementedError

    @property
    def num_coordinates(self) -> int:
        return len(self.coordinates)

    @property
    @abstractmethod
    def base_frame(self) -> str:
        """Non-empty name of the tool base, not the arm base or flange."""
        raise NotImplementedError

    @property
    @abstractmethod
    def tcp_frame(self) -> str:
        """Non-empty name of the selected TCP."""
        raise NotImplementedError

    @abstractmethod
    def tcp_transform(self, tool_state: FloatArray | None = None) -> FloatArray:
        """Return ``T_tool_base_tcp`` for a finite ``(num_coordinates,)`` state.

        ``None`` is allowed only when no coordinates are declared. Invalid
        shapes, non-finite values and invalid normalized values raise ValueError,
        even if a fixed transform does not depend on the values. Returned
        matrices must be safe for callers to mutate without altering the model.
        """
        raise NotImplementedError


class FixedToolGeometry(ToolGeometryBase):
    """A constant TCP offset, optionally retaining tool state coordinates.

    The caller supplies the transform from the tool base, excluding its mount.
    No default offset or gripper opening convention is inferred. Only finite
    values and normalized bounds are checked; radian/metre limits must be
    supplied by a concrete tool model when needed.
    """

    def __init__(
        self,
        transform: FloatArray,
        *,
        coordinates: tuple[KinematicCoordinate, ...] = (),
        base_frame: str = "tool_base",
        tcp_frame: str = "tcp",
    ) -> None:
        pose = rigid_transform(transform, "tool transform")
        layout = tuple(coordinates)
        if len({item.name for item in layout}) != len(layout):
            raise ValueError("tool coordinate names must be unique")
        self._transform = pose
        self._coordinates = layout
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
