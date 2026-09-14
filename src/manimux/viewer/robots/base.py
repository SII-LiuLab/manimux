"""Robot adapter contract used by the viewer.

The dashboard deliberately knows nothing about a robot's joint count, kinematic
groups, URDF layout, or action-to-configuration mapping.  Those details live in
implementations of :class:`RobotAdapter`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

Color = tuple[int, int, int]
Position = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]

IDENTITY_WXYZ: Quaternion = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class RobotGroup:
    """One independently visualized kinematic group, such as an arm."""

    name: str
    label: str
    base_position: Position
    prediction_color: Color
    trail_color: Color
    urdf_path: Path | None = None
    # wxyz rotation from the group's own pose()/URDF frame into the +z-up world.
    # Identity for arms authored z-up (YAM); robots whose base frame is mounted
    # sideways (Tianji) set it once instead of rotating every pose.
    base_orientation: Quaternion = IDENTITY_WXYZ


@dataclass(frozen=True, slots=True)
class SceneBox:
    """Optional static box supplied by a robot adapter for scene context."""

    name: str
    dimensions: Position
    position: Position
    color: Color = (205, 210, 218)


@dataclass(frozen=True, slots=True)
class StaticMesh:
    """A non-articulated URDF placed once, such as a torso or mounting stand.

    Unlike :class:`RobotGroup` it never receives joint state; unlike
    :class:`SceneBox` it renders real geometry.
    """

    name: str
    urdf_path: Path
    position: Position = (0.0, 0.0, 0.0)
    orientation: Quaternion = IDENTITY_WXYZ


@dataclass(frozen=True, slots=True)
class SceneView:
    """Initial camera and floor grid framing for one robot's workspace."""

    camera_position: Position = (1.25, -1.93, 1.25)
    camera_look_at: Position = (0.05, -0.13, 0.28)
    grid_position: Position = (0.25, 0.0, -0.01)
    grid_size: tuple[float, float] = (2.0, 1.6)


def aperture_closing_flags(
    values: np.ndarray,
    *,
    previous_value: float | None,
    closing_delta: float = 0.02,
    closed_threshold: float = 0.5,
    reopened_threshold: float = 0.9,
) -> np.ndarray:
    """Flag steps where a normalized aperture (0 closed, 1 open) is closing or closed."""

    apertures = np.clip(np.asarray(values, dtype=np.float64).reshape(-1), 0.0, 1.0)
    flags = np.zeros(len(apertures), dtype=np.bool_)
    if not len(apertures):
        return flags

    previous = float(apertures[0] if previous_value is None else previous_value)
    active = previous <= closed_threshold
    for index, aperture in enumerate(apertures):
        delta = float(aperture - previous)
        if delta <= -closing_delta or aperture <= closed_threshold:
            active = True
        elif active and delta >= closing_delta and aperture >= reopened_threshold:
            active = False
        flags[index] = active
        previous = float(aperture)
    return flags


def gripper_closed_steps_by_group_at(
    grouped_actions: Mapping[str, np.ndarray],
    previous_positions: Mapping[str, np.ndarray] | None,
    index: int,
) -> dict[str, np.ndarray]:
    """Return each group's closing-or-closed flags for aperture column ``index``."""

    closing_or_closed: dict[str, np.ndarray] = {}
    previous_positions = previous_positions or {}
    for group_name, actions in grouped_actions.items():
        values = np.asarray(actions, dtype=np.float64)[:, index]
        previous = previous_positions.get(group_name)
        previous_value = None
        if previous is not None:
            flat = np.asarray(previous, dtype=np.float64).reshape(-1)
            previous_value = float(flat[index]) if flat.size > index else None
        closing_or_closed[group_name] = aperture_closing_flags(
            values, previous_value=previous_value
        )
    return closing_or_closed


class RobotAdapter(ABC):
    """Boundary between the universal dashboard and robot-specific geometry."""

    name: str
    label: str
    groups: tuple[RobotGroup, ...]
    scene_boxes: tuple[SceneBox, ...] = ()
    static_meshes: tuple[StaticMesh, ...] = ()
    scene_view: SceneView = SceneView()

    @abstractmethod
    def split_actions(self, actions: np.ndarray, action_space: str) -> Mapping[str, np.ndarray]:
        """Validate and split an ``(horizon, action_dim)`` policy chunk."""

    @abstractmethod
    def split_joint_positions(self, joint_positions: np.ndarray) -> Mapping[str, np.ndarray]:
        """Validate and split one achieved robot joint vector."""

    @abstractmethod
    def pose(self, group: str, configuration: np.ndarray) -> np.ndarray:
        """Return a 4x4 end-effector transform in the group's local frame."""

    def positions(self, group: str, configurations: np.ndarray) -> np.ndarray:
        """Return end-effector positions for a configuration sequence."""

        values = np.asarray(configurations, dtype=np.float64)
        return np.stack([self.pose(group, value)[:3, 3] for value in values])

    @abstractmethod
    def visual_configuration(self, group: str, configuration: np.ndarray) -> np.ndarray:
        """Map a policy/robot vector to the URDF visualizer configuration."""

    @abstractmethod
    def initial_configuration(self, group: str) -> np.ndarray:
        """Return a safe initial URDF visualizer configuration."""

    @abstractmethod
    def demo_sample(self, elapsed_s: float, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(joint_positions, action_chunk)`` for offline demo mode."""

    def camera_slot(self, source_name: str) -> str:
        """Normalize a camera key for a dynamically created dashboard panel."""

        return source_name.removesuffix("_rgb").removesuffix("_camera")

    def gripper_closed_steps_by_group(
        self,
        grouped_actions: Mapping[str, np.ndarray],
        *,
        previous_positions: Mapping[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """Return closing-or-closed flags independently for each robot group."""

        del grouped_actions, previous_positions
        return {}

    def gripper_closed_steps(
        self,
        grouped_actions: Mapping[str, np.ndarray],
        *,
        previous_positions: Mapping[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Aggregate per-group flags for callers of the original viewer interface."""

        flags = self.gripper_closed_steps_by_group(
            grouped_actions, previous_positions=previous_positions
        )
        if not flags:
            return np.empty(0, dtype=np.bool_)
        return np.logical_or.reduce(list(flags.values()))

    def group(self, name: str) -> RobotGroup:
        for group in self.groups:
            if group.name == name:
                return group
        raise KeyError(f"unknown group {name!r} for robot {self.name!r}")
