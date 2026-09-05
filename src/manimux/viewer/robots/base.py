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


@dataclass(frozen=True, slots=True)
class RobotGroup:
    """One independently visualized kinematic group, such as an arm."""

    name: str
    label: str
    base_position: Position
    prediction_color: Color
    trail_color: Color
    urdf_path: Path | None = None


@dataclass(frozen=True, slots=True)
class SceneBox:
    """Optional static box supplied by a robot adapter for scene context."""

    name: str
    dimensions: Position
    position: Position
    color: Color = (205, 210, 218)


class RobotAdapter(ABC):
    """Boundary between the universal dashboard and robot-specific geometry."""

    name: str
    label: str
    groups: tuple[RobotGroup, ...]
    scene_boxes: tuple[SceneBox, ...] = ()

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

    def gripper_closed_steps(
        self,
        grouped_actions: Mapping[str, np.ndarray],
        *,
        previous_positions: Mapping[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Return closing-or-closed flags when the embodiment exposes a gripper."""

        del previous_positions
        if not grouped_actions:
            return np.empty(0, dtype=np.bool_)
        horizon = len(next(iter(grouped_actions.values())))
        return np.zeros(horizon, dtype=np.bool_)

    def group(self, name: str) -> RobotGroup:
        for group in self.groups:
            if group.name == name:
                return group
        raise KeyError(f"unknown group {name!r} for robot {self.name!r}")
