"""Rendering data derived from RobotModel, with per-station YAML styling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from manimux.embodiments.robot.base import RobotModel
from manimux.viewer.end_effector import (
    mounted_group_visual_configuration,
    mounted_group_visual_urdf,
)

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


class RobotView:
    """The display projection of an offline model; never owns hardware."""

    def __init__(self, model: RobotModel, options: Mapping):
        self.model = model
        self.name = model.name
        self.label = options.get("label", self.name)
        self.options = options
        self.kinematics = model.kinematics
        styles = options.get("groups", {})
        unknown = set(styles) - model.groups.keys()
        if unknown:
            raise ValueError(f"display styles name unknown groups: {sorted(unknown)}")
        self.groups = tuple(
            RobotGroup(
                name=name,
                label=styles.get(name, {}).get("label", name),
                base_position=tuple(group.base_transform[:3, 3]),
                base_orientation=tuple(
                    Rotation.from_matrix(group.base_transform[:3, :3]).as_quat()[[3, 0, 1, 2]]
                ),
                prediction_color=tuple(
                    styles.get(name, {}).get("prediction_color", [52, 111, 255])
                ),
                trail_color=tuple(styles.get(name, {}).get("trail_color", [33, 180, 106])),
                urdf_path=mounted_group_visual_urdf(group),
            )
            for name, group in model.groups.items()
        )
        self.static_meshes = tuple(
            StaticMesh(
                name,
                path,
                tuple(mount[:3, 3]),
                tuple(Rotation.from_matrix(mount[:3, :3]).as_quat()[[3, 0, 1, 2]]),
            )
            for name, path, mount in model.static_assets
        )
        scene = options.get("scene", {})
        self.scene_boxes = tuple(
            SceneBox(
                **{
                    key: tuple(value) if isinstance(value, list) else value
                    for key, value in box.items()
                }
            )
            for box in scene.get("boxes", [])
        )
        self.scene_view = SceneView(
            **{key: tuple(value) for key, value in scene.get("view", {}).items()}
        )
        if options.get("initial_pose") not in {None, "home"}:
            raise ValueError("initial_pose must be home when specified")
        for name in model.groups:
            self.initial_positions(name)

    def validate_groups(self, values: Mapping, *, sequence=False):
        if not isinstance(values, Mapping) or not values or set(values) - self.model.groups.keys():
            raise ValueError("expected named groups from the selected RobotModel")
        result = {}
        horizon = None
        for name, value in values.items():
            array = np.asarray(value, dtype=np.float64)
            width = self.model.groups[name].kinematics.num_coordinates
            if (
                array.ndim != (2 if sequence else 1)
                or array.shape[-1] != width
                or not np.isfinite(array).all()
            ):
                raise ValueError(f"{name}: invalid configuration shape/values (expected {width})")
            if sequence:
                if not len(array) or (horizon is not None and len(array) != horizon):
                    raise ValueError("action groups must share a non-empty horizon")
                horizon = len(array)
            result[name] = array
        return result

    def pose(self, group, configuration):
        # Control FK is already in the arm base. The scene root applies its mount once.
        return self.model.groups[group].kinematics.fk(configuration)

    def positions(self, group, configurations):
        return np.stack([self.pose(group, q)[:3, 3] for q in configurations])

    def visual_configuration(self, group, configuration):
        return mounted_group_visual_configuration(self.model.groups[group], configuration)

    def initial_configuration(self, group):
        return self.visual_configuration(group, self.initial_positions(group))

    def initial_positions(self, group):
        width = self.model.groups[group].kinematics.num_coordinates
        style = self.options.get("groups", {}).get(group, {})
        if self.options.get("initial_pose") == "home":
            if "initial" in style:
                raise ValueError("initial_pose: home cannot be combined with group initial values")
            if group not in self.model.home_joints:
                raise ValueError(f"{group}: initial_pose requests an unconfigured Home target")
            joints = self.model.home_joints[group]
            tool = style.get("initial_end_effector", [0.0] * (width - len(joints)))
            value = np.concatenate((joints, np.asarray(tool, dtype=np.float64)))
        else:
            value = style.get("initial", [0.0] * width)
        return self.validate_groups({group: value})[group]

    def camera_slot(self, source):
        aliases = self.options.get("camera_aliases", {})
        return aliases.get(source, source)

    def gripper_closed_steps_by_group(self, grouped_actions, *, previous_positions=None):
        result = {}
        for name, actions in grouped_actions.items():
            coordinate = self.options.get("groups", {}).get(name, {}).get("aperture")
            if coordinate is None:
                continue
            coordinates = self.model.groups[name].kinematics.coordinates
            index = next((i for i, c in enumerate(coordinates) if c.name == coordinate), None)
            if index is None or coordinates[index].unit != "normalized":
                raise ValueError(f"{name}: aperture must name a normalized model coordinate")
            result.update(
                gripper_closed_steps_by_group_at({name: actions}, previous_positions, index=index)
            )
        return result

    def group(self, name):
        return next(group for group in self.groups if group.name == name)
