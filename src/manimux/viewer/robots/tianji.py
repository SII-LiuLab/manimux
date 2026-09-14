"""Built-in Tianji Marvin dual-arm adapter.

Each arm renders from its own vendor CAD export (the two sides are mirrored
parts, not a renamed copy), with a swappable end effector from
``assets/end_effectors`` bolted to its flange, and the shared stand renders once
as a static mesh. The URDF joints use the same angle convention as the
controller's DH chain, so the radian joint vector drives them unchanged.

Each arm's base frame is mounted sideways on the stand (+x forward, +y down,
+z to that arm's left), so the groups carry the mount rotation instead of every
pose being rotated. Mounts are relative to ``Link_Base`` on the floor:

* left (A):  (0,  0.037, 1.121) m, Rx(-90 deg)
* right (B): (0, -0.037, 1.121) m, Rx(+90 deg)

Group vectors are 7 joints in radians followed by the end effector's inputs
(one normalized aperture for the default ``umi_follower``). Select another end
effector with ``--robot-option end_effector=<name>``, or ``none`` for bare arms.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from manimux.kinematics.end_effector import EndEffector, attach_end_effector, load_end_effector
from manimux.kinematics.tianji import NUM_ARM_JOINTS, TianjiKinematics

from .base import (
    RobotAdapter,
    RobotGroup,
    SceneBox,
    SceneView,
    StaticMesh,
    gripper_closed_steps_by_group_at,
)

DEFAULT_TIANJI_ROOT = Path(__file__).resolve().parents[2] / "assets" / "tianji"
DEFAULT_END_EFFECTOR = "umi_follower"

_HALF_SQRT2 = float(np.sqrt(0.5))
LEFT_BASE_POSITION = (0.0, 0.037, 1.121)
LEFT_BASE_ORIENTATION = (_HALF_SQRT2, -_HALF_SQRT2, 0.0, 0.0)
RIGHT_BASE_POSITION = (0.0, -0.037, 1.121)
RIGHT_BASE_ORIENTATION = (_HALF_SQRT2, _HALF_SQRT2, 0.0, 0.0)

# Controller home pose per arm, degrees.
HOME_JOINTS_DEG = {
    "left": (90.0, -90.0, -90.0, -90.0, 0.0, 0.0, 0.0),
    "right": (-90.0, -90.0, 90.0, -90.0, 0.0, 0.0, 0.0),
}

# Work table in front of the stand, top surface at 0.65 m. Footprint estimated.
TABLE = SceneBox("table", (0.8, 1.2, 0.04), (0.65, 0.0, 0.63))


class TianjiAdapter(RobotAdapter):
    """Two 7-DoF Marvin arms with end effectors on the Tianji stand."""

    name = "tianji"
    label = "Tianji Marvin"
    scene_boxes = (TABLE,)
    scene_view = SceneView(
        camera_position=(2.4, -2.3, 2.0),
        camera_look_at=(0.3, -0.15, 0.9),
        grid_position=(0.3, 0.0, -0.01),
        grid_size=(2.0, 2.0),
    )

    def __init__(
        self,
        model_root: Path | str | None = None,
        end_effector: EndEffector | str | None = DEFAULT_END_EFFECTOR,
        end_effector_root: Path | str | None = None,
    ) -> None:
        self.model_root = Path(model_root or DEFAULT_TIANJI_ROOT).expanduser().resolve()
        paths = {
            "left": self.model_root / "left" / "arm.urdf",
            "right": self.model_root / "right" / "arm.urdf",
            "stand": self.model_root / "stand" / "stand.urdf",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Tianji model files not found: {', '.join(missing)}. "
                "Pass --robot-model-root with a directory holding left/, right/ and stand/."
            )
        if isinstance(end_effector, str):
            end_effector = (
                None
                if end_effector == "none"
                else load_end_effector(end_effector, end_effector_root)
            )
        self.end_effector = end_effector
        self.ee_inputs = 0 if end_effector is None else end_effector.inputs
        self.group_width = NUM_ARM_JOINTS + self.ee_inputs
        if self.ee_inputs and self.group_width == 2 * NUM_ARM_JOINTS:
            raise ValueError(
                "an end effector with 7 inputs makes single- and dual-arm widths equal"
            )
        self.groups = (
            RobotGroup(
                "left",
                "Left arm (A)",
                LEFT_BASE_POSITION,
                (52, 111, 255),
                (33, 180, 106),
                attach_end_effector(paths["left"], "Flange_L", end_effector),
                LEFT_BASE_ORIENTATION,
            ),
            RobotGroup(
                "right",
                "Right arm (B)",
                RIGHT_BASE_POSITION,
                (255, 94, 87),
                (255, 170, 0),
                attach_end_effector(paths["right"], "Flange_R", end_effector),
                RIGHT_BASE_ORIENTATION,
            ),
        )
        self.static_meshes = (StaticMesh("stand", paths["stand"]),)
        self.kinematics = TianjiKinematics(end_effector=end_effector)

    def _split(self, values: np.ndarray, *, sequence: bool) -> dict[str, np.ndarray]:
        array = np.asarray(values, dtype=np.float64)
        expected_ndim = 2 if sequence else 1
        if array.ndim != expected_ndim:
            kind = "action chunk" if sequence else "joint state"
            raise ValueError(f"Tianji {kind} must be {expected_ndim}D, got {array.ndim}D")
        width = array.shape[-1]
        single = sorted({NUM_ARM_JOINTS, self.group_width})
        if width in single:
            return {"left": array}
        if width in {2 * value for value in single}:
            half = width // 2
            return {"left": array[..., :half], "right": array[..., half:]}
        allowed = [str(value) for value in (*single, *(2 * value for value in single))]
        raise ValueError(
            f"Tianji expects {', '.join(allowed[:-1])} or {allowed[-1]} values, got {width}"
        )

    def split_actions(
        self, actions: np.ndarray, action_space: str = "joint_position"
    ) -> dict[str, np.ndarray]:
        if action_space != "joint_position":
            raise ValueError(
                f"Tianji adapter supports action_space='joint_position', got {action_space!r}"
            )
        return self._split(actions, sequence=True)

    def split_joint_positions(self, joint_positions: np.ndarray) -> dict[str, np.ndarray]:
        return self._split(joint_positions, sequence=False)

    def gripper_closed_steps_by_group(
        self,
        grouped_actions: Mapping[str, np.ndarray],
        *,
        previous_positions: Mapping[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        aperture = None if self.end_effector is None else self.end_effector.spec.aperture_input
        if aperture is None:
            return {}
        index = NUM_ARM_JOINTS + aperture
        with_aperture = {
            name: actions
            for name, actions in grouped_actions.items()
            if np.asarray(actions).shape[-1] > index
        }
        return gripper_closed_steps_by_group_at(
            with_aperture, previous_positions, index=index
        )

    def pose(self, group: str, configuration: np.ndarray) -> np.ndarray:
        self.group(group)
        return self.kinematics.pose(configuration)

    def visual_configuration(self, group: str, configuration: np.ndarray) -> np.ndarray:
        self.group(group)
        values = np.asarray(configuration, dtype=np.float64).reshape(-1)
        if values.size not in (NUM_ARM_JOINTS, self.group_width):
            raise ValueError(
                f"Tianji group configuration must have {NUM_ARM_JOINTS} or "
                f"{self.group_width} values, got {values.size}"
            )
        joints = values[:NUM_ARM_JOINTS]
        if self.end_effector is None:
            return joints.copy()
        # Without reported inputs the end effector is drawn at its rest pose.
        inputs = values[NUM_ARM_JOINTS:] if values.size == self.group_width else None
        return np.concatenate((joints, self.end_effector.joint_positions(inputs)))

    def initial_configuration(self, group: str) -> np.ndarray:
        self.group(group)
        return self.visual_configuration(group, np.radians(HOME_JOINTS_DEG[group]))

    def camera_slot(self, source_name: str) -> str:
        return source_name.removesuffix("_rgb").removesuffix("_camera").removesuffix("_wrist")

    def _rest_inputs(self, phase: float) -> np.ndarray:
        if self.end_effector is None:
            return np.empty(0, dtype=np.float64)
        inputs = np.asarray(self.end_effector.spec.rest_inputs, dtype=np.float64)
        aperture = self.end_effector.spec.aperture_input
        if aperture is not None:
            inputs[aperture] = 0.5 + 0.5 * np.cos(phase * 0.5)
        return inputs

    def demo_sample(self, elapsed_s: float, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        def arm(side: str, phase: float) -> tuple[np.ndarray, np.ndarray]:
            wiggle = [0.0, 10.0 * np.sin(phase), 0.0, 8.0 * np.sin(phase * 1.3), 0.0,
                      5.0 * np.sin(phase * 0.7), 0.0]  # fmt: skip
            joints = np.radians(np.add(HOME_JOINTS_DEG[side], wiggle))
            return joints, self._rest_inputs(phase)

        (left, left_inputs), (right, right_inputs) = (
            arm("left", elapsed_s),
            arm("right", elapsed_s + np.pi / 2),
        )
        denominator = max(1, horizon - 1)
        chunk = []
        for index in range(horizon):
            delta = np.radians(
                [0.0, 4.0 * index / denominator, 0.0, 3.0 * np.sin(index / 4), 0.0, 0.0, 0.0]
            )
            chunk.append(np.concatenate((left + delta, left_inputs, right - delta, right_inputs)))
        state = np.concatenate((left, left_inputs, right, right_inputs))
        return state, np.stack(chunk)
