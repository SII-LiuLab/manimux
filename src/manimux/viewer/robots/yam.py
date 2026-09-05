"""Built-in YAM robot adapter using the official i2rt model assets."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from manimux.kinematics.yam import YamKinematics

from .base import RobotAdapter, RobotGroup, SceneBox

DEFAULT_I2RT_ROOT = Path(__file__).resolve().parents[2] / "assets"


class YamAdapter(RobotAdapter):
    """First bundled robot: one or two 7-DoF YAM arms."""

    name = "yam"
    label = "YAM"

    def __init__(
        self,
        model_root: Path | str | None = None,
        i2rt_root: Path | str | None = None,
    ) -> None:
        root_value = model_root if model_root is not None else i2rt_root
        self.i2rt_root = Path(root_value or DEFAULT_I2RT_ROOT).expanduser().resolve()
        urdf_path = self.i2rt_root / "i2rt/robot_models/arm/yam/yam.urdf"
        if not urdf_path.is_file():
            raise FileNotFoundError(
                f"YAM URDF not found: {urdf_path}. "
                "Pass --robot-model-root with the i2rt checkout path."
            )
        self.groups = (
            RobotGroup(
                "left",
                "Left arm",
                (0.0, 0.32, 0.0),
                (52, 111, 255),
                (33, 180, 106),
                urdf_path,
            ),
            RobotGroup(
                "right",
                "Right arm",
                (0.0, -0.32, 0.0),
                (255, 94, 87),
                (255, 170, 0),
                urdf_path,
            ),
        )
        self.scene_boxes = (SceneBox("table", (0.75, 0.82, 0.035), (0.38, 0.0, 0.02)),)
        self.kinematics = YamKinematics(assets_root=self.i2rt_root)

    @staticmethod
    def _split(values: np.ndarray, *, sequence: bool) -> dict[str, np.ndarray]:
        array = np.asarray(values, dtype=np.float64)
        expected_ndim = 2 if sequence else 1
        if array.ndim != expected_ndim:
            kind = "action chunk" if sequence else "joint state"
            raise ValueError(f"YAM {kind} must be {expected_ndim}D, got {array.ndim}D")
        width = array.shape[-1]
        if width == 7:
            return {"left": array}
        if width == 14:
            return {"left": array[..., :7], "right": array[..., 7:14]}
        raise ValueError(f"YAM expects 7 or 14 values, got {width}")

    def split_actions(
        self, actions: np.ndarray, action_space: str = "joint_position"
    ) -> dict[str, np.ndarray]:
        if action_space != "joint_position":
            raise ValueError(
                f"YAM adapter supports action_space='joint_position', got {action_space!r}"
            )
        return self._split(actions, sequence=True)

    def split_joint_positions(self, joint_positions: np.ndarray) -> dict[str, np.ndarray]:
        return self._split(joint_positions, sequence=False)

    def gripper_closed_steps(
        self,
        grouped_actions: Mapping[str, np.ndarray],
        *,
        previous_positions: Mapping[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        closing_or_closed: list[np.ndarray] = []
        previous_positions = previous_positions or {}
        for group_name, actions in grouped_actions.items():
            values = np.asarray(actions, dtype=np.float64)[:, 6]
            previous = previous_positions.get(group_name)
            previous_value = (
                float(np.asarray(previous, dtype=np.float64).reshape(-1)[6])
                if previous is not None
                else None
            )
            closing_or_closed.append(
                self._closing_or_closed(values, previous_value=previous_value)
            )
        if not closing_or_closed:
            return np.empty(0, dtype=np.bool_)
        return np.logical_or.reduce(closing_or_closed)

    @staticmethod
    def _closing_or_closed(
        values: np.ndarray,
        *,
        previous_value: float | None,
        closing_delta: float = 0.02,
        closed_threshold: float = 0.5,
        reopened_threshold: float = 0.9,
    ) -> np.ndarray:
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

    def pose(self, group: str, configuration: np.ndarray) -> np.ndarray:
        self.group(group)
        return self.kinematics.pose(configuration)

    def visual_configuration(self, group: str, configuration: np.ndarray) -> np.ndarray:
        self.group(group)
        action = np.asarray(configuration, dtype=np.float64).reshape(-1)
        if action.size != 7:
            raise ValueError(f"YAM group configuration must contain 7 values, got {action.size}")
        gripper = float(np.clip(action[6], 0.0, 1.0)) * -0.04695
        return np.concatenate((action[:6], np.array([gripper, gripper])))

    def initial_configuration(self, group: str) -> np.ndarray:
        self.group(group)
        return np.zeros(8, dtype=np.float64)

    def camera_slot(self, source_name: str) -> str:
        aliases = {"top_camera": "top", "front_camera": "top"}
        normalized = source_name.removesuffix("_rgb")
        return aliases.get(normalized, normalized.removesuffix("_camera"))

    def demo_sample(self, elapsed_s: float, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        left = np.array(
            [
                0.15 * np.sin(elapsed_s),
                0.8,
                1.25,
                -0.35 + 0.15 * np.sin(elapsed_s * 1.3),
                0.1,
                -0.2,
                0.7,
            ]
        )
        right = np.array(
            [
                -0.15 * np.sin(elapsed_s),
                0.75,
                1.20,
                -0.25 + 0.12 * np.cos(elapsed_s),
                -0.1,
                0.2,
                0.4,
            ]
        )
        chunk = []
        denominator = max(1, horizon - 1)
        for index in range(horizon):
            left_delta = np.array(
                [0.1 * index / denominator, 0, 0, 0.05 * np.sin(index / 4), 0, 0, 0]
            )
            right_delta = np.array(
                [-0.1 * index / denominator, 0, 0, -0.05 * np.sin(index / 4), 0, 0, 0]
            )
            chunk.append(np.r_[left + left_delta, right + right_delta])
        return np.r_[left, right], np.stack(chunk)
