"""Explicit single-arm Tianji + TacCap kinematic assembly."""

from pathlib import Path
from typing import Literal

import numpy as np

from manimux.end_effectors.taccap.geometry import TacCapGeometry
from manimux.kinematics.base import FloatArray, KinematicCoordinate
from manimux.kinematics.composed import ComposedManipulatorKinematics
from manimux.robots.tianji.kinematics import DEFAULT_CONFIG, TianjiSDKKinematics


def umi_follower_mount() -> FloatArray:
    """Return the existing CAD installation's T_flange_tool_base in metres.

    Source: assets/end_effectors/umi_follower/end_effector.yaml, mount xyz
    [-0.01575, 0, 0.0505], extrinsic xyz RPY [pi, -pi/2, 0]. This is one
    installation preset, not a universal TacCap mounting transform. A fresh
    array is returned so station-specific adjustments do not alter the preset.
    """
    return np.array(
        [
            [0, 0, 1, -0.01575],
            [0, -1, 0, 0],
            [1, 0, 0, 0.0505],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )


def build_tianji_taccap_kinematics(
    arm: Literal["left", "right"],
    *,
    mount: FloatArray,
    tool: TacCapGeometry | None = None,
    config_path: Path | str = DEFAULT_CONFIG,
    position_tolerance_m: float = 1e-5,
    orientation_tolerance_rad: float = 1e-3,
) -> ComposedManipulatorKinematics:
    """Build offline TCP FK/IK using only the official flange SDK adapter.

    Configuration order is [joint_1, ..., joint_7, gripper], J1..J7 radians
    followed by normalized aperture (0 closed, 1 open). Names are local to this
    single-arm instance. IK requires fixed_coordinates={"gripper": aperture}.
    Poses are in the selected arm's own base, not a shared world/base frame.
    Default frame names are tianji_<arm>_base and tianji_<arm>_taccap_tcp.

    Pass T_flange_tool_base explicitly, e.g. umi_follower_mount() for the
    existing CAD installation. The SDK adapter clears its native tool offset;
    mount and TCP are applied exactly once by the composition. Construction
    loads libKine only; it does not connect hardware or import the TacCap SDK.
    """
    if arm not in {"left", "right"}:
        raise ValueError("arm must be 'left' or 'right'")
    geometry = (
        tool
        if tool is not None
        else TacCapGeometry(
            base_frame=f"tianji_{arm}_taccap_base",
            tcp_frame=f"tianji_{arm}_taccap_tcp",
        )
    )
    flange = TianjiSDKKinematics(
        arm,
        config_path=config_path,
        position_tolerance_m=position_tolerance_m,
        orientation_tolerance_rad=orientation_tolerance_rad,
    )
    return ComposedManipulatorKinematics(
        flange,
        geometry,
        arm_coordinates=tuple(KinematicCoordinate(f"joint_{i}", "rad") for i in range(1, 8)),
        mount=mount,
        base_frame=f"tianji_{arm}_base",
        position_tolerance=position_tolerance_m,
        rotation_tolerance=orientation_tolerance_rad,
    )
