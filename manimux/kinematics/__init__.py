"""Arm kinematics used by policies that speak end-effector poses.

Kept out of any single integration: an embodiment implements FK/IK once and
every pose-space policy reuses it.
"""

from collections.abc import Callable

from manimux.kinematics.base import (
    ArmKinematics,
    FlangeKinematicsBase,
    IKResult,
    KinematicCoordinate,
    ManipulatorKinematicsBase,
)
from manimux.kinematics.composed import ComposedManipulatorKinematics
from manimux.kinematics.robot import RobotKinematics
from manimux.kinematics.tool import FixedToolGeometry, ToolGeometryBase
from manimux.plugins import load_plugin

_BUILTINS: dict[str, Callable[..., ArmKinematics] | str] = {
    "tianji": "manimux.kinematics.tianji:TianjiKinematics",
    "yam": "manimux.embodiments.arm.yam.kinematics:YamKinematics",
}


def build_kinematics(name: str, **options: object) -> ArmKinematics:
    factory = load_plugin(name, group="manimux.kinematics", builtins=_BUILTINS)
    return factory(**options)


__all__ = [
    "ArmKinematics",
    "ComposedManipulatorKinematics",
    "FlangeKinematicsBase",
    "FixedToolGeometry",
    "IKResult",
    "KinematicCoordinate",
    "ManipulatorKinematicsBase",
    "RobotKinematics",
    "ToolGeometryBase",
    "build_kinematics",
]
