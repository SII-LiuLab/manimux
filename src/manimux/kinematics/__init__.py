"""Arm kinematics used by policies that speak end-effector poses.

Kept out of any single integration: an embodiment implements FK/IK once and
every pose-space policy reuses it.
"""

from manimux.kinematics.base import (
    ArmKinematicsBase,
    Frame,
    IKResult,
    KinematicCoordinate,
    transform_from_xyz_rpy,
)
from manimux.kinematics.composed import (
    ComposedManipulatorKinematics,
    FixedToolGeometry,
    ManipulatorKinematicsBase,
    RobotKinematics,
    ToolGeometryBase,
)
from manimux.plugins import load_plugin

_BUILTINS: dict[str, object] = {}


def build_kinematics(name: str, **options: object) -> object:
    """Compatibility factory for integrations not yet using embodiment models."""
    factory = load_plugin(name, group="manimux.kinematics", builtins=_BUILTINS)
    return factory(**options)


__all__ = [
    "ArmKinematicsBase",
    "ComposedManipulatorKinematics",
    "Frame",
    "FixedToolGeometry",
    "IKResult",
    "KinematicCoordinate",
    "ManipulatorKinematicsBase",
    "RobotKinematics",
    "ToolGeometryBase",
    "build_kinematics",
    "transform_from_xyz_rpy",
]
