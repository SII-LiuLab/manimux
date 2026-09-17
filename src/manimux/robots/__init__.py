"""Compatibility imports; new robot construction lives in embodiments.robot."""

from manimux.embodiments.robot import RobotBase, RobotFactory, build_robot
from manimux.robots.base import RobotDriver
from manimux.robots.maniunicon import ManiUniConMeshcatDualArmDriver
from manimux.robots.mock import MockDualArmDriver

__all__ = [
    "ManiUniConMeshcatDualArmDriver", "MockDualArmDriver", "RobotBase",
    "RobotDriver", "RobotFactory", "build_robot",
]
