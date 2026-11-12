"""Shared assembly factory for ARX X5 and standard PiPER components."""

from .robot import CanArmRobot, build_robot

__all__ = ["CanArmRobot", "build_robot"]
