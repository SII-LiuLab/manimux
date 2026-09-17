"""Hardware interfaces for independently controlled end effectors."""

from manimux.end_effectors.base import EndEffectorBase
from manimux.end_effectors.gripper import GripperBase, GripperCommand, GripperState

__all__ = ["EndEffectorBase", "GripperBase", "GripperCommand", "GripperState"]
