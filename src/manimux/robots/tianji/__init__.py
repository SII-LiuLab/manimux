"""Tianji adapters built directly on the bundled official SDK."""

from manimux.robots.tianji.kinematics import TianjiSDKKinematics
from manimux.robots.tianji.tianji import TianjiArmConfig, TianjiRobot

__all__ = [
    "TianjiArmConfig",
    "TianjiRobot",
    "TianjiSDKKinematics",
]
