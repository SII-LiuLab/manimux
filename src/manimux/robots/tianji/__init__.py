"""Tianji Marvin dual-arm robot driver.

Importing this package loads no vendor SDK; the Marvin and TacCap SDKs are
loaded when the driver connects.
"""

from manimux.robots.tianji.driver import TianjiDualArmDriver, build_robot

__all__ = ["TianjiDualArmDriver", "build_robot"]
