"""RealSense RGB/depth 组件；安装的 pyrealsense2 提供设备 SDK。"""

from .sensor import RealSenseSensor, get_device_ids

__all__ = ["RealSenseSensor", "get_device_ids"]
