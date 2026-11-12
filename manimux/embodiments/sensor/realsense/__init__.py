"""RealSense RGB/depth component; the installed pyrealsense2 provides the device SDK."""

from .sensor import RealSenseSensor, get_device_ids

__all__ = ["RealSenseSensor", "get_device_ids"]
