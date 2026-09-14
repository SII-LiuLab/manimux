"""TacCap wrist camera driver (XenseRobotics, V4L2 through the TacCap SDK).

A plain camera driver with no robot or policy coupling. ``xense.taccap`` is
imported only when a camera is opened.
"""

from manimux.sensors.taccap.camera import TacCapCamera, find_camera_device

__all__ = ["TacCapCamera", "find_camera_device"]
