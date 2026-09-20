"""Multi-camera ZMQ server, its client, and the ManiMux SensorDriver plugin.

One server process owns the camera devices; every consumer (a ManiMux run,
a preview window, a legacy launcher) reads coherent frame bundles over ZMQ.
Nothing here is policy-specific.
"""

from manimux.server.sensor.taccap.client import CameraClient, CameraSubscriber
from manimux.server.sensor.taccap.driver import CameraServerSensorDriver, build_sensor

__all__ = [
    "CameraClient",
    "CameraServerSensorDriver",
    "CameraSubscriber",
    "build_sensor",
]
