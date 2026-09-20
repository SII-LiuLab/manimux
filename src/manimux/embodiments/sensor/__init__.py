"""Sensor implementations, independent of robot command dispatch."""

from collections.abc import Callable

from manimux.clock import Clock
from manimux.embodiments.sensor.base import SensorBase
from manimux.embodiments.sensor.mock import MockCameraDriver
from manimux.plugins import load_plugin

SensorFactory = Callable[[dict, Clock], SensorBase]


def _mock_camera_factory(config: dict, clock: Clock) -> SensorBase:
    return MockCameraDriver(config["name"], config["width"], config["height"], clock)


_BUILTINS: dict[str, SensorFactory | str] = {
    "mock_camera": _mock_camera_factory,
    "camera_server": "manimux.server.sensor.taccap:build_sensor",
}


def build_sensor(config: dict, clock: Clock) -> SensorBase:
    factory = load_plugin(config["driver"], group="manimux.sensors", builtins=_BUILTINS)
    return factory(config, clock)


def sensor_parameters(**options) -> dict:
    """Fill data-source settings without opening a device or network connection."""
    return {
        "service": None,
        "width": 64,
        "height": 48,
        "fps": 30.0,
        "options": {},
        **options,
    }


__all__ = ["SensorBase", "SensorFactory", "MockCameraDriver", "build_sensor", "sensor_parameters"]
