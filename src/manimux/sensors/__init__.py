from __future__ import annotations

from collections.abc import Callable

from manimux.clock import Clock
from manimux.plugins import load_plugin
from manimux.sensors.base import SensorDriver
from manimux.sensors.mock import MockCameraDriver

SensorFactory = Callable[[dict, Clock], SensorDriver]


def _mock_camera_factory(config: dict, clock: Clock) -> SensorDriver:
    return MockCameraDriver(config["name"], config["width"], config["height"], clock)


_BUILTINS: dict[str, SensorFactory | str] = {
    "mock_camera": _mock_camera_factory,
    "camera_server": "manimux.sensors.camera_server:build_sensor",
}


def build_sensor(config: dict, clock: Clock) -> SensorDriver:
    factory = load_plugin(
        config["driver"],
        group="manimux.sensors",
        builtins=_BUILTINS,
    )
    return factory(config, clock)


__all__ = [
    "MockCameraDriver",
    "SensorDriver",
    "SensorFactory",
    "build_sensor",
]


def sensor_parameters(**options) -> dict:
    """补齐数据源默认采集参数；不启动传感器。"""

    values = {
        "service": None,
        "width": 64,
        "height": 48,
        "fps": 30.0,
        "options": {},
        **options,
    }
    return values
