"""传感器组件与运行时数据源入口；构造工厂不启动设备。"""

from __future__ import annotations

from collections.abc import Callable

from manimux.clock import Clock
from manimux.embodiments.sensor.base import SensorBase, SensorRead
from manimux.plugins import load_plugin

SensorFactory = Callable[[dict, Clock], SensorBase]

_BUILTINS: dict[str, SensorFactory | str] = {
    "camera_server": "manimux.embodiments.sensor.camera_server:build_sensor",
}


def build_sensor(config: dict, clock: Clock) -> SensorBase:
    """按配置构造数据源；网络客户端与设备组件共用 start/read/close。"""
    factory = load_plugin(
        config["driver"], group="manimux.embodiments.sensor", builtins=_BUILTINS
    )
    return factory(config, clock)


def sensor_parameters(**options) -> dict:
    """补齐数据源默认参数；不打开相机或连接网络服务。"""
    return {
        "service": None,
        "width": 64,
        "height": 48,
        "fps": 30.0,
        "options": {},
        **options,
    }


__all__ = ["SensorBase", "SensorRead", "SensorFactory", "build_sensor", "sensor_parameters"]
