"""传感器组件与运行时数据源入口；构造工厂不启动设备。"""

from __future__ import annotations

from collections.abc import Callable

from manimux.clock import Clock, SystemClock
from manimux.embodiments.sensor.base import SensorBase, SensorRead
from manimux.plugins import load_plugin

SensorFactory = Callable[[dict, Clock], SensorBase]

_BUILTINS: dict[str, SensorFactory | str] = {
    "camera_server": "manimux.embodiments.sensor.camera_server:build_sensor",
}

_CAMERAS = {
    "realsense": "manimux.embodiments.sensor.realsense:RealSenseSensor",
    "orbbec": "manimux.embodiments.sensor.orbbec:OrbbecSensor",
    "taccap": "manimux.embodiments.sensor.taccap:TacCapSensor",
}


def build_camera(name: str, spec: dict, clock: Clock | None = None) -> SensorBase:
    """Construct an inert camera component; assembly implementation takes precedence.

    Camera constructors accept name, camera_serial and clock as keywords. Legacy
    standalone configurations may still spell camera_serial as device_id.
    """
    options = dict(spec)
    kind = options.pop("type", "realsense")
    implementation = options.pop("implementation", kind)
    if "device_id" in options:
        if "camera_serial" in options:
            raise ValueError("Use camera_serial or device_id, not both")
        options["camera_serial"] = options.pop("device_id")
    component = load_plugin(
        implementation, group="manimux.embodiments.camera", builtins=_CAMERAS
    )
    return component(name=name, clock=clock or SystemClock(), **options)


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


__all__ = ["SensorBase", "SensorRead", "SensorFactory", "build_sensor", "build_camera", "sensor_parameters"]
