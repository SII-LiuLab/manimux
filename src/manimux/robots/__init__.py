from __future__ import annotations

from collections.abc import Callable

from manimux.clock import Clock
from manimux.config import RobotConfig
from manimux.plugins import load_plugin
from manimux.robots.base import RobotBase, RobotDriver
from manimux.robots.maniunicon import ManiUniConMeshcatDualArmDriver
from manimux.robots.mock import MockDualArmDriver

RobotFactory = Callable[[RobotConfig, Clock], RobotDriver]


def _mock_factory(config: RobotConfig, clock: Clock) -> RobotDriver:
    return MockDualArmDriver(config.group_dims, clock)


def _maniunicon_factory(config: RobotConfig, clock: Clock) -> RobotDriver:
    if config.config is None:
        raise ValueError("maniunicon_meshcat_dual_arm requires robot.config")
    return ManiUniConMeshcatDualArmDriver.from_config_file(
        config.config,
        config.group_dims,
        clock,
    )


def _yam_factory(config: RobotConfig, clock: Clock) -> RobotDriver:
    from manimux.robots.yam import build_robot as build_yam_robot

    return build_yam_robot(config, clock)


def _tianji_factory(config: RobotConfig, clock: Clock) -> RobotDriver:
    from manimux.robots.tianji import build_robot as build_tianji_robot

    return build_tianji_robot(config, clock)


_BUILTINS: dict[str, RobotFactory] = {
    "mock_dual_arm": _mock_factory,
    "maniunicon_meshcat_dual_arm": _maniunicon_factory,
    "tianji_dual": _tianji_factory,
    "yam_dual": _yam_factory,
}


def build_robot(config: RobotConfig, clock: Clock) -> RobotDriver:
    factory = load_plugin(
        config.driver,
        group="manimux.robots",
        builtins=_BUILTINS,
    )
    return factory(config, clock)


__all__ = [
    "ManiUniConMeshcatDualArmDriver",
    "MockDualArmDriver",
    "RobotBase",
    "RobotDriver",
    "RobotFactory",
    "build_robot",
]
