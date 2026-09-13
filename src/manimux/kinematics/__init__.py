"""Arm kinematics used by policies that speak end-effector poses.

Kept out of any single integration: an embodiment implements FK/IK once and
every pose-space policy reuses it.
"""

from collections.abc import Callable

from manimux.kinematics.base import ArmKinematics
from manimux.plugins import load_plugin


def _yam_factory(**options: object) -> ArmKinematics:
    from manimux.kinematics.yam import YamKinematics

    return YamKinematics(**options)  # type: ignore[arg-type]


def _tianji_factory(**options: object) -> ArmKinematics:
    from manimux.kinematics.tianji import TianjiKinematics

    return TianjiKinematics(**options)  # type: ignore[arg-type]


_BUILTINS: dict[str, Callable[..., ArmKinematics]] = {
    "tianji": _tianji_factory,
    "yam": _yam_factory,
}


def build_kinematics(name: str, **options: object) -> ArmKinematics:
    factory = load_plugin(name, group="manimux.kinematics", builtins=_BUILTINS)
    return factory(**options)


__all__ = ["ArmKinematics", "build_kinematics"]
