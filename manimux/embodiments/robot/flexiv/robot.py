"""Two independently owned Flexiv Rizon4s controllers in the common runtime."""

import numpy as np

from manimux.clock import SystemClock
from manimux.embodiments.arm.flexiv import FlexivController
from manimux.embodiments.robot.base import RobotBase, RobotModel


class FlexivRobot(RobotBase):
    def __init__(self, model, *, clock=None, hardware=None, component_hardware=None, execute=False):
        clock = clock if clock is not None else SystemClock()
        bindings = component_hardware or {}
        arms = {}
        serials = set()
        if any(c["type"] != "arm" for c in model.components.values()):
            raise ValueError("Flexiv assembly contains arms only; use external camera clients")
        for name, group in model.groups.items():
            if group.end_effector is not None:
                raise ValueError(
                    "Flexiv gripper driver is not configured for this bare-arm assembly"
                )
            component = model.components[group.arm_name]
            controller = FlexivController(
                clock=clock,
                execute=execute,
                **{**component["hardware"], **bindings.get(group.arm_name, {})},
            )
            if controller.channel in serials:
                raise ValueError("Flexiv arms must have distinct robot serials")
            serials.add(controller.channel)
            arms[name] = component["class"](controller, kinematics=group.kinematics)
        settings = {**model.hardware, **(hardware or {})}
        super().__init__(
            arm_components=arms,
            model=model,
            models={name: group.kinematics for name, group in model.groups.items()},
            clock=clock,
            execute=execute,
            stale_timeout_s=settings.get("stale_timeout_s", 0.2),
        )

    @classmethod
    def from_config(cls, path, **options):
        return cls(RobotModel.from_config(path), **options)


def build_robot(config, clock):
    hz = config.get("control_hz", 50)
    if not np.isfinite(hz) or not 1 <= hz <= 100:
        raise ValueError("Flexiv NRT control_hz must be within 1..100 Hz")
    robot = FlexivRobot.from_config(config["config"], clock=clock, **config.get("options", {}))
    if config["group_dims"] != {
        name: model.num_coordinates for name, model in robot.kinematics.models.items()
    }:
        raise ValueError("group_dims must match the Flexiv assembly")
    return robot
