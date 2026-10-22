"""Assemble one Franka arm and optional scalar gripper through RobotBase."""

from manimux.clock import SystemClock
from manimux.embodiments.arm.franka import FrankaController
from manimux.embodiments.robot.base import RobotBase, RobotModel


class FrankaRobot(RobotBase):
    def __init__(
        self,
        model,
        *,
        clock=None,
        hardware=None,
        component_hardware=None,
        execute=False,
        home_on_close=False,
        move_to_start_on_connect=False,
    ):
        if home_on_close is not False or move_to_start_on_connect is not False:
            raise ValueError("Franka startup/Home motion is not configured")
        if len(model.groups) != 1:
            raise ValueError("Franka assembly currently supports one arm")
        clock = clock or SystemClock()
        bindings = component_hardware or {}
        arms, tools = {}, {}
        for name, group in model.groups.items():
            component = model.components[group.arm_name]
            arm_kinematics = group.arm.kinematics
            controller = FrankaController(
                joint_limits=arm_kinematics.bounds,
                clock=clock,
                execute=execute,
                **{**component["hardware"], **bindings.get(group.arm_name, {})},
            )
            arms[name] = component["class"](controller, kinematics=arm_kinematics)
            if group.end_effector is not None:
                tool = model.components[group.end_effector_name]
                tools[name] = tool["class"](
                    clock=clock,
                    execute=execute,
                    **{**tool["hardware"], **bindings.get(group.end_effector_name, {})},
                )
        settings = {**model.hardware, **(hardware or {})}
        unknown = set(settings) - {"stale_timeout_s"}
        if unknown:
            raise ValueError(f"unsupported Franka assembly hardware options: {sorted(unknown)}")
        super().__init__(
            arm_components=arms,
            end_effectors=tools,
            models={name: group.kinematics for name, group in model.groups.items()},
            model=model,
            clock=clock,
            execute=execute,
            stale_timeout_s=settings.get("stale_timeout_s", 0.2),
        )

    @classmethod
    def from_config(cls, path, **options):
        return cls(RobotModel.from_config(path), **options)


def build_robot(config, clock):
    robot = FrankaRobot.from_config(config["config"], clock=clock, **config.get("options", {}))
    if config["group_dims"] != {
        name: model.num_coordinates for name, model in robot.kinematics.models.items()
    }:
        raise ValueError("group_dims must match the Franka assembly")
    return robot
