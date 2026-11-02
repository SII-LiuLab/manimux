"""Assemble independently owned CAN arm sessions using the common RobotBase."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from manimux.clock import Clock, SystemClock
from manimux.embodiments.robot.base import RobotBase, RobotModel
from manimux.plugins import load_plugin
from manimux.types import RobotCommand


class CanArmRobot(RobotBase):
    """Component YAML selects each controller; the assembly never calls vendor APIs."""

    def __init__(
        self,
        model: RobotModel,
        *,
        clock: Clock | None = None,
        hardware: Mapping[str, Any] | None = None,
        component_hardware: Mapping[str, Mapping[str, Any]] | None = None,
        execute: bool = False,
    ) -> None:
        if execute and set(model.home_joints) != set(model.groups):
            raise ValueError("executing CAN arms requires home.joints_deg for every group in the robot assembly")
        if any(component["type"] != "arm" for component in model.components.values()):
            raise ValueError(
                "CAN arm assembly currently contains arms only; cameras use sensor clients"
            )
        clock = clock if clock is not None else SystemClock()
        bindings = component_hardware or {}
        settings = {**model.hardware, **(hardware or {})}
        self._home_motion = {
            "control_hz": 50.0, "peak_velocity_deg_s": 10.0,
            "tolerance_rad": 0.05, "speed_rad_s": 0.1, "settle_s": 0.25,
            **settings.get("home_motion", {}),
        }
        if set(self._home_motion) != {"control_hz", "peak_velocity_deg_s", "tolerance_rad", "speed_rad_s", "settle_s"}:
            raise ValueError("unknown hardware.home_motion option")
        if any(not math.isfinite(v) or v <= 0 for v in self._home_motion.values()):
            raise ValueError("hardware.home_motion values must be finite and positive")
        arms = {}
        channels = set()
        for name, group in model.groups.items():
            if group.end_effector is not None:
                raise ValueError("CAN arm assembly requires an integrated scalar gripper")
            component = model.components[group.arm_name]
            options = {
                **component["hardware"],
                **bindings.get(group.arm_name, {}),
            }
            controller_type = load_plugin(
                component["controller"], group="manimux.controllers", builtins={}
            )
            controller = controller_type(clock=clock, execute=execute, **options)
            if name in model.home_joints:
                controller.configure_home(
                    model.home_joints[name],
                    **{key: self._home_motion[key] for key in ("tolerance_rad", "speed_rad_s", "settle_s")},
                )
            if controller.channel in channels:
                raise ValueError("independent arm components must have distinct CAN channels")
            channels.add(controller.channel)
            arms[name] = component["class"](controller, kinematics=group.kinematics)
        super().__init__(
            arm_components=arms,
            models={name: group.kinematics for name, group in model.groups.items()},
            model=model,
            clock=clock,
            stale_timeout_s=settings.get("stale_timeout_s", 0.2),
            execute=execute,
        )

    def home_targets(self, state):
        if not self.model.home_joints:
            return None
        targets = {name: q.copy() for name, q in state.groups.items()}
        for name, joints in self.model.home_joints.items():
            targets[name][:6] = joints
        return targets

    def home(self) -> None:
        """Return to calibrated arm Home while retaining measured gripper opening.

        Position commands follow a bounded-speed cosine profile. Arrival uses
        measured feedback continuously; it has no elapsed-time failure deadline.
        Device readiness failures propagate to the shared GUI release gate.
        """
        if not self._execute:
            return
        with self._lock:
            for controller in self._controllers:
                controller.validate_motion_ready()
            state = self.get_state()
            targets = self.home_targets(state)
            if targets is None:
                raise ValueError("Home joints are not configured")
            start = state.groups
            distance = max(float(np.max(np.abs(targets[n][:6] - start[n][:6]))) for n in targets)
            duration = math.pi * distance / (2 * math.radians(self._home_motion["peak_velocity_deg_s"]))
            period = 1.0 / self._home_motion["control_hz"]
            ticks = max(1, math.ceil(duration / period))
            for tick in range(ticks + 1):
                fraction = (1 - math.cos(math.pi * tick / ticks)) / 2
                groups = {name: start[name] + fraction * (q - start[name]) for name, q in targets.items()}
                # Check every arm before issuing the first command in this tick.
                for controller in self._controllers:
                    controller.validate_motion_ready()
                self.send_command(RobotCommand(groups, self._clock.now_ns(), None))
                time.sleep(period)
            while True:
                pending = False
                for controller in self._controllers:
                    controller.validate_motion_ready()
                    # Invalid position/stability is a wait, not a hardware fault.
                    monitor = controller._home_monitor
                    pending |= not monitor.observe(controller.get_states()[controller.channel])
                if not pending:
                    break
                time.sleep(period)

    @classmethod
    def from_config(cls, path: Path | str, **options: Any) -> CanArmRobot:
        return cls(RobotModel.from_config(path), **options)


def build_robot(config: Mapping[str, Any], clock: Clock) -> CanArmRobot:
    robot = CanArmRobot.from_config(config["config"], clock=clock, **config.get("options", {}))
    if config["group_dims"] != {
        name: model.num_coordinates for name, model in robot.kinematics.models.items()
    }:
        raise ValueError("group_dims must match the selected CAN arm assembly")
    return robot
