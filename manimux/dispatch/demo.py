"""Synthetic robot for the hardware-free dispatch example; no SDK or devices."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from manimux.clock import Clock
from manimux.kinematics.base import IKResult, KinematicCoordinate, ManipulatorKinematicsBase
from manimux.kinematics.robot import RobotKinematics
from manimux.policies.fake import FakePolicyModel
from manimux.types import RobotCommand, RobotState, copy_group_vector


class DemoKinematics(ManipulatorKinematicsBase):
    """Analytic synthetic fixture: the first six coordinates encode a TCP pose.

    This is not a physical arm model. It exists only for the disconnected tool
    demonstration and must never be used as calibration or deployment geometry.
    """

    def __init__(self, name, width):
        self.name, self.width = name, width

    @property
    def coordinates(self):
        return tuple(KinematicCoordinate(f"joint{i}", "rad") for i in range(self.width - 1)) + (
            KinematicCoordinate("gripper", "normalized"),
        )

    @property
    def base_frame(self):
        return f"synthetic_{self.name}_base"

    @property
    def tcp_frame(self):
        return f"synthetic_{self.name}_tcp"

    def fk(self, configuration):
        pose = np.eye(4)
        pose[:3, 3] = configuration[:3]
        pose[:3, :3] = Rotation.from_euler("xyz", configuration[3:6]).as_matrix()
        return pose

    def ik(self, target_tcp, seed_configuration, *, fixed_coordinates):
        q = np.array(seed_configuration, copy=True)
        q[:3], q[3:6] = target_tcp[:3, 3], Rotation.from_matrix(target_tcp[:3, :3]).as_euler("xyz")
        q[-1] = fixed_coordinates["gripper"]
        return IKResult(True, q)


class DemoRobot:
    model = None
    recovery_actions: tuple[str, ...] = ()

    def __init__(self, config: dict[str, Any], clock: Clock) -> None:
        self.clock = clock
        self.groups = {name: np.zeros(width) for name, width in config["group_dims"].items()}
        self.kinematics = RobotKinematics(
            {name: DemoKinematics(name, width) for name, width in config["group_dims"].items()}
        )
        self.sequence = 0

    def connect(self) -> None:
        pass

    def get_state(self) -> RobotState:
        self.sequence += 1
        return RobotState(copy_group_vector(self.groups), self.clock.now_ns(), self.sequence)

    def send_command(self, command: RobotCommand) -> None:
        self.groups = copy_group_vector(command.groups)

    def runtime_metadata(self) -> dict[str, object]:
        return {"backend": "dispatch_demo", "physical_dispatch": False}

    def stop(self) -> None:
        pass

    def close(self) -> None:
        self.stop()


def build_robot(config: dict[str, Any], clock: Clock) -> DemoRobot:
    return DemoRobot(config, clock)


class ToolDemoPolicy(FakePolicyModel):
    """Synthetic motion predictions with an unchanged normalized gripper target."""

    def infer(self, request):
        chunk = super().infer(request)
        for name, values in chunk.groups.items():
            values[:, -1] = request.observation.state.groups[name][-1]
        return chunk


def build_tool_policy(config):
    return ToolDemoPolicy(
        int(config["action_dt_s"] * 1e9),
        config["horizon_policy_steps"],
        config.get("inference_delay_s", 0.01),
    )
