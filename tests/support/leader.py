"""Synthetic leader for collection tests; no public simulation mode."""

import math

import numpy as np

from manimux.collection.yam.robot.interface import RobotInterface, TeleopAgent


class SyntheticLeader(TeleopAgent):
    """Generates a deterministic sinusoidal command and applies it to a robot."""

    def __init__(self, robot: RobotInterface, num_arm_joints: int = 6):
        self._robot = robot
        self._n = num_arm_joints
        self._t = 0

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        phase = self._t * 0.05
        arm = 0.3 * np.sin(phase + np.arange(self._n) * 0.5)
        gripper = 0.5 * (1.0 + math.sin(phase))  # sweeps [0, 1]
        cmd = np.concatenate([arm, [gripper]]).astype(np.float64)
        self._robot.command_joint_pos(cmd)
        self._t += 1
        return cmd

    def engage(self, abort=None) -> None:
        # No physical leader to sync to; the synthesized trajectory starts at rest.
        pass

    def read_inputs(self) -> tuple[list[bool], float] | None:
        # The mock controller has no physical buttons; sync/record are GUI-driven.
        return None

    def leader_raw(self) -> None:
        # No encoders to read raw angles from; the trajectory is synthesized.
        return None

    def stop(self) -> None:
        self._robot.stop()
