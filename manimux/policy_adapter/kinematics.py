"""Pose adapters use the configured robot model, including in decoder processes."""

import numpy as np

from manimux.policy_adapter.base import PolicyAdapter


class KinematicAdapter(PolicyAdapter):
    """Common access for pose adapters with a trailing scalar tool coordinate."""

    def __init__(self, robot, policy, *, kinematics=None):
        if kinematics is None:
            from manimux.embodiments.robot.base import RobotModel

            kinematics = RobotModel.from_config(robot["config"]).kinematics
        super().__init__(robot, policy, kinematics=kinematics)
        if "kinematics" in policy["adapter"] or "kinematics_options" in policy["adapter"]:
            raise ValueError("kinematics belongs in robot.config")

    def _fk(self, group, joints, gripper):
        return self.kinematics.models[group].fk(np.r_[joints, gripper])

    def _ik(self, group, target, joints, gripper):
        model = self.kinematics.models[group]
        result = model.ik(
            target,
            np.r_[joints, gripper],
            fixed_coordinates={model.coordinates[-1].name: gripper},
        )
        # Existing adapters retain their distinct hold/reject policies on failure.
        solved = result.joints[: len(joints)] if result.converged else np.asarray(joints).copy()
        return result.converged, solved
