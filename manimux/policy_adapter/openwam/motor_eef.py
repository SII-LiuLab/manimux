"""Keep the model EEF contract while the runtime publishes motor positions."""

from copy import deepcopy

from manimux.policy_adapter.openwam.native_eef import OpenWAMNativeEEFAdapter


class OpenWAMMotorEEFAdapter(OpenWAMNativeEEFAdapter):
    def validate(self, robot, policy):
        if robot["options"].get("command_action_space") != "joint_position":
            raise ValueError("runtime motor control requires joint_position commands")
        reference = deepcopy(robot)
        reference["options"]["command_action_space"] = "eef_pose"
        super().validate(reference, policy)
