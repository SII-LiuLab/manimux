"""Pi05 zero-pose observations over the shared Tianji absolute EE decoder."""

import numpy as np

from manimux.policy_adapter.tianji_ee import TianjiAbsoluteEEAdapter


class Pi05TianjiTacCapAdapter(TianjiAbsoluteEEAdapter):
    def validate(self, robot: dict, policy: dict) -> None:
        super().validate(robot, policy)
        if self._horizon_steps != 32:
            raise ValueError("Pi05 pass-ball requires a 32-step horizon")
        if abs(self._action_dt_s - 1 / 30) > 1e-9:
            raise ValueError("Pi05 pass-ball requires a 30 Hz action interval")
        identity = (policy.get("expected_backend") or {}).get("model", {})
        required = {
            "policy_name": "Pi_05",
            "observation_profile": "tianji_taccap_pi05_zero_pose",
            "model_state_encoding": "zero_pose",
            "output_format": "xpolicylab",
            "action_semantics": "absolute_per_arm_base_xyz_wxyz",
            "action_horizon": 32,
        }
        if any(identity.get(key) != value for key, value in required.items()):
            raise ValueError("Pi05 Tianji backend identity mismatch")

    def prepare_request(self, request):
        if getattr(request, "action_condition", None) is not None:
            raise ValueError("Pi05 zero-pose profile currently supports default sampling only")
        return super().prepare_request(request)


class Pi05PackPlateTacCapAdapter(TianjiAbsoluteEEAdapter):
    """Pack-plate identity gate over the same Tianji absolute EE decoder."""

    def validate(self, robot: dict, policy: dict) -> None:
        super().validate(robot, policy)
        if self._horizon_steps != 32 or abs(self._action_dt_s - 1 / 30) > 1e-9:
            raise ValueError("Pi05 pack-plate requires 32 steps at 30 Hz")
        identity = (policy.get("expected_backend") or {}).get("model", {})
        required = {
            "policy_name": "Pi_05",
            "task_name": "plate",
            "checkpoint_variant": "pi05_pack_plate_wrist_only_step59999",
            "observation_profile": "tianji_taccap_pi05_pack_plate",
            "model_state_encoding": "zero_pose",
            "output_format": "xpolicylab",
            "action_semantics": "absolute_per_arm_base_xyz_wxyz",
            "action_horizon": 32,
        }
        if any(identity.get(key) != value for key, value in required.items()):
            raise ValueError("Pi05 pack-plate backend identity mismatch")

    def _absolute_condition(self, condition, weights):
        poses, weights = super()._absolute_condition(condition, weights)
        active = weights > 0
        deadzone = self._gripper_output_deadzone
        exponent = self._gripper_output_exponent
        for index in range(7, poses.shape[1], 8):
            opening = poses[active, index]
            if np.any((opening < 0) | (opening > 1)):
                raise ValueError("Pi05 RTC gripper condition must be in [0,1]")
            poses[active, index] = np.where(
                opening == 0,
                0.0,
                deadzone + (1.0 - deadzone) * opening ** (1.0 / exponent),
            )
        return poses, weights
