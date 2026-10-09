"""Pi05 zero-pose observations over the shared Tianji absolute EE decoder."""

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

    _CHECKPOINT_VARIANTS = {
        "pi05_pack_plate_wrist_only_step59999",
        "pi05_pack_plate_full_zero_state_pack_instruction_step59999",
    }

    def validate(self, robot: dict, policy: dict) -> None:
        super().validate(robot, policy)
        if self._horizon_steps != 32 or abs(self._action_dt_s - 1 / 30) > 1e-9:
            raise ValueError("Pi05 pack-plate requires 32 steps at 30 Hz")
        identity = (policy.get("expected_backend") or {}).get("model", {})
        required = {
            "policy_name": "Pi_05",
            "task_name": "plate",
            "observation_profile": "tianji_taccap_pi05_pack_plate",
            "model_state_encoding": "zero_pose",
            "output_format": "xpolicylab",
            "action_semantics": "absolute_per_arm_base_xyz_wxyz",
            "action_horizon": 32,
        }
        if (
            any(identity.get(key) != value for key, value in required.items())
            or identity.get("checkpoint_variant") not in self._CHECKPOINT_VARIANTS
        ):
            raise ValueError("Pi05 pack-plate backend identity mismatch")
