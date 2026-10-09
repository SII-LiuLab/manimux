"""Xiaomi-specific identity checks over the shared Tianji EE decoder."""

from manimux.policy_adapter.tianji_ee import (
    ACTION_SEMANTICS,
    OBSERVATION_PROFILE,
    TianjiAbsoluteEEAdapter,
    TianjiEERequest,
)

# Preserve the existing public request type/import.
XR1TianjiRequest = TianjiEERequest


class XR1TianjiTacCapAdapter(TianjiAbsoluteEEAdapter):
    def validate(self, robot: dict, policy: dict) -> None:
        super().validate(robot, policy)
        if self._horizon_steps != 30:
            raise ValueError("the pass-ball XR-1 checkpoint requires a 30-step horizon")
        expected = policy.get("expected_backend") or {}
        identity = expected.get("model", {})
        required = {
            "policy_name": "Xiaomi_Robotics_1",
            "observation_profile": OBSERVATION_PROFILE,
            "output_format": "xpolicylab",
            "ego_view_mode": "black",
            "action_semantics": ACTION_SEMANTICS,
        }
        mismatches = {
            name: (identity.get(name), value)
            for name, value in required.items()
            if identity.get(name) != value
        }
        if mismatches:
            raise ValueError(f"XR-1 Tianji backend identity mismatch: {mismatches}")
