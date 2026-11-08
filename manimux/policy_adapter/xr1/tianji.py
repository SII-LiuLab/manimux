"""XR-1 contracts for the Tianji TacCap absolute TCP adapter."""

from __future__ import annotations

from dataclasses import dataclass

from manimux.policy_adapter.tianji import (
    ACTION_SEMANTICS,
    TianjiAbsoluteEEAdapter,
    TianjiEERequest,
)

OBSERVATION_PROFILE = "tianji_taccap_two_wrist_black_ego"


@dataclass(slots=True)
class XR1TianjiRequest(TianjiEERequest):
    """XR-1 request carrying the assembled Tianji TCP state."""


class XR1TianjiTacCapAdapter(TianjiAbsoluteEEAdapter):
    _model_name = "XR-1"
    _plan_id_prefix = "xr1-tianji"
    _request_type = XR1TianjiRequest

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
