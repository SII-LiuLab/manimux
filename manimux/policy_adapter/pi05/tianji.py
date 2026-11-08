"""Pi05 contracts for the Tianji TacCap absolute TCP adapter."""

from __future__ import annotations

from dataclasses import dataclass

from manimux.policy_adapter.base import require_request_type
from manimux.policy_adapter.tianji import (
    ACTION_SEMANTICS,
    TianjiAbsoluteEEAdapter,
    TianjiEERequest,
)
from manimux.runtime.rtc.request import RtcInferenceRequest
from manimux.types import InferenceRequest


@dataclass(slots=True)
class Pi05TianjiRequest(TianjiEERequest):
    """Pi05 request carrying the assembled Tianji TCP state."""


class _Pi05TianjiEEAdapter(TianjiAbsoluteEEAdapter):
    _model_name = "Pi05"
    _plan_id_prefix = "pi05-tianji"
    _request_type = Pi05TianjiRequest

    def prepare_request(self, request: InferenceRequest) -> TianjiEERequest:
        require_request_type(request, (InferenceRequest, RtcInferenceRequest), "Pi05 Tianji")
        return super().prepare_request(request)


class Pi05TianjiTacCapAdapter(_Pi05TianjiEEAdapter):
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
            "action_semantics": ACTION_SEMANTICS,
            "action_horizon": 32,
        }
        if any(identity.get(key) != value for key, value in required.items()):
            raise ValueError("Pi05 Tianji backend identity mismatch")

    def prepare_request(self, request):
        if getattr(request, "action_condition", None) is not None:
            raise ValueError("Pi05 zero-pose profile currently supports default sampling only")
        return super().prepare_request(request)


class Pi05PackPlateTacCapAdapter(_Pi05TianjiEEAdapter):
    """Pack-plate identity gate over the same Tianji absolute EE decoder."""

    _CHECKPOINT_VARIANTS = {
        "pi05_pack_plate_wrist_only_step59999": "zero_pose",
        "pi05_pack_plate_full_zero_state_pack_instruction_step59999": "zero_pose",
        "pi05_pack_plate_full_zero_state_pack_instruction_step20000": "zero_pose",
        "pi05_pack_plate_inter_hand_relative_step20000": "inter_hand_relative",
        "pi05_pack_plate_inter_hand_relative_step59999": "inter_hand_relative",
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
            "model_state_encoding": self._CHECKPOINT_VARIANTS.get(
                identity.get("checkpoint_variant")
            ),
            "output_format": "xpolicylab",
            "action_semantics": ACTION_SEMANTICS,
            "action_horizon": 32,
        }
        if (
            any(identity.get(key) != value for key, value in required.items())
            or identity.get("checkpoint_variant") not in self._CHECKPOINT_VARIANTS
        ):
            raise ValueError("Pi05 pack-plate backend identity mismatch")
