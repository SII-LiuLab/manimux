"""The shared policy-to-robot adapter interface; implementations do not own hardware connections."""

from __future__ import annotations

from abc import ABC, abstractmethod

from manimux.types import ActionChunk, ActionContext, InferenceRequest, ObservationSnapshot


class PolicyAdapter(ABC):
    """Models may share a conversion flow; policy.adapter.type selects the implementation.

    policy.adapter holds mapping parameters; other policy fields hold action timing and client configuration.
    kinematics is an assembled offline model; passing it does not connect cameras or robots.
    """

    # The runtime selects pose decoding only after validating step IK support.
    decode_pose_only = False
    supports_step_ik = False
    pose_condition_modes = ()

    # True when decode_action honors ActionContext.handoff_reference.
    supports_waypoint_handoff = False

    # Decode model apertures and invert the mapping for RTC conditioning.
    supports_gripper_mapping = False

    # Each candidate decodes independently into complete, untrimmed absolute-joint
    # rows on the policy time grid, with no runtime trajectory or handoff attached.
    # Used by BID and deferred AAC selection; this does not declare model sampling.
    supports_candidate_decoding = False

    # Select the IK starting state, independently of the measured observation.
    decode_seed_source = "execution_reference"

    def decode_seed_time_ns(self, context: ActionContext) -> int | None:
        """Exact seed time for timed decoders; None keeps the runtime start/hold policy."""
        return None

    def __init__(self, robot: dict, policy: dict, *, kinematics=None):
        self.robot = robot
        self.policy = policy
        self.kinematics = kinematics

    def build_observation(self, snapshot: ObservationSnapshot) -> ObservationSnapshot:
        """Adapters that do not rewrite observations preserve the original snapshot and timestamps."""
        return snapshot

    def prepare_request(self, request: InferenceRequest) -> InferenceRequest:
        """Specialized adapters may add poses or history; the default preserves the request."""
        return request

    @abstractmethod
    def decode_action(self, raw: object, context: ActionContext) -> ActionChunk:
        """Convert explicitly contracted server outputs to action sequences; the base class does not infer formats."""

    def validate(self, robot: dict, policy: dict) -> None:
        """Preserve specialized adapters' contract-check entry points; the base class adds no checks."""
        return None
