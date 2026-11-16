"""Future-state-conditioned asynchronous execution for VLASH joint policies.

Images and measured state retain their capture clock. Only model input state is
replaced by the outgoing action immediately before the planned takeover. The
returned prediction starts at that takeover, not at image capture; latency must
therefore never be trimmed a second time against the original image clock.
"""
from dataclasses import dataclass, replace

from manimux.policies.base import action_interval
from manimux.runtime.inference import DefaultChunkStrategy, InferenceSubmission
from manimux.runtime.safety import RuntimeState
from manimux.types import InferenceRequest, ObservationSnapshot, RobotState, copy_group_vector


def vlash_parameters(**options):
    defaults = {"delay_policy_steps": 6, "min_execute_policy_steps": 32}
    unknown = options.keys() - defaults.keys()
    if unknown:
        raise ValueError(f"unsupported vlash fields: {sorted(unknown)}")
    defaults.update(options)
    for key, minimum in (("delay_policy_steps", 0), ("min_execute_policy_steps", 1)):
        if type(defaults[key]) is not int or defaults[key] < minimum:
            raise ValueError(f"vlash.{key} must be an integer >= {minimum}")
    return defaults


@dataclass(slots=True)
class VlashInferenceRequest(InferenceRequest):
    model_observation: ObservationSnapshot | None = None


class VlashInferenceStrategy(DefaultChunkStrategy):
    discard_plans_while_paused = True

    def __init__(self, config):
        settings = config["inference"]
        self.delay = settings["vlash"]["delay_policy_steps"]
        self.window = settings["vlash"]["min_execute_policy_steps"]
        self.dt_ns = int(action_interval(config["policy"]) * 1e9)
        contract = (config["policy"].get("expected_backend") or {}).get("model", {}).get("vlash", {})
        if contract.get("state_source") != "previous_action" or self.delay > contract.get("max_delay_steps", -1):
            raise ValueError("VLASH requires previous_action checkpoint metadata covering deployment delay")
        if self.window + self.delay > config["policy"]["horizon_policy_steps"] or self.window <= self.delay:
            raise ValueError("VLASH requires delay < execution window and window + delay <= horizon")
        limits = config["executor"].get("motion_limits") or {}
        rate_limited = any(value is not None for options in limits.values()
            for key, value in options.items() if key in {"max_velocity", "max_acceleration", "max_closing_velocity"})
        if (settings["inference_schedule"] != "single_inflight"
                or settings["action_start_mode"] != "drop_infer_latency"
                or settings["blend_policy_steps"] or settings["handoff_skip_steps"]
                or settings["handoff"] != "blend" or settings["min_plan_hold_policy_steps"]
                or settings["temporal_ensemble"]["enabled"]
                or config["executor"]["type"] != "direct"
                or rate_limited
                or config["policy"]["action_decoding"] != "inline"
                or config["policy"]["adapter"]["type"] != "manimux.policy_adapter.joint:JointAdapter"
                or config["policy"]["adapter"].get("gripper_mapping")):
            raise ValueError("VLASH currently requires single-inflight, inline absolute joints, Direct, no blending or rate mapping")
        self.warmup_required_calls = {"ordinary": 1, **({"conditioned": 2} if self.delay else {})}
        super().__init__(config)

    @property
    def name(self):
        return "vlash"

    @property
    def control_mode(self):
        return "managed"

    def reset(self):
        super().reset()
        self._requests = {}

    def build_submission(self, *, snapshot, timeline, now_ns, runtime_state, request_state, **kwargs):
        if runtime_state != RuntimeState.RUNNING or request_state.in_flight:
            return None
        progress = 0 if self._active_source_offset is None else self._active_source_offset + timeline.cursor(now_ns)
        if self._active_source_offset is not None and progress < self.window:
            return None
        origin = snapshot.state.monotonic_ns
        delay = self.delay if timeline.active_plan_id is not None else 0
        # Training uses action[t+d-1] as state for action[t+d:t+d+H].
        future = timeline.sample(origin + (delay - 1) * self.dt_ns) if delay else None
        if delay and future is None:
            # No invented extrapolation beyond the available outgoing trajectory.
            delay = 0
        model_snapshot = snapshot if not delay else ObservationSnapshot(
            state=RobotState(copy_group_vector(future), origin, snapshot.state.sequence),
            frames=snapshot.frames,
        )
        submission = super().build_submission(snapshot=snapshot, timeline=timeline, now_ns=now_ns,
            runtime_state=runtime_state, request_state=request_state, **kwargs)
        request = submission.request
        request = VlashInferenceRequest(
            session_id=request.session_id, request_seq=request.request_seq,
            observation_time_ns=origin, deadline_ns=request.deadline_ns,
            observation=request.observation, instruction=request.instruction,
            model_observation=model_snapshot,
        )
        self._requests[request.request_seq] = delay
        return InferenceSubmission(request, {
            "conditioned": bool(delay), "vlash_delay_steps": delay,
            "vlash_state_source": "previous_action" if delay else "measured_bootstrap",
            "vlash_prediction_start_ns": origin + delay * self.dt_ns,
            "vlash_state_time_ns": origin + (delay - 1) * self.dt_ns if delay else origin,
            "executed_steps": progress, "chunk_policy_steps": self.window,
        })

    def prepare_chunk(self, *, chunk, response, now_ns):
        delay = self._requests[response.request_seq]
        if (chunk.action_space != "joint_position" or chunk.source_offset_steps
                or chunk.hold_from_step or chunk.runtime_trajectory is not None or chunk.handoff is not None
                or chunk.dt_ns != self.dt_ns
                or chunk.horizon_steps != self._config["policy"]["horizon_policy_steps"]):
            raise ValueError("VLASH requires full absolute joint predictions on the configured clock")
        age_limit = self._config["inference"]["max_plan_age_s"]
        if age_limit is not None and now_ns - response.observation_time_ns > age_limit * 1e9:
            raise ValueError("VLASH observation exceeded max_plan_age_s")
        return replace(chunk, observation_time_ns=response.observation_time_ns + delay * self.dt_ns,
            metadata={**chunk.metadata, "vlash_delay_steps": delay,
                "observation_capture_time_ns": response.observation_time_ns,
                "vlash_prediction_start_ns": response.observation_time_ns + delay * self.dt_ns})

    def on_plan_accepted(self, *, chunk, result, response, now_ns):
        delay = self._requests.pop(response.request_seq)
        # Match TT-RTC's observation-relative trigger: its trimmed prefix also
        # counts toward the 32-row source-progress window.
        self._active_source_offset = result.trimmed_steps + delay
        return {"trimmed_steps": result.trimmed_steps,
            "request_to_commit_ms": (now_ns - response.observation_time_ns) / 1e6,
            "vlash_late_ms": max(0, now_ns - chunk.observation_time_ns) / 1e6 if delay else 0}

    def on_response_rejected(self, response):
        self._requests.pop(response.request_seq, None)
