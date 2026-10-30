"""Hardware-free planning context shared by rollout and warmup.

Both callers supply the same resolved configuration and isolated planning state.
No device, recorder or dispatch operation belongs in this layer.
"""

from dataclasses import replace

from manimux.policies.base import action_interval
from manimux.policy_adapter.pose_output import POSE_SPACE
from manimux.runtime.timeline import ActionTimeline
from manimux.types import ActionContext, RobotState


def condition_overlap_fields(fields: dict, horizon_steps: int) -> dict[str, int]:
    """RoboGUI/record overlap of a conditioned submission, in source policy rows.

    RTC conditions on the unexecuted remainder and freezes its forecast delay
    (ChunkTrust RTC reports that budget as frozen_budget). PAINT conditions only
    on its delay prefix.
    """
    if "delay_steps" in fields:
        delay = max(0, int(fields["delay_steps"]))
        return {"conditioned_overlap_steps": delay, "frozen_steps": delay}
    return {
        "conditioned_overlap_steps": max(0, horizon_steps - int(fields.get("executed_steps", 0))),
        "frozen_steps": int(fields.get("frozen_budget", fields.get("forecast_delay", 0))),
    }


class PlanningContext:
    def _build_timeline(self) -> ActionTimeline:
        return ActionTimeline(
            ({name: 8 for name in self._config["robot"]["group_dims"]}
             if self._step_ik else self._config["robot"]["group_dims"]),
            action_space=POSE_SPACE if self._step_ik else "joint_position",
            max_source_steps=self._config["inference"]["max_chunk_policy_steps"],
            blend_mode=self._config["inference"].get("blend_mode", "linear"),
            gripper_indices=((self._config["executor"].get("smooth") or {}).get(
                "gripper") or {}).get("group_indices", {}),
            action_start_mode=self._config["inference"]["action_start_mode"],
            allow_retimed_suffix=getattr(self._strategy, "allow_retimed_suffix", False),
            hold_last_step=(
                (
                    self._config["inference"]["inference_schedule"] == "serial"
                    and self._config["inference"]["algorithm"] in {"async", "serial"}
                )
                or getattr(self._strategy, "hold_last_step", False)
            ),
        )

    def _decode_seed(
        self,
        state: RobotState,
        now_ns: int,
        observation_state: RobotState | None = None,
        *,
        seed_time_ns: int | None = None,
    ) -> tuple[int, RobotState, str]:
        """Expected plan start and IK seed for a process decode.

        The adapter may declare an exact seed time independently of its source.
        Complete Tianji chunks use observation time; waypoint suffixes use the
        aligned handoff time. Neither choice changes the request observation.
        """
        if self._step_ik:
            return now_ns, state, "measured_state"
        expected_decode_s = self._decode_forecast.seconds
        start_ns = now_ns + int(expected_decode_s * 1e9)
        if self._decode_seed_source == "observation_state":
            if observation_state is None:
                raise ValueError("request observation state is unavailable for action decoding")
            return start_ns, observation_state, "observation_state"
        if seed_time_ns is not None:
            groups = self._timeline.sample_runtime(seed_time_ns)
            if groups is not None:
                seed = RobotState(groups, seed_time_ns, state.sequence)
                return start_ns, seed, "active_reference"
            if (
                observation_state is not None
                and seed_time_ns == observation_state.monotonic_ns
            ):
                return start_ns, observation_state, "observation_state"
            raise ValueError("no reference covers the requested IK seed time")
        remaining_ns = self._timeline.remaining_ns(now_ns)
        if expected_decode_s > 0 and remaining_ns > 0:
            reference = self._timeline.sample(min(start_ns, now_ns + remaining_ns))
            if reference is not None:
                return start_ns, RobotState(reference, start_ns, state.sequence), "active_reference"
        return start_ns, state, "measured_state"

    def _decode_context(self, response, state, now_ns, observation_state) -> ActionContext:
        """Keep observations intact and resolve the final IK seed once, before decoding."""
        start_ns = now_ns + int(self._decode_forecast.seconds * 1e9)
        handoff_reference = None
        if self._config["inference"]["handoff"] == "waypoint" and self._strategy.decode_handoff(
            response=response
        ):
            start_ns += int(self._config["inference"]["handoff_margin_s"] * 1e9)
            handoff_reference = self._timeline.handoff_reference(
                start_ns, round(action_interval(self._config["policy"]) * 1e9)
            )
        context = ActionContext(
            request_seq=response.request_seq,
            observation_time_ns=response.observation_time_ns,
            created_time_ns=response.finished_time_ns,
            execution_time_ns=(
                start_ns
                if self._config["inference"]["action_start_mode"] == "drop_infer_latency"
                and (self._decoder is not None
                     or self._strategy.name in {"async", "serial", "rtc", "train_time_rtc"})
                else None
            ),
            observation_state=observation_state,
            decode_reference=(
                self._timeline.active_horizon()
                if getattr(self._adapter, "uses_decode_reference", False) else None
            ),
            handoff_reference=handoff_reference,
            max_source_steps=self._config["inference"]["max_chunk_policy_steps"],
            independent_groups=self._config["inference"]["independent_group_decoding"],
            decode_budget_ms=(
                self._config["inference"]["decode_budget_ms"]
                if self._config["inference"]["independent_group_decoding"] else None
            ),
        )
        _, seed, source = self._decode_seed(
            state, now_ns, observation_state,
            seed_time_ns=self._adapter.decode_seed_time_ns(context),
        )
        return replace(context, ik_seed=seed, ik_seed_source=source)

    def _response_rejection_reason(
        self, response, *, last_submitted_seq, last_request_deadline_ns,
        discard_responses_through,
    ) -> str | None:
        # A completed response can wait while a newer request is still in flight.
        # Its own deadline and the last accepted sequence remain authoritative.
        latest_completed = self._multi_inflight or self._min_execute_steps > 0
        if response.error is not None:
            return response.error
        if response.session_id != self._session_id:
            return "session_mismatch"
        if response.request_seq <= discard_responses_through:
            return "invalidated_by_pause_or_home"
        if not latest_completed and response.request_seq < last_submitted_seq:
            return "superseded_response"
        if latest_completed and response.request_seq <= self._timeline.accepted_request_seq:
            return "stale_request_seq"
        deadline = response.deadline_ns if latest_completed else last_request_deadline_ns
        if response.finished_time_ns > deadline:
            return "inference_deadline_exceeded"
        if latest_completed and self._clock.now_ns() > response.deadline_ns:
            return "deadline_exceeded_before_decode"
        if response.raw_action is None:
            return "missing_action"
        return None
