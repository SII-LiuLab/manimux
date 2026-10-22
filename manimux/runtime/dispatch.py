"""Bounded policy calls in one rollout; transport and planners live outside the loop."""

from __future__ import annotations

import json
import math
import threading
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from manimux.runtime.inference import DefaultChunkStrategy, InferenceSubmission
from manimux.runtime.motion import MotionTools, ToolMotion
from manimux.runtime.timeline import CommitResult
from manimux.types import (
    ActionChunk,
    InferenceResponse,
    ObservationSnapshot,
    RobotState,
    SensorFrame,
    copy_group_vector,
)

TERMINAL_STATES = frozenset({"completed", "cancelled", "failed"})


@dataclass
class DispatchCall:
    request_id: str
    prompt: str
    chunks: int
    use_length: int
    timeout_s: float
    deadline: float
    state: str = "queued"
    reason: str = ""
    completed_chunks: int = 0
    dispatched_control_steps: int = 0
    request_seq: int | None = None
    plan_id: str | None = None
    plan_end_ns: int | None = None
    last_row_ns: int | None = None
    final_reference_sent: bool = False
    finishing: str | None = None
    held: bool = False
    observation: ObservationSnapshot | None = None
    plans: list[str] = field(default_factory=list)
    kind: str = "policy"
    arguments_key: str = ""
    motion: ToolMotion | None = None
    motion_dispatched: bool = False
    arrival: dict[str, Any] = field(default_factory=dict)
    settle_deadline: float | None = None
    planning_pending: bool = False


class PolicyDispatch:
    """Single-call mailbox with idempotent submission and runtime-acknowledged stops."""

    def __init__(self, config: dict[str, Any], *, policy_enabled: bool = True) -> None:
        if type(policy_enabled) is not bool:
            raise ValueError("policy_enabled must be a boolean")
        self.validate_config(config)
        self.policy_enabled = policy_enabled
        self.horizon = config["policy"]["horizon_policy_steps"]
        self.session_id = uuid.uuid4().hex
        self._lock = threading.RLock()
        self._calls: dict[str, DispatchCall] = {}
        self._active: DispatchCall | None = None
        self._snapshot: ObservationSnapshot | None = None
        self._updated_at = 0.0
        self._ready = False
        self._enabled = False
        self._closed = False
        self.stop_requested = False
        self._metadata: dict[str, Any] = {}
        self._events: list[tuple[str, dict[str, Any]]] = []
        self.motion_tools: MotionTools | None = None
        self._reference: dict[str, np.ndarray] | None = None
        self._agent_decisions: dict[str, dict[str, Any]] = {}
        self._agent_status: dict[str, Any] = {}
        self._agent_messages: list[dict[str, Any]] = []
        self._agent_message_count = 0

    def agent_status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._agent_status)

    def set_agent_status(self, phase: str, **fields: Any) -> None:
        with self._lock:
            self._agent_status = {"phase": phase, **fields}
            self._events.append(("agent_status", {**self._metadata, **self._agent_status}))

    @staticmethod
    def validate_config(config: dict[str, Any]) -> None:
        settings = config["inference"]
        if (
            settings["algorithm"] != "serial"
            or settings["inference_schedule"] != "serial"
            or settings["action_start_mode"] != "first_step_when_ready"
            or settings.get("strategy")
            or settings["blend_policy_steps"]
            or settings["handoff_skip_steps"]
            or settings["max_chunk_policy_steps"] is not None
            or (settings.get("temporal_ensemble") or {}).get("enabled")
        ):
            raise ValueError(
                "policy dispatch requires plain serial inference "
                "without cropping, skipping or blending"
            )
        if config["policy"]["action_decoding"] != "inline":
            raise ValueError("policy dispatch requires inline action decoding")
        if config.get("execution", {}).get("ik", {}).get("mode", "chunk") != "chunk":
            raise ValueError("policy dispatch currently requires joint chunks (chunk IK)")

    def start(self, **metadata: Any) -> None:
        with self._lock:
            self._metadata = metadata
            self._ready = True

    def _check_session(self, session_id: str) -> None:
        if session_id != self.session_id:
            raise ValueError("dispatch session mismatch; reconnect after runtime restart")

    def status(self) -> dict[str, Any]:
        with self._lock:
            fresh = self._snapshot is not None and time.monotonic() - self._updated_at < 1.0
            return {
                "protocol_version": 1,
                "session_id": self.session_id,
                "ready": self._ready and fresh and not self._closed,
                "enabled": self._enabled and fresh and not self.stop_requested,
                "closed": self._closed,
                "horizon_policy_steps": self.horizon,
                "active_request_id": None if self._active is None else self._active.request_id,
                "reset_scope": "rollout",
                "agent_decision_telemetry": True,
                "motion": None if self.motion_tools is None else self.motion_tools.describe(),
                **self._metadata,
                "policy_enabled": self.policy_enabled,
            }

    def report_agent_decision(
        self, *, session_id: str, decision_id: str, tool: str,
        arguments: dict[str, Any], phase: str, planner: str,
        outcome: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Retain bounded, session-scoped tool decisions for recording and RoboGUI."""
        if tool not in {"observe", "move_to", "move_delta", "rotate", "set_gripper",
                        "execute_policy", "finish"}:
            raise ValueError("unsupported agent decision tool")
        if phase not in {"selected", "completed", "failed"}:
            raise ValueError("unsupported agent decision phase")
        if not isinstance(decision_id, str) or not 1 <= len(decision_id) <= 128:
            raise ValueError("decision_id must contain 1..128 characters")
        if not isinstance(planner, str) or len(planner) > 128:
            raise ValueError("planner must contain at most 128 characters")
        if not isinstance(arguments, dict) or (
            outcome is not None and not isinstance(outcome, dict)
        ):
            raise ValueError("arguments and outcome must be mappings")
        payload = dict(decision_id=decision_id, tool=tool, arguments=arguments,
                       phase=phase, planner=planner, outcome=outcome or {})
        encoded = json.dumps(payload, allow_nan=False)
        if len(encoded.encode()) > 16384:
            raise ValueError("agent decision exceeds 16384 bytes")
        with self._lock:
            self._check_session(session_id)
            if self._closed:
                raise RuntimeError("runtime is closed")
            previous = self._agent_decisions.get(decision_id)
            if previous is not None:
                if any(previous[key] != payload[key] for key in ("tool", "arguments", "planner")):
                    raise ValueError("decision_id was already used with different arguments")
                if previous["phase"] != "selected":
                    return {"session_id": self.session_id, "recorded": True}
            elif len(self._agent_decisions) >= 128:
                raise RuntimeError("agent decision limit reached")
            payload = json.loads(encoded)
            payload["sequence"] = (
                previous["sequence"] if previous else len(self._agent_decisions) + 1
            )
            self._agent_decisions[decision_id] = payload
            self._events.append(("agent_decision", {**self._metadata, **payload,
                                                   "session_id": self.session_id}))
            return {"session_id": self.session_id, "recorded": True}

    def agent_decisions(self) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(list(self._agent_decisions.values())[-8:])

    def add_agent_messages(self, messages: list[dict[str, str]]) -> None:
        """Record planner answers, thinking and tool lines; RoboGUI shows the recent ones."""
        with self._lock:
            for message in messages:
                self._agent_message_count += 1
                entry = {**message, "text": message["text"][:16384],
                         "sequence": self._agent_message_count, "wall_time": time.time()}
                self._agent_messages = [*self._agent_messages[-199:], entry]
                self._events.append(("agent_message", {**self._metadata, **entry,
                                                      "session_id": self.session_id}))

    def agent_messages(self) -> list[dict[str, Any]]:
        # Heartbeats carry a short tail; RoboGUI accumulates by sequence.
        with self._lock:
            return [{**m, "text": m["text"][:600]} for m in self._agent_messages[-12:]]

    def submit(
        self,
        *,
        session_id: str,
        request_id: str,
        prompt: str,
        chunks: int,
        use_length: int,
        timeout_s: float = 120.0,
    ) -> dict[str, Any]:
        with self._lock:
            self._check_session(session_id)
            if not self.policy_enabled:
                raise RuntimeError(
                    "execute_policy is disabled for this runtime; only motion primitives are available"
                )
            if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
                raise ValueError("request_id must contain 1..128 characters")
            if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 16384:
                raise ValueError("prompt must contain 1..16384 characters")
            if type(chunks) is not int or not 1 <= chunks <= 1000:
                raise ValueError("chunks must be an integer in 1..1000")
            if type(use_length) is not int or not 1 <= use_length <= self.horizon:
                raise ValueError(f"use_length must be an integer in 1..{self.horizon}")
            if (
                isinstance(timeout_s, bool)
                or not isinstance(timeout_s, int | float)
                or not math.isfinite(timeout_s)
                or not 0 < timeout_s <= 3600
            ):
                raise ValueError("timeout_s must be finite and in (0, 3600]")
            previous = self._calls.get(request_id)
            if previous is not None:
                if previous.kind != "policy" or (
                    previous.prompt,
                    previous.chunks,
                    previous.use_length,
                    previous.timeout_s,
                ) != (
                    prompt,
                    chunks,
                    use_length,
                    timeout_s,
                ):
                    raise ValueError("request_id was already used with different arguments")
                return self.result(request_id)
            status = self.status()
            if not status["ready"] or not status["enabled"]:
                raise RuntimeError("runtime is not ready; start the rollout in RoboGUI first")
            if self._active is not None:
                raise RuntimeError("another policy call is active")
            # Retain IDs for the entire rollout, so a late retry cannot replay motion.
            if len(self._calls) >= 128:
                raise RuntimeError("dispatch call limit reached; start a new rollout")
            call = DispatchCall(
                request_id,
                prompt,
                chunks,
                use_length,
                float(timeout_s),
                time.monotonic() + timeout_s,
            )
            self._calls[request_id] = self._active = call
            self._events.append(("policy_call_submitted", self.result(request_id)))
            return self.result(request_id)

    def submit_motion(
        self,
        *,
        session_id: str,
        request_id: str,
        tool: str,
        arguments: dict[str, Any],
        timeout_s: float = 120.0,
    ) -> dict[str, Any]:
        """Reserve the same mailbox before planning; cancellation cannot revive a plan."""
        key = json.dumps([tool, arguments, timeout_s], sort_keys=True, allow_nan=False)
        with self._lock:
            self._check_session(session_id)
            if self.motion_tools is None:
                raise ValueError("motion tools are not configured for this runtime")
            if tool not in self.motion_tools.describe()["tools"] or not isinstance(arguments, dict):
                raise ValueError("unsupported motion tool or invalid arguments mapping")
            if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
                raise ValueError("request_id must contain 1..128 characters")
            if (
                isinstance(timeout_s, bool)
                or not isinstance(timeout_s, int | float)
                or not math.isfinite(timeout_s)
                or not 0 < timeout_s <= 3600
            ):
                raise ValueError("timeout_s must be finite and in (0,3600]")
            previous = self._calls.get(request_id)
            if previous is not None:
                if previous.kind != tool or previous.arguments_key != key:
                    raise ValueError("request_id was already used with different arguments")
                return self.result(request_id)
            status = self.status()
            if not status["ready"] or not status["enabled"]:
                raise RuntimeError("runtime is not ready; start the rollout in RoboGUI first")
            if self._active is not None:
                raise RuntimeError("another policy or motion call is active")
            if len(self._calls) >= 128:
                raise RuntimeError("dispatch call limit reached; start a new rollout")
            assert self._snapshot is not None and self._reference is not None
            snapshot, reference = self._snapshot, copy_group_vector(self._reference)
            call = DispatchCall(
                request_id,
                tool,
                1,
                0,
                float(timeout_s),
                time.monotonic() + timeout_s,
                state="planning",
                kind=tool,
                arguments_key=key,
                planning_pending=True,
            )
            self._calls[request_id] = self._active = call
            self._events.append(
                ("motion_call_submitted", {**self.result(request_id), "arguments": arguments})
            )
        # Bounded IK must not hold the mailbox lock or block the hardware writer.
        try:
            motion = self.motion_tools.plan(tool, arguments, snapshot, reference)
        except Exception as exc:
            with self._lock:
                call.planning_pending = False
                if call.finishing is None:
                    self._stop(call, "failed", f"motion_plan_rejected:{type(exc).__name__}:{exc}")
                return self.result(request_id)
        with self._lock:
            call.planning_pending = False
            if call.finishing is None and call.state not in TERMINAL_STATES:
                call.motion = motion
                call.use_length = len(next(iter(motion.groups.values())))
                call.state = "queued"
            return self.result(request_id)

    def take_motion(self, request_seq: int, now_ns: int) -> ActionChunk | None:
        with self._lock:
            call = self._active
            if (
                call is None
                or call.state != "running"
                or call.motion is None
                or call.motion_dispatched
            ):
                return None
            motion = call.motion
            call.motion_dispatched = True
            return ActionChunk(
                f"motion-{call.request_id}",
                request_seq,
                motion.observation_time_ns,
                now_ns,
                "joint_position",
                motion.dt_ns,
                motion.groups,
                metadata={"dispatch_request_id": call.request_id, "tool": call.kind},
            )

    def plan_accepted(self, chunk: ActionChunk, now_ns: int) -> None:
        with self._lock:
            call = self._active
            if call is None or call.finishing is not None:
                return
            call.request_seq = None
            call.plan_id = chunk.plan_id
            call.plan_end_ns = now_ns + chunk.horizon_steps * chunk.dt_ns
            call.last_row_ns = now_ns + (chunk.horizon_steps - 1) * chunk.dt_ns
            call.final_reference_sent = False
            call.plans.append(chunk.plan_id)

    def motion_rejected(self, reason: str) -> None:
        with self._lock:
            if self._active is not None and self._active.finishing is None:
                self._stop(self._active, "failed", reason)

    def result(self, request_id: str) -> dict[str, Any]:
        with self._lock:
            call = self._calls[request_id]
            return {
                "session_id": self.session_id,
                "request_id": call.request_id,
                "state": call.state,
                "reason": call.reason,
                "prompt": call.prompt,
                "call_type": call.kind,
                "chunks": call.chunks,
                "use_length": call.use_length,
                "completed_chunks": call.completed_chunks,
                "completed_policy_steps": call.completed_chunks * call.use_length
                if call.kind == "policy"
                else 0,
                "completed_motion_steps": call.completed_chunks * call.use_length
                if call.kind != "policy"
                else 0,
                "arrival": dict(call.arrival),
                "dispatched_control_steps": call.dispatched_control_steps,
                "plan_ids": list(call.plans),
                **self._metadata,
            }

    def observation(self, request_id: str | None = None) -> ObservationSnapshot:
        with self._lock:
            snapshot = self._snapshot if request_id is None else self._calls[request_id].observation
            if snapshot is None:
                raise RuntimeError("observation is not available yet")
            return snapshot

    def cancel(self, *, session_id: str, request_id: str) -> dict[str, Any]:
        with self._lock:
            self._check_session(session_id)
            call = self._calls[request_id]
            if call.state not in TERMINAL_STATES:
                self._stop(call, "cancelled", "client_cancelled")
            return self.result(request_id)

    def finish(self, *, session_id: str) -> dict[str, Any]:
        with self._lock:
            self._check_session(session_id)
            self.stop_requested = True
            return {"session_id": self.session_id, "finish_requested": True}

    def _stop(self, call: DispatchCall, state: str, reason: str) -> None:
        call.state, call.finishing, call.reason = "stopping", state, reason

    def _complete(self, call: DispatchCall) -> None:
        call.state = call.finishing or "failed"
        call.observation = self._snapshot
        self._events.append(
            (
                "policy_call_finished" if call.kind == "policy" else "motion_call_finished",
                self.result(call.request_id),
            )
        )
        self._active = None

    def before_tick(
        self,
        snapshot: ObservationSnapshot,
        *,
        paused: bool,
        home: bool,
        reference: dict[str, np.ndarray] | None = None,
    ) -> bool:
        """Publish feedback; return True when the runtime must hold and fence replies."""
        with self._lock:
            old_frames = {} if self._snapshot is None else self._snapshot.frames
            frames = {}
            for name, frame in snapshot.frames.items():
                old = old_frames.get(name)
                frames[name] = (
                    old
                    if old is not None
                    and old.sequence == frame.sequence
                    and old.capture_monotonic_ns == frame.capture_monotonic_ns
                    else SensorFrame(
                        name,
                        frame.data.copy(),
                        frame.capture_monotonic_ns,
                        frame.sequence,
                        depth=None if frame.depth is None else frame.depth.copy(),
                        camera_meta=deepcopy(frame.camera_meta),
                    )
                )
            state = snapshot.state
            self._snapshot = ObservationSnapshot(
                RobotState(
                    copy_group_vector(state.groups),
                    state.monotonic_ns,
                    state.sequence,
                    grippers=deepcopy(state.grippers),
                ),
                frames,
            )
            self._updated_at = time.monotonic()
            self._reference = copy_group_vector(
                snapshot.state.groups if reference is None else reference
            )
            self._enabled = not (paused or home)
            call = self._active
            if call is None:
                return True
            if call.finishing is not None and call.held:
                if call.planning_pending:
                    return True
                self._complete(call)
                return True
            if paused or home:
                self._stop(call, "cancelled", "operator_home" if home else "operator_paused")
            elif time.monotonic() >= call.deadline:
                self._stop(call, "failed", "call_timeout")
            if call.finishing is not None:
                return True
            if call.state == "planning":
                return True
            if call.settle_deadline is not None:
                assert self.motion_tools is not None and call.motion is not None
                call.arrival = self.motion_tools.arrival(call.motion, snapshot)
                if call.arrival["reached"]:
                    self._stop(call, "completed", "measured_arrival")
                    return True
                if time.monotonic() >= call.settle_deadline:
                    self._stop(call, "failed", "motion_arrival_timeout")
                    return True
            call.state = "running"
            return False

    def command_sent(
        self, *, now_ns: int, plan_id: str | None, reference_plan_id: str | None, holding: bool
    ) -> None:
        with self._lock:
            call = self._active
            if call is None:
                return
            if holding:
                if call.finishing is not None:
                    call.held = True
                return
            if call.plan_id is not None and plan_id == call.plan_id:
                if reference_plan_id == call.plan_id:
                    call.dispatched_control_steps += 1
                    if call.last_row_ns is not None and now_ns >= call.last_row_ns:
                        call.final_reference_sent = True
                if call.plan_end_ns is not None and now_ns >= call.plan_end_ns:
                    if not call.final_reference_sent:
                        self._stop(call, "failed", "final_reference_not_dispatched")
                        return
                    call.completed_chunks += 1
                    call.plan_id = call.plan_end_ns = None
                    if call.completed_chunks == call.chunks:
                        if call.motion is None:
                            self._stop(call, "completed", "budget_completed")
                        else:
                            assert self.motion_tools is not None
                            call.settle_deadline = (
                                time.monotonic() + self.motion_tools.settle_timeout_s
                            )

    def close(self, reason: str) -> None:
        with self._lock:
            self._closed = True
            self._enabled = False
            if self._active is not None:
                self._stop(self._active, "failed", reason)
                self._complete(self._active)

    def fail_active(self, reason: str) -> None:
        with self._lock:
            self._ready = False
            if self._active is not None and self._active.finishing is None:
                self._stop(self._active, "failed", reason)

    def take_events(self) -> list[tuple[str, dict[str, Any]]]:
        with self._lock:
            events, self._events = self._events, []
            return events


def validate_dispatch_chunk(
    chunk: ActionChunk, *, use_length: int, motion: MotionTools | None,
    reference: dict[str, np.ndarray] | None,
) -> None:
    """Apply the same action contract to isolated previews and formal policy calls."""
    if (
        chunk.action_space != "joint_position"
        or chunk.runtime_trajectory is not None
        or chunk.handoff is not None
        or chunk.source_offset_steps
        or chunk.hold_from_step
    ):
        raise ValueError("dispatch requires complete, untrimmed joint chunks without dense IK")
    if chunk.horizon_steps < use_length:
        raise ValueError("policy response is shorter than the requested execution prefix")
    if motion is not None:
        for name, index in motion.grippers.items():
            values = chunk.groups[name][:, index]
            invalid = np.flatnonzero((values < 0) | (values > 1) | ~np.isfinite(values))
            if len(invalid):
                raise ValueError(
                    f"{name}.gripper: expected [0,1]; invalid rows={invalid.tolist()}, "
                    f"count={len(invalid)}, values={values[invalid].tolist()}"
                )
        motion.validate_policy_path(
            {k: v[:use_length] for k, v in chunk.groups.items()}, reference,
        )


class DispatchWarmupStrategy(DefaultChunkStrategy):
    """Serial preview with the dispatch action contract and no live call mailbox."""

    def __init__(self, config: dict[str, Any], motion: MotionTools | None) -> None:
        super().__init__(config)
        self._motion = motion
        self._horizon = config["policy"]["horizon_policy_steps"]
        self._preview_chunk: ActionChunk | None = None

    def reset(self) -> None:
        super().reset()
        self._preview_chunk = None

    def prepare_chunk(
        self, *, chunk: ActionChunk, response: InferenceResponse, now_ns: int,
    ) -> ActionChunk:
        self._preview_chunk = chunk
        return chunk

    def commit_settings(self, *, response, measured, last_command):
        # PreviewPlanner supplies its own evolving reference here. Never read
        # the formal dispatch reference or create a synthetic live tool call.
        assert self._preview_chunk is not None
        validate_dispatch_chunk(
            self._preview_chunk, use_length=self._horizon,
            motion=self._motion, reference=last_command,
        )
        return super().commit_settings(
            response=response, measured=measured, last_command=last_command,
        )


class DispatchStrategy(DefaultChunkStrategy):
    """Use existing serial scheduling and decode/execution with a per-call prefix."""

    def __init__(self, config: dict[str, Any], dispatch: PolicyDispatch) -> None:
        super().__init__(config)
        self.dispatch = dispatch

    def build_submission(self, **kwargs: Any) -> InferenceSubmission | None:
        with self.dispatch._lock:
            call = self.dispatch._active
            if (
                call is None
                or call.kind != "policy"
                or call.state != "running"
                or call.request_seq is not None
                or call.plan_id is not None
            ):
                return None
            submission = super().build_submission(**kwargs)
            if submission is not None:
                call.request_seq = submission.request.request_seq
                submission = replace(
                    submission,
                    request=replace(submission.request, instruction=call.prompt),
                    event_fields={
                        **submission.event_fields,
                        "dispatch_request_id": call.request_id,
                        "policy_instruction": call.prompt,
                    },
                )
            return submission

    def prepare_chunk(
        self, *, chunk: ActionChunk, response: InferenceResponse, now_ns: int
    ) -> ActionChunk:
        del now_ns
        with self.dispatch._lock:
            call = self.dispatch._active
            if call is None or call.state != "running" or call.request_seq != response.request_seq:
                raise ValueError("policy call is no longer active")
            validate_dispatch_chunk(
                chunk, use_length=call.use_length,
                motion=self.dispatch.motion_tools, reference=self.dispatch._reference,
            )
            return replace(
                chunk,
                groups={k: v[: call.use_length].copy() for k, v in chunk.groups.items()},
                metadata={
                    **chunk.metadata,
                    "dispatch_request_id": call.request_id,
                    "policy_instruction": call.prompt,
                },
            )

    def on_plan_accepted(
        self, *, chunk: ActionChunk, result: CommitResult, response: InferenceResponse, now_ns: int
    ) -> dict[str, object]:
        fields = super().on_plan_accepted(
            chunk=chunk, result=result, response=response, now_ns=now_ns
        )
        self.dispatch.plan_accepted(chunk, now_ns)
        return fields

    def on_response_rejected(self, response: InferenceResponse) -> None:
        with self.dispatch._lock:
            call = self.dispatch._active
            # Pause/cancel/deadline already owns the stop result. Invalidating a
            # late reply must not replace that result while the hold is pending.
            if (
                call is not None
                and call.finishing is None
                and call.request_seq == response.request_seq
            ):
                self.dispatch._stop(call, "failed", response.error or "policy_response_rejected")

    def take_runtime_events(self, *, step: int) -> list[tuple[str, dict[str, object]]]:
        del step
        return self.dispatch.take_events()
