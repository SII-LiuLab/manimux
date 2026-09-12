from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

from manimux.clock import Clock, SystemClock
from manimux.config import ManiMuxConfig
from manimux.policies import ActionDecoderClient, PolicyCapabilities, build_policy_adapter
from manimux.policies.base import decode_policy_action, prepare_policy_request
from manimux.policies.worker import PolicyWorkerClient
from manimux.recording import EpisodeRecorder
from manimux.robots import build_robot
from manimux.robots.base import RobotDriver
from manimux.runtime.diagnostics import build_plan_boundary_payload
from manimux.runtime.executors import DirectExecutor, Executor, MPCExecutor, SmoothExecutor
from manimux.runtime.inference import (
    DefaultChunkStrategy,
    InferenceStrategy,
    RequestState,
    prepare_strategy_chunk,
)
from manimux.runtime.safety import RuntimeState, SafetyGuard
from manimux.runtime.timeline import ActionTimeline, CommitResult
from manimux.sensors import build_sensor
from manimux.types import (
    ActionContext,
    GroupVector,
    ObservationSnapshot,
    RobotCommand,
    RobotState,
    SensorFrame,
    copy_action_chunk,
    copy_group_vector,
)
from manimux.viewer import ViewerBridge


@dataclass(frozen=True, slots=True)
class RunResult:
    episode_dir: Path
    steps: int
    accepted_plans: int
    rejected_plans: int
    success: bool
    terminal_reason: str


def _next_rollout_id(run_dir: Path) -> str:
    highest = 0
    pattern = re.compile(r"^rollout-(\d+)(?:\.partial)?$")
    if not run_dir.exists():
        return "rollout-001"
    for path in run_dir.iterdir():
        match = pattern.fullmatch(path.name)
        if match is not None:
            highest = max(highest, int(match.group(1)))
    return f"rollout-{highest + 1:03d}"


def _metadata_mismatches(
    expected: dict[str, object],
    actual: dict[str, object],
    *,
    path: str = "backend",
) -> list[str]:
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        field_path = f"{path}.{key}"
        if key not in actual:
            mismatches.append(f"{field_path} is missing (expected {expected_value!r})")
            continue
        actual_value = actual[key]
        if isinstance(expected_value, dict):
            if not isinstance(actual_value, dict):
                mismatches.append(
                    f"{field_path} expected a mapping, got {type(actual_value).__name__}"
                )
                continue
            mismatches.extend(_metadata_mismatches(expected_value, actual_value, path=field_path))
            continue
        if actual_value != expected_value:
            mismatches.append(f"{field_path} expected {expected_value!r}, got {actual_value!r}")
    return mismatches


class EdgeRuntime:
    """One real-robot control loop with a replaceable inference strategy."""

    def __init__(
        self,
        config: ManiMuxConfig,
        run_dir: Path,
        *,
        clock: Clock | None = None,
        strategy: InferenceStrategy | None = None,
        launch_mode: str = "run",
    ) -> None:
        self._config = config
        self._run_dir = run_dir
        self._clock = clock or SystemClock()
        self._control_dt_ns = int(1_000_000_000 / config.robot.control_hz)
        self._robot = self._build_robot()
        self._sensors = [build_sensor(sensor, self._clock) for sensor in config.sensors]
        self._adapter = build_policy_adapter(config.robot, config.policy)
        self._adapter.validate(config.robot, config.policy)
        if config.execution.independent_group_decoding and not getattr(
            self._adapter, "supports_independent_group_decode", False
        ):
            raise ValueError("adapter does not support independent group decoding")
        self._decoder = None
        if config.policy.action_decoding == "process":
            if config.execution.runtime != "manimux":
                raise ValueError("process action decoding currently requires runtime=manimux")
            self._decoder = ActionDecoderClient(config.robot, config.policy, self._adapter)
        self._session_id = f"session-{uuid.uuid4().hex}"
        self._worker = PolicyWorkerClient(config.policy, self._session_id)
        self._timeline = self._build_timeline()
        self._executor = self._build_executor()
        self._strategy = strategy or DefaultChunkStrategy(config)
        self._launch_mode = launch_mode
        position_limit_abs = None
        if config.execution.executor == "smooth":
            position_limit_abs = config.execution.smooth.position_limit_abs
        elif config.execution.executor == "mpc":
            position_limit_abs = config.execution.mpc.position_limit_abs
        command_safety = config.execution.command_safety
        self._safety = SafetyGuard(
            config.robot.group_dims,
            position_limit_abs,
            position_lower=command_safety.position_lower,
            position_upper=command_safety.position_upper,
            max_velocity=command_safety.max_velocity,
            max_acceleration=command_safety.max_acceleration,
            control_dt_s=self._control_dt_ns / 1_000_000_000,
        )
        self._viewer = ViewerBridge(
            enabled=config.viewer.enabled,
            robot_adapter=config.viewer.robot_adapter,
            group_order=list(config.robot.group_dims),
            policy=config.viewer.policy_label or "manimux-local",
            instruction=config.run.task if config.viewer.policy_label else "",
            camera_hz=config.viewer.camera_hz,
        )
        self._state = RuntimeState.DISCONNECTED

    def _build_robot(self) -> RobotDriver:
        return build_robot(self._config.robot, self._clock)

    def _build_executor(self) -> Executor:
        control_dt_s = self._control_dt_ns / 1_000_000_000
        if self._config.execution.executor == "direct":
            return DirectExecutor(self._config.execution.motion_limits, control_dt_s)
        if self._config.execution.executor == "smooth":
            return SmoothExecutor(self._config.execution.smooth, control_dt_s)
        if self._config.execution.motion_limits is not None:
            raise ValueError("shared motion_limits currently support direct and smooth, not mpc")
        return MPCExecutor(self._config.execution.mpc, control_dt_s)

    def _build_timeline(self) -> ActionTimeline:
        return ActionTimeline(
            self._config.robot.group_dims,
            max_source_steps=self._config.execution.max_chunk_steps,
            start_on_commit=self._config.execution.inference_schedule == "serial",
        )

    def _hold_command(self, now_ns: int, groups: GroupVector) -> RobotCommand:
        return RobotCommand(
            groups=copy_group_vector(groups),
            monotonic_ns=now_ns,
            plan_id=self._timeline.active_plan_id,
        )

    def _validate_policy_capabilities(self) -> None:
        capabilities = getattr(self._worker, "capabilities", PolicyCapabilities())
        missing = self._strategy.required_sampling_modes.difference(capabilities.sampling_modes)
        if missing:
            raise RuntimeError(
                f"execution strategy {self._strategy.name!r} requires sampling modes "
                f"{sorted(missing)} that the policy server does not advertise"
            )
        expected = self._config.policy.expected_backend
        if expected is None:
            return
        expected_metadata = expected.model_dump(mode="python", exclude_none=True)
        mismatches = _metadata_mismatches(expected_metadata, capabilities.backend_metadata)
        if mismatches:
            details = "; ".join(mismatches)
            raise RuntimeError(f"policy backend identity mismatch: {details}")

    def run(self) -> RunResult:  # noqa: C901 - the safety-critical loop stays linear
        started_wall = time.perf_counter()
        episode_id = _next_rollout_id(self._run_dir)
        recorder = EpisodeRecorder(
            self._run_dir,
            episode_id,
            self._config.robot.group_dims,
            metadata={
                "episode_id": episode_id,
                "session_id": self._session_id,
                "task": self._config.run.task,
                "executor_kind": self._config.execution.executor,
                "smooth": (
                    self._config.execution.smooth.model_dump(mode="json")
                    if self._config.execution.executor == "smooth"
                    else None
                ),
                "runtime": self._strategy.name,
                "policy_label": self._config.viewer.policy_label,
                "policy_worker": self._config.policy.worker,
                "policy_adapter": self._config.policy.adapter,
                "action_dt_s": self._config.policy.effective_action_dt_s,
                "horizon_steps": self._config.policy.horizon_steps,
                "max_chunk_steps": self._config.execution.max_chunk_steps,
                "blend_steps": self._config.execution.blend_steps,
                "experiment_mode": self._config.run.experiment_mode,
                "layout_id": self._config.run.layout_id,
                "launch_mode": self._launch_mode,
                "policy_backend": {},
            },
            video_fps=self._config.recording.video_fps,
            video_codec=self._config.recording.video_codec,
            video_queue_size=self._config.recording.video_queue_size,
        )
        accepted_plans = 0
        rejected_plans = 0
        request_seq = 0
        last_submitted_seq = -1
        last_request_deadline_ns = 0
        request_in_flight = False
        last_inference_ms: float | None = None
        discard_responses_through = -1
        pending_visuals: dict[int, dict[str, object]] = {}
        worker_failure_reported = False
        robot_connected = False
        steps = 0
        completed = False
        terminal_reason = "completed"
        abort_reason = "runtime_exception"
        try:
            for sensor in self._sensors:
                sensor.start()
                sensor.read()
            self._worker.start()
            if self._decoder is not None:
                self._decoder.start()
            self._validate_policy_capabilities()
            capabilities = getattr(self._worker, "capabilities", PolicyCapabilities())
            recorder.update_metadata(
                policy_backend=dict(capabilities.backend_metadata),
            )
            self._robot.connect()
            robot_connected = True
            initial_state = self._robot.get_state()
            self._safety.reset(initial_state)
            self._executor.reset(initial_state)
            self._strategy.reset()
            previous_command = copy_group_vector(initial_state.groups)
            last_command = copy_group_vector(initial_state.groups)
            self._state = RuntimeState.RUNNING
            viewer_episode_metadata = {
                "episode_active": True,
                "episode_id": episode_id,
                "episode_dir": str(recorder.final_dir.resolve()),
                "run_dir": str(self._run_dir.resolve()),
                "instruction": self._config.run.task,
                "max_steps": self._config.run.max_steps,
                "control_mode": self._strategy.control_mode,
                "runtime": self._strategy.name,
                "executor": self._config.execution.executor,
                "policy_label": self._config.viewer.policy_label,
                "experiment_mode": self._config.run.experiment_mode,
                "layout_id": self._config.run.layout_id,
                "launch_mode": self._launch_mode,
            }
            self._viewer.set_state_metadata(viewer_episode_metadata)
            self._viewer.publish_event(
                "episode_started",
                metadata=viewer_episode_metadata,
            )
            next_tick_ns = self._clock.now_ns()

            while steps < self._config.run.max_steps:
                loop_start_ns = self._clock.now_ns()
                now_ns = loop_start_ns
                state = self._robot.get_state()
                self._safety.validate_state(state)
                frames: dict[str, SensorFrame] = {}
                for sensor in self._sensors:
                    reading = sensor.read()
                    batch = {reading.name: reading} if isinstance(reading, SensorFrame) else reading
                    overlap = set(frames).intersection(batch)
                    if overlap:
                        raise RuntimeError(f"duplicate sensor frames: {sorted(overlap)}")
                    frames.update(batch)

                viewer_control = self._viewer.poll_control()
                if viewer_control.finish_requested:
                    terminal_reason = "viewer_finish_requested"
                    recorder.event("viewer_finish_requested", step=steps)
                    break
                if viewer_control.home_requested:
                    self._robot.home()
                    state = self._robot.get_state()
                    self._safety.reset(state)
                    self._timeline = self._build_timeline()
                    self._executor.reset(state)
                    self._strategy.reset()
                    previous_command = copy_group_vector(state.groups)
                    last_command = copy_group_vector(state.groups)
                    discard_responses_through = max(discard_responses_through, request_seq)
                    self._state = RuntimeState.PAUSED
                    recorder.event("viewer_home_requested", step=steps)
                    next_tick_ns = self._clock.now_ns()
                    continue
                if viewer_control.paused and (
                    self._decoder is not None
                    or self._config.execution.inference_schedule == "serial"
                ):
                    self._timeline = self._build_timeline()
                    discard_responses_through = max(discard_responses_through, request_seq)
                self._state = (
                    RuntimeState.RUNNING if not viewer_control.paused else RuntimeState.PAUSED
                )

                if not self._worker.is_alive and not worker_failure_reported:
                    worker_failure_reported = True
                    recorder.event("policy_worker_stopped", step=steps)

                decoded_chunk = None
                response = self._worker.poll()
                if self._decoder is not None:
                    decoded = self._decoder.poll()
                    if decoded is not None:
                        if response is not None:
                            raise RuntimeError("model response arrived while decode was in flight")
                        response = decoded.response
                        decoded_chunk = decoded.chunk
                        if decoded.error is not None:
                            response = replace(response, error=decoded.error)
                    elif (
                        response is not None
                        and response.error is None
                        and response.session_id == self._session_id
                        and response.request_seq > discard_responses_through
                        and response.request_seq >= last_submitted_seq
                        and response.finished_time_ns <= last_request_deadline_ns
                        and response.raw_action is not None
                    ):
                        if self._clock.now_ns() > last_request_deadline_ns:
                            response = replace(response, error="deadline_exceeded_before_decode")
                        else:
                            self._decoder.submit(
                                response,
                                ActionContext(
                                    request_seq=response.request_seq,
                                    observation_time_ns=response.observation_time_ns,
                                    created_time_ns=response.finished_time_ns,
                                    execution_time_ns=self._clock.now_ns()
                                    + int(self._config.execution.commit_lead_s * 1e9),
                                    measured_state=state,
                                    max_source_steps=self._config.execution.max_chunk_steps,
                                    independent_groups=self._config.execution.independent_group_decoding,
                                    decode_budget_ms=(
                                        self._config.execution.decode_budget_ms
                                        if self._config.execution.independent_group_decoding
                                        else None
                                    ),
                                ),
                                last_request_deadline_ns,
                            )
                            recorder.event(
                                "decode_submitted",
                                request_seq=response.request_seq,
                                seed_time_ns=state.monotonic_ns,
                            )
                            # Keep inference+decode in flight until both arms finish.
                            response = None
                if response is not None:
                    request_in_flight = False
                    submission_visuals = pending_visuals.get(response.request_seq, {})
                    rejection_reason = None
                    if response.error is not None:
                        rejection_reason = response.error
                    elif (
                        response.session_id != self._session_id
                        or response.request_seq <= discard_responses_through
                        or response.request_seq < last_submitted_seq
                        or response.finished_time_ns > last_request_deadline_ns
                        or response.raw_action is None
                    ):
                        rejection_reason = "stale_or_expired_response"
                    if rejection_reason is not None:
                        self._strategy.on_response_rejected(response)
                        rejected_plans += 1
                        recorder.event(
                            "inference_rejected",
                            request_seq=response.request_seq,
                            reason=rejection_reason,
                        )
                        self._viewer.publish_event(
                            "inference_rejected",
                            step=steps,
                            chunk_id=response.request_seq,
                            metadata={"reason": rejection_reason},
                        )
                        pending_visuals.pop(response.request_seq, None)
                    else:
                        try:
                            decode_start = time.perf_counter_ns()
                            chunk = (
                                decoded_chunk
                                if decoded_chunk is not None
                                else decode_policy_action(
                                    self._adapter,
                                    response.raw_action,
                                    ActionContext(
                                        request_seq=response.request_seq,
                                        observation_time_ns=response.observation_time_ns,
                                        created_time_ns=response.finished_time_ns,
                                        execution_time_ns=(
                                            now_ns
                                            + int(
                                                self._config.execution.commit_lead_s * 1_000_000_000
                                            )
                                            if self._strategy.name == "manimux"
                                            else None
                                        ),
                                        measured_state=state,
                                        max_source_steps=self._config.execution.max_chunk_steps,
                                    ),
                                )
                            )
                            if decoded_chunk is None:
                                chunk.metadata["decode_ms"] = (
                                    time.perf_counter_ns() - decode_start
                                ) / 1e6
                        except (TypeError, ValueError) as exc:
                            self._strategy.on_response_rejected(response)
                            rejected_plans += 1
                            reason = f"invalid_action:{type(exc).__name__}:{exc}"
                            recorder.event(
                                "plan_rejected",
                                request_seq=response.request_seq,
                                reason=reason,
                            )
                            self._viewer.publish_event(
                                "plan_rejected",
                                step=steps,
                                chunk_id=response.request_seq,
                                metadata={"reason": reason},
                            )
                            pending_visuals.pop(response.request_seq, None)
                            chunk = None
                        now_ns = self._clock.now_ns()
                        if chunk is not None and chunk.action_space != "joint_position":
                            self._strategy.on_response_rejected(response)
                            rejected_plans += 1
                            recorder.event(
                                "plan_rejected",
                                request_seq=response.request_seq,
                                reason=(
                                    "invalid_action:canonical action_space must be "
                                    f"'joint_position', got {chunk.action_space!r}"
                                ),
                            )
                            self._viewer.publish_event(
                                "plan_rejected",
                                step=steps,
                                chunk_id=response.request_seq,
                                metadata={"reason": "invalid_action_space"},
                            )
                            pending_visuals.pop(response.request_seq, None)
                            chunk = None
                        canonical_raw = None if chunk is None else copy_action_chunk(chunk)
                        if chunk is not None:
                            try:
                                chunk = prepare_strategy_chunk(
                                    self._strategy,
                                    chunk=chunk,
                                    response=response,
                                    now_ns=now_ns,
                                )
                            except (TypeError, ValueError) as exc:
                                self._strategy.on_response_rejected(response)
                                rejected_plans += 1
                                reason = f"invalid_strategy_chunk:{type(exc).__name__}:{exc}"
                                recorder.event(
                                    "plan_rejected",
                                    request_seq=response.request_seq,
                                    reason=reason,
                                )
                                self._viewer.publish_event(
                                    "plan_rejected",
                                    step=steps,
                                    chunk_id=response.request_seq,
                                    metadata={"reason": reason},
                                )
                                pending_visuals.pop(response.request_seq, None)
                                chunk = None
                        if chunk is not None:
                            previous_reference = self._timeline.sample(now_ns)
                            previous_horizon = self._timeline.active_horizon()
                            previous_chunk_id = (
                                None
                                if self._timeline.accepted_request_seq < 0
                                else self._timeline.accepted_request_seq
                            )
                            previous_chunk_index = self._timeline.cursor(now_ns)
                            commit = self._strategy.commit_settings(
                                response=response,
                                measured=state.groups,
                                last_command=last_command,
                            )
                            # FK/decoding may consume time. Crop against the actual commit time.
                            now_ns = self._clock.now_ns()
                            chunk.metadata["observation_to_commit_ms"] = (
                                now_ns - chunk.observation_time_ns
                            ) / 1e6
                            commit_lead_ns = int(self._config.execution.commit_lead_s * 1e9)
                            source_end_ns = (
                                chunk.observation_time_ns
                                + (chunk.source_offset_steps + chunk.horizon_steps - 1)
                                * chunk.dt_ns
                            )
                            if (
                                self._decoder is not None
                                and now_ns + commit_lead_ns > source_end_ns
                            ):
                                result = CommitResult(False, "no_future_horizon")
                            else:
                                result = self._timeline.commit(
                                    chunk,
                                    now_ns=now_ns,
                                    commit_lead_ns=commit_lead_ns,
                                    max_plan_age_ns=int(
                                        self._config.execution.max_plan_age_s * 1_000_000_000
                                    ),
                                    current_command=commit.current_command,
                                    blend_steps=commit.blend_steps,
                                )
                            if result.accepted:
                                accepted_plans += 1
                                last_inference_ms = response.inference_ms
                                committed = self._timeline.active_horizon()
                                if committed is None:
                                    raise RuntimeError("accepted plan missing committed horizon")
                                if canonical_raw is None:
                                    raise RuntimeError("accepted plan missing canonical raw chunk")
                                recorder.record_plan(
                                    canonical_raw=canonical_raw,
                                    infra_output=chunk,
                                    committed=committed,
                                )
                                event_fields = self._strategy.on_plan_accepted(
                                    chunk=chunk,
                                    result=result,
                                    response=response,
                                    now_ns=now_ns,
                                )
                                recorder.event(
                                    "plan_accepted",
                                    plan_id=chunk.plan_id,
                                    request_seq=chunk.request_seq,
                                    **chunk.metadata,
                                    **event_fields,
                                )
                                recorder.event(
                                    "plan_boundary",
                                    **build_plan_boundary_payload(
                                        step=steps,
                                        monotonic_ns=now_ns,
                                        blend_anchor_source=commit.anchor_source,
                                        blend_steps=commit.blend_steps,
                                        trimmed_steps=result.trimmed_steps,
                                        previous_reference=previous_reference,
                                        previous_command=previous_command,
                                        last_command=last_command,
                                        measured=state.groups,
                                        chunk=chunk,
                                        committed=committed,
                                    ),
                                )
                                self._viewer.publish_plan(
                                    chunk,
                                    response.inference_ms,
                                    committed=committed,
                                    metadata={
                                        "runtime": self._strategy.name,
                                        "raw_horizon_steps": chunk.horizon_steps,
                                        "committed_horizon_steps": committed.horizon_steps,
                                        "trimmed_steps": result.trimmed_steps,
                                        "previous_chunk_id": previous_chunk_id,
                                        "previous_chunk_index": previous_chunk_index,
                                        "previous_chunk_horizon_steps": (
                                            0
                                            if previous_horizon is None
                                            else previous_horizon.horizon_steps
                                        ),
                                        "superseded_steps": (
                                            0
                                            if previous_horizon is None
                                            else max(
                                                0,
                                                previous_horizon.horizon_steps
                                                - previous_chunk_index,
                                            )
                                        ),
                                        **submission_visuals,
                                        **event_fields,
                                    },
                                )
                                pending_visuals.pop(response.request_seq, None)
                            else:
                                self._strategy.on_response_rejected(response)
                                rejected_plans += 1
                                recorder.event(
                                    "plan_rejected",
                                    plan_id=chunk.plan_id,
                                    request_seq=chunk.request_seq,
                                    reason=result.reason,
                                )
                                self._viewer.publish_event(
                                    "plan_rejected",
                                    step=steps,
                                    chunk_id=response.request_seq,
                                    metadata={"reason": result.reason},
                                )
                                pending_visuals.pop(response.request_seq, None)

                if self._worker.is_alive and not (
                    self._decoder is not None
                    and self._state != RuntimeState.RUNNING
                ):
                    snapshot = ObservationSnapshot(state=state, frames=frames)
                    submission = self._strategy.build_submission(
                        session_id=self._session_id,
                        request_seq=request_seq + 1,
                        now_ns=now_ns,
                        snapshot=snapshot,
                        adapter=self._adapter,
                        timeline=self._timeline,
                        request_state=RequestState(
                            in_flight=request_in_flight,
                            last_submitted_seq=last_submitted_seq,
                            last_deadline_ns=last_request_deadline_ns,
                        ),
                        runtime_state=self._state,
                    )
                    if submission is not None:
                        prepared_request = prepare_policy_request(
                            self._adapter,
                            submission.request,
                        )
                        request_seq = submission.request.request_seq
                        self._worker.submit_latest(prepared_request)
                        request_in_flight = True
                        last_submitted_seq = request_seq
                        last_request_deadline_ns = prepared_request.deadline_ns
                        recorder.event(
                            "inference_submitted",
                            request_seq=request_seq,
                            **submission.event_fields,
                        )
                        active_horizon = self._timeline.active_horizon()
                        active_chunk_id = (
                            None
                            if self._timeline.accepted_request_seq < 0
                            else self._timeline.accepted_request_seq
                        )
                        active_chunk_index = self._timeline.cursor(now_ns)
                        visual_fields: dict[str, object] = {
                            "runtime": self._strategy.name,
                            "horizon_steps": self._config.policy.horizon_steps,
                            "active_chunk_id": active_chunk_id,
                            "active_chunk_index": active_chunk_index,
                            "active_horizon_steps": (
                                0 if active_horizon is None else active_horizon.horizon_steps
                            ),
                            **submission.event_fields,
                        }
                        if bool(submission.event_fields.get("conditioned", False)):
                            executed_steps = int(submission.event_fields.get("executed_steps", 0))
                            visual_fields["conditioned_overlap_steps"] = max(
                                0, self._config.policy.horizon_steps - executed_steps
                            )
                            visual_fields["frozen_steps"] = int(
                                submission.event_fields.get("forecast_delay", 0)
                            )
                        pending_visuals[request_seq] = visual_fields
                        self._viewer.publish_event(
                            "inference_submitted",
                            step=steps,
                            chunk_id=request_seq,
                            metadata=visual_fields,
                        )
                for kind, fields in self._strategy.take_runtime_events(step=steps):
                    recorder.event(kind, **fields)
                    self._viewer.publish_event(kind, step=steps, metadata=fields)

                now_ns = self._clock.now_ns()
                reference = self._timeline.reference_horizon(
                    now_ns=now_ns,
                    dt_ns=self._control_dt_ns,
                    horizon_steps=self._executor.horizon_steps,
                )
                scheduled = copy_group_vector(state.groups)
                if self._state == RuntimeState.RUNNING and reference is not None:
                    scheduled = {
                        name: values[0].copy() for name, values in reference.groups.items()
                    }
                    command = self._executor.step(now_ns, state, reference)
                    if (
                        isinstance(self._executor, SmoothExecutor)
                        and self._config.execution.smooth.release_guard is not None
                    ):
                        recorder.event(
                            "gripper_decision", step=steps, monotonic_ns=now_ns,
                            plan_id=reference.plan_id, groups=self._executor.gripper_diagnostics,
                        )
                elif (
                    self._state == RuntimeState.RUNNING
                    and self._config.execution.inference_schedule == "serial"
                ):
                    # Keep the last command fixed while waiting for the next chunk.
                    # Reset executor velocity history to the held command, so a new
                    # chunk does not resume with velocity left over before the wait.
                    self._executor.reset(RobotState(
                        groups=copy_group_vector(last_command),
                        monotonic_ns=now_ns, sequence=state.sequence,
                    ))
                    command = self._hold_command(now_ns, last_command)
                elif (
                    self._state == RuntimeState.RUNNING
                    and isinstance(self._executor, SmoothExecutor)
                    and self._executor.braking_tracking
                ):
                    command = self._executor.brake_hold(now_ns, state)
                    if self._executor.has_pending_gripper_event:
                        recorder.event(
                            "gripper_decision", step=steps, monotonic_ns=now_ns,
                            plan_id=None, groups=self._executor.gripper_diagnostics,
                        )
                else:
                    self._executor.reset(state)
                    self._safety.reset(state)
                    command = self._hold_command(now_ns, state.groups)
                self._safety.validate_command(command)
                self._robot.send_command(command)
                previous_command = copy_group_vector(last_command)
                last_command = copy_group_vector(command.groups)
                self._viewer.publish_state(
                    state,
                    frames,
                    step=steps,
                    max_steps=self._config.run.max_steps,
                    chunk_index=self._timeline.cursor(now_ns),
                    active_chunk_id=(
                        None
                        if self._timeline.accepted_request_seq < 0
                        else self._timeline.accepted_request_seq
                    ),
                )
                if self._state == RuntimeState.RUNNING:
                    recorder.record_tick(
                        monotonic_ns=now_ns,
                        state=state,
                        scheduled=scheduled,
                        optimized=command.groups,
                        command=command.groups,
                        plan_id=command.plan_id,
                        inference_ms=last_inference_ms,
                        camera_times_ns={
                            name: frame.capture_monotonic_ns for name, frame in frames.items()
                        },
                        frames=frames,
                    )
                    steps += 1

                self._strategy.on_tick(
                    steps=steps,
                    loop_ms=(self._clock.now_ns() - loop_start_ns) / 1e6,
                    control_dt_ns=self._control_dt_ns,
                )
                next_tick_ns += self._control_dt_ns
                finished_tick_ns = self._clock.now_ns()
                if next_tick_ns <= finished_tick_ns:
                    if self._state == RuntimeState.RUNNING:
                        recorder.event(
                            "control_overrun",
                            lag_ns=finished_tick_ns - next_tick_ns,
                            step=steps,
                        )
                    next_tick_ns = finished_tick_ns + self._control_dt_ns
                self._clock.sleep_until_ns(next_tick_ns)

            episode_dir = recorder.finish(
                success=True,
                terminal_reason=terminal_reason,
                steps=steps,
                wall_time_s=time.perf_counter() - started_wall,
            )
            self._viewer.publish_event(
                "episode_finished",
                step=steps,
                metadata={
                    "episode_id": episode_id,
                    "episode_dir": str(episode_dir.resolve()),
                    "reason": terminal_reason,
                    "launch_mode": self._launch_mode,
                },
            )
            completed = True
            return RunResult(
                episode_dir=episode_dir,
                steps=steps,
                accepted_plans=accepted_plans,
                rejected_plans=rejected_plans,
                success=True,
                terminal_reason=terminal_reason,
            )
        except KeyboardInterrupt:
            abort_reason = "KeyboardInterrupt"
            raise
        except BaseException as exc:
            abort_reason = type(exc).__name__
            raise
        finally:
            faulted = not completed and abort_reason != "KeyboardInterrupt"
            self._state = RuntimeState.IDLE
            cleanup_errors: list[BaseException] = []
            stopped_cleanly = True
            try:
                self._robot.stop()
            except BaseException as exc:
                stopped_cleanly = False
                cleanup_errors.append(exc)
            if (
                robot_connected
                and stopped_cleanly
                and not faulted
                and bool(self._config.robot.options.get("home_on_close", False))
            ):
                try:
                    self._robot.home()
                except BaseException as exc:
                    cleanup_errors.append(exc)
            closers = [self._robot.close, self._worker.close]
            if self._decoder is not None:
                closers.append(self._decoder.close)
            for closer in closers:
                try:
                    closer()
                except BaseException as exc:
                    cleanup_errors.append(exc)
            for sensor in self._sensors:
                try:
                    sensor.close()
                except BaseException as exc:
                    cleanup_errors.append(exc)
            try:
                self._viewer.close()
            except BaseException as exc:
                cleanup_errors.append(exc)
            if not completed:
                try:
                    recorder.abort(abort_reason)
                except BaseException as exc:
                    cleanup_errors.append(exc)
            if cleanup_errors:
                raise BaseExceptionGroup("runtime cleanup failed", cleanup_errors)
