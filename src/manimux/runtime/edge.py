from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from json import dumps, loads
from pathlib import Path

import numpy as np

from manimux.clock import Clock, SystemClock
from manimux.embodiments.robot import RobotBase, build_robot
from manimux.embodiments.sensor import build_sensor
from manimux.policies import ActionDecoderClient, PolicyCapabilities, build_policy_adapter
from manimux.policies.base import action_interval, decode_policy_action, prepare_policy_request
from manimux.policies.worker import PolicyWorkerClient
from manimux.recording import EpisodeRecorder
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

logger = logging.getLogger(__name__)


def _action_value(value: object) -> object:
    """Compact numeric action values without dumping an entire chunk."""
    try:
        array = np.asarray(value)
    except Exception:  # noqa: BLE001 - diagnostics must never alter execution
        return type(value).__name__
    if array.dtype.kind not in "biuf" or array.size > 16:
        return {"type": type(value).__name__, "shape": list(array.shape)}
    return np.round(array.astype(float), 5).tolist()


def _action_step(step: object) -> object:
    if not isinstance(step, Mapping):
        return _action_value(step)
    return {str(key): _action_value(value) for key, value in step.items()}


def _raw_action_summary(raw: object) -> dict[str, object]:
    actions = raw.get("actions") if isinstance(raw, Mapping) and "actions" in raw else raw
    if isinstance(actions, Sequence) and not isinstance(actions, str | bytes):
        if not actions:
            return {"type": type(raw).__name__, "steps": 0}
        return {
            "type": type(raw).__name__,
            "steps": len(actions),
            "first": _action_step(actions[0]),
            "last": _action_step(actions[-1]),
        }
    return {"type": type(raw).__name__}


def _group_action_summary(groups: Mapping[str, np.ndarray]) -> dict[str, object]:
    return {
        name: {
            "shape": list(np.asarray(values).shape),
            "first": np.round(np.asarray(values)[0], 5).tolist(),
            "last": np.round(np.asarray(values)[-1], 5).tolist(),
        }
        for name, values in groups.items()
    }


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
        config: dict,
        run_dir: Path,
        *,
        clock: Clock | None = None,
        strategy: InferenceStrategy | None = None,
        launch_mode: str = "run",
    ) -> None:
        self._config = config
        self._run_dir = run_dir
        self._clock = clock or SystemClock()
        self._control_dt_ns = int(1_000_000_000 / config["robot"]["control_hz"])
        self._robot = self._build_robot()
        self._sensors = [build_sensor(sensor, self._clock) for sensor in config["sensors"]]
        # New embodiment adapters share robot FK/IK. Legacy factories keep their
        # existing call signature; decoder subprocesses reconstruct offline models.
        if isinstance(self._robot, RobotBase):
            self._adapter = build_policy_adapter(
                config["robot"], config["policy"], kinematics=self._robot.kinematics
            )
        else:
            self._adapter = build_policy_adapter(config["robot"], config["policy"])
        self._adapter.validate(config["robot"], config["policy"])
        if config["execution"]["independent_group_decoding"] and not getattr(
            self._adapter, "supports_independent_group_decode", False
        ):
            raise ValueError("adapter does not support independent group decoding")
        self._strategy = strategy or DefaultChunkStrategy(config)
        self._decoder = None
        if config["policy"]["action_decoding"] == "process":
            # A plugin may wrap the default strategy (the UMI history plugin
            # does); the constructed strategy decides, not the plugin path.
            if self._strategy.name not in {"manimux", "rtc"}:
                raise ValueError("process action decoding requires the manimux or rtc strategy")
            self._decoder = ActionDecoderClient(config["robot"], config["policy"], self._adapter)
        self._session_id = f"session-{uuid.uuid4().hex}"
        self._worker = PolicyWorkerClient(config["policy"], self._session_id)
        self._timeline = self._build_timeline()
        self._executor = self._build_executor()
        self._launch_mode = launch_mode
        position_limit_abs = None
        if config["execution"]["executor"] == "smooth":
            position_limit_abs = config["execution"]["smooth"]["position_limit_abs"]
        elif config["execution"]["executor"] == "mpc":
            position_limit_abs = config["execution"]["mpc"]["position_limit_abs"]
        command_safety = config["execution"]["command_safety"]
        self._safety = SafetyGuard(
            config["robot"]["group_dims"],
            position_limit_abs,
            position_lower=command_safety["position_lower"],
            position_upper=command_safety["position_upper"],
            max_velocity=command_safety["max_velocity"],
            max_acceleration=command_safety["max_acceleration"],
            control_dt_s=self._control_dt_ns / 1_000_000_000,
        )
        self._viewer = ViewerBridge(
            enabled=config["viewer"]["enabled"],
            robot=config["viewer"]["robot"],
            policy=config["viewer"]["policy_label"] or "manimux-local",
            instruction=config["run"]["task"] if config["viewer"]["policy_label"] else "",
            camera_hz=config["viewer"]["camera_hz"],
        )
        self._state = RuntimeState.DISCONNECTED
        logger.info(
            "runtime_config robot=%s execute=%s end_effector_control=%s worker=%s "
            "policy_endpoint=%s strategy=%s executor=%s",
            config["robot"].get("type", config["robot"].get("driver")),
            config["robot"]["options"].get("execute"),
            config["robot"]["options"].get("end_effector_control"),
            config["policy"]["worker"],
            config["policy"]["options"].get("server"),
            self._strategy.name,
            config["execution"]["executor"],
        )

    def _build_robot(self) -> RobotBase:
        return build_robot(self._config["robot"], self._clock)

    def _build_executor(self) -> Executor:
        control_dt_s = self._control_dt_ns / 1_000_000_000
        if self._config["execution"]["executor"] == "direct":
            return DirectExecutor(self._config["execution"]["motion_limits"], control_dt_s)
        if self._config["execution"]["executor"] == "smooth":
            return SmoothExecutor(self._config["execution"]["smooth"], control_dt_s)
        if self._config["execution"]["motion_limits"] is not None:
            raise ValueError("shared motion_limits currently support direct and smooth, not mpc")
        return MPCExecutor(self._config["execution"]["mpc"], control_dt_s)

    def _build_timeline(self) -> ActionTimeline:
        return ActionTimeline(
            self._config["robot"]["group_dims"],
            max_source_steps=self._config["execution"]["max_chunk_steps"],
            start_on_commit=self._config["execution"]["inference_schedule"] == "serial",
        )

    def _hold_command(self, now_ns: int, groups: GroupVector) -> RobotCommand:
        return RobotCommand(
            groups=copy_group_vector(groups),
            monotonic_ns=now_ns,
            plan_id=self._timeline.active_plan_id,
        )

    def _decode_seed(self, state: RobotState, now_ns: int) -> tuple[int, RobotState, str]:
        """Expected plan start and IK seed for a process decode.

        The arm keeps following the active plan while decoding runs, so a seed
        measured at submission starts the new plan behind the arm. With
        expected_decode_s the seed is the active reference at the expected start,
        or at its end if it finishes first. Without an active reference the arm
        holds still and the measurement remains the seed.
        """
        execution = self._config["execution"]
        start_ns = now_ns + int((execution["commit_lead_s"] + execution["expected_decode_s"]) * 1e9)
        remaining_ns = self._timeline.remaining_ns(now_ns)
        if execution["expected_decode_s"] > 0 and remaining_ns > 0:
            reference = self._timeline.sample(min(start_ns, now_ns + remaining_ns))
            if reference is not None:
                return start_ns, RobotState(reference, start_ns, state.sequence), "active_reference"
        return start_ns, state, "measured_state"

    def _validate_policy_capabilities(self) -> None:
        capabilities = getattr(self._worker, "capabilities", PolicyCapabilities())
        missing = self._strategy.required_sampling_modes.difference(capabilities.sampling_modes)
        if missing:
            raise RuntimeError(
                f"execution strategy {self._strategy.name!r} requires sampling modes "
                f"{sorted(missing)} that the policy server does not advertise"
            )
        expected = self._config["policy"]["expected_backend"]
        if expected is None:
            return
        expected_metadata = {
            key: value for key, value in deepcopy(expected).items() if value is not None
        }
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
            self._config["robot"]["group_dims"],
            metadata={
                "episode_id": episode_id,
                "session_id": self._session_id,
                "task": self._config["run"]["task"],
                "executor_kind": self._config["execution"]["executor"],
                "smooth": (
                    loads(dumps(deepcopy(self._config["execution"]["smooth"]), default=str))
                    if self._config["execution"]["executor"] == "smooth"
                    else None
                ),
                "runtime": self._strategy.name,
                "policy_label": self._config["viewer"]["policy_label"],
                "policy_worker": self._config["policy"]["worker"],
                "policy_adapter": self._config["policy"]["adapter"],
                "view_profile": self._config["policy"]["options"].get("view_profile"),
                "camera_map": self._config["policy"]["options"].get("camera_map", {}),
                "action_dt_s": action_interval(self._config["policy"]),
                "horizon_steps": self._config["policy"]["horizon_steps"],
                "max_chunk_steps": self._config["execution"]["max_chunk_steps"],
                "blend_steps": self._config["execution"]["blend_steps"],
                "experiment_mode": self._config["run"]["experiment_mode"],
                "layout_id": self._config["run"]["layout_id"],
                "launch_mode": self._launch_mode,
                "policy_backend": {},
            },
            video_fps=self._config["recording"]["video_fps"],
            video_codec=self._config["recording"]["video_codec"],
            video_queue_size=self._config["recording"]["video_queue_size"],
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
        last_dispatch_log_ns = 0
        last_dispatch_plan_id: object = object()
        robot_connected = False
        steps = 0
        completed = False
        terminal_reason = "completed"
        abort_reason = "runtime_exception"
        home_on_close = bool(self._config["robot"]["options"].get("home_on_close", False))
        try:
            for sensor in self._sensors:
                sensor.start()
                sensor.read()
            self._worker.start()
            logger.info("policy_worker_ready session=%s", self._session_id)
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
            logger.info(
                "robot_connected groups=%s initial=%s",
                list(initial_state.groups),
                {
                    name: np.round(values, 5).tolist()
                    for name, values in initial_state.groups.items()
                },
            )
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
                "instruction": self._config["run"]["task"],
                "max_steps": self._config["run"]["max_steps"],
                "control_mode": self._strategy.control_mode,
                "runtime": self._strategy.name,
                "executor": self._config["execution"]["executor"],
                "policy_label": self._config["viewer"]["policy_label"],
                "experiment_mode": self._config["run"]["experiment_mode"],
                "camera_map": self._config["policy"]["options"].get("camera_map", {}),
                "layout_id": self._config["run"]["layout_id"],
                "launch_mode": self._launch_mode,
            }
            viewer_episode_metadata["recovery_available"] = False
            self._viewer.set_state_metadata(viewer_episode_metadata)
            self._viewer.publish_event(
                "episode_started",
                metadata=viewer_episode_metadata,
            )
            next_tick_ns = self._clock.now_ns()

            while steps < self._config["run"]["max_steps"]:
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
                    if viewer_control.finish_home is not None:
                        home_on_close = viewer_control.finish_home
                    terminal_reason = "viewer_finish_requested"
                    recorder.event("viewer_finish_requested", step=steps, home=home_on_close)
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
                    or self._config["execution"]["inference_schedule"] == "serial"
                    or getattr(self._strategy, "discard_plans_while_paused", False)
                ):
                    # RTC conditions must not refer to the timeline discarded
                    # by Pause/Hold while a decoder response is still pending.
                    if self._state != RuntimeState.PAUSED:
                        self._strategy.reset()
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
                if response is not None:
                    logger.info(
                        "policy_response seq=%d inference_ms=%.1f error=%s raw=%s",
                        response.request_seq,
                        response.inference_ms,
                        response.error,
                        _raw_action_summary(response.raw_action),
                    )
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
                            start_ns, seed, seed_source = self._decode_seed(
                                state, self._clock.now_ns()
                            )
                            self._decoder.submit(
                                response,
                                ActionContext(
                                    request_seq=response.request_seq,
                                    observation_time_ns=response.observation_time_ns,
                                    created_time_ns=response.finished_time_ns,
                                    execution_time_ns=start_ns,
                                    measured_state=seed,
                                    max_source_steps=self._config["execution"]["max_chunk_steps"],
                                    independent_groups=self._config["execution"][
                                        "independent_group_decoding"
                                    ],
                                    decode_budget_ms=(
                                        self._config["execution"]["decode_budget_ms"]
                                        if self._config["execution"]["independent_group_decoding"]
                                        else None
                                    ),
                                ),
                                last_request_deadline_ns,
                            )
                            recorder.event(
                                "decode_submitted",
                                request_seq=response.request_seq,
                                seed_time_ns=seed.monotonic_ns,
                                seed_source=seed_source,
                                expected_start_ns=start_ns,
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
                        logger.warning(
                            "inference_rejected seq=%d reason=%s",
                            response.request_seq,
                            rejection_reason,
                        )
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
                                                self._config["execution"]["commit_lead_s"]
                                                * 1_000_000_000
                                            )
                                            if self._strategy.name in {"manimux", "rtc"}
                                            else None
                                        ),
                                        measured_state=state,
                                        max_source_steps=self._config["execution"][
                                            "max_chunk_steps"
                                        ],
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
                            logger.warning(
                                "action_decode_rejected seq=%d reason=%s",
                                response.request_seq,
                                reason,
                            )
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
                            logger.info(
                                "action_decoded seq=%d plan=%s source_offset=%d "
                                "dt_ms=%.3f groups=%s",
                                chunk.request_seq,
                                chunk.plan_id,
                                chunk.source_offset_steps,
                                chunk.dt_ns / 1e6,
                                _group_action_summary(chunk.groups),
                            )
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
                            commit_lead_ns = int(self._config["execution"]["commit_lead_s"] * 1e9)
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
                                        self._config["execution"]["max_plan_age_s"] * 1_000_000_000
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
                                logger.info(
                                    "plan_accepted seq=%d plan=%s trimmed=%d committed_steps=%d",
                                    chunk.request_seq,
                                    chunk.plan_id,
                                    result.trimmed_steps,
                                    committed.horizon_steps,
                                )
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
                                logger.warning(
                                    "plan_rejected seq=%d plan=%s reason=%s",
                                    chunk.request_seq,
                                    chunk.plan_id,
                                    result.reason,
                                )
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
                    (
                        self._decoder is not None
                        or getattr(self._strategy, "discard_plans_while_paused", False)
                    )
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
                        logger.info(
                            "inference_submitted seq=%d observation_ns=%d deadline_ns=%d "
                            "state_seq=%d cameras=%s",
                            request_seq,
                            prepared_request.observation_time_ns,
                            prepared_request.deadline_ns,
                            state.sequence,
                            {
                                name: {
                                    "sequence": frame.sequence,
                                    "capture_ns": frame.capture_monotonic_ns,
                                }
                                for name, frame in frames.items()
                            },
                        )
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
                            "horizon_steps": self._config["policy"]["horizon_steps"],
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
                                0, self._config["policy"]["horizon_steps"] - executed_steps
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
                    if isinstance(self._executor, SmoothExecutor) and (
                        self._config["execution"]["smooth"]["release_guard"] is not None
                        or self._executor.uses_close_latch
                    ):
                        recorder.event(
                            "gripper_decision",
                            step=steps,
                            monotonic_ns=now_ns,
                            plan_id=reference.plan_id,
                            groups=self._executor.gripper_diagnostics,
                        )
                elif (
                    self._state == RuntimeState.RUNNING
                    and self._config["execution"]["inference_schedule"] == "serial"
                ):
                    # Keep the last command fixed while waiting for the next chunk.
                    # Reset executor velocity history to the held command, so a new
                    # chunk does not resume with velocity left over before the wait.
                    held_state = RobotState(
                        groups=copy_group_vector(last_command),
                        monotonic_ns=now_ns,
                        sequence=state.sequence,
                    )
                    if isinstance(self._executor, SmoothExecutor):
                        command = self._executor.hold(now_ns, held_state)
                        command.plan_id = self._timeline.active_plan_id
                    else:
                        self._executor.reset(held_state)
                        command = self._hold_command(now_ns, last_command)
                elif (
                    self._state == RuntimeState.RUNNING
                    and isinstance(self._executor, SmoothExecutor)
                    and self._executor.braking_tracking
                ):
                    command = self._executor.brake_hold(now_ns, state)
                    if self._executor.has_pending_gripper_event:
                        recorder.event(
                            "gripper_decision",
                            step=steps,
                            monotonic_ns=now_ns,
                            plan_id=None,
                            groups=self._executor.gripper_diagnostics,
                        )
                else:
                    if self._state == RuntimeState.RUNNING and isinstance(
                        self._executor, SmoothExecutor
                    ):
                        command = self._executor.hold(now_ns, state)
                        command.plan_id = self._timeline.active_plan_id
                    else:
                        self._executor.reset(state)
                        command = self._hold_command(now_ns, state.groups)
                    # Arm holds retain their measured anchor; a latched gripper
                    # retains its already-sent command, which can differ at contact.
                    self._safety.reset(
                        RobotState(
                            copy_group_vector(command.groups),
                            now_ns,
                            state.sequence,
                        )
                    )
                self._safety.validate_command(command)
                log_dispatch = (
                    command.plan_id != last_dispatch_plan_id
                    or now_ns - last_dispatch_log_ns >= 1_000_000_000
                )
                if log_dispatch:
                    logger.info(
                        "command_ready runtime_state=%s plan=%s execute=%s "
                        "end_effector_control=%s max_command_minus_state=%s",
                        self._state.value,
                        command.plan_id,
                        self._config["robot"]["options"].get("execute"),
                        self._config["robot"]["options"].get("end_effector_control"),
                        {
                            name: round(float(np.max(np.abs(values - state.groups[name]))), 6)
                            for name, values in command.groups.items()
                        },
                    )
                try:
                    self._robot.send_command(command)
                except Exception:
                    logger.exception(
                        "command_dispatch_failed runtime_state=%s plan=%s execute=%s "
                        "end_effector_control=%s groups=%s",
                        self._state.value,
                        command.plan_id,
                        self._config["robot"]["options"].get("execute"),
                        self._config["robot"]["options"].get("end_effector_control"),
                        list(command.groups),
                    )
                    raise
                if log_dispatch:
                    logger.info(
                        "command_sent plan=%s physical_dispatch=%s",
                        command.plan_id,
                        self._config["robot"]["options"].get("execute") is True,
                    )
                    last_dispatch_log_ns = now_ns
                    last_dispatch_plan_id = command.plan_id
                previous_command = copy_group_vector(last_command)
                last_command = copy_group_vector(command.groups)
                self._viewer.publish_state(
                    state,
                    frames,
                    step=steps,
                    max_steps=self._config["run"]["max_steps"],
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
            if robot_connected and stopped_cleanly and not faulted and home_on_close:
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
