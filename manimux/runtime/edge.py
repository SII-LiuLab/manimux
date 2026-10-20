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
from manimux.configuration import initial_pose_metadata, initial_pose_targets
from manimux.embodiments.robot import RobotBase, build_robot
from manimux.embodiments.robot.release import observe_release
from manimux.embodiments.sensor import build_sensor
from manimux.embodiments.sensor.reader import SensorReader
from manimux.evaluation.identity import rollout_identity, start_identity
from manimux.evaluation.rubric import evaluation_parameters
from manimux.failures import failure_context
from manimux.policies import ActionDecoderClient, PolicyCapabilities, metadata_mismatches
from manimux.policies.base import action_interval
from manimux.policies.worker import PolicyWorkerClient
from manimux.policy_adapter import build_policy_adapter
from manimux.recording import EpisodeRecorder
from manimux.recording.provenance import state_evidence
from manimux.robogui import RoboGUIBridge
from manimux.runtime.decode_forecast import DecodeForecast
from manimux.runtime.diagnostics import build_plan_boundary_payload
from manimux.runtime.execution import ExecutionPipeline
from manimux.runtime.executors import DirectExecutor, Executor, MPCExecutor, SmoothExecutor
from manimux.runtime.ik import step_ik_enabled, validate_ik_config
from manimux.runtime.inference import (
    DefaultChunkStrategy,
    InferenceStrategy,
    RequestState,
    StrategyChunkSelector,
    build_inference_strategy,
    decode_strategy_action,
    prepare_strategy_chunk,
    seed_strategy_warmup,
)
from manimux.runtime.lifecycle import OperationStatus, ShutdownStatus
from manimux.runtime.planning import PlanningContext
from manimux.runtime.safety import RuntimeState, SafetyGuard
from manimux.runtime.scheduling import RequestScheduler
from manimux.runtime.timeline import CommitResult
from manimux.runtime.warmup import PolicyWarmup
from manimux.timing import LoopTiming, stage
from manimux.types import (
    GroupVector,
    ObservationSnapshot,
    RobotCommand,
    RobotState,
    copy_action_chunk,
    copy_group_vector,
)

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
    if isinstance(raw, Mapping) and raw.get("format") in {"joint", "pose"}:
        return {"type": raw["format"], "groups": _group_action_summary(actions)}
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


def _exception_detail(exc: BaseException, *, depth: int = 3) -> str:
    """Render a failure in one line, expanding groups and the chained cause.

    ``robot/base.py`` reports a failed command as an ExceptionGroup holding both
    the original error and the emergency stop error, so the type name alone says
    nothing about what actually went wrong.
    """
    text = f"{type(exc).__name__}: {exc}"
    origin = getattr(exc, "manimux_failure", None)
    if origin:
        location = " / ".join(str(origin[key]) for key in ("source", "stage", "target") if origin.get(key))
        text = f"[{location}] {text}"
    if depth <= 0:
        return text
    if isinstance(exc, BaseExceptionGroup):
        # exc.message drops str()'s redundant "(N sub-exceptions)" suffix.
        inner = "; ".join(_exception_detail(item, depth=depth - 1) for item in exc.exceptions)
        return f"{type(exc).__name__}: {exc.message} [{inner}]"
    cause = exc.__cause__
    if cause is None and not exc.__suppress_context__:
        cause = exc.__context__
    if cause is None:
        return text
    return f"{text} <- {_exception_detail(cause, depth=depth - 1)}"


@dataclass(frozen=True, slots=True)
class RunResult:
    """Runtime completion only; human task outcomes are stored separately."""

    episode_dir: Path
    steps: int
    accepted_plans: int
    rejected_plans: int
    run_completed: bool
    terminal_reason: str


def _next_rollout_id(run_dir: Path) -> str:
    highest = 0
    pattern = re.compile(r"^rollout-(\d+)(?:\.partial|\.deleted)?$")
    if not run_dir.exists():
        return "rollout-001"
    for path in run_dir.iterdir():
        match = pattern.fullmatch(path.name)
        if match is not None:
            highest = max(highest, int(match.group(1)))
    return f"rollout-{highest + 1:03d}"


class EdgeRuntime(PlanningContext):
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
        self._rollout_identity = rollout_identity(config["run"])
        self._evaluation = evaluation_parameters(config.get("evaluation"))
        self._config = config
        self._run_dir = run_dir
        self._clock = clock or SystemClock()
        self._control_dt_ns = int(1_000_000_000 / config["robot"]["control_hz"])
        validate_ik_config(config)
        self._step_ik = step_ik_enabled(config)
        self._robot = self._build_robot()
        self._sensors = [build_sensor(sensor, self._clock) for sensor in config["sensors"]]
        # Runtime reuses robot kinematics; decoding subprocesses load the offline
        # model from the same configuration.
        self._adapter = self._build_adapter()
        if (
            (config["inference"].get("temporal_ensemble") or {}).get("enabled", False)
            or (
                config["inference"]["algorithm"] == "act_temporal_ensemble"
                and config["inference"]["inference_schedule"] == "serial"
            )
            or (
                config["inference"]["algorithm"] in {"aac", "autohorizon"}
                and config["inference"]["inference_schedule"] != "serial"
            )
        ) and not getattr(self._adapter, "supports_candidate_decoding", False):
            raise ValueError("inference composition requires complete absolute-joint decoding")
        self._scheduler = RequestScheduler(config["inference"])
        self._multi_inflight = self._scheduler.multi_inflight
        self._min_execute_steps = config["inference"].get("min_plan_hold_policy_steps", 0)
        if self._multi_inflight and not getattr(self._adapter, "supports_multi_inflight", False):
            raise ValueError("multi_inflight requires an adapter with streaming-safe decoding")
        self._decode_seed_source = getattr(
            self._adapter, "decode_seed_source", "execution_reference"
        )
        if self._multi_inflight and self._decode_seed_source != "execution_reference":
            raise ValueError("multi_inflight does not retain request-observation decode seeds")
        if self._decode_seed_source not in {"execution_reference", "observation_state"}:
            raise ValueError(
                "adapter decode_seed_source must be 'execution_reference' or "
                f"'observation_state', got {self._decode_seed_source!r}"
            )
        if config["inference"]["independent_group_decoding"] and not getattr(
            self._adapter, "supports_independent_group_decode", False
        ):
            raise ValueError("adapter does not support independent group decoding")
        if config["inference"]["handoff"] == "waypoint" and not getattr(
            self._adapter, "supports_waypoint_handoff", False
        ):
            raise ValueError("adapter does not support waypoint handoff")
        if (
            config["inference"]["handoff"] == "waypoint"
            and self._decode_seed_source != "execution_reference"
        ):
            raise ValueError("waypoint handoff requires an execution_reference IK seed")
        self._strategy = strategy or DefaultChunkStrategy(config)
        if self._strategy.name == "bid_backward" and not getattr(
            self._adapter, "supports_candidate_decoding", False,
        ):
            raise ValueError("BID requires an adapter with candidate decoding support")
        self._decoder = None
        if config["policy"]["action_decoding"] == "process":
            # A plugin may wrap the default strategy (the UMI history plugin
            # does); the constructed strategy decides, not the plugin path.
            if self._strategy.name not in {"async", "serial", "rtc", "train_time_rtc"}:
                raise ValueError("process decoding requires async, serial, rtc or train_time_rtc")
            self._decoder = ActionDecoderClient(
                config["robot"],
                config["policy"],
                self._adapter,
                motion_limits=config["executor"]["motion_limits"],
            )
        self._decode_forecast = DecodeForecast(
            floor_s=config["inference"]["expected_decode_s"],
            size=config["inference"]["decode_forecast_size"],
            mode=config["inference"]["decode_forecast_mode"],
        )
        self._session_id = f"session-{uuid.uuid4().hex}"
        self._worker = PolicyWorkerClient(
            config["policy"], self._session_id, multi_inflight=self._multi_inflight,
        )
        self._timeline = self._build_timeline()
        self._launch_mode = launch_mode
        position_limit_abs = None
        if config["executor"]["type"] == "smooth":
            position_limit_abs = config["executor"]["smooth"]["position_limit_abs"]
        elif config["executor"]["type"] == "mpc":
            position_limit_abs = config["executor"]["mpc"]["position_limit_abs"]
        command_safety = config["executor"]["command_safety"]
        self._safety = SafetyGuard(
            config["robot"]["group_dims"],
            position_limit_abs,
            position_lower=command_safety["position_lower"],
            position_upper=command_safety["position_upper"],
            max_velocity=command_safety["max_velocity"],
            max_acceleration=command_safety["max_acceleration"],
            control_dt_s=self._control_dt_ns / 1_000_000_000,
            device_bounds=getattr(self._robot, "command_bounds", lambda: {})() or {},
        )
        self._executor = self._build_executor()
        self._execution = ExecutionPipeline(
            self._executor,
            kinematics=self._robot.kinematics,
            ik_options=config.get("execution", {}).get("ik"),
            control_dt_ns=self._control_dt_ns,
        )
        if (config["inference"].get("rtc") or {}).get("condition_source") == "executor":
            predictor = getattr(self._executor, "predict_commands", None)
            setter = getattr(self._strategy, "set_condition_predictor", None)
            if not callable(predictor) or not callable(setter):
                raise ValueError(
                    "RTC executor conditions require a forecast-capable executor and strategy"
                )
            setter(predictor)
        self._robogui = RoboGUIBridge(
            enabled=config["robogui"]["enabled"],
            robot=config["robogui"]["robot"],
            policy=config["robogui"]["policy_label"] or "manimux-local",
            instruction=config["run"]["task_prompt"] if config["robogui"]["policy_label"] else "",
            camera_hz=config["robogui"]["camera_hz"],
            control=config["robogui"].get("control", {}),
        )
        self._state = RuntimeState.DISCONNECTED
        logger.info(
            "runtime_config robot=%s execute=%s end_effector_control=%s worker=%s "
            "policy_endpoint=%s strategy=%s executor=%s action_start_mode=%s",
            config["robot"].get("type", config["robot"].get("driver")),
            config["robot"]["options"].get("execute"),
            config["robot"]["options"].get("end_effector_control"),
            config["policy"]["worker"],
            config["policy"]["options"].get("server"),
            self._strategy.name,
            config["executor"]["type"],
            config["inference"]["action_start_mode"],
        )

    def _build_robot(self) -> RobotBase:
        return build_robot(self._config["robot"], self._clock)

    def _build_adapter(self):
        adapter = build_policy_adapter(
            self._config["robot"], self._config["policy"],
            kinematics=self._robot.kinematics,
            motion_limits=self._config["executor"]["motion_limits"],
        )
        adapter.decode_pose_only = step_ik_enabled(self._config)
        adapter.validate(self._config["robot"], self._config["policy"])
        return adapter

    def _build_executor(self) -> Executor:
        control_dt_s = self._control_dt_ns / 1_000_000_000
        if self._config["executor"]["type"] == "direct":
            return DirectExecutor(self._config["executor"]["motion_limits"], control_dt_s)
        if self._config["executor"]["type"] == "smooth":
            return SmoothExecutor(
                self._config["executor"]["smooth"], control_dt_s,
                command_limits=self._safety.execution_limits(),
            )
        if self._config["executor"]["type"] == "mpc":
            if self._config["executor"]["motion_limits"] is not None:
                raise ValueError(
                    "shared motion_limits currently support direct and smooth, not mpc"
                )
            return MPCExecutor(self._config["executor"]["mpc"], control_dt_s)
        raise ValueError(f"unknown executor type: {self._config['executor']['type']!r}")

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
        expected = self._config["policy"]["expected_backend"]
        if expected is None:
            return
        expected_metadata = {
            key: value for key, value in deepcopy(expected).items() if value is not None
        }
        mismatches = metadata_mismatches(expected_metadata, capabilities.backend_metadata)
        if mismatches:
            details = "; ".join(mismatches)
            raise RuntimeError(f"policy backend identity mismatch: {details}")

    def _send_command(
        self, command: RobotCommand, state: RobotState, recorder: EpisodeRecorder, *, step: int,
    ) -> None:
        """Validate and dispatch; retain a failed attempt outside successful tick data."""
        failure_stage = "safety_validation"
        try:
            with stage("safety_command"), failure_context("action", "command safety validation"):
                self._safety.validate_command(command)
            failure_stage = "robot_dispatch"
            with failure_context("robot", "command dispatch"):
                self._robot.send_command(command)
        except Exception as exc:
            # A controller can fail after submitting only part of its batch.
            # Never label such a failure as a successfully sent command.
            try:
                fields = dict(
                    step=step, monotonic_ns=command.monotonic_ns, stage=failure_stage,
                    plan_id=command.plan_id, request_seq=self._timeline.accepted_request_seq,
                    attempted_command={
                        name: np.asarray(q).tolist() for name, q in command.groups.items()
                    },
                    measured_state={
                        name: np.asarray(q).tolist() for name, q in state.groups.items()
                    },
                    measured_time_ns=state.monotonic_ns,
                    limits=self._safety.execution_limits(), error=_exception_detail(exc),
                    dispatch={"dispatch_attempted": False}
                    if failure_stage == "safety_validation" else {},
                )
                try:
                    dispatch = dict(getattr(self._robot, "command_status", lambda: {})() or {})
                    fields["channels"] = dispatch.pop("group_channels", {})
                    if failure_stage == "robot_dispatch":
                        fields.update(stage=dispatch.get("stage", failure_stage), dispatch=dispatch)
                    fields["coordinates"] = {
                        name: [{"name": c.name, "unit": c.unit} for c in model.coordinates]
                        for name, model in getattr(self._robot.kinematics, "models", {}).items()
                    }
                except Exception as diagnostic_error:
                    fields["diagnostic_error"] = _exception_detail(diagnostic_error)
                recorder.event("command_rejected", **fields)
            except Exception:
                logger.exception("failed to record rejected command")
            logger.exception("command_rejected stage=%s plan=%s", failure_stage, command.plan_id)
            raise

    def run(self) -> RunResult:  # noqa: C901 - the safety-critical loop stays linear
        started_wall = time.perf_counter()
        sensor_reader = SensorReader(
            self._sensors, self._clock, **self._config["run"].get("sensor_reading", {}),
        )
        episode_id = _next_rollout_id(self._run_dir)
        timing = LoopTiming(
            enabled=self._config["run"].get("control_timing", False),
            max_cycles=self._config["run"].get("timing_max_cycles", 20000),
            period_ns=self._control_dt_ns,
            clock_source=type(self._clock).__name__,
        )
        recorder = EpisodeRecorder(
            self._run_dir,
            episode_id,
            self._config["robot"]["group_dims"],
            metadata={
                "episode_id": episode_id,
                "session_id": self._session_id,
                "task": self._config["run"]["task_prompt"],
                "initial_pose": initial_pose_metadata(self._config),
                "collection": deepcopy(self._config.get("collection")),
                "evaluation": dict(self._evaluation),
                "executor_kind": self._config["executor"]["type"],
                "execution_ik": self._config.get("execution", {}).get("ik", {"mode": "chunk"}),
                "smooth": (
                    loads(dumps(deepcopy(self._config["executor"]["smooth"]), default=str))
                    if self._config["executor"]["type"] == "smooth"
                    else None
                ),
                "runtime": self._strategy.name,
                "inference_schedule": self._scheduler.mode,
                "request_trigger": self._scheduler.trigger,
                "observation_hz": self._config["inference"]["observation_hz"],
                "min_plan_hold_policy_steps": self._min_execute_steps,
                "command_limits": self._safety.execution_limits(),
                "policy_label": self._config["robogui"]["policy_label"],
                "policy_worker": self._config["policy"]["worker"],
                "policy_adapter": self._config["policy"]["adapter"]["type"],
                "view_profile": self._config["policy"]["adapter"].get("view_profile"),
                "camera_map": self._config["policy"]["adapter"].get("camera_map", {}),
                "action_dt_s": action_interval(self._config["policy"]),
                "horizon_steps": self._config["policy"]["horizon_policy_steps"],
                "max_chunk_steps": self._config["inference"]["max_chunk_policy_steps"],
                "blend_steps": self._config["inference"]["blend_policy_steps"],
                "action_start_mode": self._config["inference"]["action_start_mode"],
                **self._rollout_identity,
                "identity_frozen": False,
                "launch_mode": self._launch_mode,
                "policy_backend": {},
            },
            video_fps=self._config["recording"]["video_fps"],
            video_codec=self._config["recording"]["video_codec"],
            video_queue_size=self._config["recording"]["video_queue_size"],
            control_timing=timing,
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
        pending_observation_states: dict[int, RobotState] = {}
        pending_response = None

        def reject_response(response, reason):
            nonlocal rejected_plans
            logger.warning("inference_rejected seq=%d reason=%s", response.request_seq, reason)
            self._strategy.on_response_rejected(response)
            rejected_plans += 1
            recorder.event("inference_rejected", request_seq=response.request_seq, reason=reason)
            with stage("robogui_publish_event"):
                self._robogui.publish_event(
                    "inference_rejected", step=steps, chunk_id=response.request_seq,
                    metadata={"reason": reason},
                )
            pending_visuals.pop(response.request_seq, None)
            pending_observation_states.pop(response.request_seq, None)

        worker_failure_reported = False
        last_dispatch_log_ns = 0
        last_dispatch_plan_id: object = object()
        robot_connected = False
        steps = 0
        formal_start_recorded = False
        identity_frozen = False
        last_control_status = None
        completed = False
        terminal_reason = "completed"
        abort_reason = "runtime_exception"
        abort_detail = ""
        warmup = None
        shutdown_attempted = False
        shutdown_status = ShutdownStatus(
            enabled=self._config["robogui"]["enabled"],
            robot=self._config["robogui"]["robot"],
            policy=self._config["robogui"]["policy_label"],
            run_dir=str(self._run_dir.resolve()), episode_id=episode_id,
            home=(self._robot.home
                  if self._config["robot"]["options"].get("execute", False) else None),
        )

        preparation = OperationStatus(
            enabled=self._config["robogui"]["enabled"],
            robot=self._config["robogui"]["robot"],
            policy=self._config["robogui"]["policy_label"], event="runtime_preparing",
            identity={"run_dir": str(self._run_dir.resolve()), "episode_id": episode_id},
        )

        def shutdown_runtime(*, faulted: bool) -> list[BaseException]:
            """Release control resources before waiting for recording I/O."""
            nonlocal shutdown_attempted
            if shutdown_attempted:
                return []
            shutdown_attempted = True
            preparation.close()
            shutdown_status.update("stopping")
            self._state = RuntimeState.IDLE
            errors: list[BaseException] = []
            try:
                self._worker.request_stop()
            except BaseException as exc:
                errors.append(exc)
            # The dispatch loop has stopped. Home while actuator control is
            # still owned: some SDK stop() calls disable position control.
            # Robot.close() performs stop only after its measured release gate.
            if robot_connected:
                try:
                    logger.warning("shutdown_home_started faulted=%s", faulted)
                    shutdown_status.update("homing")
                    self._robot.home()
                    logger.info("shutdown_home_completed")
                except BaseException as exc:
                    logger.error("shutdown_home_failed: %s", _exception_detail(exc))
                    errors.append(exc)
            closers = [self._robot.close, self._worker.close]
            if self._decoder is not None:
                closers.append(self._decoder.close)
            closers.append(sensor_reader.close)
            for closer in closers:
                try:
                    shutdown_status.update("closing")
                    with observe_release(shutdown_status.update, recover=shutdown_status.wait_for_release):
                        closer()
                except BaseException as exc:
                    errors.append(exc)
            shutdown_status.update("finalizing")
            return errors

        try:
            if self._config["run"].get("warmup_before_start", False):
                warmup = PolicyWarmup(
                    self._config, worker=self._worker,
                    strategy=build_inference_strategy(self._config),
                    adapter=self._build_adapter(), session_id=self._session_id, clock=self._clock,
                    executor=self._build_executor(), kinematics=self._robot.kinematics,
                    safety=deepcopy(self._safety), decoder=self._decoder,
                )
            preparation.update("Starting cameras")
            with failure_context("camera", "sensor startup"):
                sensor_reader.start()
            preparation.update("Connecting policy backend")
            with failure_context("model", "model connection", str(self._config["policy"].get("options", {}).get("server", ""))):
                self._worker.start()
            logger.info("policy_worker_ready session=%s", self._session_id)
            if self._decoder is not None:
                with failure_context("action", "decoder startup"):
                    self._decoder.start()
            self._validate_policy_capabilities()
            capabilities = getattr(self._worker, "capabilities", PolicyCapabilities())
            recorder.update_metadata(
                policy_backend=dict(capabilities.backend_metadata),
            )
            # Policy startup can be slow; recheck sensors before enabling hardware.
            preparation.update("Checking fresh camera frames")
            with failure_context("camera", "fresh frame check before robot connection"):
                sensor_reader.read()
            preparation.update("Connecting robot and moving to the selected initial pose")
            with observe_release(shutdown_status.update, recover=shutdown_status.wait_for_release):
                channels = self._config["robot"].get("options", {}).get("component_hardware", {})
                target = ", ".join(f"{name}: {binding['channel']}" for name, binding in channels.items() if "channel" in binding)
                with failure_context("robot", "robot connection", target):
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
            self._scheduler.reset()
            recorder.update_metadata(
                robot_backend=self._robot.runtime_metadata(),
                initial_state=state_evidence(
                    initial_state,
                    targets=initial_pose_targets(self._config),
                ),
                inference_seed=capabilities.backend_metadata.get("model", {}).get("inference_seed"),
            )
            previous_command = copy_group_vector(initial_state.groups)
            preparation.update("Preparing action execution")
            self._execution.warmup(initial_state.groups)
            last_command = copy_group_vector(initial_state.groups)
            self._state = RuntimeState.RUNNING
            robogui_episode_metadata = {
                "episode_active": True,
                "identity_frozen": False,
                "episode_id": episode_id,
                "episode_dir": str(recorder.final_dir.resolve()),
                "run_dir": str(self._run_dir.resolve()),
                "instruction": self._config["run"]["task_prompt"],
                "evaluation": dict(self._evaluation),
                "max_steps": self._config["run"]["max_control_steps"],
                "control_mode": self._strategy.control_mode,
                "runtime": self._strategy.name,
                "executor": self._config["executor"]["type"],
                "policy_label": self._config["robogui"]["policy_label"],
                **self._rollout_identity,
                "camera_map": self._config["policy"]["adapter"].get("camera_map", {}),
                "launch_mode": self._launch_mode,
                **({"warmup": warmup.metadata()} if warmup is not None else {}),
            }
            # The current robot does not offer manual-drag recovery; RoboGUI does
            # not expose retired driver capabilities.
            robogui_episode_metadata["recovery_available"] = False
            # The active runtime owns Home; idle recovery remains service-owned.
            robogui_episode_metadata["home_available"] = (
                "home" in self._robot.recovery_actions
            )
            preparation.close()
            self._robogui.set_state_metadata(robogui_episode_metadata)
            with stage("robogui_publish_event"):
                self._robogui.publish_event(
                    "episode_started",
                    metadata=robogui_episode_metadata,
                )
            next_tick_ns = self._clock.now_ns()

            # Zero disables only the tick budget; Finish/Home and faults still exit.
            while (
                self._config["run"]["max_control_steps"] == 0
                or steps < self._config["run"]["max_control_steps"]
            ):
                timing.begin(
                    scheduled_start_ns=next_tick_ns,
                    phase=(
                        f"warmup_{warmup.phase}"
                        if warmup is not None and not warmup.complete else self._state.value.lower()
                    ),
                    step=steps,
                )
                loop_start_ns = self._clock.now_ns()
                now_ns = loop_start_ns
                with failure_context("robot", "robot feedback"):
                    state = self._robot.get_state()
                with stage("safety_state"):
                    self._safety.validate_state(state)
                with stage("camera_read", mode=sensor_reader.mode), failure_context("camera", "frame acquisition"):
                    frames = sensor_reader.read()

                with stage("robogui_control"):
                    robogui_control = self._robogui.poll_control()
                control_diagnostics = getattr(robogui_control, "diagnostics", None)
                if control_diagnostics is not None:
                    control_status = (
                        robogui_control.paused, control_diagnostics.get("reason"),
                        control_diagnostics.get("timeout_count"),
                        control_diagnostics.get("error_count"),
                        control_diagnostics.get("transport_status"),
                    )
                    if control_status != last_control_status:
                        recorder.event(
                            "robogui_control_state", step=steps,
                            monotonic_ns=self._clock.now_ns(),
                            paused=robogui_control.paused, **control_diagnostics,
                        )
                        last_control_status = control_status
                request_matches = (
                    getattr(robogui_control, "episode_id", "") == episode_id
                    and getattr(robogui_control, "control_run_dir", "")
                    == robogui_episode_metadata["run_dir"]
                )
                robogui_episode_metadata["control_status"] = {
                    "paused": robogui_control.paused,
                    "revision": getattr(robogui_control, "control_revision", ""),
                    "reason": (control_diagnostics or {}).get("reason", "unknown"),
                }
                self._robogui.set_state_metadata(robogui_episode_metadata)
                if robogui_control.finish_requested and (
                    not getattr(robogui_control, "finish_request_id", "") or request_matches
                ):
                    timing.set_phase("stopped")
                    # Stop model requests before Home/recording cleanup can take time.
                    self._worker.request_stop()
                    terminal_reason = "robogui_finish_requested"
                    recorder.event("robogui_finish_requested", step=steps, home=True)
                    break
                if robogui_control.home_requested and (
                    not getattr(robogui_control, "home_request_id", "") or request_matches
                ):
                    timing.set_phase("homing")
                    if pending_response is not None:
                        reject_response(pending_response, "invalidated_by_pause_or_home")
                        pending_response = None
                    home_status = OperationStatus(
                        enabled=self._config["robogui"]["enabled"],
                        robot=self._config["robogui"]["robot"],
                        policy=self._config["robogui"]["policy_label"],
                        event="runtime_home", identity={
                            "run_dir": str(self._run_dir.resolve()), "episode_id": episode_id,
                            "home_request_id": getattr(robogui_control, "home_request_id", ""),
                        },
                    )
                    home_status.update("Returning Home and verifying arrival")
                    try:
                        self._robot.home()
                    finally:
                        home_status.close()
                    robogui_episode_metadata["home_ack"] = getattr(
                        robogui_control, "home_request_id", ""
                    )
                    self._robogui.set_state_metadata(robogui_episode_metadata)
                    state = self._robot.get_state()
                    self._safety.reset(state)
                    self._timeline = self._build_timeline()
                    self._executor.reset(state)
                    self._strategy.reset()
                    self._scheduler.reset()
                    previous_command = copy_group_vector(state.groups)
                    last_command = copy_group_vector(state.groups)
                    discard_responses_through = max(discard_responses_through, request_seq)
                    pending_observation_states.clear()
                    self._state = RuntimeState.PAUSED
                    recorder.event(
                        "robogui_home_requested", step=steps,
                        state=state_evidence(
                            state, targets=self._robot.home_targets(state)
                        ),
                    )
                    next_tick_ns = self._clock.now_ns()
                    timing.end()
                    continue
                if not identity_frozen and not robogui_control.paused:
                    if self._config["robogui"].get("enabled", False):
                        if (robogui_control.episode_id != episode_id
                                or robogui_control.control_run_dir
                                != robogui_episode_metadata["run_dir"]
                                or robogui_control.start_identity is None):
                            raise ValueError("Start requires a selection for the active episode")
                        self._rollout_identity = start_identity(
                            self._rollout_identity, robogui_control.start_identity,
                        )
                    identity_frozen = True
                    self._config["run"].update(self._rollout_identity)
                    recorder.update_metadata(**self._rollout_identity, identity_frozen=True)
                    recorder.event("rollout_identity_frozen", **self._rollout_identity)
                    robogui_episode_metadata.update(self._rollout_identity, identity_frozen=True)
                    self._robogui.set_state_metadata(robogui_episode_metadata)
                if warmup is not None and not warmup.complete:
                    self._state = RuntimeState.PAUSED
                    before = warmup.metadata()
                    if (not self._config["robogui"].get("enabled", False)
                            or (robogui_control.episode_id == episode_id
                                and robogui_control.control_run_dir
                                == robogui_episode_metadata["run_dir"]
                                and robogui_control.warmup_requested)):
                        warmup.activate()
                    with stage("warmup_advance"):
                        warmup.advance(
                            paused=robogui_control.paused,
                            snapshot=ObservationSnapshot(state=state, frames=frames),
                        )
                    status = warmup.metadata()
                    timing.set_phase(f"warmup_{warmup.phase}")
                    preview = warmup.take_preview()
                    if preview is not None:
                        preview_chunk, preview_inference_ms, preview_metadata = preview
                        with stage("robogui_publish_plan"):
                            self._robogui.publish_plan(
                                preview_chunk, preview_inference_ms,
                                metadata={
                                    **preview_metadata,
                                    "episode_id": episode_id,
                                    "run_dir": str(self._run_dir.resolve()),
                                    "runtime": self._strategy.name,
                                },
                            )
                    if warmup.complete:
                        self._validate_policy_capabilities()
                        self._strategy.reset()
                        self._scheduler.reset()
                        seed_strategy_warmup(self._strategy, latency_ns=list(warmup.latency_ns))
                        self._timeline = self._build_timeline()
                        recorder.event(
                            "formal_rollout_ready",
                            inference_seed=capabilities.backend_metadata.get("model", {}).get(
                                "inference_seed"
                            ), latency_sample_count=len(warmup.latency_ns),
                        )
                    for kind, fields in warmup.take_events():
                        recorder.event(kind, **fields)
                        logger.info("%s %s", kind, fields)
                        with stage("robogui_publish_event"):
                            self._robogui.publish_event(
                                kind, step=steps,
                                metadata={
                                    **fields, "episode_id": episode_id,
                                    "run_dir": str(self._run_dir.resolve()),
                                    "runtime": self._strategy.name,
                                },
                            )
                    if status != before:
                        recorder.update_metadata(warmup=status)
                    robogui_episode_metadata["warmup"] = status
                    self._robogui.set_state_metadata(robogui_episode_metadata)
                    # Preserve the existing paused hold; model output never reaches it.
                    self._executor.reset(state)
                    command = self._hold_command(self._clock.now_ns(), state.groups)
                    self._safety.reset(state)
                    self._send_command(command, state, recorder, step=steps)
                    previous_command = copy_group_vector(last_command)
                    last_command = copy_group_vector(command.groups)
                    with stage("robogui_publish_state"):
                        self._robogui.publish_state(
                            state, frames, step=steps,
                            max_steps=self._config["run"]["max_control_steps"],
                        )
                    next_tick_ns = max(
                        next_tick_ns + self._control_dt_ns,
                        self._clock.now_ns(),
                    )
                    timing.sleep_until(self._clock, next_tick_ns)
                    timing.end()
                    # Even after RESET, reacquire a fresh observation on the next tick.
                    continue
                if robogui_control.paused and (
                    self._decoder is not None
                    or self._multi_inflight
                    or self._min_execute_steps
                    or (
                        self._config["inference"]["inference_schedule"] == "serial"
                        and self._config["inference"]["algorithm"] in {"async", "serial"}
                    )
                    or getattr(self._strategy, "discard_plans_while_paused", False)
                ):
                    # RTC conditions must not refer to the timeline discarded
                    # by Pause/Hold while a decoder response is still pending.
                    if pending_response is not None:
                        reject_response(pending_response, "invalidated_by_pause_or_home")
                        pending_response = None
                    if self._state != RuntimeState.PAUSED:
                        self._strategy.reset()
                        self._scheduler.reset()
                    self._timeline = self._build_timeline()
                    discard_responses_through = max(discard_responses_through, request_seq)
                    pending_observation_states.clear()
                self._state = (
                    RuntimeState.RUNNING if not robogui_control.paused else RuntimeState.PAUSED
                )
                timing.set_phase(self._state.value.lower())
                if self._state == RuntimeState.RUNNING and not formal_start_recorded:
                    recorder.update_metadata(
                        formal_start_state=state_evidence(
                            state, targets=initial_pose_targets(self._config)
                        ),
                    )
                    formal_start_recorded = True

                if not self._worker.is_alive and not worker_failure_reported:
                    worker_failure_reported = True
                    recorder.event("policy_worker_stopped", step=steps)

                decoded_chunk = None
                with stage("worker_poll"):
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
                    with stage("decoder_poll"):
                        decoded = self._decoder.poll()
                    if decoded is not None:
                        if response is not None:
                            raise RuntimeError("model response arrived while decode was in flight")
                        response = decoded.response
                        decoded_chunk = decoded.chunk
                        if decoded_chunk is not None:
                            self._decode_forecast.observe(decoded_chunk.metadata["decode_stage_ms"])
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
                            observation_state = pending_observation_states.pop(
                                response.request_seq, None
                            )
                            try:
                                context = self._decode_context(
                                    response, state, self._clock.now_ns(), observation_state
                                )
                            except ValueError as exc:
                                response = replace(response, error=f"decode_seed_unavailable:{exc}")
                            else:
                                with stage("decoder_submit"):
                                    self._decoder.submit(
                                        response, context, last_request_deadline_ns
                                    )
                                seed, seed_source = context.ik_seed, context.ik_seed_source
                                start_ns = context.execution_time_ns
                                handoff_reference = context.handoff_reference
                                recorder.event(
                                    "decode_submitted",
                                    request_seq=response.request_seq,
                                    seed_time_ns=seed.monotonic_ns,
                                    seed_source=seed_source,
                                    expected_start_ns=start_ns,
                                    handoff_plan_id=(
                                        None
                                        if handoff_reference is None
                                        else handoff_reference.plan_id
                                    ),
                                    # first_step_when_ready has no expected start until commit.
                                    seed_to_expected_start_ms=(
                                        None
                                        if start_ns is None
                                        else (start_ns - seed.monotonic_ns) / 1e6
                                    ),
                                    expected_decode_ms=round(
                                        self._decode_forecast.seconds * 1e3, 3
                                    ),
                                )
                                # Keep inference+decode in flight until both arms finish.
                                response = None
                if self._min_execute_steps:
                    # Buffer raw results so decoding, candidate selection and time
                    # trimming all use the eventual handoff clock. Receipt frees
                    # the inference slot; consuming an older buffered result does not.
                    if response is not None:
                        if response.request_seq == last_submitted_seq:
                            request_in_flight = False
                        reason = self._response_rejection_reason(
                            response, last_submitted_seq=last_submitted_seq,
                            last_request_deadline_ns=last_request_deadline_ns,
                            discard_responses_through=discard_responses_through,
                        )
                        if reason is None and pending_response is not None:
                            if response.request_seq <= pending_response.request_seq:
                                reason = "superseded_pending_response"
                            else:
                                reject_response(pending_response, "superseded_pending_response")
                        if reason is None:
                            pending_response = response
                        else:
                            reject_response(response, reason)
                        response = None
                    now_ns = self._clock.now_ns()
                    if pending_response is not None and self._timeline.replacement_ready(
                        now_ns, self._min_execute_steps,
                    ):
                        response, pending_response = pending_response, None

                if response is not None:
                    if not self._min_execute_steps:
                        request_in_flight = False
                    observation_state = pending_observation_states.pop(
                        response.request_seq, None
                    )
                    submission_visuals = pending_visuals.get(response.request_seq, {})
                    rejection_reason = self._response_rejection_reason(
                        response, last_submitted_seq=last_submitted_seq,
                        last_request_deadline_ns=last_request_deadline_ns,
                        discard_responses_through=discard_responses_through,
                    )
                    if rejection_reason is not None:
                        reject_response(response, rejection_reason)
                    else:
                        try:
                            decode_start = time.perf_counter_ns()
                            with stage("policy_decode"):
                                chunk = (
                                    decoded_chunk
                                    if decoded_chunk is not None
                                    else decode_strategy_action(
                                        self._strategy, self._adapter,
                                        response.raw_action,
                                        self._decode_context(
                                            response, state, now_ns, observation_state
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
                            with stage("robogui_publish_event"):
                                self._robogui.publish_event(
                                    "plan_rejected",
                                    step=steps,
                                    chunk_id=response.request_seq,
                                    metadata={"reason": reason},
                                )
                            pending_visuals.pop(response.request_seq, None)
                            chunk = None
                        now_ns = self._clock.now_ns()
                        if chunk is not None and chunk.action_space != self._timeline.action_space:
                            self._strategy.on_response_rejected(response)
                            rejected_plans += 1
                            recorder.event(
                                "plan_rejected",
                                request_seq=response.request_seq,
                                reason=(
                                    "invalid_action:canonical action_space must be "
                                    f"{self._timeline.action_space!r}, got {chunk.action_space!r}"
                                ),
                            )
                            with stage("robogui_publish_event"):
                                self._robogui.publish_event(
                                    "plan_rejected",
                                    step=steps,
                                    chunk_id=response.request_seq,
                                    metadata={"reason": "invalid_action_space"},
                                )
                            pending_visuals.pop(response.request_seq, None)
                            chunk = None
                        canonical_raw = None if chunk is None else copy_action_chunk(chunk)
                        selector = (
                            self._strategy
                            if isinstance(self._strategy, StrategyChunkSelector) else None
                        )
                        if chunk is not None and selector is None and not self._min_execute_steps:
                            try:
                                with stage("strategy_prepare_chunk"):
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
                                with stage("robogui_publish_event"):
                                    self._robogui.publish_event(
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
                            source_end_ns = (
                                chunk.observation_time_ns
                                + (chunk.source_offset_steps + chunk.horizon_steps - 1)
                                * chunk.dt_ns
                            )
                            if (
                                self._decoder is not None
                                and self._config["inference"]["action_start_mode"]
                                == "drop_infer_latency"
                                and now_ns > source_end_ns
                            ):
                                result = CommitResult(False, "no_future_horizon")
                            else:
                                with stage("timeline_commit"):
                                    try:
                                        if selector is not None:
                                            # Selection and trimming share one handoff clock.
                                            chunk = selector.select_chunk(
                                                chunk=chunk, now_ns=now_ns,
                                            )
                                            canonical_raw = copy_action_chunk(chunk)
                                        if selector is not None or self._min_execute_steps:
                                            chunk = prepare_strategy_chunk(
                                                self._strategy, chunk=chunk,
                                                response=response, now_ns=now_ns,
                                            )
                                    except (TypeError, ValueError) as exc:
                                        result = CommitResult(
                                            False,
                                            f"invalid_strategy_chunk:{type(exc).__name__}:{exc}",
                                        )
                                    else:
                                        result = self._timeline.commit(
                                            chunk,
                                            now_ns=now_ns,
                                            max_plan_age_ns=int(
                                                self._config["inference"]["max_plan_age_s"]
                                                * 1_000_000_000
                                            ),
                                            current_command=self._execution.reference_from_joints(commit.current_command),
                                            blend_steps=commit.blend_steps,
                                            command_velocity={
                                                name: (last_command[name]-previous_command[name])
                                                / (self._control_dt_ns/1e9)
                                                for name in last_command
                                            },
                                            handoff_skip_steps=self._config["inference"][
                                                "handoff_skip_steps"
                                            ],
                                        )
                            if result.accepted:
                                accepted_plans += 1
                                last_inference_ms = response.inference_ms
                                chunk.metadata["timeline_latency_ms"] = (
                                    result.timeline_latency_ns / 1_000_000
                                )
                                chunk.metadata["blend_mode"] = self._config["inference"].get(
                                    "blend_mode", "linear"
                                )
                                chunk.metadata["time_trimmed_steps"] = result.time_trimmed_steps
                                chunk.metadata["handoff_skipped_steps"] = (
                                    result.handoff_skipped_steps
                                )
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
                                with stage("strategy_feedback"):
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
                                with stage("plan_boundary_diagnostics"):
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
                                with stage("robogui_publish_plan"):
                                    self._robogui.publish_plan(
                                        chunk,
                                        response.inference_ms,
                                        committed=committed,
                                        metadata={
                                            "runtime": self._strategy.name,
                                            "raw_horizon_steps": chunk.horizon_steps,
                                            "committed_horizon_steps": committed.horizon_steps,
                                            "time_trimmed_steps": result.time_trimmed_steps,
                                            "handoff_skipped_steps": (
                                                result.handoff_skipped_steps
                                            ),
                                            "timeline_latency_ms": (
                                                result.timeline_latency_ns / 1_000_000
                                            ),
                                            "decode_stage_ms": chunk.metadata.get(
                                                "decode_stage_ms"
                                            ),
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
                                            # After event_fields, which repeat the timeline's count.
                                            "trimmed_steps": result.trimmed_steps,
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
                                with stage("robogui_publish_event"):
                                    self._robogui.publish_event(
                                        "plan_rejected",
                                        step=steps,
                                        chunk_id=response.request_seq,
                                        metadata={"reason": result.reason},
                                    )
                                pending_visuals.pop(response.request_seq, None)

                if self._worker.is_alive and not (
                    (
                        self._decoder is not None
                        or self._multi_inflight
                        or self._min_execute_steps
                        or getattr(self._strategy, "discard_plans_while_paused", False)
                    )
                    and self._state != RuntimeState.RUNNING
                ):
                    snapshot = ObservationSnapshot(state=state, frames=frames)
                    request_state = RequestState(
                        in_flight=request_in_flight,
                        last_submitted_seq=last_submitted_seq,
                        last_deadline_ns=last_request_deadline_ns,
                        multi_flight=self._multi_inflight,
                        expected_decode_ns=round(self._decode_forecast.seconds * 1e9),
                    )
                    with stage("policy_schedule"):
                        submission = self._scheduler.build_submission(
                            self._strategy,
                            session_id=self._session_id,
                            request_seq=request_seq + 1,
                            now_ns=now_ns,
                            snapshot=snapshot,
                            adapter=self._adapter,
                            timeline=self._timeline,
                            request_state=request_state,
                            runtime_state=self._state,
                        )
                    if submission is not None:
                        with stage("policy_prepare"):
                            prepared_request = self._adapter.prepare_request(
                                submission.request,
                            )
                        request_seq = submission.request.request_seq
                        observation_state = prepared_request.observation.state
                        pending_observation_states[request_seq] = RobotState(
                            groups=copy_group_vector(observation_state.groups),
                            monotonic_ns=observation_state.monotonic_ns,
                            sequence=observation_state.sequence,
                        )
                        with stage("worker_submit"):
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
                        self._scheduler.on_submitted(now_ns)
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
                            "horizon_steps": self._config["policy"]["horizon_policy_steps"],
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
                                0, self._config["policy"]["horizon_policy_steps"] - executed_steps
                            )
                            visual_fields["frozen_steps"] = int(
                                submission.event_fields.get("forecast_delay", 0)
                            )
                        pending_visuals[request_seq] = visual_fields
                        if self._multi_inflight:
                            # Coalesced uploads never produce a plan. Keep diagnostic
                            # history bounded independently of model/network latency.
                            while len(pending_visuals) > 128:
                                pending_visuals.pop(next(iter(pending_visuals)))
                        with stage("robogui_publish_event"):
                            self._robogui.publish_event(
                                "inference_submitted",
                                step=steps,
                                chunk_id=request_seq,
                                metadata=visual_fields,
                            )
                for kind, fields in self._strategy.take_runtime_events(step=steps):
                    recorder.event(kind, **fields)
                    with stage("robogui_publish_event"):
                        self._robogui.publish_event(kind, step=steps, metadata=fields)

                now_ns = self._clock.now_ns()
                with stage("timeline_reference"):
                    reference = self._timeline.reference_horizon(
                        now_ns=now_ns,
                        dt_ns=self._control_dt_ns,
                        horizon_steps=self._execution.horizon_steps,
                    )
                scheduled = copy_group_vector(state.groups)
                if self._state == RuntimeState.RUNNING and reference is not None:
                    execution = self._execution.step(
                        now_ns, state, reference, last_command=last_command,
                    )
                    reference, command = execution.reference, execution.command
                    scheduled = {name: values[0].copy() for name, values in reference.groups.items()}
                    if execution.ik_status is not None and not execution.ik_status["success"]:
                        recorder.event("step_ik_hold", step=steps, **execution.ik_status)
                    if isinstance(self._executor, SmoothExecutor) and (
                        self._config["executor"]["smooth"]["release_guard"] is not None
                        or self._executor.uses_close_latch
                    ):
                        recorder.event(
                            "gripper_decision",
                            step=steps,
                            monotonic_ns=now_ns,
                            plan_id=reference.plan_id,
                            groups=self._executor.gripper_diagnostics,
                        )
                elif self._state == RuntimeState.RUNNING:
                    # During a timeline gap, target the last sent command. Bounded
                    # Smooth decelerates while retaining the emitted velocity;
                    # other executors retain their existing hold/reset behavior.
                    held_state = RobotState(
                        groups=copy_group_vector(last_command),
                        monotonic_ns=now_ns,
                        sequence=state.sequence,
                    )
                    if isinstance(self._executor, SmoothExecutor):
                        with stage("executor_hold"):
                            command = self._executor.hold(now_ns, held_state)
                        command.plan_id = self._timeline.active_plan_id
                    else:
                        self._executor.reset(held_state)
                        command = self._hold_command(now_ns, last_command)
                else:
                    # Pause holds measured state and clears execution history.
                    self._executor.reset(state)
                    command = self._hold_command(now_ns, state.groups)
                    self._safety.reset(
                        RobotState(
                            copy_group_vector(command.groups),
                            now_ns,
                            state.sequence,
                        )
                    )
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
                self._send_command(command, state, recorder, step=steps)
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
                with stage("robogui_publish_state"):
                    self._robogui.publish_state(
                        state,
                        frames,
                        step=steps,
                        max_steps=self._config["run"]["max_control_steps"],
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
                        command=command.groups,
                        sent_commands=getattr(self._robot, "sent_command_snapshots", lambda: {})(),
                        plan_id=command.plan_id,
                        inference_ms=last_inference_ms,
                        camera_times_ns={
                            name: frame.capture_monotonic_ns for name, frame in frames.items()
                        },
                        frames=frames,
                    )
                    steps += 1

                with stage("strategy_tick"):
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
                    # Resume immediately; rebase so missed periods are not replayed.
                    next_tick_ns = finished_tick_ns
                timing.sleep_until(self._clock, next_tick_ns)
                timing.end()

            timing.end(completed=False)
            # Keep the rollout duration independent of Home and cleanup latency.
            wall_time_s = time.perf_counter() - started_wall
            cleanup_errors = shutdown_runtime(faulted=False)
            if cleanup_errors:
                raise BaseExceptionGroup("runtime cleanup failed", cleanup_errors)
            episode_dir = recorder.finish(
                run_completed=True,
                terminal_reason=terminal_reason,
                steps=steps,
                wall_time_s=wall_time_s,
            )
            preparation.close()
            shutdown_status.close()
            with stage("robogui_publish_event"):
                self._robogui.publish_event(
                    "episode_finished",
                    step=steps,
                    metadata={
                        "episode_id": episode_id,
                        "episode_dir": str(episode_dir.resolve()),
                        "reason": terminal_reason,
                        "launch_mode": self._launch_mode,
                        "evaluation": dict(self._evaluation),
                        **self._rollout_identity,
                    },
                )
            completed = True
            return RunResult(
                episode_dir=episode_dir,
                steps=steps,
                accepted_plans=accepted_plans,
                rejected_plans=rejected_plans,
                run_completed=True,
                terminal_reason=terminal_reason,
            )
        except KeyboardInterrupt:
            abort_reason = "KeyboardInterrupt"
            raise
        except BaseException as exc:
            abort_reason = type(exc).__name__
            abort_detail = _exception_detail(exc)
            logger.error("episode_aborted %s", abort_detail)
            raise
        finally:
            timing.end(completed=False)
            faulted = not completed and abort_reason != "KeyboardInterrupt"
            cleanup_errors = shutdown_runtime(faulted=faulted)
            preparation.close()
            shutdown_status.close()
            try:
                self._robogui.close()
            except BaseException as exc:
                cleanup_errors.append(exc)
            if not completed or cleanup_errors:
                detail = "; ".join(filter(None, [
                    abort_detail, *(_exception_detail(exc) for exc in cleanup_errors),
                ]))
                try:
                    recorder.abort(
                        "runtime_cleanup_failed" if completed else abort_reason, detail=detail,
                    )
                except BaseException as exc:
                    cleanup_errors.append(exc)
            if cleanup_errors:
                raise BaseExceptionGroup("runtime cleanup failed", cleanup_errors)
