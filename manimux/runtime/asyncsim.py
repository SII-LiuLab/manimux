"""Independent simulation control loop; no physical robot or camera lifecycle."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from typing import Any
import uuid

from manimux.clock import Clock, SystemClock
from manimux.embodiments.asyncsim import AsyncSimClient, AsyncSimRobot, AsyncSimSensor
from manimux.policies import PolicyCapabilities
from manimux.policies.worker import PolicyWorkerClient
from manimux.policy_adapter import build_policy_adapter
from manimux.runtime.executors import DirectExecutor, SmoothExecutor
from manimux.runtime.inference import (
    CommitSettings, RequestState, build_inference_strategy, prepare_strategy_chunk,
)
from manimux.runtime.safety import RuntimeState, SafetyGuard
from manimux.runtime.timeline import ActionTimeline, CommitResult
from manimux.types import (
    ActionChunk, ActionContext, InferenceResponse, ObservationSnapshot, RobotCommand,
    copy_group_vector,
)


class AsyncSimRuntime:
    """Own one AsyncSim session, one timeline, and one canonical command stream."""

    def __init__(
        self, config: dict, *, client: Any = None, clock: Clock | None = None, worker: Any = None,
    ) -> None:
        if config["executor"]["type"] not in {"direct", "smooth"}:
            raise ValueError("AsyncSim currently supports direct and smooth executors")
        if config["inference"]["algorithm"] not in {"manimux", "act_temporal_ensemble"}:
            raise ValueError("AsyncSim supports default chunks and ACT temporal ensemble")
        self.config = config
        self.clock = clock or SystemClock()
        self.session_id = f"asyncsim-{uuid.uuid4().hex}"
        self.worker = worker
        self.strategy = build_inference_strategy(config)
        self.adapter = build_policy_adapter(
            config["robot"], config["policy"], motion_limits=config["executor"]["motion_limits"],
        )
        self.adapter.validate(config["robot"], config["policy"])
        robot = config["robot"]
        options = robot["options"]
        self.client = client or AsyncSimClient(options["endpoint"])
        self.robot = AsyncSimRobot(
            self.client, robot["group_dims"], options["state_keys"],
            state_stream=options.get("state_stream", "proprio.joint_state"),
            env_index=options.get("env_index", 0),
            command_ttl_s=options.get("command_ttl_s", 0.2),
        )
        sensor_specs = [item for item in config["sensors"] if item["driver"] == "asyncsim"]
        if len(sensor_specs) != 1 or len(config["sensors"]) != 1:
            raise ValueError("AsyncSim requires one asyncsim sensor bundle")
        self.sensor = AsyncSimSensor(
            self.client, sensor_specs[0]["options"]["streams"],
            env_index=options.get("env_index", 0),
        )
        self.streams = [self.robot.state_stream, *self.sensor.streams.values()]
        self.dt_ns = round(1_000_000_000 / robot["control_hz"])
        inference = config["inference"]
        self.timeline = ActionTimeline(
            robot["group_dims"],
            max_source_steps=inference["max_chunk_policy_steps"],
            action_start_mode=inference["action_start_mode"],
        )
        self.executor = (
            SmoothExecutor(config["executor"]["smooth"], self.dt_ns / 1e9)
            if config["executor"]["type"] == "smooth"
            else DirectExecutor(config["executor"]["motion_limits"], self.dt_ns / 1e9)
        )
        limits = config["executor"]["command_safety"]
        self.safety = SafetyGuard(
            robot["group_dims"], (
                config["executor"]["smooth"]["position_limit_abs"]
                if config["executor"]["type"] == "smooth" else None
            ),
            position_lower=limits["position_lower"], position_upper=limits["position_upper"],
            max_velocity=limits["max_velocity"], max_acceleration=limits["max_acceleration"],
            control_dt_s=self.dt_ns / 1e9,
        )
        self._last_command = None
        self._started = False

    def start(self, *, seed: int | None = None) -> dict[str, Any]:
        if self._started:
            raise RuntimeError("AsyncSim runtime already started")
        self.robot.connect()
        try:
            episode = self.client.reset(seed=seed)
            self.robot.reset()
            inference = self.config["inference"]
            self.timeline = ActionTimeline(
                self.robot.group_dims,
                max_source_steps=inference["max_chunk_policy_steps"],
                action_start_mode=inference["action_start_mode"],
            )
            self._last_command = None
            self.client.subscribe(self.streams)
            self.sensor.start()
            self._started = True
            return episode
        except BaseException:
            self.close()
            raise

    def observe(self) -> ObservationSnapshot:
        if not self._started:
            raise RuntimeError("start the AsyncSim runtime before observing")
        snapshot = self.client.read_snapshot(self.streams)
        state = self.robot.use_snapshot(snapshot)
        frames = self.sensor.use_snapshot(snapshot)
        self.safety.validate_state(state)
        if self._last_command is None:
            self._last_command = copy_group_vector(state.groups)
            self.executor.reset(state)
            self.safety.reset(state)
        return ObservationSnapshot(state, frames)

    def step(self, chunk: ActionChunk | None = None) -> tuple[ObservationSnapshot, CommitResult | None, dict]:
        observation = self.observe()
        result, ack = self.execute(observation, chunk)
        return observation, result, ack

    def execute(
        self, observation: ObservationSnapshot, chunk: ActionChunk | None = None,
        *, commit_settings: CommitSettings | None = None,
    ) -> tuple[CommitResult | None, dict]:
        now_ns = self.clock.now_ns()
        state = observation.state
        result = None
        if chunk is not None:
            if chunk.action_space != "joint_position":
                raise ValueError("AsyncSim requires a canonical joint_position chunk")
            result = self.timeline.commit(
                chunk, now_ns=now_ns,
                commit_lead_ns=round(self.config["inference"]["commit_lead_s"] * 1e9),
                max_plan_age_ns=round(self.config["inference"]["max_plan_age_s"] * 1e9),
                current_command=(
                    self._last_command if commit_settings is None else commit_settings.current_command
                ),
                blend_steps=(
                    self.config["inference"]["blend_policy_steps"]
                    if commit_settings is None else commit_settings.blend_steps
                ),
            )
        horizon = self.timeline.reference_horizon(
            now_ns=now_ns, dt_ns=self.dt_ns, horizon_steps=self.executor.horizon_steps
        )
        if horizon is None:
            if isinstance(self.executor, SmoothExecutor) and self.executor.braking_tracking:
                command = self.executor.brake_hold(now_ns, state)
            elif isinstance(self.executor, SmoothExecutor):
                held_state = type(state)(copy_group_vector(self._last_command), now_ns, state.sequence)
                command = self.executor.hold(now_ns, held_state)
            else:
                command = RobotCommand(copy_group_vector(self._last_command), now_ns, "hold")
        else:
            command = self.executor.step(now_ns, state, horizon)
        self.safety.validate_command(command)
        ack = self.robot.send_command(command)
        self._last_command = copy_group_vector(command.groups)
        return result, ack

    def _check_capabilities(self, worker: Any) -> None:
        capabilities = getattr(worker, "capabilities", PolicyCapabilities())
        if not isinstance(capabilities, PolicyCapabilities):
            raise TypeError("policy worker returned invalid capabilities")
        missing = self.strategy.required_sampling_modes.difference(capabilities.sampling_modes)
        if missing:
            raise RuntimeError(f"policy backend does not support sampling modes {sorted(missing)}")
        expected = self.config["policy"]["expected_backend"]
        if expected is not None:
            def compare(required: dict, actual: dict, path: str) -> None:
                for key, value in required.items():
                    if key not in actual:
                        raise RuntimeError(f"policy backend identity missing {path}.{key}")
                    if isinstance(value, dict):
                        if not isinstance(actual[key], dict):
                            raise RuntimeError(f"policy backend identity mismatch at {path}.{key}")
                        compare(value, actual[key], f"{path}.{key}")
                    elif value is not None and value != actual[key]:
                        raise RuntimeError(f"policy backend identity mismatch at {path}.{key}")
            compare(deepcopy(expected), capabilities.backend_metadata, "backend")

    def run_policy(self, *, seed: int | None = None, max_steps: int | None = None) -> dict[str, Any]:
        """Execute ManiMux's worker, adapter, strategy, timeline and direct executor."""
        worker = self.worker or PolicyWorkerClient(self.config["policy"], self.session_id)
        max_steps = max_steps or self.config["run"]["max_control_steps"]
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        accepted = rejected = 0
        rejection_reasons: list[str] = []
        responses = 0
        command_acks: list[dict[str, Any]] = []
        request_seq = 0
        last_submitted = -1
        last_deadline = 0
        in_flight = False
        pending_observation_ns: dict[int, int] = {}
        next_tick = self.clock.now_ns()
        self.start(seed=seed)
        try:
            worker.start()
            self._check_capabilities(worker)
            self.strategy.reset()
            for step_index in range(max_steps):
                if not worker.is_alive:
                    raise RuntimeError("policy worker stopped")
                observation = self.observe()
                now_ns = self.clock.now_ns()
                response: InferenceResponse | None = worker.poll()
                chunk = None
                settings = None
                if response is not None:
                    responses += 1
                    if response.request_seq == last_submitted:
                        in_flight = False
                    observation_ns = pending_observation_ns.pop(response.request_seq, None)
                    invalid_reason = (
                        response.error
                        or ("stale_session" if response.session_id != self.session_id else None)
                        or ("stale_request" if response.request_seq != last_submitted else None)
                        or ("observation_mismatch" if response.observation_time_ns != observation_ns else None)
                        or ("deadline_exceeded" if response.finished_time_ns > last_deadline or now_ns > last_deadline else None)
                        or ("empty_action" if response.raw_action is None else None)
                    )
                    if invalid_reason is not None:
                        self.strategy.on_response_rejected(response)
                        rejected += 1
                        rejection_reasons.append(invalid_reason)
                    else:
                        try:
                            chunk = self.adapter.decode_action(
                                response.raw_action,
                                ActionContext(
                                    response.request_seq, response.observation_time_ns,
                                    response.finished_time_ns, execution_time_ns=now_ns,
                                    measured_state=observation.state,
                                    max_source_steps=self.config["inference"]["max_chunk_policy_steps"],
                                ),
                            )
                            if chunk.action_space != "joint_position":
                                raise ValueError("policy adapter must return joint_position")
                            chunk = prepare_strategy_chunk(
                                self.strategy, chunk=chunk, response=response, now_ns=now_ns,
                            )
                            settings = self.strategy.commit_settings(
                                response=response, measured=observation.state.groups,
                                last_command=self._last_command,
                            )
                        except (TypeError, ValueError) as exc:
                            self.strategy.on_response_rejected(response)
                            rejected += 1
                            rejection_reasons.append(f"invalid_action:{exc}")
                            chunk = None
                result, ack = self.execute(observation, chunk, commit_settings=settings)
                command_acks.append(ack)
                if result is not None and response is not None and chunk is not None:
                    if result.accepted:
                        accepted += 1
                        self.strategy.on_plan_accepted(
                            chunk=chunk, result=result, response=response, now_ns=self.clock.now_ns(),
                        )
                    else:
                        rejected += 1
                        rejection_reasons.append(result.reason)
                        self.strategy.on_response_rejected(response)
                now_ns = self.clock.now_ns()
                submission = self.strategy.build_submission(
                    session_id=self.session_id, request_seq=request_seq + 1, now_ns=now_ns,
                    snapshot=observation, adapter=self.adapter, timeline=self.timeline,
                    request_state=RequestState(in_flight, last_submitted, last_deadline),
                    runtime_state=RuntimeState.RUNNING,
                )
                if submission is not None:
                    prepared = self.adapter.prepare_request(submission.request)
                    worker.submit_latest(prepared)
                    request_seq = prepared.request_seq
                    last_submitted = request_seq
                    last_deadline = prepared.deadline_ns
                    pending_observation_ns = {request_seq: prepared.observation_time_ns}
                    in_flight = True
                self.strategy.on_tick(steps=step_index + 1, loop_ms=0.0, control_dt_ns=self.dt_ns)
                health = self.client.health()
                if health["episode"]["state"] not in {"ready", "running"}:
                    break
                next_tick += self.dt_ns
                self.clock.sleep_until_ns(next_tick)
            return {
                "steps": step_index + 1, "responses": responses,
                "accepted_plans": accepted, "rejected_plans": rejected,
                "rejection_reasons": rejection_reasons,
                "command_acks": command_acks,
                "asyncsim_result": self.client.result(),
            }
        finally:
            worker.close()
            self.close()

    def run(self, chunks: Iterable[ActionChunk | None], *, seed: int | None = None) -> list[dict]:
        """Run a fake-policy sequence; worker submission is added separately."""
        acknowledgements = []
        self.start(seed=seed)
        next_tick = self.clock.now_ns()
        try:
            for chunk in chunks:
                _, _, ack = self.step(chunk)
                acknowledgements.append(ack)
                next_tick += self.dt_ns
                self.clock.sleep_until_ns(next_tick)
            return acknowledgements
        finally:
            self.close()

    def close(self) -> None:
        self._started = False
        self.sensor.close()
        self.robot.close()
