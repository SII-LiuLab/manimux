"""Independent simulation control loop; no physical robot or camera lifecycle."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from manimux.clock import Clock, SystemClock
from manimux.embodiments.asyncsim import AsyncSimClient, AsyncSimRobot, AsyncSimSensor
from manimux.runtime.executors import DirectExecutor
from manimux.runtime.safety import SafetyGuard
from manimux.runtime.timeline import ActionTimeline, CommitResult
from manimux.types import ActionChunk, ObservationSnapshot, RobotCommand, copy_group_vector


class AsyncSimRuntime:
    """Own one AsyncSim session, one timeline, and one canonical command stream."""

    def __init__(self, config: dict, *, client: Any = None, clock: Clock | None = None) -> None:
        if config["executor"]["type"] != "direct":
            raise ValueError("AsyncSim currently supports only direct executor")
        if config["inference"]["algorithm"] != "manimux":
            raise ValueError("AsyncSim currently supports only the default inference strategy")
        self.config = config
        self.clock = clock or SystemClock()
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
        self.executor = DirectExecutor(config["executor"]["motion_limits"], self.dt_ns / 1e9)
        limits = config["executor"]["command_safety"]
        self.safety = SafetyGuard(
            robot["group_dims"], None,
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

    def step(self, chunk: ActionChunk | None = None) -> tuple[ObservationSnapshot, CommitResult | None, dict]:
        if not self._started:
            raise RuntimeError("start the AsyncSim runtime before stepping")
        now_ns = self.clock.now_ns()
        snapshot = self.client.read_snapshot(self.streams)
        state = self.robot.use_snapshot(snapshot)
        frames = self.sensor.use_snapshot(snapshot)
        self.safety.validate_state(state)
        if self._last_command is None:
            self._last_command = copy_group_vector(state.groups)
            self.executor.reset(state)
            self.safety.reset(state)
        observation = ObservationSnapshot(state, frames)
        result = None
        if chunk is not None:
            if chunk.action_space != "joint_position":
                raise ValueError("AsyncSim requires a canonical joint_position chunk")
            result = self.timeline.commit(
                chunk, now_ns=now_ns,
                commit_lead_ns=round(self.config["inference"]["commit_lead_s"] * 1e9),
                max_plan_age_ns=round(self.config["inference"]["max_plan_age_s"] * 1e9),
                current_command=self._last_command,
                blend_steps=self.config["inference"]["blend_policy_steps"],
            )
        horizon = self.timeline.reference_horizon(
            now_ns=now_ns, dt_ns=self.dt_ns, horizon_steps=self.executor.horizon_steps
        )
        if horizon is None:
            command = RobotCommand(copy_group_vector(self._last_command), now_ns, "hold")
        else:
            command = self.executor.step(now_ns, state, horizon)
        self.safety.validate_command(command)
        ack = self.robot.send_command(command)
        self._last_command = copy_group_vector(command.groups)
        return observation, result, ack

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
