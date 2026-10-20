"""Run configured planning and execution on an isolated, non-dispatching timeline."""

from manimux.runtime.decode_forecast import DecodeForecast
from manimux.runtime.execution import ExecutionPipeline
from manimux.runtime.executors import SmoothExecutor
from manimux.runtime.ik import step_ik_enabled
from manimux.runtime.inference import RequestState, StrategyChunkSelector, prepare_strategy_chunk
from manimux.runtime.planning import PlanningContext
from manimux.runtime.safety import RuntimeState
from manimux.runtime.scheduling import RequestScheduler
from manimux.runtime.timeline import CommitResult
from manimux.types import RobotCommand, RobotState, copy_group_vector


class PreviewPlanner(PlanningContext):
    """The hardware boundary is structural: no robot or command sink is stored.

    Observations remain measured, not fabricated from predicted commands. Only
    reference/executor history advances, so this is a dry run, not a simulator.
    """

    def __init__(
        self, config, *, strategy, adapter, executor, kinematics, safety, decoder, session_id, clock
    ):
        self._config = config
        self._strategy = strategy
        self._adapter = adapter
        self._decoder = decoder
        if config["policy"].get("action_decoding") == "process" and decoder is None:
            raise ValueError("process-configured warmup requires the runtime action decoder")
        self._session_id = session_id
        self._clock = clock
        self._step_ik = step_ik_enabled(config)
        self._control_dt_ns = int(1e9 / config["robot"]["control_hz"])
        self._decode_seed_source = getattr(adapter, "decode_seed_source", "execution_reference")
        inference = config["inference"]
        self._decode_forecast = DecodeForecast(
            floor_s=inference["expected_decode_s"],
            size=inference["decode_forecast_size"],
            mode=inference["decode_forecast_mode"],
        )
        self.scheduler = RequestScheduler(inference)
        self._multi_inflight = self.scheduler.multi_inflight
        self._min_execute_steps = inference.get("min_plan_hold_policy_steps", 0)
        self._timeline = self._build_timeline()
        self.execution = ExecutionPipeline(
            executor,
            kinematics=kinematics,
            ik_options=config.get("execution", {}).get("ik"),
            control_dt_ns=self._control_dt_ns,
        )
        self.safety = safety
        self.last_command = None
        self.previous_command = None
        self.last_seq = -1
        self.last_deadline_ns = 0
        self.steps = 0
        if (inference.get("rtc") or {}).get("condition_source") == "executor":
            predictor = getattr(executor, "predict_commands", None)
            setter = getattr(strategy, "set_condition_predictor", None)
            if not callable(predictor) or not callable(setter):
                raise ValueError(
                    "RTC executor conditions require a forecast-capable executor and strategy"
                )
            setter(predictor)

    def initialize(self, state):
        if self.last_command is None:
            self.last_command = copy_group_vector(state.groups)
            self.previous_command = copy_group_vector(state.groups)
            self.execution.executor.reset(state)
            self.execution.warmup(state.groups)
            self.safety.reset(state)

    def submission(self, snapshot, *, in_flight):
        return self.scheduler.build_submission(
            self._strategy,
            session_id=self._session_id,
            request_seq=self.last_seq + 1,
            now_ns=self._clock.now_ns(),
            snapshot=snapshot,
            adapter=self._adapter,
            timeline=self._timeline,
            runtime_state=RuntimeState.RUNNING,
            request_state=RequestState(
                in_flight=in_flight,
                last_submitted_seq=self.last_seq,
                last_deadline_ns=self.last_deadline_ns,
                multi_flight=self._multi_inflight,
                expected_decode_ns=round(self._decode_forecast.seconds * 1e9),
            ),
        )

    def submitted(self, request, now_ns):
        self.last_seq = request.request_seq
        self.last_deadline_ns = request.deadline_ns
        self.scheduler.on_submitted(now_ns)

    def rejection_reason(self, response):
        return self._response_rejection_reason(
            response,
            last_submitted_seq=self.last_seq,
            last_request_deadline_ns=self.last_deadline_ns,
            discard_responses_through=-1,
        )

    def replacement_ready(self):
        return self._timeline.replacement_ready(self._clock.now_ns(), self._min_execute_steps)

    def accept(self, chunk, response, state):
        if chunk.action_space != self._timeline.action_space:
            raise ValueError(f"Warmup requires canonical {self._timeline.action_space} output")
        now_ns = self._clock.now_ns()
        if isinstance(self._strategy, StrategyChunkSelector):
            chunk = self._strategy.select_chunk(chunk=chunk, now_ns=now_ns)
        chunk = prepare_strategy_chunk(
            self._strategy, chunk=chunk, response=response, now_ns=now_ns
        )
        commit = self._strategy.commit_settings(
            response=response,
            measured=state.groups,
            last_command=self.last_command,
        )
        anchor = self.execution.reference_from_joints(commit.current_command)
        now_ns = self._clock.now_ns()
        source_end_ns = (
            chunk.observation_time_ns
            + (chunk.source_offset_steps + chunk.horizon_steps - 1) * chunk.dt_ns
        )
        settings = self._config["inference"]
        if (
            self._decoder is not None
            and settings["action_start_mode"] == "drop_infer_latency"
            and now_ns > source_end_ns
        ):
            return chunk, CommitResult(False, "no_future_horizon"), {}
        result = self._timeline.commit(
            chunk,
            now_ns=now_ns,
            max_plan_age_ns=int(settings["max_plan_age_s"] * 1e9),
            current_command=anchor,
            blend_steps=commit.blend_steps,
            command_velocity={
                name: (self.last_command[name] - self.previous_command[name])
                / (self._control_dt_ns / 1e9)
                for name in self.last_command
            },
            handoff_skip_steps=settings["handoff_skip_steps"],
        )
        fields = {}
        if result.accepted:
            fields = self._strategy.on_plan_accepted(
                chunk=chunk,
                result=result,
                response=response,
                now_ns=now_ns,
            )
        return chunk, result, fields

    def tick(self, state, *, loop_started_ns):
        started = self._clock.now_ns()
        reference = self._timeline.reference_horizon(
            now_ns=started,
            dt_ns=self._control_dt_ns,
            horizon_steps=self.execution.horizon_steps,
        )
        ik_status = None
        if reference is not None:
            result = self.execution.step(started, state, reference, last_command=self.last_command)
            command, ik_status = result.command, result.ik_status
        else:
            held = RobotState(copy_group_vector(self.last_command), started, state.sequence)
            executor = self.execution.executor
            if isinstance(executor, SmoothExecutor):
                command = executor.hold(started, held)
                command.plan_id = self._timeline.active_plan_id
            else:
                executor.reset(held)
                command = RobotCommand(held.groups, started, self._timeline.active_plan_id)
        # Same validation as dispatch, with independent command-rate history.
        self.safety.validate_command(command)
        self.previous_command, self.last_command = (
            self.last_command,
            copy_group_vector(command.groups),
        )
        elapsed = (self._clock.now_ns() - started) / 1e6
        self.steps += 1
        self._strategy.on_tick(
            steps=self.steps,
            loop_ms=(self._clock.now_ns() - loop_started_ns) / 1e6,
            control_dt_ns=self._control_dt_ns,
        )
        return elapsed, ik_status

    def metadata(self):
        return {
            "active_chunk_id": -(self._timeline.accepted_request_seq + 1)
            if self._timeline.accepted_request_seq >= 0
            else None,
            "chunk_index": self._timeline.cursor(self._clock.now_ns()),
            "preview_control_steps": self.steps,
        }
