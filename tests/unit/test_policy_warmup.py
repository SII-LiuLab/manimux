from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from manimux.policies.capabilities import PolicyCapabilities
from manimux.policies.worker import PolicyResetResult
from manimux.runtime.inference import DefaultChunkStrategy, seed_strategy_warmup
from manimux.runtime.rtc.strategy import RtcInferenceStrategy
from manimux.runtime.warmup import PolicyWarmup
from manimux.types import ActionChunk, InferenceResponse, ObservationSnapshot, RobotState


class Clock:
    value = 1_000_000_000

    def now_ns(self):
        return self.value


class Worker:
    is_alive = True

    def __init__(self):
        self.requests = []
        self.responses = deque()
        self.resets = []
        self.reset_results = deque()

    def submit_latest(self, request):
        self.requests.append(request)

    def poll(self):
        return self.responses.popleft() if self.responses else None

    def submit_reset(self, session_id):
        self.resets.append(session_id)
        return len(self.resets)

    def poll_reset(self):
        return self.reset_results.popleft() if self.reset_results else None


class Adapter:
    def __init__(self, clock):
        self.clock = clock
        self.decode_error = False

    def build_observation(self, snapshot):
        self.clock.value += 10_000_000
        return snapshot

    def prepare_request(self, request):
        self.clock.value += 20_000_000
        return request

    def decode_seed_time_ns(self, context):
        return None

    def decode_action(self, raw_action, context):
        self.clock.value += 30_000_000
        if self.decode_error:
            raise ValueError("invalid action")
        return ActionChunk(
            plan_id="warmup-output-must-be-discarded",
            request_seq=context.request_seq,
            observation_time_ns=context.observation_time_ns,
            created_time_ns=context.created_time_ns,
            action_space="joint_position",
            dt_ns=33_333_333,
            groups={"arm": np.zeros((50, 2))},
        )


@pytest.fixture
def setup():
    config = {
        "robot": {"group_dims": {"arm": 2}},
        "run": {'task_prompt': "Put bottles into the bin"},
        "policy": {
            "timeout_s": 2.0,
            "horizon_policy_steps": 50,
            "action_dt_s": 1 / 30,
            "trajectory_duration_s": None,
        },
        "inference": {
            "inference_schedule": "single_inflight",
            "handoff": "blend",
            "handoff_skip_steps": 0,
            "rtc": {
                "delay_buffer_size": 10,
                "initial_delay_policy_steps": None,
                "min_execute_policy_steps": 16,
                "beta": 5.0,
            },
        },
    }
    from manimux.runtime import inference_parameters
    config['inference'] = {**inference_parameters(executor={'type': 'direct'}), **config['inference']}
    config['inference'].update(request_trigger='continuous', blend_policy_steps=0)
    config['robot']['control_hz'] = 100
    config['executor'] = {'type': 'direct', 'motion_limits': None}
    clock = Clock()
    worker = Worker()
    adapter = Adapter(clock)
    snapshot = ObservationSnapshot(RobotState({"arm": np.zeros(2)}, clock.value, 0))

    def create(strategy=None):
        return PolicyWarmup(
            config,
            worker=worker,
            strategy=strategy or DefaultChunkStrategy(config),
            adapter=adapter,
            session_id="trial",
            clock=clock,
            executor=__import__('manimux.runtime.executors', fromlist=['DirectExecutor']).DirectExecutor(),
            kinematics=None,
            safety=__import__('manimux.runtime.safety', fromlist=['SafetyGuard']).SafetyGuard({'arm': 2}, None),
        )

    return config, clock, worker, adapter, snapshot, create


def respond(worker, clock, *, error=None, session_id="trial"):
    request = worker.requests[-1]
    worker.responses.append(InferenceResponse(
        session_id=session_id,
        request_seq=request.request_seq,
        finished_time_ns=clock.value,
        inference_ms=5.0,
        raw_action=None if error else object(),
        error=error,
        observation_time_ns=request.observation_time_ns,
        deadline_ns=request.deadline_ns,
    ))


def test_strategy_tick_accounts_for_decode_time_separately_from_execution(setup):
    _, clock, worker, _, snapshot, create = setup
    warmup = create()
    ticks = []
    warmup.strategy.on_tick = lambda **fields: ticks.append(fields)
    warmup.activate()
    warmup.advance(paused=True, snapshot=snapshot)
    respond(worker, clock)
    warmup.advance(paused=True, snapshot=snapshot)
    assert warmup.error == ""
    assert ticks[-1]["loop_ms"] == pytest.approx(30.0)
    assert warmup.metadata()["execution_ms"] == pytest.approx(0.0)


def test_start_waits_for_inflight_response_then_reset_ack(setup):
    _, clock, worker, _, snapshot, create = setup
    warmup = create()
    warmup.activate()
    assert warmup.advance(paused=True, snapshot=snapshot) is None
    assert worker.requests[0].request_seq < 0
    warmup.advance(paused=False, snapshot=snapshot)
    assert warmup.phase == "draining"
    assert not worker.resets
    assert len(worker.requests) == 1
    respond(worker, clock)
    assert warmup.advance(paused=False, snapshot=snapshot) is None
    assert worker.resets == ["trial"]
    assert warmup.phase == "resetting"
    assert not warmup.complete
    worker.reset_results.append(PolicyResetResult("trial", 1, clock.value))
    warmup.advance(paused=False, snapshot=snapshot)
    assert warmup.complete
    warmup.advance(paused=False, snapshot=snapshot)
    assert len(worker.requests) == 1
    assert all(name.startswith("warmup_") for name, _ in warmup.take_events())
    assert warmup.take_preview() is None


def test_warmup_preview_is_single_use_and_disappears_on_start(setup):
    _, clock, worker, _, snapshot, create = setup
    warmup = create()
    warmup.activate()
    warmup.advance(paused=True, snapshot=snapshot)
    respond(worker, clock)
    warmup.advance(paused=True, snapshot=snapshot)
    chunk, inference_ms, metadata = warmup.take_preview()
    assert chunk.request_seq < 0
    assert metadata["warmup_preview"] is True
    assert metadata["warmup_branch"] == "ordinary"
    assert inference_ms >= 0
    assert warmup.take_preview() is None
    respond(worker, clock)
    warmup.advance(paused=False, snapshot=snapshot)
    assert warmup.take_preview() is None


def test_e2e_counts_only_observation_age_preparation_wait_and_decode(setup):
    _, clock, worker, _, snapshot, create = setup
    snapshot.state.monotonic_ns -= 40_000_000
    warmup = create()
    warmup.activate()
    warmup.advance(paused=True, snapshot=snapshot)
    clock.value += 50_000_000
    respond(worker, clock)
    warmup.advance(paused=False, snapshot=snapshot)
    # 40 observation age + 10 build + 20 prepare + 50 wait + 30 decode.
    assert not warmup.latency_ns  # The first branch call is displayed, not calibrated.
    assert warmup.metadata()["recent_inference_ms"] == [150.0]
    assert warmup.first_inference_ms == {"ordinary": 150.0}
    completed = dict(warmup.take_events())["warmup_inference_completed"]
    assert completed["e2e_ms"] == 150.0
    assert completed["inference_ms"] == 5.0


def next_request(warmup, worker, clock, snapshot, *, paused=True):
    count = len(worker.requests)
    for _ in range(250):
        clock.value += 10_000_000
        snapshot.state.monotonic_ns = clock.value
        warmup.advance(paused=paused, snapshot=snapshot)
        if len(worker.requests) > count:
            return worker.requests[-1]
        assert warmup.phase != 'error', warmup.error
    raise AssertionError('configured scheduler did not submit')


def test_calibration_excludes_first_call_per_branch_and_bounds_recent_samples(setup):
    config, clock, worker, _, snapshot, create = setup
    config['inference']['algorithm'] = 'rtc'
    config['inference']['request_trigger'] = 'algorithm'
    config['inference']['rtc']['delay_buffer_size'] = 3
    warmup = create(RtcInferenceStrategy(config))
    warmup.activate()
    for index in range(6):
        next_request(warmup, worker, clock, snapshot)
        clock.value += 50_000_000
        respond(worker, clock)
        warmup.advance(paused=True, snapshot=snapshot)
        assert not warmup.error
        if index < 2:
            assert not warmup.latency_ns
    assert warmup.completed_requests == 6
    assert len(warmup.latency_ns) == 3
    completions = [f for k, f in warmup.take_events() if k == 'warmup_inference_completed']
    assert [c['branch'] for c in completions] == ['ordinary'] + ['conditioned'] * 5
    assert [c['first_call'] for c in completions] == [True, True, False, False, False, False]
    assert list(warmup.latency_ns) == [round(c['e2e_ms'] * 1e6) for c in completions[-3:]]


@pytest.mark.parametrize("failure", ["model", "decode", "identity"])
def test_failed_warmup_does_not_enter_success_samples_and_retries(setup, failure):
    _, clock, worker, adapter, snapshot, create = setup
    warmup = create()
    warmup.activate()
    warmup.advance(paused=True, snapshot=snapshot)
    adapter.decode_error = failure == "decode"
    respond(
        worker, clock,
        error="model failure" if failure == "model" else None,
        session_id="other trial" if failure == "identity" else "trial",
    )
    warmup.advance(paused=True, snapshot=snapshot)
    assert not warmup.latency_ns
    assert warmup.completed_requests == 0
    assert warmup.error
    assert len(worker.requests) == 1
    snapshot.state.monotonic_ns = clock.value
    warmup.advance(paused=True, snapshot=snapshot)
    assert len(worker.requests) == 2
    assert worker.requests[-1].request_seq < worker.requests[0].request_seq
    assert any(name == "warmup_inference_failed" for name, _ in warmup.take_events())


def test_reset_error_requires_another_start_and_successful_ack(setup):
    _, clock, worker, _, snapshot, create = setup
    warmup = create()
    warmup.activate()
    # Start still waits for one valid configured plan and backend reset.
    warmup.advance(paused=False, snapshot=snapshot)
    assert len(worker.requests) == 1
    respond(worker, clock)
    warmup.advance(paused=False, snapshot=snapshot)
    assert worker.resets == ["trial"]
    worker.reset_results.append(PolicyResetResult("trial", 1, clock.value, "timeout"))
    warmup.advance(paused=False, snapshot=snapshot)
    assert warmup.phase == "error"
    assert not warmup.complete
    warmup.advance(paused=False, snapshot=snapshot)
    assert len(worker.resets) == 1
    warmup.advance(paused=True, snapshot=snapshot)
    warmup.advance(paused=False, snapshot=snapshot)
    assert len(worker.resets) == 2
    worker.reset_results.append(PolicyResetResult("trial", 2, clock.value))
    warmup.advance(paused=False, snapshot=snapshot)
    assert warmup.complete
    assert not warmup.latency_ns


def test_rtc_warms_both_branches_without_polluting_formal_scheduler(setup):
    config, clock, worker, _, snapshot, create = setup
    config['inference'].update(algorithm='rtc', request_trigger='algorithm')
    strategy = RtcInferenceStrategy(config)
    formal = RtcInferenceStrategy(config)
    warmup = create(strategy)
    warmup.activate()
    ordinary = next_request(warmup, worker, clock, snapshot)
    assert ordinary.action_condition is None
    respond(worker, clock)
    warmup.advance(paused=True, snapshot=snapshot)
    conditioned = next_request(warmup, worker, clock, snapshot)
    assert conditioned.action_condition.shape == (50, 2)
    assert conditioned.condition_weights.shape == (50,)
    assert conditioned.rtc_beta == 5.0
    respond(worker, clock)
    warmup.advance(paused=True, snapshot=snapshot)
    assert set(warmup.first_inference_ms) == {'ordinary', 'conditioned'}
    assert strategy._active_rows is not None  # isolated preview history really advances
    assert formal._active_rows is None
    assert not formal._request_started_ns
    assert not formal._latency


def test_rtc_automatic_seed_uses_recent_e2e_samples_and_survives_reset(setup):
    config, _, _, _, _, _ = setup
    strategy = RtcInferenceStrategy(config)
    seed_strategy_warmup(strategy, latency_ns=[33_333_334, 66_666_665])
    assert list(strategy._delay_forecast) == [2, 2]
    strategy.reset()
    assert list(strategy._delay_forecast) == [2, 2]
    seed_strategy_warmup(strategy, latency_ns=[])
    assert list(strategy._delay_forecast) == [0]


def test_rtc_explicit_initial_delay_is_not_overridden_by_warmup(setup):
    config, _, _, _, _, _ = setup
    config["inference"]["rtc"]["initial_delay_policy_steps"] = 4
    strategy = RtcInferenceStrategy(config)
    seed_strategy_warmup(strategy, latency_ns=[1_000_000_000])
    strategy.reset()
    assert list(strategy._delay_forecast) == [4]


def test_dead_worker_never_completes_or_resets(setup):
    _, _, worker, _, snapshot, create = setup
    warmup = create()
    warmup.activate()
    worker.is_alive = False
    warmup.advance(paused=False, snapshot=snapshot)
    assert warmup.phase == "error"
    assert not warmup.complete
    assert not worker.requests
    assert not worker.resets


@pytest.mark.parametrize("method", ["poll", "poll_reset"])
def test_ipc_failure_is_visible_and_cannot_complete_warmup(setup, monkeypatch, method):
    _, clock, worker, _, snapshot, create = setup
    warmup = create()
    warmup.activate()
    warmup.advance(paused=method == "poll", snapshot=snapshot)
    if method == 'poll_reset':
        respond(worker, clock)
        warmup.advance(paused=False, snapshot=snapshot)

    def fail():
        raise OSError("worker pipe closed")

    monkeypatch.setattr(worker, method, fail)
    warmup.advance(paused=method == "poll", snapshot=snapshot)
    assert warmup.phase == "error"
    assert "worker pipe closed" in warmup.error
    assert not warmup.complete


@pytest.mark.parametrize("control_timing", [False, True])
@pytest.mark.parametrize("sensor_mode", ["inline", "background"])
@pytest.mark.parametrize("algorithm", ["default", "paint"])
def test_edge_discards_warmup_actions_and_keeps_formal_counts_zero(
    setup, monkeypatch, tmp_path, control_timing, sensor_mode, algorithm,
):
    import threading

    import manimux.runtime.edge as edge
    from manimux.types import SensorFrame

    config, clock, worker, adapter, _, _ = setup
    config["robot"]["options"] = {}
    config["run"].update(
        warmup_before_start=True, max_control_steps=1, control_timing=control_timing,
        sensor_reading={"mode": sensor_mode},
    )
    config["policy"].update(worker="fake", adapter={"type": "fake"})
    config["inference"].update(
        max_chunk_policy_steps=50, blend_policy_steps=0,
        action_start_mode="first_step_when_ready",
    )
    config["executor"] = {"type": "direct", "motion_limits": None}
    config["robogui"] = {"policy_label": "warmup isolation", "enabled": False, "robot": "fake"}
    config["recording"] = {"video_fps": 30, "video_codec": "unused", "video_queue_size": 1}
    cycle_starts, waits = [], []

    def sleep_until(target):
        work_ns = clock.value - cycle_starts[-1]
        waits.append(target - clock.value)
        assert target - clock.value == max(0, 10_000_000 - work_ns)
        clock.value = max(clock.value, target)
        if sensor_mode == 'background':
            __import__('time').sleep(0.002)  # Let the real sensor thread catch up to the fake clock.

    clock.sleep_until_ns = sleep_until
    commands, events, metadata_updates, formal_samples, previews, stops = [], [], [], [], [], []

    def forbidden(*args, **kwargs):
        raise AssertionError("warmup entered the formal execution or recording path")

    def noop(*args, **kwargs):
        return None

    def record_event(kind, **fields):
        events.append((kind, fields))

    recorder = SimpleNamespace(
        final_dir=tmp_path / "rollout-001",
        update_metadata=lambda **fields: metadata_updates.append(fields),
        event=record_event,
        record_tick=forbidden,
        finish=lambda **fields: tmp_path / "rollout-001",
        abort=forbidden,
    )
    monkeypatch.setattr(edge, "EpisodeRecorder", lambda *args, **kwargs: recorder)
    monkeypatch.setattr(
        edge, "build_inference_strategy",
        lambda cfg: paint_strategy(cfg) if algorithm == "paint" else DefaultChunkStrategy(cfg),
    )
    worker.capabilities = PolicyCapabilities(backend_metadata={"model": {"inference_seed": 0}})
    worker.start = noop
    worker.close = noop
    worker.request_stop = lambda: stops.append("stop")
    submit = worker.submit_latest

    def auto_respond(request):
        submit(request)
        respond(worker, clock)

    worker.submit_latest = auto_respond
    reset = worker.submit_reset

    def auto_ack(session):
        sequence = reset(session)
        worker.reset_results.append(PolicyResetResult(session, sequence, clock.value))
        return sequence

    worker.submit_reset = auto_ack
    controls = iter([
        SimpleNamespace(paused=True, finish_requested=False, home_requested=False),
        SimpleNamespace(paused=True, finish_requested=False, home_requested=False),
        SimpleNamespace(paused=True, finish_requested=False, home_requested=False),
        SimpleNamespace(paused=False, finish_requested=False, home_requested=False),
        *([
            SimpleNamespace(paused=False, finish_requested=False, home_requested=False),
            SimpleNamespace(paused=False, finish_requested=False, home_requested=False),
            SimpleNamespace(paused=False, finish_requested=False, home_requested=False),
            SimpleNamespace(paused=False, finish_requested=False, home_requested=False),
        ] if algorithm == "paint" else []),
        SimpleNamespace(paused=False, finish_requested=True, finish_home=False),
    ])

    def poll_control():
        cycle_starts.append(clock.value)
        if algorithm == 'paint':
            assert len(cycle_starts) < 300
            return SimpleNamespace(paused=len(cycle_starts) <= 3,
                                   finish_requested=bool(formal_samples), home_requested=False)
        return next(controls)

    runtime = edge.EdgeRuntime.__new__(edge.EdgeRuntime)
    runtime._dispatch = None
    runtime._external = None
    runtime._step_ik = False
    runtime._ik_resolver = None
    runtime._config = config
    runtime._run_dir = tmp_path
    from manimux.runtime import inference_parameters
    from manimux.runtime.scheduling import RequestScheduler
    runtime._config["inference"] = {
        **inference_parameters(executor={"type": "direct"}), **runtime._config["inference"],
    }
    runtime._scheduler = RequestScheduler(inference_parameters(executor={"type": "direct"}))
    runtime._multi_inflight = False
    runtime._min_execute_steps = 0
    runtime._clock = clock
    runtime._session_id = "trial"
    runtime._control_dt_ns = 10_000_000
    runtime._rollout_identity = {}
    runtime._evaluation = {"kind": "binary"}
    runtime._launch_mode = "serve"
    sensor_owners = []

    def sensor_start():
        sensor_owners.append(threading.get_ident())

    def sensor_read():
        sensor_owners.append(threading.get_ident())
        return SensorFrame("any_camera", np.zeros((2, 2, 3), np.uint8), clock.value, 1)

    def sensor_close():
        sensor_owners.append(threading.get_ident())

    runtime._sensors = [SimpleNamespace(start=sensor_start, read=sensor_read, close=sensor_close)]
    runtime._decoder = None
    runtime._adapter = SimpleNamespace(prepare_request=forbidden, decode_action=forbidden)
    runtime._build_adapter = lambda: adapter
    runtime._timeline = SimpleNamespace(active_plan_id=None)
    runtime._build_timeline = lambda: SimpleNamespace(active_plan_id=None)
    runtime._validate_policy_capabilities = noop
    runtime._worker = worker
    runtime._strategy = SimpleNamespace(
        name="manimux", control_mode="managed", reset=noop,
        seed_warmup=lambda **fields: formal_samples.append(fields),
        build_submission=forbidden, on_tick=forbidden,
    )
    runtime._robot = SimpleNamespace(
        kinematics=None,
        connect=noop, stop=noop, close=noop, home=lambda: stops.append('home'),
        runtime_metadata=lambda: {"source": "mock"}, recovery_actions=(),
        get_state=lambda: RobotState({"arm": np.ones(2)}, clock.value, 0),
        send_command=commands.append,
    )
    runtime._executor = SimpleNamespace(reset=noop, step=forbidden)
    from manimux.runtime.execution import ExecutionPipeline
    runtime._execution = ExecutionPipeline(runtime._executor, kinematics=None,
        ik_options=None, control_dt_ns=10_000_000)
    runtime._safety = SimpleNamespace(reset=noop, validate_state=noop, validate_command=noop,
                                      execution_limits=lambda: {})
    runtime._robogui = SimpleNamespace(
        poll_control=poll_control, set_state_metadata=noop,
        publish_event=noop, publish_state=noop, close=noop,
        publish_plan=lambda *args, **kwargs: previews.append((args, kwargs)),
    )
    result = runtime.run()
    assert len(set(sensor_owners)) == 1
    assert (sensor_owners[0] == threading.get_ident()) == (sensor_mode == "inline")
    from manimux.timing import active_timing
    assert active_timing.get() is None
    assert result.steps == result.accepted_plans == result.rejected_plans == 0
    assert len(commands) >= 4 if algorithm == 'paint' else len(commands) == 4
    assert len(waits) == len(commands)
    assert 0 in waits
    for command in commands:
        np.testing.assert_array_equal(command.groups["arm"], np.ones(2))
    assert len(worker.requests) == (3 if algorithm == "paint" else 2)
    # Finish fences worker requests; shared cleanup repeats the idempotent stop
    # and always Homes the connected robot, including warmup-only episodes.
    assert stops == ["stop", "stop", "home"]
    assert len(previews) == 1
    assert previews[0][1]["metadata"]["warmup_preview"] is True
    assert all(request.request_seq < 0 for request in worker.requests)
    assert len(formal_samples) == 1
    assert len(formal_samples[0]["latency_ns"]) == 1
    ready = dict(events)["formal_rollout_ready"]
    assert ready["inference_seed"] == 0
    assert ready["latency_sample_count"] == len(formal_samples[0]["latency_ns"])
    assert any(item.get("warmup", {}).get("phase") == "complete" for item in metadata_updates)


def test_idle_warmup_never_submits_until_explicit_activation(setup):
    _, _, worker, _, snapshot, create = setup
    warmup = create()
    for paused in (True, False, True):
        warmup.advance(paused=paused, snapshot=snapshot)
    assert warmup.phase == "idle"
    assert not worker.requests and not worker.resets
    warmup.activate()
    warmup.activate()
    warmup.advance(paused=True, snapshot=snapshot)
    assert len(worker.requests) == 1
    assert [name for name, _ in warmup.take_events()].count("warmup_activated") == 1


def paint_strategy(config):
    from manimux.runtime.paint import PaintInferenceStrategy
    config['inference']['algorithm'] = 'paint'
    config['inference']['request_trigger'] = 'algorithm'
    config['inference']['inference_schedule'] = 'single_inflight'
    config['inference']['paint'] = {
        'execution_policy_steps': 16,
        'initial_delay_policy_steps': 10,
        'delay_buffer_size': 3,
    }
    return PaintInferenceStrategy(config)


def test_paint_early_start_finishes_both_branches_and_calibrates_before_reset(setup):
    config, clock, worker, _, snapshot, create = setup
    strategy = paint_strategy(config)
    warmup = create(strategy)
    warmup.activate()
    branches = []
    for index in range(3):
        request = next_request(warmup, worker, clock, snapshot, paused=False)
        branches.append(getattr(request, 'paint_action_prefix', None) is not None)
        if branches[-1]:
            assert request.paint_delay_steps >= 10
            np.testing.assert_array_equal(request.paint_action_prefix, np.zeros((request.paint_delay_steps, 2)))
        clock.value += 100_000_000
        respond(worker, clock)
        warmup.advance(paused=False, snapshot=snapshot)
        assert not warmup.error
        assert warmup.take_preview() is None
        if index < 2:
            assert not worker.resets
    assert branches == [False, True, True]
    assert warmup.metadata()['completed_branch_calls'] == {'ordinary': 1, 'conditioned': 2}
    assert warmup.ready
    assert len(warmup.latency_ns) == 1
    assert warmup.latency_ns.maxlen == 3
    assert worker.resets == ['trial']
    worker.reset_results.append(PolicyResetResult('trial', 1, clock.value))
    warmup.advance(paused=False, snapshot=snapshot)
    assert warmup.complete
    assert strategy._active_rows is None
    assert not strategy._request_started_ns


def test_paint_failed_conditioned_warmup_cannot_reset_or_start(setup):
    config, clock, worker, _, snapshot, create = setup
    strategy = paint_strategy(config)
    warmup = create(strategy)
    warmup.activate()
    warmup.advance(paused=False, snapshot=snapshot)
    respond(worker, clock)
    warmup.advance(paused=False, snapshot=snapshot)
    next_request(warmup, worker, clock, snapshot, paused=False)
    assert worker.requests[-1].paint_delay_steps == 10
    respond(worker, clock, error='PAINT sampler failed')
    warmup.advance(paused=False, snapshot=snapshot)
    assert warmup.phase == 'error'
    assert not warmup.complete
    assert not worker.resets
    assert warmup.error == 'PAINT sampler failed'
    assert warmup.metadata()['completed_branch_calls'] == {'ordinary': 1}


def test_paint_warmup_sampling_reaches_transport(setup):
    from manimux.policies.xpolicylab.client import XPolicyLabWsPolicyModel
    config, clock, _, adapter, snapshot, _ = setup
    strategy = paint_strategy(config)
    snapshot.state.groups['arm'] = np.array([0.3, 0.7])
    request = strategy.build_warmup_submission(
        session_id='trial', request_seq=-2, now_ns=clock.value,
        snapshot=snapshot, adapter=adapter, conditioned=True,
    ).request
    client = XPolicyLabWsPolicyModel({
        **config['policy'], 'options': {}, 'adapter': {}, 'startup_timeout_s': 2.0,
    })
    sampling = client._sampling(adapter.prepare_request(request))
    assert sampling['mode'] == 'paint'
    assert sampling['delay_steps'] == 10
    np.testing.assert_allclose(sampling['action_prefix'], np.tile([0.3, 0.7], (10, 1)))


def test_slow_decode_defers_submission_until_fresh_observation(setup):
    _, clock, worker, adapter, snapshot, create = setup
    warmup = create()
    warmup.activate()
    warmup.advance(paused=True, snapshot=snapshot)
    clock.value += 275_000_000
    respond(worker, clock)
    warmup.advance(paused=True, snapshot=snapshot)
    assert len(worker.requests) == 1
    completed = [f for k,f in warmup.take_events() if k == 'warmup_inference_completed'][0]
    assert completed['decode_ms'] == 30.0
    assert completed['e2e_ms'] == sum(completed[k] for k in (
        'observation_age_ms', 'request_to_decode_ms', 'decode_ms', 'post_decode_ms'))
    snapshot.state.monotonic_ns = clock.value
    fresh = clock.value
    warmup.advance(paused=True, snapshot=snapshot)
    assert len(worker.requests) == 2
    assert worker.requests[-1].observation_time_ns == fresh

@pytest.mark.parametrize('mode', ['serial', 'async'])
def test_warmup_obeys_configured_execution_window(setup, mode):
    config, clock, worker, _, snapshot, create = setup
    config['inference'].update(
        algorithm=mode, inference_schedule='serial' if mode == 'serial' else 'single_inflight',
        request_trigger='algorithm' if mode == 'serial' else 'progress',
        chunk_policy_steps=19, max_chunk_policy_steps=19 if mode == 'serial' else None,
        action_start_mode='first_step_when_ready' if mode == 'serial' else 'drop_infer_latency',
    )
    warmup = create(); warmup.activate()
    next_request(warmup, worker, clock, snapshot)
    respond(worker, clock); warmup.advance(paused=True, snapshot=snapshot)
    assert not warmup.error
    count = len(worker.requests)
    clock.value += 100_000_000; snapshot.state.monotonic_ns = clock.value
    warmup.advance(paused=True, snapshot=snapshot)
    assert len(worker.requests) == count  # Not a free-running sampler.
    next_request(warmup, worker, clock, snapshot)
    assert warmup.planner._timeline.cursor(clock.value) >= (17 if mode == 'async' else 18)
    assert warmup.planner._timeline.active_horizon().horizon_steps == (19 if mode == 'serial' else 48)


def test_serial_autohorizon_uses_specialized_sampler_and_selected_horizon(setup):
    from manimux.runtime.autohorizon import AutoHorizonInferenceStrategy
    config, clock, worker, _, snapshot, create = setup
    config['inference'].update(algorithm='autohorizon', inference_schedule='serial', request_trigger='algorithm',
                               action_start_mode='first_step_when_ready')
    warmup = create(AutoHorizonInferenceStrategy(config)); warmup.activate()
    request = next_request(warmup, worker, clock, snapshot)
    assert request.autohorizon is True
    respond(worker, clock)
    worker.responses[-1].raw_action = {'autohorizon': {'execution_steps': 7}}
    warmup.advance(paused=True, snapshot=snapshot)
    assert not warmup.error
    assert warmup.planner._timeline.active_horizon().horizon_steps == 7
    assert warmup.take_preview()[0].horizon_steps == 7


def test_warmup_uses_real_step_ik_executor_and_safety_without_dispatch(setup):
    from manimux.runtime.executors import DirectExecutor
    from manimux.runtime.safety import SafetyGuard
    from manimux.policy_adapter.pose_output import pose_chunk
    from step_ik_fixture import inputs
    config, clock, worker, adapter, _, _ = setup
    kin, groups, rows = inputs()
    config['robot']['group_dims'] = {g: 7 for g in groups}
    config['execution'] = {'ik': {'mode': 'step', 'budget_ms': 100}}
    config['inference']['action_start_mode'] = 'first_step_when_ready'
    snapshot = ObservationSnapshot(RobotState(groups, clock.value, 1))
    adapter.decode_action = lambda raw, ctx: pose_chunk(ctx, rows, 100_000_000)
    safety = SafetyGuard(config['robot']['group_dims'], None)
    validated = []
    validate = safety.validate_command
    def check(command):
        validated.append(command); validate(command)
    safety.validate_command = check
    warmup = PolicyWarmup(config, worker=worker, strategy=DefaultChunkStrategy(config),
        adapter=adapter, session_id='trial', clock=clock, executor=DirectExecutor(),
        kinematics=kin, safety=safety)
    warmup.activate(); next_request(warmup, worker, clock, snapshot)
    respond(worker, clock); warmup.advance(paused=True, snapshot=snapshot)
    assert not warmup.error
    assert warmup.metadata()['ik_status']['success']
    assert all(m.calls for m in kin.models.values())
    assert validated and warmup.planner.steps == len(validated)
    assert not hasattr(warmup, 'robot') and not hasattr(warmup.planner, 'robot')
    kin.models['right_arm'].fail = True
    clock.value += 100_000_000; snapshot.state.monotonic_ns = clock.value
    warmup.advance(paused=True, snapshot=snapshot)
    assert not warmup.metadata()['ik_status']['success']


def test_process_decoder_is_used_and_drained_before_start(setup):
    from manimux.runtime.executors import DirectExecutor
    from manimux.runtime.safety import SafetyGuard
    config, clock, worker, adapter, snapshot, _ = setup
    config['policy']['action_decoding'] = 'process'
    config['inference']['decode_forecast_size'] = 3
    class Decoder:
        busy = False
        calls = []
        output = None
        def poll(self):
            output, self.output = self.output, None
            if output is not None: self.busy = False
            return output
        def submit(self, response, context, deadline):
            self.busy = True
            self.calls.append((response, context, deadline))
    decoder = Decoder()
    warmup = PolicyWarmup(config, worker=worker, strategy=DefaultChunkStrategy(config),
        adapter=adapter, session_id='trial', clock=clock, executor=DirectExecutor(),
        kinematics=None, safety=SafetyGuard({'arm': 2}, None), decoder=decoder)
    warmup.activate(); next_request(warmup, worker, clock, snapshot)
    respond(worker, clock); warmup.advance(paused=False, snapshot=snapshot)
    assert decoder.busy and len(decoder.calls) == 1 and not worker.resets
    response, context, _ = decoder.calls[0]
    chunk = adapter.decode_action(None, context); chunk.metadata['decode_stage_ms'] = 30
    decoder.output = SimpleNamespace(response=response, chunk=chunk, error=None)
    adapter.decode_action = lambda *a: pytest.fail('inline decoding bypassed configured process')
    warmup.advance(paused=False, snapshot=snapshot)
    assert not warmup.error and worker.resets == ['trial']
    assert warmup.planner._decode_forecast.seconds >= .03


def test_preview_safety_failure_cannot_become_a_ready_start(setup):
    _, clock, worker, _, snapshot, create = setup
    warmup = create(); warmup.activate()
    next_request(warmup, worker, clock, snapshot)
    def fail(command): raise ValueError('command limit exceeded')
    warmup.planner.safety.validate_command = fail
    respond(worker, clock); warmup.advance(paused=False, snapshot=snapshot)
    assert warmup.phase == 'error'
    assert 'command limit exceeded' in warmup.error
    assert not worker.resets


def test_multi_inflight_drains_coalesced_previews_before_reset(setup):
    config, clock, worker, _, snapshot, create = setup
    config['inference'].update(inference_schedule='multi_inflight', observation_hz=100)
    warmup = create(); warmup.activate()
    for _ in range(3):
        next_request(warmup, worker, clock, snapshot)
    assert len(warmup._pending) == 3
    respond(worker, clock)  # Worker coalesced uploads; only newest result survives.
    warmup.advance(paused=False, snapshot=snapshot)
    assert not warmup.error
    assert not warmup._pending
    assert worker.resets == ['trial']
    assert warmup.completed_requests == 1
