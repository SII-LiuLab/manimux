"""Offline lifecycle checks through EdgeRuntime with fake devices and real recording."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import manimux.runtime.edge as edge
from manimux.cli import prepare_experiment
from manimux.policies.capabilities import PolicyCapabilities
from manimux.recording.episode import EpisodeRecorder
from manimux.runtime.inference import INFERENCE_ALGORITHMS
from manimux.types import RobotState


@pytest.fixture
def runtime_case(tmp_path, monkeypatch):
    events = []

    class Clock:
        value = 1_000_000_000

        def now_ns(self):
            return self.value

        def sleep_until_ns(self, value):
            self.value = max(self.value, value)

    clock = Clock()
    config = prepare_experiment(
        run={"task_prompt": "shutdown test", "max_control_steps": 1},
        robot={"type": "fake", "group_dims": {"arm": 2}, "options": {}},
        policy={"worker": "fake", "adapter": {"type": "fake"},
                "action_dt_s": 0.1, "horizon_policy_steps": 5},
        inference={"algorithm": "serial", "action_start_mode": "first_step_when_ready",
                   "blend_policy_steps": 0},
        executor={"type": "direct"}, robogui={"enabled": False},
    )

    def record(name):
        return lambda *args, **kwargs: events.append(name)

    robot = Mock(kinematics=None, recovery_actions=())
    robot.command_bounds.return_value = {}
    robot.runtime_metadata.return_value = {}
    robot.sent_command_snapshots.return_value = {}
    robot.get_state.side_effect = lambda: RobotState({"arm": np.zeros(2)}, clock.now_ns(), 0)
    for operation in ("stop", "home", "close"):
        getattr(robot, operation).side_effect = record("robot." + operation)
    def close_robot():
        try:
            robot.stop()
        finally:
            events.append("robot.close")
    robot.close.side_effect = close_robot
    worker = Mock(is_alive=False, capabilities=PolicyCapabilities())
    worker.poll.return_value = None
    worker.request_stop.side_effect = record("worker.request_stop")
    worker.close.side_effect = record("worker.close")
    gui = Mock()
    gui.poll_control.return_value = SimpleNamespace(
        paused=False, finish_requested=False, home_requested=False,
    )
    gui.publish_event.side_effect = lambda name, **kwargs: events.append("gui." + name)
    gui.close.side_effect = record("gui.close")
    sensors = Mock()
    sensors.read.return_value = {}
    sensors.close.side_effect = record("sensors.close")
    monkeypatch.setattr(edge.EdgeRuntime, "_build_robot", lambda self: robot)
    monkeypatch.setattr(edge.EdgeRuntime, "_build_adapter", lambda self: SimpleNamespace())
    monkeypatch.setattr(edge, "PolicyWorkerClient", lambda *args, **kwargs: worker)
    monkeypatch.setattr(edge, "RoboGUIBridge", lambda *args, **kwargs: gui)
    monkeypatch.setattr(edge, "SensorReader", lambda *args, **kwargs: sensors)
    recorders = []

    def recorder_factory(*args, **kwargs):
        recorder = EpisodeRecorder(*args, **kwargs)
        close_video = recorder._video.close
        write_zarr = recorder._write_zarr

        def video_close():
            events.append("video.close")
            return close_video()

        def zarr_write():
            events.append("zarr.write")
            write_zarr()

        recorder._video.close = video_close
        recorder._write_zarr = zarr_write
        recorders.append(recorder)
        return recorder

    monkeypatch.setattr(edge, "EpisodeRecorder", recorder_factory)
    runtime = edge.EdgeRuntime(config, tmp_path, clock=clock)
    return SimpleNamespace(
        runtime=runtime, robot=robot, worker=worker, gui=gui, sensors=sensors,
        events=events, recorders=recorders, root=tmp_path,
    )


@pytest.mark.parametrize("finish_button", [False, True])
@pytest.mark.parametrize("home", [False, True])
def test_control_cleanup_precedes_recording_and_completion(runtime_case, finish_button, home):
    case = runtime_case
    case.runtime._config["robot"]["options"]["home_on_close"] = home
    if finish_button:
        case.gui.poll_control.return_value = SimpleNamespace(
            paused=False, finish_requested=True, finish_home=home,
        )
    result = case.runtime.run()
    for operation in ("robot.stop", "worker.request_stop", "robot.close",
                      "worker.close", "sensors.close"):
        assert case.events.index(operation) < case.events.index("video.close")
    assert case.events.index("zarr.write") < case.events.index("gui.episode_finished")
    case.robot.stop.assert_called_once()
    case.robot.close.assert_called_once()
    assert case.robot.home.call_count == 1
    if home:
        assert case.events.index("robot.home") < case.events.index("robot.stop")
        assert case.events.index("robot.home") < case.events.index("robot.close")
    saved = json.loads((result.episode_dir / "result.json").read_text())
    assert saved["run_completed"] is True
    expected_reason = "robogui_finish_requested" if finish_button else "completed"
    assert saved["terminal_reason"] == expected_reason


@pytest.mark.parametrize("owner,operation", [
    ("robot", "stop"), ("robot", "home"), ("robot", "close"),
    ("worker", "request_stop"), ("worker", "close"), ("sensors", "close"),
])
def test_cleanup_failure_is_saved_before_reporting_success(runtime_case, owner, operation):
    case = runtime_case
    case.runtime._config["robot"]["options"]["home_on_close"] = True
    message = f"{owner}.{operation} failed"
    getattr(getattr(case, owner), operation).side_effect = RuntimeError(message)
    with pytest.raises(BaseExceptionGroup, match="runtime cleanup failed"):
        case.runtime.run()
    saved = json.loads((case.root / "rollout-001.partial/result.json").read_text())
    assert saved["run_completed"] is False and saved["incomplete"] is True
    assert message in saved["detail"]
    assert not (case.root / "rollout-001").exists()
    assert "gui.episode_finished" not in case.events
    if operation != "close":
        case.robot.stop.assert_called_once()
    case.robot.close.assert_called_once()
    case.worker.close.assert_called_once()
    case.sensors.close.assert_called_once()
    case.gui.close.assert_called_once()
    case.robot.home.assert_called_once()


@pytest.mark.parametrize("interrupted", [False, True])
def test_fault_and_interrupt_preserve_home_policy_and_partial_result(runtime_case, interrupted):
    case = runtime_case
    case.runtime._config["robot"]["options"]["home_on_close"] = True
    error = KeyboardInterrupt() if interrupted else ValueError("control failed")
    case.gui.poll_control.side_effect = error
    with pytest.raises(type(error)):
        case.runtime.run()
    assert case.robot.home.call_count == 1
    saved = json.loads((case.root / "rollout-001.partial/result.json").read_text())
    assert saved["run_completed"] is False
    assert saved["terminal_reason"] == type(error).__name__
    assert case.events.index("robot.close") < case.events.index("video.close")


def test_runtime_and_cleanup_errors_both_survive_in_saved_evidence(runtime_case):
    case = runtime_case
    case.gui.poll_control.side_effect = ValueError("control failed")
    case.robot.close.side_effect = RuntimeError("close failed")
    with pytest.raises(BaseExceptionGroup, match="runtime cleanup failed"):
        case.runtime.run()
    saved = json.loads((case.root / "rollout-001.partial/result.json").read_text())
    assert "control failed" in saved["detail"] and "close failed" in saved["detail"]


def test_late_gui_cleanup_failure_corrects_saved_result_without_reencoding(runtime_case):
    case = runtime_case
    case.gui.close.side_effect = RuntimeError("GUI close failed")
    with pytest.raises(BaseExceptionGroup, match="runtime cleanup failed"):
        case.runtime.run()
    saved = json.loads((case.root / "rollout-001/result.json").read_text())
    assert saved["run_completed"] is False and saved["incomplete"] is True
    assert saved["terminal_reason"] == "runtime_cleanup_failed"
    assert "GUI close failed" in saved["detail"]
    assert case.events.count("video.close") == case.events.count("zarr.write") == 1
    assert "episode_aborted" in (case.root / "rollout-001/events.jsonl").read_text()


def test_recording_write_failure_still_leaves_failed_result(runtime_case, monkeypatch):
    case = runtime_case
    monkeypatch.setattr(EpisodeRecorder, "_write_zarr", Mock(side_effect=OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        case.runtime.run()
    saved = json.loads((case.root / "rollout-001.partial/result.json").read_text())
    assert saved["run_completed"] is False and saved["incomplete"] is True
    assert "disk full" in saved["detail"]
    case.robot.stop.assert_called_once()
    case.robot.close.assert_called_once()


def test_home_latency_does_not_change_recorded_rollout_duration(runtime_case, monkeypatch):
    case = runtime_case
    elapsed = [10.0]
    monkeypatch.setattr(edge.time, "perf_counter", lambda: elapsed[0])
    case.runtime._config["robot"]["options"]["home_on_close"] = True

    def control():
        elapsed[0] += 2.0
        return SimpleNamespace(paused=False, finish_requested=False, home_requested=False)

    def home():
        elapsed[0] += 100.0

    case.gui.poll_control.side_effect = control
    case.robot.home.side_effect = home
    result = case.runtime.run()
    saved = json.loads((result.episode_dir / "result.json").read_text())
    assert elapsed[0] == 112.0
    assert saved["wall_time_s"] == 2.0


@pytest.mark.parametrize('trigger', ['finish_false', 'step_limit', 'camera', 'control', 'interrupt', 'worker_stop', 'robot_stop'])
def test_required_home_cannot_be_disabled_by_config_ui_or_software_failure(runtime_case, trigger):
    c = runtime_case
    c.robot.home_before_close = True
    c.runtime._config['robot']['options']['home_on_close'] = False
    if trigger == 'finish_false':
        c.gui.poll_control.return_value = SimpleNamespace(paused=False, finish_requested=True, finish_home=False)
    elif trigger == 'camera':
        c.sensors.read.side_effect = [{}, RuntimeError('camera lost')]
    elif trigger == 'control':
        c.gui.poll_control.side_effect = ValueError('GUI software error')
    elif trigger == 'interrupt':
        c.gui.poll_control.side_effect = KeyboardInterrupt()
    elif trigger == 'worker_stop':
        c.worker.request_stop.side_effect = RuntimeError('policy shutdown error')
    elif trigger == 'robot_stop':
        c.robot.stop.side_effect = ValueError('stop software error')
    if trigger in {'finish_false', 'step_limit'}:
        result = c.runtime.run()
        if trigger == 'finish_false':
            events = [json.loads(line) for line in (result.episode_dir / 'events.jsonl').read_text().splitlines()]
            assert next(e for e in events if e['kind'] == 'robogui_finish_requested')['home'] is True
    else:
        with pytest.raises((Exception, KeyboardInterrupt)):
            c.runtime.run()
    c.robot.home.assert_called_once()
    assert c.events.index('robot.home') < c.events.index('robot.close')


def test_shutdown_reporter_start_failure_cannot_prevent_home(runtime_case, monkeypatch):
    import manimux.runtime.lifecycle as lifecycle

    c = runtime_case
    c.robot.home_before_close = True
    c.runtime._config['robogui']['enabled'] = True
    c.gui.poll_control.return_value = SimpleNamespace(
        paused=False, finish_requested=True, finish_home=None,
    )
    reporter = Mock()
    reporter.start.side_effect = RuntimeError('cannot start reporter thread')
    monkeypatch.setattr(lifecycle.threading, 'Thread', lambda **kwargs: reporter)
    c.runtime.run()
    c.robot.home.assert_called_once()
    assert c.events.index('robot.home') < c.events.index('robot.close')


@pytest.mark.parametrize('algorithm', sorted(INFERENCE_ALGORITHMS))
def test_registered_strategies_share_fault_home_contract(runtime_case, algorithm):
    c = runtime_case
    c.runtime._config['inference']['algorithm'] = algorithm
    c.runtime._strategy = edge.build_inference_strategy(c.runtime._config)
    modes = c.runtime._strategy.required_sampling_modes
    c.worker.capabilities = PolicyCapabilities(sampling_modes=frozenset({'default', *modes}))
    c.robot.home_before_close = True
    c.gui.poll_control.side_effect = RuntimeError('injected operator channel failure')
    with pytest.raises(RuntimeError, match='injected operator channel failure'):
        c.runtime.run()
    c.robot.home.assert_called_once()
    assert c.events.index('robot.home') < c.events.index('robot.stop') < c.events.index('robot.close')
