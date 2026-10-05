from __future__ import annotations

import json
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from manimux.cli import _create_run_dir, _handle_termination, build_parser, load_config
from manimux.runtime import RunResult
from manimux.runtime.edge import _next_rollout_id
from manimux.session import RuntimeSessionService, _TianjiRecovery


class _FakeControl:
    def __init__(self, states: list[dict[str, Any]]) -> None:
        self._states = iter(states)
        self.closed = False

    def poll(self) -> dict[str, Any]:
        return next(self._states, {"new_rollout_requested": True})

    def close(self) -> None:
        self.closed = True


class _FakePublisher:
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._messages = messages
        self.closed = False

    def publish(self, message: Any) -> None:
        self._messages.append(message.to_wire())

    def close(self) -> None:
        self.closed = True


class _FakeRuntime:
    def __init__(self, result: RunResult) -> None:
        self._result = result
        self.run_count = 0

    def run(self) -> RunResult:
        self.run_count += 1
        return self._result


class _FailingRuntime:
    def run(self) -> RunResult:
        raise RuntimeError("model unavailable")


class _FakeRecoveryRobot:
    """Records the robot-level recovery calls; hardware behaviour is tested with the robot."""

    def __init__(self, *, fail: str = "") -> None:
        self.calls: list[tuple] = []
        self.fail = fail
        self.drag_started = threading.Event()

    def clear_errors(self) -> None:
        self.calls.append(("clear_errors",))
        if self.fail == "clear_errors":
            raise RuntimeError("E-stop still engaged")

    def connect(self) -> None:
        self.calls.append(("connect",))

    def home(self) -> None:
        self.calls.append(("home",))
        if self.fail == "home":
            raise RuntimeError("home timed out")

    def drag(self, sides, stop, on_active) -> None:
        self.calls.append(("drag", tuple(sides)))
        if self.fail == "drag":
            raise RuntimeError("right: failed to enter torque mode")
        on_active()
        self.drag_started.set()
        stop.wait(2.0)
        self.calls.append(("drag_exit",))

    def close(self) -> None:
        self.calls.append(("close",))


_TIANJI_CONFIG = {
    "robot": {
        "type": "tianji_taccap",
        "options": {"execute": True, "hardware": {"ip": "192.168.1.190"}},
    }
}


def _recovery(robot: _FakeRecoveryRobot, built: list[dict] | None = None) -> _TianjiRecovery:
    def factory(robot_config, _clock):
        if built is not None:
            built.append(robot_config)
        return robot

    return _TianjiRecovery(deepcopy(_TIANJI_CONFIG), robot_factory=factory)


def test_session_service_waits_for_viewer_then_runs_one_isolated_episode(tmp_path: Path) -> None:
    config = load_config("tests/fixtures/runtime.yaml")
    config["viewer"]["enabled"] = True
    config["policy"]["adapter"]["camera_map"] = {"cam_head": "gemini305", "cam_left": "left_camera"}
    run_dir = tmp_path / "run-session"
    run_dir.mkdir()
    episode_dir = run_dir / "episode-one"
    episode_dir.mkdir()
    runtime = _FakeRuntime(
        RunResult(
            episode_dir=episode_dir,
            steps=10,
            accepted_plans=2,
            rejected_plans=0,
            success=True,
            terminal_reason="viewer_finish_requested",
        )
    )
    controls: list[_FakeControl] = []
    messages: list[dict[str, Any]] = []

    def control_factory() -> _FakeControl:
        control = _FakeControl([{}, {"new_rollout_requested": True}])
        controls.append(control)
        return control

    service = RuntimeSessionService(
        config,
        run_dir,
        runtime_factory=lambda _config, _run_dir: runtime,
        control_factory=control_factory,
        publisher_factory=lambda: _FakePublisher(messages),
        poll_interval_s=0,
        announcement_interval_s=0,
    )

    service.serve(max_rollout_attempts=1)

    assert runtime.run_count == 1
    assert controls[0].closed
    ready = next(message for message in messages if message["event"] == "runtime_service_ready")
    assert ready["metadata"]["camera_map"] == config["policy"]["adapter"]["camera_map"]


def test_cli_keeps_run_and_adds_serve() -> None:
    parser = build_parser()
    assert parser.parse_args(["run", "--config", "tests/fixtures/runtime.yaml"]).command == "run"
    assert parser.parse_args(["serve", "--config", "tests/fixtures/runtime.yaml"]).command == "serve"


def test_sigterm_uses_keyboard_interrupt_cleanup_path() -> None:
    try:
        _handle_termination(15, None)
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("SIGTERM handler did not request orderly shutdown")


def test_rollout_ids_are_readable_and_include_partial_attempts(tmp_path: Path) -> None:
    assert _next_rollout_id(tmp_path / "missing") == "rollout-001"
    assert _next_rollout_id(tmp_path) == "rollout-001"
    (tmp_path / "rollout-001").mkdir()
    (tmp_path / "rollout-002.partial").mkdir()
    assert _next_rollout_id(tmp_path) == "rollout-003"


def test_session_manifest_records_config_identity(tmp_path: Path) -> None:
    config_path = Path("tests/fixtures/runtime.yaml")
    config = load_config(config_path)
    config["run"]["output_dir"] = tmp_path

    run_dir = _create_run_dir(config, config_path, mode="serve")
    manifest = json.loads((run_dir / "session-manifest.json").read_text(encoding="utf-8"))

    assert run_dir.name.startswith("session-")
    assert manifest["session_id"] == run_dir.name
    assert manifest["config_path"] == str(config_path.resolve())
    assert len(manifest["config_sha256"]) == 64


def test_session_service_builds_a_fresh_runtime_for_every_episode(tmp_path: Path) -> None:
    config = load_config("tests/fixtures/runtime.yaml")
    config["viewer"]["enabled"] = True
    run_dir = tmp_path / "run-session"
    run_dir.mkdir()
    runtimes: list[_FakeRuntime] = []

    def runtime_factory(_config, _run_dir) -> _FakeRuntime:
        index = len(runtimes)
        episode_dir = run_dir / f"episode-{index}"
        episode_dir.mkdir()
        runtime = _FakeRuntime(RunResult(episode_dir, 1, 1, 0, True, "viewer_finish_requested"))
        runtimes.append(runtime)
        return runtime

    service = RuntimeSessionService(
        config,
        run_dir,
        runtime_factory=runtime_factory,
        control_factory=lambda: _FakeControl([{"new_rollout_requested": True}]),
        publisher_factory=lambda: _FakePublisher([]),
        poll_interval_s=0,
    )

    service.serve(max_rollout_attempts=2)

    assert len(runtimes) == 2
    assert runtimes[0] is not runtimes[1]
    assert [runtime.run_count for runtime in runtimes] == [1, 1]


def test_viewer_request_selects_task_and_experiment_metadata(tmp_path: Path) -> None:
    config = load_config("tests/fixtures/runtime.yaml")
    config["viewer"]["enabled"] = True
    run_dir = tmp_path / "session"
    run_dir.mkdir()
    captured = []

    def runtime_factory(rollout_config, _run_dir):
        captured.append(rollout_config)
        episode_dir = run_dir / "rollout-001"
        episode_dir.mkdir()
        return _FakeRuntime(RunResult(episode_dir, 1, 1, 0, True, "completed"))

    service = RuntimeSessionService(
        config,
        run_dir,
        runtime_factory=runtime_factory,
        control_factory=lambda: _FakeControl(
            [
                {
                    "new_rollout_requested": True,
                    "task_command": "fold the towel",
                    "experiment_mode": True,
                    "layout_id": "layout-03",
                }
            ]
        ),
        publisher_factory=lambda: _FakePublisher([]),
        poll_interval_s=0,
    )

    service.serve(max_rollout_attempts=1)

    assert captured[0]["run"]["task"] == "fold the towel"
    assert captured[0]["run"]["experiment_mode"] is True
    assert captured[0]["run"]["layout_id"] == "layout-03"
    assert config["run"]["task"] != "fold the towel"


def test_session_service_survives_one_failed_rollout_attempt(tmp_path: Path) -> None:
    config = load_config("tests/fixtures/runtime.yaml")
    config["viewer"]["enabled"] = True
    run_dir = tmp_path / "run-session"
    run_dir.mkdir()
    messages: list[dict[str, Any]] = []
    factory_calls = 0

    def runtime_factory(_config, _run_dir):
        nonlocal factory_calls
        factory_calls += 1
        if factory_calls == 1:
            return _FailingRuntime()
        episode_dir = run_dir / "episode-success"
        episode_dir.mkdir()
        return _FakeRuntime(RunResult(episode_dir, 1, 1, 0, True, "completed"))

    service = RuntimeSessionService(
        config,
        run_dir,
        runtime_factory=runtime_factory,
        control_factory=lambda: _FakeControl([{"new_rollout_requested": True}]),
        publisher_factory=lambda: _FakePublisher(messages),
        poll_interval_s=0,
    )

    service.serve(max_rollout_attempts=2)

    assert factory_calls == 2
    assert any(message["event"] == "episode_failed" for message in messages)


def test_repeated_failures_have_distinct_ids_republished_in_idle_heartbeats(tmp_path: Path) -> None:
    config = load_config("tests/fixtures/runtime.yaml")
    config["viewer"]["enabled"] = True
    messages: list[dict[str, Any]] = []
    attempts = 0

    def runtime_factory(_config, _run_dir):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return _FailingRuntime()
        return _FakeRuntime(RunResult(tmp_path, 1, 1, 0, True, "completed"))

    service = RuntimeSessionService(
        config,
        tmp_path,
        runtime_factory=runtime_factory,
        control_factory=lambda: _FakeControl([{"new_rollout_requested": True}]),
        publisher_factory=lambda: _FakePublisher(messages),
        poll_interval_s=0,
    )
    service.serve(max_rollout_attempts=3)

    failures = [message["metadata"] for message in messages if message["event"] == "episode_failed"]
    heartbeats = [
        message["metadata"] for message in messages if message["event"] == "runtime_service_ready"
    ]
    assert len(failures) == 2
    assert failures[0]["last_error"] == failures[1]["last_error"]
    assert failures[0]["last_failure_id"] != failures[1]["last_failure_id"]
    for failure, heartbeat in zip(failures, heartbeats[1:], strict=True):
        assert heartbeat["last_error"] == failure["last_error"]
        assert heartbeat["last_failure_id"] == failure["last_failure_id"]


def test_tianji_clear_error_recovery_only_clears_faults() -> None:
    robot = _FakeRecoveryRobot()
    built: list[dict] = []
    recovery = _recovery(robot, built)

    recovery.update({"recovery_request": "clear_error", "recovery_request_id": "request-1"})

    assert robot.calls == [("clear_errors",)]
    assert built[0]["options"]["end_effector_control"] is False
    assert recovery.metadata() == {
        "available": True,
        "actions": ["clear_error", "home", "drag"],
        "busy": False,
        "state": "cleared",
        "ack": "request-1",
        "error": "",
        "arm": "AB",
    }


def test_tianji_clear_error_failure_is_reported() -> None:
    recovery = _recovery(_FakeRecoveryRobot(fail="clear_errors"))

    recovery.update({"recovery_request": "clear_error", "recovery_request_id": "request-1"})

    metadata = recovery.metadata()
    assert metadata["state"] == "error"
    assert metadata["error"] == "RuntimeError: E-stop still engaged"


def test_tianji_recovery_home_clears_connects_homes_and_closes() -> None:
    robot = _FakeRecoveryRobot()
    built: list[dict] = []
    recovery = _recovery(robot, built)

    recovery.update({"recovery_request": "home", "recovery_request_id": "home-1"})
    _wait_for_recovery(recovery, "idle")
    recovery.close()

    assert robot.calls == [("clear_errors",), ("connect",), ("home",), ("close",)]
    assert built[0]["options"]["end_effector_control"] is True
    assert "end_effector_control" not in _TIANJI_CONFIG["robot"]["options"]
    assert recovery.metadata()["error"] == ""
    assert recovery.metadata()["ack"] == "home-1"


def test_tianji_recovery_home_failure_still_closes_robot() -> None:
    robot = _FakeRecoveryRobot(fail="home")
    recovery = _recovery(robot)

    recovery.update({"recovery_request": "home", "recovery_request_id": "home-1"})
    failed = _wait_for_recovery(recovery, "error")

    assert failed["error"] == "RuntimeError: home timed out"
    assert robot.calls[-1] == ("close",)


def test_executing_tianji_session_advertises_recovery_actions(tmp_path: Path) -> None:
    config = load_config("tests/fixtures/runtime.yaml")
    config["viewer"]["enabled"] = True
    config["robot"] = {
        "type": "tianji_taccap",
        "options": {"execute": True, "hardware": {"ip": "192.168.1.190"}},
    }

    metadata = RuntimeSessionService(config, tmp_path)._ready_metadata()

    assert metadata["recovery"]["available"] is True
    assert metadata["recovery"]["actions"] == ["clear_error", "home", "drag"]


def _wait_for_recovery(recovery: _TianjiRecovery, state: str, timeout_s: float = 2.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        metadata = recovery.metadata()
        if metadata["state"] == state or (state == "idle" and not metadata["busy"]):
            return metadata
        time.sleep(0.001)
    raise AssertionError(f"recovery did not reach {state}: {recovery.metadata()}")


def test_tianji_drag_maps_viewer_arms_and_stops_on_request() -> None:
    robot = _FakeRecoveryRobot()
    built: list[dict] = []
    recovery = _recovery(robot, built)

    recovery.update(
        {
            "recovery_request": "drag:AB",
            "recovery_request_id": "drag-1",
            "recovery_lease": True,
        }
    )
    active = _wait_for_recovery(recovery, "active")
    assert active["busy"] is True
    assert active["arm"] == "AB"
    assert robot.calls == [("clear_errors",), ("drag", ("left", "right"))]
    assert built[0]["options"]["end_effector_control"] is False

    recovery.update(
        {
            "recovery_request": "stop",
            "recovery_request_id": "stop-1",
            "recovery_lease": False,
        }
    )
    stopped = _wait_for_recovery(recovery, "idle")
    assert stopped["error"] == ""
    assert stopped["ack"] == "stop-1"
    assert stopped["arm"] == ""
    assert robot.calls[-1] == ("drag_exit",)


def test_tianji_drag_lost_viewer_lease_exits() -> None:
    robot = _FakeRecoveryRobot()
    recovery = _recovery(robot)
    recovery.update(
        {
            "recovery_request": "drag:A",
            "recovery_request_id": "drag-1",
            "recovery_lease": True,
        }
    )
    _wait_for_recovery(recovery, "active")
    assert ("drag", ("left",)) in robot.calls

    recovery.update({})

    _wait_for_recovery(recovery, "idle")
    assert robot.calls[-1] == ("drag_exit",)


def test_tianji_drag_failure_is_reported_and_rejects_parallel_requests() -> None:
    robot = _FakeRecoveryRobot(fail="drag")
    recovery = _recovery(robot)
    recovery.update(
        {
            "recovery_request": "drag:B",
            "recovery_request_id": "drag-1",
            "recovery_lease": True,
        }
    )

    failed = _wait_for_recovery(recovery, "error")

    assert "right: failed to enter torque mode" in failed["error"]
    assert failed["arm"] == "B"
    recovery.update({"recovery_request": "drag:C", "recovery_request_id": "drag-2"})
    assert recovery.metadata()["error"] == "Unsupported drag arm: C"
