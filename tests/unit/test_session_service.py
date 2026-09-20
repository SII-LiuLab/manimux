from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from manimux.cli import _create_run_dir, _handle_termination, build_parser, load_config
from manimux.runtime import RunResult
from manimux.runtime.edge import _next_rollout_id
from manimux.session import RuntimeSessionService


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
