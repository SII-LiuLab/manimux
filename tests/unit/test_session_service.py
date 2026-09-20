from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from manimux.cli import _create_run_dir, _handle_termination, build_parser, load_config
from manimux.runtime import RunResult
from manimux.runtime.edge import _next_rollout_id
from manimux.session import RuntimeSessionService, _TianjiRecovery
from manimux.types import RobotState


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


class _FakeMarvin:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []
        self.faults = [13, 0]
        self.modes = [100, 0]

    def connect(self, ip: str) -> bool:
        self.calls.append(("connect", ip))
        return True

    def clear_error(self, arm: str) -> bool:
        self.calls.append(("clear_error", arm))
        index = 0 if arm == "A" else 1
        self.faults[index] = 0
        self.modes[index] = 0
        return False

    def subscribe(self, _buffer) -> dict:
        self.calls.append(("subscribe", None))
        return {
            "states": [
                {"err_code": self.faults[index], "cur_state": self.modes[index]}
                for index in range(2)
            ]
        }

    def release_robot(self) -> bool:
        self.calls.append(("release_robot", None))
        return True


class _FakeTime:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, duration: float) -> None:
        self.value += duration


class _FakeHomeRobot:
    def __init__(self) -> None:
        self.model = SimpleNamespace(
            home_joints={
                "left_arm": np.full(7, 0.02),
                "right_arm": np.full(7, -0.02),
            }
        )
        self.groups = {
            "left_arm": np.r_[np.zeros(7), 0.3],
            "right_arm": np.r_[np.zeros(7), 0.4],
        }
        self.calls: list[str] = []
        self.commands = []

    def connect(self) -> None:
        self.calls.append("connect")

    def get_state(self) -> RobotState:
        return RobotState(
            {name: values.copy() for name, values in self.groups.items()},
            0,
            len(self.commands),
        )

    def send_command(self, command) -> None:
        self.commands.append(command)
        self.groups = {name: values.copy() for name, values in command.groups.items()}

    def close(self) -> None:
        self.calls.append("close")


class _FakeDragMarvin:
    def __init__(self, *, failed_arm: str | None = None) -> None:
        self.calls: list[tuple] = []
        self.modes = [0, 0]
        self.faults = [0, 0]
        self.drag_types = [0, 0]
        self.frames = [0, 0]
        self.joints = [[0.0] * 7, [0.0] * 7]
        self.failed_arm = failed_arm

    def connect(self, ip: str) -> bool:
        self.calls.append(("connect", ip))
        return True

    def subscribe(self, _buffer) -> dict:
        self.frames = [frame + 1 for frame in self.frames]
        return {
            "states": [
                {"err_code": self.faults[index], "cur_state": self.modes[index]}
                for index in range(2)
            ],
            "outputs": [
                {"frame_serial": self.frames[index], "fb_joint_pos": self.joints[index]}
                for index in range(2)
            ],
            "inputs": [
                {"drag_sp_type": self.drag_types[index]} for index in range(2)
            ],
        }

    def clear_error(self, arm: str) -> bool:
        self.calls.append(("clear_error", arm))
        index = 0 if arm == "A" else 1
        self.faults[index] = self.modes[index] = 0
        return True

    def clear_set(self) -> bool:
        self.calls.append(("clear_set",))
        return True

    def set_state(self, *, arm: str, state: int) -> bool:
        self.calls.append(("state", arm, state))
        if not (state == 3 and arm == self.failed_arm):
            self.modes[0 if arm == "A" else 1] = state
        return True

    def set_impedance_type(self, **kwargs) -> bool:
        self.calls.append(("impedance", kwargs))
        return True

    def set_tool(self, **kwargs) -> bool:
        self.calls.append(("tool", kwargs))
        return True

    def set_joint_kd_params(self, **kwargs) -> bool:
        self.calls.append(("kd", kwargs))
        return True

    def set_drag_space(self, *, arm: str, dgType: int) -> bool:
        self.calls.append(("drag", arm, dgType))
        self.drag_types[0 if arm == "A" else 1] = dgType
        return True

    def set_joint_cmd_pose(self, *, arm: str, joints: list[float]) -> bool:
        self.calls.append(("joints", arm, list(joints)))
        return True

    def send_cmd(self) -> bool:
        self.calls.append(("send",))
        return True

    def release_robot(self) -> bool:
        self.calls.append(("release",))
        return True


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


def test_tianji_clear_error_recovery_acknowledges_selected_controller() -> None:
    sdk = _FakeMarvin()
    config = {
        "robot": {
            "type": "tianji_taccap",
            "options": {"execute": True, "hardware": {"ip": "192.168.1.190"}},
        }
    }
    clock = _FakeTime()
    recovery = _TianjiRecovery(
        config,
        sdk_factory=lambda: (sdk, object()),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    recovery.update(
        {
            "recovery_request": "clear_error",
            "recovery_request_id": "request-1",
        }
    )

    assert sdk.calls == [
        ("connect", "192.168.1.190"),
        ("subscribe", None),
        ("clear_error", "A"),
        ("subscribe", None),
        ("release_robot", None),
    ]
    assert recovery.metadata() == {
        "available": True,
        "actions": ["clear_error", "home", "drag"],
        "busy": False,
        "state": "cleared",
        "ack": "request-1",
        "error": "",
        "arm": "AB",
    }


def test_tianji_recovery_moves_to_configured_home_without_moving_grippers() -> None:
    sdk = _FakeMarvin()
    sdk.faults = [0, 0]
    sdk.modes = [0, 0]
    robot = _FakeHomeRobot()
    clock = _FakeTime()
    config = {
        "robot": {
            "type": "tianji_taccap",
            "control_hz": 10.0,
            "options": {"execute": True, "hardware": {"ip": "192.168.1.190"}},
        }
    }
    recovery = _TianjiRecovery(
        config,
        sdk_factory=lambda: (sdk, object()),
        robot_factory=lambda _config, _clock: robot,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    recovery.update({"recovery_request": "home", "recovery_request_id": "home-1"})
    while recovery.metadata()["busy"]:
        time.sleep(0.001)
    recovery.close()

    assert robot.calls == ["connect", "close"]
    assert robot.commands
    np.testing.assert_allclose(robot.groups["left_arm"][:7], 0.02)
    np.testing.assert_allclose(robot.groups["right_arm"][:7], -0.02)
    assert robot.groups["left_arm"][7] == 0.3
    assert robot.groups["right_arm"][7] == 0.4
    assert recovery.metadata()["error"] == ""
    assert recovery.metadata()["ack"] == "home-1"


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


def test_tianji_drag_runs_teleop_sdk_sequence_and_stops_on_request() -> None:
    sdk = _FakeDragMarvin()
    config = {
        "robot": {
            "type": "tianji_taccap",
            "options": {"execute": True, "hardware": {"ip": "192.168.1.190"}},
        }
    }
    recovery = _TianjiRecovery(
        config,
        sdk_factory=lambda: (sdk, object()),
        sleep=lambda duration: time.sleep(min(duration, 0.001)),
    )

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
    for arm, expected_mass in (("A", 0.7592901345868115), ("B", 0.7388190421557731)):
        assert ("state", arm, 3) in sdk.calls
        assert ("drag", arm, 1) in sdk.calls
        assert ("kd", {"arm": arm, "K": [1.0] * 7, "D": [0.3] * 7}) in sdk.calls
        tool = next(call[1] for call in sdk.calls if call[0] == "tool" and call[1]["arm"] == arm)
        assert tool["dynamicParams"][0] == expected_mass
        assert tool["kineParams"] == [-36.745, 0.0, 169.45, 0.0, -90.0, 180.0]

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
    for arm in "AB":
        assert ("drag", arm, 0) in sdk.calls
        assert ("state", arm, 0) in sdk.calls
    assert sdk.calls[-1] == ("release",)


def test_tianji_drag_lost_viewer_lease_exits_and_disables() -> None:
    sdk = _FakeDragMarvin()
    config = {
        "robot": {
            "type": "tianji_taccap",
            "options": {"execute": True, "hardware": {"ip": "192.168.1.190"}},
        }
    }
    recovery = _TianjiRecovery(
        config,
        sdk_factory=lambda: (sdk, object()),
        sleep=lambda duration: time.sleep(min(duration, 0.001)),
    )
    recovery.update(
        {
            "recovery_request": "drag:A",
            "recovery_request_id": "drag-1",
            "recovery_lease": True,
        }
    )
    _wait_for_recovery(recovery, "active")

    recovery.update({})

    _wait_for_recovery(recovery, "idle")
    assert ("drag", "A", 0) in sdk.calls
    assert ("state", "A", 0) in sdk.calls
    assert sdk.calls[-1] == ("release",)


def test_tianji_drag_partial_start_failure_still_disables_and_releases() -> None:
    sdk = _FakeDragMarvin(failed_arm="B")
    config = {
        "robot": {
            "type": "tianji_taccap",
            "options": {"execute": True, "hardware": {"ip": "192.168.1.190"}},
        }
    }
    recovery = _TianjiRecovery(
        config,
        sdk_factory=lambda: (sdk, object()),
        sleep=lambda duration: time.sleep(min(duration, 0.001)),
    )
    recovery.update(
        {
            "recovery_request": "drag:AB",
            "recovery_request_id": "drag-1",
            "recovery_lease": True,
        }
    )

    failed = _wait_for_recovery(recovery, "error")

    assert "Arm B failed to enter torque mode" in failed["error"]
    for arm in "AB":
        assert ("drag", arm, 0) in sdk.calls
        assert ("state", arm, 0) in sdk.calls
    assert sdk.calls[-1] == ("release",)
