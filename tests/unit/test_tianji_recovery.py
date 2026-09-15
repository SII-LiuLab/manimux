from __future__ import annotations

import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from manimux.config import load_config
from manimux.robots.tianji import recovery, teleop_drag
from manimux.session import RuntimeSessionService
from manimux.viewer.dashboard import PolicyViewer


def _config():
    config = load_config("configs/mock.yaml")
    config.viewer.enabled = True
    config.viewer.robot_adapter = "tianji"
    config.robot.driver = "tianji_dual"
    config.robot.options = {"execute": True, "robot_ip": "test-controller"}
    return config


@pytest.mark.parametrize("arms", ["A", "B", "AB"])
@pytest.mark.parametrize("failed_arm", [None, "B"])
def test_drag_uses_paired_umi_params_tracks_and_cleans_partial_startup(
    monkeypatch,
    arms,
    failed_arm,
):
    monkeypatch.setattr(teleop_drag.time, "sleep", lambda _: None)
    stop = threading.Event()
    calls = []
    modes = {"A": 0, "B": 0}
    drag_types = {"A": 0, "B": 0}

    class Controller:
        def clear_set(self):
            pass

        def send_cmd(self):
            pass

        def set_state(self, *, arm, state):
            modes[arm] = 0 if arm == failed_arm else state
            calls.append(("state", arm, state))

        def set_drag_space(self, *, arm, dgType):
            drag_types[arm] = dgType
            calls.append(("drag", arm, dgType))

        def set_impedance_type(self, **kwargs):
            calls.append(("impedance", kwargs))

        def set_tool(self, **kwargs):
            calls.append(("tool", kwargs))

        def set_joint_kd_params(self, **kwargs):
            calls.append(("kd", kwargs))

    class Driver:
        def __init__(self, arm):
            self.arm = arm
            self.idx = "AB".index(arm)
            self.engaged = False

        def state(self):
            return {"cur": modes[self.arm], "err": 0, "frame": 1, "q": [1.0] * 7}

        def joints(self):
            return [0.0] * 7

        def disable(self):
            assert self.engaged
            calls.append(("disable", self.arm))
            modes[self.arm] = 0

    def send(_conn, commands):
        calls.append(("commands", commands.copy()))
        stop.set()

    def ensure_clear(_conn, driver, arm):
        calls.append(("checked_clear", arm))
        return True, driver.state()

    api = SimpleNamespace(
        STATE_TORQUE=3, STATE_DISABLED=0, STATE_ERROR=100, send_joint_commands=send,
        ensure_clear=ensure_clear,
    )
    conn = SimpleNamespace(
        robot=Controller(),
        close=lambda: calls.append(("close",)),
        subscribe=lambda: {"inputs": [{"drag_sp_type": drag_types[a]} for a in "AB"]},
    )
    drivers = {arm: Driver(arm) for arm in arms}
    tools = {
        arm: SimpleNamespace(
            kine_params=lambda: [0.0] * 6,
            dynamic_params=lambda a=arm: [0.759 if a == "A" else 0.739],
        )
        for arm in arms
    }
    if failed_arm is not None and failed_arm in arms:
        with pytest.raises(RuntimeError, match="torque mode"):
            teleop_drag._drag_loop(api, conn, drivers, tools, stop)
        assert not any(call[0] == "commands" for call in calls)
    else:
        teleop_drag._drag_loop(api, conn, drivers, tools, stop)
        commands = next(call[1] for call in calls if call[0] == "commands")
        assert set(commands) == set(arms)
        for command in commands.values():
            assert command == pytest.approx([0.06] * 7)
    for arm in arms:
        assert ("drag", arm, 0) in calls
        assert ("disable", arm) in calls
        assert ("kd", {"arm": arm, "K": [1.0] * 7, "D": [0.3] * 7}) in calls
        tool = next(c[1] for c in calls if c[0] == "tool" and c[1]["arm"] == arm)
        assert tool["dynamicParams"] == [0.759 if arm == "A" else 0.739]
    assert calls[-1] == ("close",)
    assert calls[:len(arms)] == [("checked_clear", arm) for arm in arms]
    assert sum(c[0] == "disable" for c in calls) == len(arms)


@pytest.mark.parametrize("arm", ["A", "B", "AB"])
def test_managed_process_passes_umi_and_exits_on_parent_pipe_close(tmp_path, monkeypatch, arm):
    for relative in (".venv/bin/python3", "drivers/arm_driver.py", "configs/tool/umi.yaml"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    actual_popen = subprocess.Popen
    commands = []

    def popen(argv, **kwargs):
        commands.append(argv)
        # Exercise real pipe/process lifetime without importing any hardware SDK.
        return actual_popen(
            [
                sys.executable,
                "-u",
                "-c",
                'import sys; print(\'MANIMUX_DRAG {"state": "active"}\'); sys.stdin.read()',
            ],
            **kwargs,
        )

    monkeypatch.setattr(teleop_drag.subprocess, "Popen", popen)
    drag = teleop_drag.TeleopDragProcess(tmp_path, "test-controller")
    drag.start(arm)
    try:
        assert drag._ready.wait(3)
        drag.poll()
        assert drag.state == "active" and drag.busy
        with pytest.raises(RuntimeError, match="Exit"):
            drag.start(arm)
        assert commands[0][-6:] == ["--arm", arm, "--tool", "umi", "--robot-ip", "test-controller"]
    finally:
        drag.close()
    assert not drag.busy and drag.state == "idle"


class FakeDrag:
    def __init__(self, *_args):
        self.busy = False
        self.state = "idle"
        self.arm = ""
        self.error = ""
        self.starts = []
        self.stops = 0

    def start(self, arm):
        self.starts.append(arm)
        self.arm, self.state, self.busy = arm, "active", True

    def stop(self):
        self.stops += 1
        self.state = "stopping"

    def poll(self):
        if self.state == "stopping":
            self.state, self.arm, self.busy = "idle", "", False

    def close(self):
        self.busy = False


def test_recovery_deduplicates_rejects_overlaps_and_stops_on_disconnect(monkeypatch):
    monkeypatch.setattr(recovery, "TeleopDragProcess", FakeDrag)
    controller = recovery.TianjiRecovery(_config())
    request = {"recovery_request_id": "1", "recovery_request": "drag:AB", "recovery_lease": True}
    controller.update(request)
    controller.update(request)
    assert controller._drag.starts == ["AB"]
    controller.update({**request, "recovery_request_id": "2", "recovery_request": "home"})
    assert "Exit drag" in controller.error
    assert controller._home_thread is None
    controller.update({})
    assert controller._drag.stops == 1
    assert controller.busy
    controller.update({})
    assert not controller.busy


def test_recovery_home_uses_existing_driver_and_waits_for_close(monkeypatch):
    monkeypatch.setattr(recovery, "TeleopDragProcess", FakeDrag)
    calls = []
    robot = SimpleNamespace(
        connect=lambda **kwargs: calls.append(("connect", kwargs)),
        **{name: lambda n=name: calls.append(n) for name in ("home", "stop", "close")}
    )
    monkeypatch.setattr(recovery, "build_robot", lambda *_: robot)
    controller = recovery.TianjiRecovery(_config())
    controller.update({"recovery_request_id": "1", "recovery_request": "home"})
    assert controller.busy
    controller._home_thread.join(2)
    controller.update({})
    assert not controller.busy
    assert calls == [("connect", {"recover_errors": True}), "home", "stop", "close"]


def test_recovery_reports_uncleared_error_without_homing_and_releases_driver(monkeypatch):
    monkeypatch.setattr(recovery, "TeleopDragProcess", FakeDrag)
    calls = []

    def connect(*, recover_errors):
        assert recover_errors
        raise RuntimeError("arm B state 100 err_code 13 (emergency stop)")

    robot = SimpleNamespace(
        connect=connect,
        **{name: lambda n=name: calls.append(n) for name in ("home", "stop", "close")},
    )
    monkeypatch.setattr(recovery, "build_robot", lambda *_: robot)
    controller = recovery.TianjiRecovery(_config())
    controller.update({"recovery_request_id": "1", "recovery_request": "home"})
    controller._home_thread.join(2)
    controller.update({})
    assert not controller.busy
    assert "err_code 13" in controller.metadata()["error"]
    assert calls == ["stop", "close"]


@pytest.mark.parametrize(
    "options, error",
    [
        ({"execute": False}, "executing Tianji"),
        ({"execute": True, "active_arms": ["left_arm"]}, "active_arms"),
    ],
)
def test_recovery_respects_runtime_hardware_scope(monkeypatch, options, error):
    monkeypatch.setattr(recovery, "TeleopDragProcess", FakeDrag)
    config = _config()
    config.robot.options = options
    controller = recovery.TianjiRecovery(config)
    controller.update(
        {"recovery_request_id": "1", "recovery_request": "drag:AB", "recovery_lease": True}
    )
    assert error in controller.error
    assert controller._drag.starts == []


def test_session_never_hands_off_to_rollout_while_drag_owns_connection(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "TeleopDragProcess", FakeDrag)
    base = {
        "recovery_service_id": str(tmp_path),
        "recovery_lease": True,
        "recovery_request_id": "1",
        "recovery_request": "drag:A",
    }
    states = iter(
        [
            base,
            {**base, "new_rollout_requested": True},
            {
                **base,
                "recovery_request_id": "2",
                "recovery_request": "stop",
                "recovery_lease": False,
                "new_rollout_requested": True,
            },
            {"new_rollout_requested": True, "task_command": "after cleanup"},
        ]
    )
    messages = []
    service = RuntimeSessionService(
        _config(),
        tmp_path,
        control_factory=lambda: SimpleNamespace(poll=lambda: next(states), close=lambda: None),
        publisher_factory=lambda: SimpleNamespace(publish=messages.append, close=lambda: None),
        poll_interval_s=0,
        announcement_interval_s=0,
    )
    result = service._wait_for_rollout_request()
    assert result["task_command"] == "after cleanup"
    assert service._recovery._drag.starts == ["A"]
    assert any(m.metadata["recovery"]["state"] == "active" for m in messages)


def _viewer():
    viewer = PolicyViewer.__new__(PolicyViewer)
    viewer.lock = threading.RLock()
    viewer._clear_recovery()
    viewer.service_ready = True
    viewer.service_id = "test-service"
    viewer.robot = SimpleNamespace(name="tianji")
    viewer.launch_mode = "serve"
    viewer.observe_only = False
    viewer.rollout_started = False
    viewer.episode_active = False
    viewer.preparing_rollout = False
    viewer.evaluation_saved = True
    viewer.recovery_available = True
    viewer.paused = True
    viewer.home_requested = viewer.finish_requested = viewer.new_rollout_requested = False
    viewer.experiment_mode = False
    for name in (
        "drag_btn",
        "drag_arm",
        "stop_drag_btn",
        "home_btn",
        "prepare_normal_btn",
        "prepare_experiment_btn",
        "task",
        "layout_id",
        "finish_btn",
        "finish_no_home_btn",
        "start_btn",
        "pause_btn",
    ):
        setattr(viewer, name, SimpleNamespace(disabled=False, value=""))
    viewer.recovery_status = SimpleNamespace(content="")
    for name in ("status", "rollout_setup_status", "evaluation_status"):
        setattr(viewer, name, SimpleNamespace(content=""))
    for name in ("policy_name", "runtime_name", "episode_path", "executor_info"):
        setattr(viewer, name, SimpleNamespace(value=""))
    for name in ("new_rollout_folder", "policy_control_folder", "recovery_folder",
                 "evaluation_folder", "overlay_folder", "run_folder"):
        setattr(viewer, name, SimpleNamespace(visible=False))
    viewer.camera_view = SimpleNamespace(
        set_policy_map=lambda *_args, **_kwargs: None, clear_images=lambda: None,
    )
    viewer.current_episode_dir = None
    viewer.episode_finalized = False
    viewer._set_evaluation_enabled = lambda _: None
    viewer._set_instruction = lambda _: None
    return viewer


def test_viewer_waits_for_ack_keeps_stop_available_and_locks_out_prepare():
    viewer = _viewer()
    viewer._request_recovery("drag:AB")
    request = viewer.control_state()
    assert request["recovery_lease"]
    assert request["recovery_request"] == "drag:AB"
    assert viewer.control_state()["recovery_request"] == "drag:AB"
    assert viewer.prepare_normal_btn.disabled and viewer.home_btn.disabled
    assert not viewer.stop_drag_btn.disabled
    viewer._prepare_rollout(experiment_mode=False)
    assert not viewer.new_rollout_requested
    # An old heartbeat cannot release controls while a request is unacknowledged.
    viewer._update_recovery({"recovery": {"available": True, "busy": False, "ack": "old"}})
    assert viewer.home_btn.disabled
    viewer._update_recovery(
        {
            "recovery": {
                "available": True,
                "busy": True,
                "state": "active",
                "arm": "AB",
                "ack": request["recovery_request_id"],
            }
        }
    )
    assert not viewer.recovery_request and viewer.recovery_lease
    viewer._request_recovery("stop")
    assert not viewer.recovery_lease
    viewer._update_recovery(
        {
            "recovery": {
                "available": True,
                "busy": False,
                "state": "idle",
                "ack": viewer.recovery_request_id,
            }
        }
    )
    viewer._update_prepare_enabled()
    assert not viewer.home_btn.disabled and not viewer.prepare_normal_btn.disabled


@pytest.mark.parametrize("stage", ["waiting", "setup", "preparing", "control",
                                  "evaluation", "complete"])
def test_recovery_panel_is_visible_at_every_stage(stage):
    viewer = _viewer()
    viewer._set_stage(stage)
    assert viewer.recovery_folder.visible


@pytest.mark.parametrize("paused", [False, True])
@pytest.mark.parametrize("arm", ["A", "B", "AB"])
def test_drag_during_rollout_requests_finish_without_home_and_queues_drag(paused, arm):
    viewer = _viewer()
    viewer.episode_active = True
    viewer.service_ready = False
    viewer.paused = paused
    viewer._update_recovery_controls()
    assert not viewer.drag_btn.disabled
    assert viewer.drag_btn.label == "Stop rollout & drag"
    viewer._request_recovery(f"drag:{arm}")
    request = viewer.control_state()
    assert request["finish_requested"]
    assert request["finish_home"] is False
    assert request["recovery_request"] == f"drag:{arm}"
    assert viewer.control_state()["recovery_request"] == f"drag:{arm}"
    assert viewer.drag_btn.disabled and viewer.home_btn.disabled
    assert not viewer.stop_drag_btn.disabled
    # Cancellation must work while the service is still releasing the robot.
    viewer._request_recovery("stop")
    assert not viewer.recovery_lease
    assert viewer.control_state()["recovery_request"] == "stop"


@pytest.mark.parametrize("mode", ["absent", "observe", "one-shot", "preparing"])
def test_viewer_rejects_drag_without_an_eligible_runtime(mode):
    viewer = _viewer()
    viewer.service_ready = False
    viewer.episode_active = mode in {"observe", "one-shot"}
    viewer.observe_only = mode == "observe"
    viewer.launch_mode = "run" if mode == "one-shot" else "serve"
    viewer.preparing_rollout = mode == "preparing"
    viewer._request_recovery("drag:A")
    assert not viewer.recovery_request


@pytest.mark.parametrize("event", ["episode_failed", "runtime_service_ready"])
@pytest.mark.parametrize("active", [True, False])
def test_estop_failure_unlocks_recovery_even_when_failure_event_was_lost(event, active):
    viewer = _viewer()
    viewer.episode_active = active
    viewer.preparing_rollout = not active
    viewer.service_ready = False
    viewer.evaluation_saved = False
    viewer.paused = False
    metadata = {
        "run_dir": "test-service", "last_failure_id": "failure-1",
        "last_error": "RuntimeError: left_arm state 1 err_code 13 (emergency stop)",
        "recovery": {"available": True, "busy": False, "state": "idle"},
    }
    viewer._update_event({"event": event, "metadata": metadata})
    assert viewer.service_ready and not viewer.episode_active
    assert not viewer.preparing_rollout and viewer.paused
    assert viewer.evaluation_saved and viewer.recovery_folder.visible
    assert not viewer.drag_btn.disabled and not viewer.home_btn.disabled
    assert "physical E-stop" in viewer.recovery_status.content
    assert "before execution" not in viewer.status.content
    viewer._request_recovery("drag:AB")
    viewer._update_event({"event": "runtime_service_ready", "metadata": {
        **metadata,
        "recovery": {"available": True, "busy": True, "state": "active", "arm": "AB",
                     "ack": viewer.recovery_request_id},
    }})
    assert viewer.recovery_lease
    assert "Drag AB" in viewer.recovery_status.content
    assert "Manual recovery" in viewer.status.content


def test_repeated_failure_heartbeat_does_not_cancel_a_new_preparation():
    viewer = _viewer()
    metadata = {"run_dir": "test-service", "last_failure_id": "failure-1",
                "last_error": "emergency stop", "recovery": {"available": True}}
    viewer._update_event({"event": "runtime_service_ready", "metadata": metadata})
    viewer._prepare_rollout(experiment_mode=False)
    viewer._update_event({"event": "runtime_service_ready", "metadata": metadata})
    assert viewer.preparing_rollout and not viewer.service_ready
    # A fresh failure with the identical text must still unblock recovery.
    viewer._update_event({"event": "runtime_service_ready", "metadata": {
        **metadata, "last_failure_id": "failure-2",
    }})
    assert not viewer.preparing_rollout and not viewer.drag_btn.disabled


@pytest.mark.parametrize("clear_ok, remaining_error", [(False, 13), (True, 13), (False, 0)])
def test_uncleared_estop_never_enables_either_arm(clear_ok, remaining_error):
    calls = []
    def ensure_clear(conn, driver, arm):
        calls.append(("check", arm))
        return ((True, {"cur": 0, "err": 0}) if arm == "A" else
                (clear_ok, {"cur": 0, "err": remaining_error}))
    api = SimpleNamespace(ensure_clear=ensure_clear, STATE_ERROR=100)
    conn = SimpleNamespace(close=lambda: calls.append(("close",)))
    with pytest.raises(RuntimeError, match="physical E-stop"):
        teleop_drag._drag_loop(api, conn, {"A": object(), "B": object()}, {}, threading.Event())
    assert calls == [("check", "A"), ("check", "B"), ("close",)]


def test_drag_cancellation_during_error_check_does_not_enable_arm():
    stop = threading.Event()
    def ensure_clear(*_args):
        stop.set()
        return True, {"cur": 0, "err": 0}
    closed = []
    api = SimpleNamespace(ensure_clear=ensure_clear, STATE_ERROR=100)
    conn = SimpleNamespace(close=lambda: closed.append(True))
    teleop_drag._drag_loop(api, conn, {"A": object(), "B": object()}, {}, stop)
    assert closed == [True]
