import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from manimux.collection.yam.backend import CollectionBackend, load_backend_config
from manimux.collection.yam.config import CameraConfig, build_station_config
from manimux.collection.yam.data.recorder import EpisodeRecorder
from manimux.collection.yam.robot.yam_adapter import YamLeaderPolicy
from manimux.runtime.executors import DirectExecutor, SmoothExecutor
from manimux.runtime.lock import RuntimeInstanceLock, RuntimeLockError


def test_orbbec_missing_dependency_is_not_reported_as_missing_camera(monkeypatch):
    from manimux.collection.yam.camera.orbbec import discover_orbbec

    monkeypatch.setitem(sys.modules, "pyudev", None)
    with pytest.raises(RuntimeError, match="requires pyudev"):
        discover_orbbec()


def station(tmp_path):
    cfg = build_station_config("configs/collection/yam/station.yaml")
    cfg.save_root = str(tmp_path / "episodes")
    cfg.cameras = []
    return cfg


def test_camera_timeout_reports_capture_error():
    from manimux.collection.yam.camera.interface import CameraMode
    from manimux.collection.yam.camera.mock_camera import MockCamera
    from manimux.collection.yam.camera.worker import CameraWorker

    class FailingCamera(MockCamera):
        def read(self):
            raise RuntimeError("device disconnected")

    worker = CameraWorker(FailingCamera("left", "left", CameraMode.MONO, 64, 48))
    with pytest.raises(RuntimeError, match="last capture error: RuntimeError: device disconnected"):
        worker.start(warmup_timeout=0.05)
    assert worker._thread is None


def test_camera_start_failure_closes_all_drivers(tmp_path, monkeypatch):
    from manimux.collection.yam.camera.interface import CameraMode
    from manimux.collection.yam.camera.mock_camera import MockCamera
    from manimux.collection.yam.gui import session as session_module

    closed = []

    class TrackingCamera(MockCamera):
        def stop(self):
            closed.append(self.name)

    drivers = [
        TrackingCamera(name, name, CameraMode.MONO, 64, 48) for name in ("left", "right")
    ]
    monkeypatch.setattr(
        session_module, "build_cameras_from_config", lambda *args, **kwargs: drivers
    )

    def refuse_start(self):
        raise RuntimeError("warmup failed")

    monkeypatch.setattr(session_module.CameraWorker, "start", refuse_start)
    config = station(tmp_path)
    session = session_module.CollectSession(config, mock=True)
    with pytest.raises(RuntimeError, match="warmup failed"):
        session.connect_cameras(config)
    assert closed == ["left", "right"]
    assert session.workers == []
    assert not session.cameras_connected


@pytest.fixture
def backend(tmp_path):
    config = load_backend_config(station(tmp_path), mock=True)
    instance = CollectionBackend(config, mock=True, lock_dir=tmp_path, execution_mode="threaded")
    instance.connect(start_thread=False)
    try:
        yield instance
    finally:
        instance.close()


def test_direct_batch_pause_and_stop(backend):
    assert isinstance(backend.executor, DirectExecutor)
    backend.tick()
    assert backend._last_command is None
    target = np.full(7, 0.2)
    with backend.target_batch():
        backend.set_target("left_arm", target)
        assert backend._sequence == 0
        backend.set_target("right_arm", target)
    assert backend._sequence == 1
    backend.tick()
    for values in backend._last_command.groups.values():
        np.testing.assert_array_equal(values, target)
    backend.pause()
    assert not backend._enabled
    backend.set_target("left_arm", target)
    backend.pause(halt=True)
    with pytest.raises(RuntimeError, match="Reset Session"):
        backend.set_target("left_arm", target)


def test_batch_failure_does_not_publish(backend):
    with pytest.raises(ValueError), backend.target_batch():
        backend.set_target("left_arm", np.ones(7))
        raise ValueError("bad right leader")
    assert not backend._enabled
    with pytest.raises(ValueError, match="gripper"):
        backend.set_target("left_arm", np.full(7, 2.0))
    assert not backend._enabled


def test_stale_leader_latches_stop(backend):
    backend.set_target("left_arm", np.zeros(7))
    backend._target_ns -= int(1e9)
    backend._run()
    assert backend._halted
    assert "timed out" in backend._fault
    with pytest.raises(RuntimeError, match="executor failed"):
        backend.observation("left_arm")


def test_shared_lease_is_acquired_before_driver_connect(tmp_path):
    config = load_backend_config(station(tmp_path), mock=True)
    with RuntimeInstanceLock("yam", mode="test", config_path=Path("test"), lock_dir=tmp_path):
        from manimux.clock import SystemClock
        from manimux.robots.mock import MockDualArmDriver

        driver = MockDualArmDriver(config.robot.group_dims, SystemClock())
        instance = CollectionBackend(config, driver=driver, lock_dir=tmp_path)
        with pytest.raises(RuntimeLockError):
            instance.connect()
        assert not driver._connected


def test_smooth_executor_is_shared(tmp_path):
    config = load_backend_config(station(tmp_path), mock=True)
    config.execution.executor = "smooth"
    instance = CollectionBackend(config, mock=True, lock_dir=tmp_path, execution_mode="threaded")
    instance.connect(start_thread=False)
    try:
        assert isinstance(instance.executor, SmoothExecutor)
        instance.set_target("left_arm", np.full(7, 0.5))
        instance.tick()
        assert instance._last_command.groups["left_arm"][0] < 0.5
    finally:
        instance.close()


def test_recorder_preserves_source_and_executor_trace(backend, tmp_path):
    cfg = station(tmp_path)
    recorder = EpisodeRecorder(cfg.save_root, cfg, [], ["left", "right"], backend=backend)
    recorder.start("bottles")
    target = np.full(7, 0.1)
    backend.set_target("left_arm", target)
    backend.tick()
    actions = {side: backend._target[f"{side}_arm"] for side in ["left", "right"]}
    obs = {side: backend.observation(f"{side}_arm") for side in actions}
    recorder.tick(
        actions,
        obs,
        {},
        controller_inputs={side: backend.controller_input(f"{side}_arm") for side in actions},
    )
    output = recorder.stop()
    assert (output / "write_complete.flag").exists()
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["extra"]["manimux"]["backend"] == "manimux"
    trace = [
        json.loads(line) for line in (output / "manimux-control.jsonl").read_text().splitlines()
    ]
    assert len(trace) == 1
    assert trace[0]["source"] == trace[0]["command"]
    np.testing.assert_allclose(np.load(output / "action-left-joint.npy")[0], target[:6])


@pytest.mark.parametrize("execution_mode", ["synchronous", "threaded"])
def test_gui_uses_mock_backend_and_pause(tmp_path, monkeypatch, execution_mode):
    from fastapi.testclient import TestClient

    from manimux.collection.yam.gui.server import create_app

    cfg = station(tmp_path)
    cfg.execution_mode = execution_mode
    if execution_mode == "threaded":
        cfg.manimux_config = "configs/collection/yam/control-threaded.yaml"
    cfg.cameras = [CameraConfig("test_top", "mock", "top", width=64, height=48)]
    monkeypatch.setenv("YAM_ABC_VIDEO_ENCODER", "libx264")
    app = create_app(cfg, mock=True)
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.post("/api/deploy/start", json={}).status_code == 409
        assert client.post("/api/maintenance/reset-can").status_code == 409
        session = app.state.session
        assert client.post("/api/collect/start-teleop").status_code == 200
        time.sleep(0.15)
        assert session.loop.sync_enabled
        assert session.units[0].robot.backend._sequence > 0
        assert (
            client.post(
                "/api/collect/start-recording",
                json={
                    "task_name": "bottles",
                    "include_eepose": False,
                },
            ).status_code
            == 200
        )
        time.sleep(0.2)
        response = client.post("/api/collect/stop-recording")
        assert response.status_code == 200, response.text
        episode = Path(response.json()["path"])
        assert (episode / "write_complete.flag").exists()
        import av

        with av.open(str(episode / "top-images-rgb.mp4")) as video:
            frames = list(video.decode(video=0))
        assert len(frames) == response.json()["frames"]
        assert (frames[0].width, frames[0].height) == (64, 48)
        assert client.post("/api/collect/stop-teleop").status_code == 200
        sequence = session.units[0].robot.backend._sequence
        time.sleep(0.08)
        assert session.units[0].robot.backend._sequence == sequence
        assert not session.units[0].robot.backend._enabled
        assert client.post("/api/estop").status_code == 200
        assert session.loop.estopped
        assert client.post("/api/session/reset").status_code == 200
        assert not session.units
        assert client.post("/api/collect/start-teleop").status_code == 200
        active_backend = session.units[0].robot.backend
    assert not active_backend._connected
    assert active_backend._thread is None or not active_backend._thread.is_alive()


def test_synchronous_sends_once_per_target_without_thread(tmp_path, monkeypatch):
    config = load_backend_config(station(tmp_path), mock=True)
    instance = CollectionBackend(config, mock=True, lock_dir=tmp_path)
    commands = []
    send = instance.driver.send_command

    def capture(command):
        commands.append(command)
        send(command)

    monkeypatch.setattr(instance.driver, "send_command", capture)
    instance.connect()
    try:
        assert instance._thread is None
        assert instance.dt == pytest.approx(1 / 30)
        assert not commands
        with instance.target_batch():
            instance.set_target("left_arm", np.full(7, 0.2))
            instance.set_target("right_arm", np.full(7, 0.3))
        assert len(commands) == 1
        np.testing.assert_array_equal(commands[0].groups["right_arm"], np.full(7, 0.3))
        time.sleep(0.08)
        assert len(commands) == 1
        instance.set_target("left_arm", np.full(7, 0.4))
        assert len(commands) == 2
        assert instance.metadata()["execution_mode"] == "synchronous"
    finally:
        instance.close()


@pytest.mark.parametrize("frequency", [30.0, 100.0])
def test_threaded_frequency_is_configured(tmp_path, frequency):
    config = load_backend_config(station(tmp_path), mock=True)
    config.robot.control_hz = frequency
    instance = CollectionBackend(config, mock=True, lock_dir=tmp_path, execution_mode="threaded")
    instance.connect()
    try:
        assert instance._thread.is_alive()
        assert instance.dt == pytest.approx(1 / frequency)
        instance.start_trace()
        instance.set_target("left_arm", np.full(7, 0.2))
        time.sleep(0.15)
        assert len(instance.finish_trace()) >= 2
        assert instance._sequence == 1
    finally:
        instance.close()


def test_synchronous_rejects_stale_resume(tmp_path):
    config = load_backend_config(station(tmp_path), mock=True)
    instance = CollectionBackend(config, mock=True, lock_dir=tmp_path)
    instance.connect()
    try:
        instance.set_target("left_arm", np.zeros(7))
        instance._target_ns -= int(1e9)
        with pytest.raises(RuntimeError, match="timed out"):
            instance.set_target("left_arm", np.ones(7))
        assert instance._halted
        assert instance.driver._stopped
    finally:
        instance.close()


def test_synchronous_send_failure_stops_driver(tmp_path, monkeypatch):
    config = load_backend_config(station(tmp_path), mock=True)
    instance = CollectionBackend(config, mock=True, lock_dir=tmp_path)
    instance.connect()
    try:

        def failed_send(command):
            raise RuntimeError("simulated send failure")

        monkeypatch.setattr(instance.driver, "send_command", failed_send)
        with pytest.raises(RuntimeError, match="send failure"):
            instance.set_target("left_arm", np.zeros(7))
        assert instance._halted
        assert instance.driver._stopped
    finally:
        instance.close()


def test_synchronous_requires_matching_frequencies(tmp_path):
    cfg = station(tmp_path)
    cfg.manimux_config = "configs/collection/yam/control-threaded.yaml"
    with pytest.raises(ValueError, match="matching robot and station"):
        load_backend_config(cfg, mock=True)
    cfg.execution_mode = "threaded"
    assert load_backend_config(cfg, mock=True).robot.control_hz == 100
    cfg.execution_mode = "typo"
    with pytest.raises(ValueError, match="execution_mode"):
        load_backend_config(cfg, mock=True)


def test_collection_cli_dispatches_configured_embodiment(tmp_path, monkeypatch):
    from manimux.collection.cli import main
    from manimux.collection.yam import cli as yam_cli

    calls = []
    monkeypatch.setattr(yam_cli, "run_gui", calls.append)
    main(["--config", "configs/collection/yam/station.yaml", "--mock"])
    assert len(calls) == 1 and calls[0].mock
    invalid = tmp_path / "unknown.yaml"
    invalid.write_text("collector: unknown\n")
    with pytest.raises(SystemExit):
        main(["--config", str(invalid)])
    assert len(calls) == 1


def test_leader_policy_keeps_analog_gripper():
    class Leader:
        def get_state(self):
            return np.arange(6) / 10, 0.37, [False, False]

    class Follower:
        def num_dofs(self):
            return 7

        def command_joint_pos(self, command):
            self.command = command

    follower = Follower()
    policy = YamLeaderPolicy(Leader(), follower)
    policy.read_inputs()
    result = policy.act({})
    np.testing.assert_allclose(result, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.37])


def test_native_default_call_is_unchanged(monkeypatch):
    import importlib
    from types import SimpleNamespace

    from i2rt.robots.utils import ArmType, GripperType

    from manimux.robots.yam.arm import YAMRobot

    calls = []

    def fake_factory(**options):
        calls.append(options)
        return SimpleNamespace(get_joint_pos=lambda: np.zeros(7))

    module = importlib.import_module("i2rt.robots.get_robot")
    monkeypatch.setattr(module, "get_yam_robot", fake_factory)
    YAMRobot(channel="test_bus")
    assert calls[-1] == {"channel": "test_bus", "gripper_type": GripperType.LINEAR_4310}
    YAMRobot(channel="test_bus", gripper_force_limit=50.0)
    assert calls[-1] == {"channel": "test_bus", "gripper_type": GripperType.LINEAR_4310}
    count = len(calls)
    with pytest.raises(ValueError, match="fixes gripper force at 50 N"):
        YAMRobot(channel="test_bus", gripper_force_limit=60.0)
    assert len(calls) == count
    YAMRobot(channel="leader_bus", arm_type="yam", gripper_type="yam_teaching_handle")
    assert calls[-1]["arm_type"] == ArmType.from_string_name("yam")
    assert calls[-1]["gripper_type"] == GripperType.from_string_name("yam_teaching_handle")


def test_invalid_controller_binding_rejected_before_connect(tmp_path):
    cfg = station(tmp_path)
    cfg.robot.controllers[1].controls = "yam_left"
    with pytest.raises(ValueError, match="one leader per follower"):
        load_backend_config(cfg, mock=True)


def test_collection_profile_rejects_gui_hardware_override(tmp_path):
    cfg = station(tmp_path)
    cfg.robot.gripper_force_limit = 60
    with pytest.raises(ValueError, match="hardware conflicts with control_profile"):
        load_backend_config(cfg, mock=True)


@pytest.mark.parametrize("executor", ["direct", "smooth"])
def test_shared_command_limits_apply_to_both_executors(tmp_path, executor):
    from manimux.config import CommandSafetyConfig

    config = load_backend_config(station(tmp_path), mock=True)
    config.execution.executor = executor
    config.execution.command_safety = CommandSafetyConfig(
        position_lower={group: [-10.0] * 7 for group in config.robot.group_dims},
        position_upper={group: [10.0] * 7 for group in config.robot.group_dims},
        max_velocity={group: [0.001] * 7 for group in config.robot.group_dims},
        max_acceleration={group: [0.001] * 7 for group in config.robot.group_dims},
    )
    instance = CollectionBackend(config, mock=True, lock_dir=tmp_path)
    instance.connect()
    try:
        with pytest.raises(ValueError, match="velocity"):
            instance.set_target("left_arm", np.ones(7))
        assert instance._fault is not None
    finally:
        instance.close()


def test_collection_profile_applies_shared_station_defaults(tmp_path):
    import yaml

    profile = yaml.safe_load(Path("configs/robots/yam/common.yaml").read_text())
    for side in ("left", "right"):
        profile["robot"]["options"][f"{side}_hardware_options"]["ee_mass"] = 0.8
    (tmp_path / "shared.yaml").write_text(yaml.safe_dump(profile))
    control = yaml.safe_load(Path("configs/collection/yam/control.yaml").read_text())
    control["control_profile"] = "shared.yaml"
    (tmp_path / "control.yaml").write_text(yaml.safe_dump(control))
    raw = yaml.safe_load(Path("configs/collection/yam/station.yaml").read_text())
    raw["manimux_config"] = str(tmp_path / "control.yaml")
    path = tmp_path / "station.yaml"
    path.write_text(yaml.safe_dump(raw))
    cfg = build_station_config(path)
    assert cfg.robot.ee_mass == 0.8
    assert cfg.robot.gripper_force_limit == 50
    assert cfg.robot.gripper_close_duration_s == 1.0
    assert cfg.control_hz == 30
    hardware = load_backend_config(cfg, mock=True).robot.options["left_hardware_options"]
    assert hardware["ee_mass"] == 0.8
    raw["robot"]["ee_mass"] = 1.0
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="ee_mass conflicts with control_profile"):
        build_station_config(path)


def test_toggle_gripper_matches_original_curve_without_double_slowdown(tmp_path):
    from types import SimpleNamespace

    from manimux.types import ActionHorizon, RobotState

    cfg = station(tmp_path)
    runtime = load_backend_config(cfg, mock=True)
    follower = SimpleNamespace(
        num_dofs=lambda: 7, get_joint_pos=lambda: np.array([0.0] * 6 + [1.0]),
    )
    policy = YamLeaderPolicy(
        None, follower, gripper_mode="toggle", control_hz=cfg.control_hz,
        gripper_close_duration_s=cfg.robot.gripper_close_duration_s,
    )
    executor = DirectExecutor(runtime.execution.motion_limits, control_dt_s=1 / cfg.control_hz)
    state = RobotState(
        groups={group: follower.get_joint_pos() for group in runtime.robot.group_dims},
        monotonic_ns=0, sequence=0,
    )
    assert policy._gripper_command(1.0) == 1.0
    for index in range(30):
        target = policy._gripper_command(0.0)
        assert target == pytest.approx(max(0.0, 1 - (index + 1) / 30))
        reference = ActionHorizon(
            start_time_ns=0, dt_ns=33_333_333, plan_id="teleop",
            groups={group: np.tile([0.0] * 6 + [target], (2, 1)) for group in state.groups},
        )
        command = executor.step(index * 33_333_333, state, reference)
        for values in command.groups.values():
            assert values[6] == pytest.approx(target)
    policy._gripper_command(1.0)
    assert policy._gripper_command(0.0) == 1.0
