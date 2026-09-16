"""Collection timing tests with synthetic inputs; no hardware or GUI servers."""

from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pytest
import yaml

from manimux.collection.yam.backend import CollectionBackend, FollowerView, load_backend_config
from manimux.collection.yam.camera.interface import CameraFrame, CameraMode
from manimux.collection.yam.camera.mock_camera import MockCamera
from manimux.collection.yam.camera.worker import CameraWorker
from manimux.collection.yam.config import CameraConfig, StationConfig, build_station_config
from manimux.collection.yam.data.formats.default_format import DefaultFormat
from manimux.collection.yam.data.recorder import EpisodeRecorder
from manimux.collection.yam.data.replay import load_joint_replay
from manimux.collection.yam.data.timing import uniform_training_buffers
from manimux.collection.yam.robot.yam_adapter import YamLeaderPolicy
from manimux.collection.yam.runtime import ArmUnit
from manimux.collection.yam.teleop.loop import ControlLoop


@pytest.mark.parametrize("hz", [30, 100])
def test_one_station_setting_changes_both_target_and_record_clock(tmp_path, hz):
    raw = yaml.safe_load(Path("configs/collection/yam/station.yaml").read_text())
    raw["collection_hz"] = hz
    path = tmp_path / "station.yaml"
    path.write_text(yaml.safe_dump(raw))
    cfg = build_station_config(path)
    config = load_backend_config(cfg, mock=True)
    assert cfg.control_hz == hz
    assert config.robot.control_hz == hz
    assert config.policy.action_dt_s == pytest.approx(1 / hz)
    assert all(c.fps == 30 for c in cfg.cameras)
    # The profile on disk still specifies the existing deployment action period.
    from manimux.config import load_config

    assert load_config(cfg.manimux_config).policy.action_dt_s == pytest.approx(1 / 30)


@pytest.mark.parametrize("hz", [0, -1, float("nan"), float("inf"), True])
def test_invalid_collection_frequency_rejected(hz):
    with pytest.raises(ValueError, match="collection_hz"):
        StationConfig(collection_hz=hz)


@pytest.mark.parametrize("hz", [0.5, 1, 2, 30, 100])
def test_leader_timeout_allows_normal_period_but_still_rejects_stale_target(
    tmp_path,
    monkeypatch,
    hz,
):
    cfg = build_station_config("configs/collection/yam/station.yaml")
    cfg.collection_hz = hz
    backend = CollectionBackend(load_backend_config(cfg, mock=True), mock=True, lock_dir=tmp_path)
    now = [time.monotonic_ns()]
    monkeypatch.setattr(backend.clock, "now_ns", lambda: now[0])
    backend.connect(start_thread=False)
    try:
        timeout = max(0.5, 2 / hz)
        assert backend.leader_timeout_s == pytest.approx(timeout)
        for _ in range(4):
            backend.set_target("left_arm", np.zeros(7))
            now[0] += round(1.1 / hz * 1e9)  # normal period with 10% scheduling jitter
        assert backend._sequence == 4 and backend._fault is None
        now[0] = backend._target_ns + round((timeout + 0.001) * 1e9)
        with pytest.raises(RuntimeError, match="leader target timed out"):
            backend.set_target("left_arm", np.zeros(7))
        assert backend._halted and backend.driver._stopped
    finally:
        backend.close()


def test_slow_to_fast_switch_preserves_only_the_pending_old_interval(tmp_path, monkeypatch):
    cfg = build_station_config("configs/collection/yam/station.yaml")
    cfg.collection_hz = 1
    backend = CollectionBackend(load_backend_config(cfg, mock=True), mock=True, lock_dir=tmp_path)
    now = [time.monotonic_ns()]
    monkeypatch.setattr(backend.clock, "now_ns", lambda: now[0])
    backend.connect(start_thread=False)
    try:
        backend.set_target("left_arm", np.zeros(7))
        now[0] += 750_000_000
        backend.set_control_hz(100)
        assert backend.leader_timeout_s == 2.0
        now[0] += 350_000_000  # already sleeping in an old one-second period
        backend.set_target("left_arm", np.zeros(7))
        assert backend.leader_timeout_s == 0.5
        now[0] += 501_000_000
        with pytest.raises(RuntimeError, match="timed out"):
            backend.set_control_hz(1)
        assert backend.config.robot.control_hz == 100  # stale target was not revived
        with pytest.raises(RuntimeError, match="timed out"):
            backend.set_target("left_arm", np.zeros(7))
        assert backend._halted
    finally:
        backend.close()


def test_each_new_leader_value_is_submitted_without_interpolation(tmp_path):
    cfg = build_station_config("configs/collection/yam/station.yaml")
    cfg.collection_hz = 100
    config = load_backend_config(cfg, mock=True)
    backend = CollectionBackend(config, mock=True, lock_dir=tmp_path)
    backend.connect(start_thread=False)
    try:
        leaders, units = {}, []
        for side in ("left", "right"):
            leader = SimpleNamespace(value=np.zeros(6))
            leader.get_state = lambda leader=leader: (leader.value.copy(), 0.4, [False, False])
            leaders[side] = leader
            follower = FollowerView(backend, f"{side}_arm")
            units.append(ArmUnit(side, follower, YamLeaderPolicy(leader, follower, control_hz=100)))
        loop = ControlLoop(units, [], cfg.control_hz)
        loop.sync_enabled = True
        assert loop.dt == 0.01
        backend.start_trace()
        expected = []
        for index in range(21):
            # Alternating targets catch hidden smoothing, interpolation or 30 Hz gating.
            value = 0.02 if index % 2 else -0.03
            leaders["left"].value[:] = value
            leaders["right"].value[:] = -value
            loop._step()
            expected.append(value)
        trace = backend.finish_trace()
        assert len(trace) == 21
        assert [r["command"]["left_arm"][0] for r in trace] == expected
        assert [r["command"]["right_arm"][0] for r in trace] == [-v for v in expected]
    finally:
        backend.close()


def test_online_hz_changes_keep_devices_and_episode_rates(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from manimux.collection.yam.gui.server import create_app

    raw = yaml.safe_load(Path("configs/collection/yam/station.yaml").read_text())
    raw["collection_hz"] = 30
    raw["save_root"] = str(tmp_path / "episodes")
    raw["cameras"] = [dict(name="top", type="mock", role="top", width=64, height=48, fps=30)]
    station_path = tmp_path / "station.yaml"
    station_path.write_text(yaml.safe_dump(raw))
    cfg = build_station_config(station_path)
    app = create_app(cfg, mock=True, station_path=str(station_path))
    with TestClient(app) as client:
        for bad in (0, -1, True, "100", None):
            assert (
                client.post("/api/collect/timing", json={"collection_hz": bad}).status_code == 422
            )
        assert cfg.collection_hz == 30
        # Selection survives config reloads for Preview and Start Teleop.
        assert client.post("/api/collect/timing", json={"collection_hz": 100}).status_code == 200
        assert client.post("/api/collect/connect").json()["collection_hz"] == 100
        assert client.post("/api/collect/start-teleop").status_code == 200
        session = app.state.session
        loop, recorder = session.loop, session.recorder
        workers, units = list(session.workers), list(session.units)
        backend = units[0].robot.backend
        thread = loop._thread
        deadline = time.monotonic() + 1
        while not backend._enabled and time.monotonic() < deadline:
            time.sleep(0.005)
        assert backend._enabled  # allow normal first-target initialization before guarding resets

        def forbidden(*args, **kwargs):
            raise AssertionError("online Hz switch must not reconnect, reset, or send a hold")

        with monkeypatch.context() as patch:
            for owner, names in (
                (backend.driver, ("connect", "close", "stop")),
                (backend, ("pause",)),
                (backend.executor, ("reset",)),
                (backend.safety, ("reset",)),
                (workers[0], ("start", "stop")),
            ):
                for name in names:
                    patch.setattr(owner, name, forbidden)
            for hz in (30, 100, 30):
                response = client.post("/api/collect/timing", json={"collection_hz": hz})
                assert response.status_code == 200, response.text
                assert response.json()["teleop_running"]
                assert backend.dt == loop.dt == pytest.approx(1 / hz)
                assert backend.config.policy.action_dt_s == pytest.approx(1 / hz)
                assert loop._thread is thread and thread.is_alive()
                assert session.workers == workers and session.units == units
                assert session.recorder is recorder
                assert recorder.station.control_hz == hz
                assert recorder.independent_cameras
                time.sleep(0.05)
                assert loop.last_error is None

        for hz in (30, 100):
            assert client.post("/api/collect/timing", json={"collection_hz": hz}).status_code == 200
            response = client.post(
                "/api/collect/start-recording",
                json={
                    "task_name": "rate_switch",
                    "include_eepose": False,
                },
            )
            assert response.status_code == 200, response.text
            refused = client.post("/api/collect/timing", json={"collection_hz": 60})
            assert refused.status_code == 409
            assert session.cfg.collection_hz == hz and loop.dt == pytest.approx(1 / hz)
            time.sleep(0.12)
            saved = client.post("/api/collect/stop-recording")
            assert saved.status_code == 200, saved.text
            meta = json.loads((Path(saved.json()["path"]) / "metadata.json").read_text())
            assert meta["schema_version"] == 2 and meta["control_hz"] == hz
            assert meta["cameras"][0]["fps"] == 30
            assert meta["extra"]["manimux"]["robot"]["control_hz"] == hz

        recorder.is_saving = True
        try:
            assert client.post("/api/collect/timing", json={"collection_hz": 30}).status_code == 409
        finally:
            recorder.is_saving = False
        assert yaml.safe_load(station_path.read_text())["collection_hz"] == 30
        assert client.post("/api/session/reset").status_code == 200
        assert client.post("/api/collect/start-teleop").json()["collection_hz"] == 100
        # Internal loop faults latch e-stop while session.live remains true;
        # this is the timeout path, distinct from the operator's /api/estop.
        session.loop.estop()
        session.loop.last_error = "RuntimeError: leader target timed out"
        stopped_loop = session.loop
        refused = client.post("/api/collect/start-teleop")
        assert refused.status_code == 409
        assert "leader target timed out" in refused.json()["detail"]
        assert "Reset Session" in refused.json()["detail"]
        assert session.loop is stopped_loop and stopped_loop.estopped


@pytest.mark.parametrize("executor_name", ["direct", "smooth"])
def test_frequency_switch_preserves_executor_and_gripper_history(tmp_path, executor_name):
    from manimux.types import ActionHorizon, RobotState

    cfg = build_station_config("configs/collection/yam/station.yaml")
    cfg.collection_hz = 30
    config = load_backend_config(cfg, mock=True)
    config.execution.executor = executor_name
    # Exercise an explicitly configured ramp; the default profile now closes directly.
    config.execution.motion_limits.gripper.max_closing_velocity = 1.0
    config.execution.smooth.gripper.max_closing_velocity = 1.0
    backend = CollectionBackend(config, mock=True, lock_dir=tmp_path)
    backend.connect(start_thread=False)
    try:
        follower = SimpleNamespace(
            num_dofs=lambda: 7, get_joint_pos=lambda: np.r_[np.zeros(6), 1.0]
        )
        policy = YamLeaderPolicy(
            None, follower, gripper_mode="toggle", control_hz=30, gripper_close_duration_s=1.0
        )
        policy._gripper_command(1.0)
        for _ in range(15):
            policy._gripper_command(0.0)
        assert policy._toggle_gripper_command == pytest.approx(0.5)
        state = RobotState(
            groups={g: np.r_[np.zeros(6), 0.5] for g in config.robot.group_dims},
            monotonic_ns=0,
            sequence=0,
        )
        backend.executor.reset(state)
        backend.safety.reset(state)
        previous = backend.executor._previous
        velocities = backend.executor._previous_velocity
        safety_previous = backend.safety._previous_command
        policy.set_control_hz(100)
        backend.set_control_hz(100)
        assert backend.executor._previous is previous
        assert backend.executor._previous_velocity is velocities
        assert backend.safety._previous_command is safety_previous
        for index in range(50):
            target = policy._gripper_command(0.0)
            assert target == pytest.approx(max(0, 0.5 - (index + 1) / 100))
        if executor_name == "direct":
            reference = ActionHorizon(
                0, 10_000_000, "test", {g: np.zeros((2, 7)) for g in config.robot.group_dims}
            )
            command = backend.executor.step(0, state, reference)
            # Shared closing speed stays 1 normalized unit/s after the change.
            assert command.groups["left_arm"][-1] == pytest.approx(0.49)
        else:
            import math

            rc = 1 / (2 * math.pi * config.execution.smooth.cutoff_hz)
            assert backend.executor._alpha == pytest.approx(0.01 / (rc + 0.01))
        assert backend.safety._control_dt_s == 0.01
    finally:
        backend.close()


def test_online_switch_waits_until_leader_read_and_command_complete(tmp_path, monkeypatch):
    from manimux.collection.yam.gui.session import CollectSession
    from manimux.collection.yam.runtime import build_arm_units

    cfg = build_station_config("configs/collection/yam/station.yaml")
    cfg.collection_hz = 30
    cfg.__post_init__()
    cfg.save_root = str(tmp_path)
    session = CollectSession(cfg, mock=True)
    session.units = build_arm_units(cfg, mock=True)
    session.loop = ControlLoop(session.units, [], 30)
    session.loop.sync_enabled = True
    entered, release = threading.Event(), threading.Event()
    policy = session.units[0].agent
    act = policy.act

    def blocked_act(obs):
        entered.set()
        assert release.wait(2)
        return act(obs)

    monkeypatch.setattr(policy, "act", blocked_act)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            step = pool.submit(session.loop._step)
            assert entered.wait(1)
            change = pool.submit(session.set_collection_hz, 100)
            try:
                assert not change.done()
                assert session.loop.dt == 1 / 30
            finally:
                release.set()
            step.result(timeout=2)
            change.result(timeout=2)
        assert session.loop.dt == 0.01
        assert session.units[0].robot.backend.config.robot.control_hz == 100
    finally:
        release.set()
        session.units[0].robot.backend.close()


@pytest.fixture
def multirate_episode(tmp_path, monkeypatch):
    from manimux.collection.yam.data import codec

    # Tiny synthetic videos must work on machines with or without a GPU.
    monkeypatch.setattr(codec, "encoder", lambda: "libx264")
    monkeypatch.setattr(codec, "hw_decoder", lambda: None)
    cfg = StationConfig(collection_hz=100, save_root=str(tmp_path))
    cfg.cameras = [CameraConfig("physical_top", "mock", "top", width=32, height=24, fps=30)]
    worker = CameraWorker(MockCamera("physical_top", "top", CameraMode.MONO, 32, 24))
    rec = EpisodeRecorder(tmp_path, cfg, [worker], ["left", "right"])
    rec.start("timing")
    origin = time.time_ns() + 1_000_000_000
    # Synthetic events have independent clocks: 201 control ticks and 60 images.
    events = [(origin + i * 10_000_000, "tick", i) for i in range(201)]
    events += [(origin + 2_000_000 + round(i * 1e9 / 30), "image", i) for i in range(60)]
    for timestamp, kind, index in sorted(events):
        if kind == "image":
            rgb = np.full((24, 32, 3), index, dtype=np.uint8)
            frame = CameraFrame({"rgb": rgb}, timestamp / 1e6)
            for listener in tuple(worker._frame_listeners):
                listener(worker.name, frame)
                listener(worker.name, frame)  # duplicate capture notification
            rgb[:] = 255  # a driver reusing its buffer must not corrupt saved frames
        else:
            obs, actions, controller = {}, {}, {}
            for side in ("left", "right"):
                q = np.full(6, index / 1000)
                obs[side] = {
                    "joint_pos": q,
                    "gripper_pos": [0.4],
                    "feedback_timestamp_ns": timestamp - 1_000_000,
                }
                actions[side] = np.r_[q + 0.1, 0.4]
                controller[side] = {
                    "joint_pos": actions[side],
                    "timestamp_ns": timestamp + 1_000_000,
                }
            rec.tick(
                actions,
                obs,
                {},
                controller,
                tick_timestamp_ns=timestamp,
                tick_monotonic_ns=10_000_000_000 + index * 10_000_000,
            )
    path = rec.stop()
    assert not worker._frame_listeners
    return path


def test_freeze_excludes_home_frames_and_defers_writer_until_stop(tmp_path):
    cfg = StationConfig(collection_hz=60, save_root=str(tmp_path))
    cfg.cameras = [CameraConfig("top", "mock", "top", width=32, height=24, fps=30)]
    worker = CameraWorker(MockCamera("top", "top", CameraMode.MONO, 32, 24))
    rec = EpisodeRecorder(tmp_path, cfg, [worker], ["left"])
    written = []
    rec.writer = SimpleNamespace(
        write_episode=lambda path, meta, data: written.append((meta, data))
    )
    path = rec.start("freeze")
    t = time.time_ns() + 1_000_000
    obs = {"left": {"joint_pos": np.zeros(6), "gripper_pos": [0.4],
                    "feedback_timestamp_ns": t}}
    rec.tick({"left": np.r_[np.ones(6), 0.4]}, obs, {}, tick_timestamp_ns=t,
             tick_monotonic_ns=1)
    rec.capture_frame("top", CameraFrame({"rgb": np.zeros((24, 32, 3), dtype=np.uint8)}, t / 1e6))
    native_events = []
    rec._native_recording = SimpleNamespace(
        request_stop=lambda: native_events.append("native-freeze"),
        stop=lambda: native_events.append("native-save") or {"complete": True},
    )
    rec.freeze()
    rec.freeze()  # idempotent before finalization
    assert not rec.is_recording and rec.is_saving
    assert not worker._frame_listeners and not written
    assert native_events == ["native-freeze"]
    with pytest.raises(RuntimeError):
        rec.start("must-not-overwrite-frozen-episode")
    # A Home operation taking time must not enter the frozen episode.
    rec.tick({"left": np.zeros(7)}, obs, {}, tick_timestamp_ns=t + 5_000_000_000)
    rec.capture_frame("top", CameraFrame({"rgb": np.ones((24, 32, 3), dtype=np.uint8)},
                                        (t + 5_000_000_000) / 1e6))
    assert rec.stop() == path
    assert native_events == ["native-freeze", "native-save"]
    assert not rec.is_saving and len(written) == 1
    meta, data = written[0]
    assert meta.num_frames == 1
    assert len(data["top-images-rgb"]) == 1
    np.testing.assert_array_equal(data["action-left-joint"], np.ones((1, 6)))


def test_100hz_records_and_30fps_video_are_independent(multirate_episode):
    p = multirate_episode
    meta = json.loads((p / "metadata.json").read_text())
    assert meta["schema_version"] == 2 and meta["num_frames"] == 201
    assert meta["extra"]["timing"]["commands"]["left"]["actual_hz"] == 100
    for side in ("left", "right"):
        assert np.load(p / f"action-{side}-joint.npy").shape == (201, 6)
        assert np.load(p / f"{side}-joint_pos.npy").shape == (201, 6)
    assert len(np.load(p / "top-timestamp.npy")) == 60
    indices = np.load(p / "top-frame-index.npy")
    assert len(indices) == 201 and indices[0] == -1
    assert indices[1] == indices[2] == indices[3] == 0
    with av.open(str(p / "top-images-rgb.mp4")) as video:
        assert float(video.streams.video[0].average_rate) == 30
        frames = list(video.decode(video=0))
        assert len(frames) == 60
        assert float(frames[0].to_ndarray(format="rgb24").mean()) < 2
        assert float(video.duration) / av.time_base == pytest.approx(2, abs=0.04)


@pytest.mark.parametrize("source", ["feedback", "command"])
def test_replay_handles_v2_without_native_sidecars(multirate_episode, source):
    replay = load_joint_replay(multirate_episode, source)
    assert replay.native
    assert len(replay.cameras["physical_top"].timestamps_ns) == 60
    assert "已录" in replay.reference_label
    assert "100 Hz" in replay.source_note
    assert len(replay.time_s) > 100


def test_lerobot_projection_aligns_by_time_and_rejects_gaps(multirate_episode):
    meta, buffers = DefaultFormat().read_episode(multirate_episode)
    aligned = uniform_training_buffers(meta, buffers)
    n = len(aligned["action-left-joint"])
    assert 190 <= n <= 200
    assert len(aligned["top-images-rgb"]) == n
    assert aligned["top-images-rgb"][0] is aligned["top-images-rgb"][1]
    np.testing.assert_allclose(
        aligned["action-left-joint"][:, 0] - aligned["left-joint_pos"][:, 0], 0.099, atol=1e-12
    )
    broken = dict(buffers)
    broken["top-timestamp"] = np.array([buffers["top-timestamp"][0], buffers["top-timestamp"][-1]])
    broken["top-images-rgb"] = [buffers["top-images-rgb"][0], buffers["top-images-rgb"][-1]]
    with pytest.raises(ValueError, match="gaps"):
        uniform_training_buffers(meta, broken)


def test_camera_copy_does_not_hold_control_recorder_lock(tmp_path, monkeypatch):
    from manimux.collection.yam.data import recorder as module

    worker = CameraWorker(MockCamera("top", "top", CameraMode.MONO, 32, 24))
    rec = EpisodeRecorder(tmp_path, StationConfig(collection_hz=100), [worker], ["left"])
    rec.start("copy")
    entered, release = threading.Event(), threading.Event()
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    original = np.array

    def slow_copy(value, *args, **kwargs):
        if value is rgb:
            entered.set()
            assert release.wait(3)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(module.np, "array", slow_copy)
    thread = threading.Thread(
        target=rec.capture_frame,
        args=("top", CameraFrame({"rgb": rgb}, (time.time_ns() + 1_000_000) / 1e6)),
    )
    thread.start()
    try:
        assert entered.wait(2)
        assert rec._lock.acquire(timeout=0.2)
        rec._lock.release()
        rec.abort()
        rec.start("next")
    finally:
        release.set()
        thread.join(3)
    assert not rec._buf["top-timestamp"]  # old callback cannot enter the new episode
    rec.abort()
    assert not worker._frame_listeners


def test_lerobot_writer_retains_control_rows_with_held_images(multirate_episode):
    from manimux.collection.yam.data.formats.lerobot_format import LeRobotFormat

    meta, buffers = DefaultFormat().read_episode(multirate_episode)
    rows, saved = [], []
    writer = LeRobotFormat()
    # Check our writer boundary without requiring the optional LeRobot package.
    writer._ds = SimpleNamespace(add_frame=rows.append, save_episode=lambda: saved.append(True))
    writer._fps = 100
    writer.add_episode(meta, buffers)
    assert saved == [True] and 190 <= len(rows) <= 200
    assert rows[0]["action"].shape == rows[0]["observation.state"].shape == (14,)
    assert rows[0]["observation.images.top_rgb"] is rows[1]["observation.images.top_rgb"]
    meta.control_hz = 30
    with pytest.raises(ValueError, match="mix collection frequencies"):
        writer._ensure_dataset(meta)


def test_abc_writer_keeps_independent_counts_and_message_times(
    multirate_episode,
    tmp_path,
    monkeypatch,
):
    from manimux.collection.yam.data.formats import abc_format

    meta, buffers = DefaultFormat().read_episode(multirate_episode)
    messages, finished, video_rates = [], [], []
    transport = SimpleNamespace(
        Writer=lambda handle: SimpleNamespace(
            write_message=lambda **kwargs: messages.append(kwargs),
            finish=lambda: finished.append(True),
        )
    )
    monkeypatch.setitem(sys.modules, "mcap_protobuf", SimpleNamespace(writer=transport))
    monkeypatch.setitem(sys.modules, "mcap_protobuf.writer", transport)

    def message(**kwargs):
        return SimpleNamespace(timestamp=SimpleNamespace(), **kwargs)

    monkeypatch.setattr(
        abc_format,
        "_load_message_classes",
        lambda: {
            name: message
            for name in (
                "RobotState",
                "GripperState",
                "Instructions",
                "foxglove.CompressedVideo",
            )
        },
    )

    def packets(frames, fps):
        video_rates.append(fps)
        return [b"test packet"] * len(frames)

    monkeypatch.setattr(abc_format, "_h264_packets", packets)
    writer = abc_format.ABCFormat()
    writer.begin("test", str(tmp_path / "abc"))
    writer.add_episode(meta, buffers)
    assert finished == [True] and video_rates == [30]
    origin = int(buffers["left-feedback-timestamp-ns"][0])
    for topic, time_key, count, scale in (
        ("/left-arm-state", "left-feedback-timestamp-ns", 201, 1),
        ("/right-arm-action", "controller-right-timestamp-ns", 201, 1),
        ("/top-camera", "top-timestamp", 60, 1e6),
    ):
        rows = [row for row in messages if row["topic"] == topic]
        assert len(rows) == count
        for row, timestamp in zip(rows, buffers[time_key], strict=True):
            expected = int(round(timestamp * scale)) - origin
            assert row["log_time"] == row["publish_time"] == expected
            stamp = row["message"].timestamp
            assert stamp.seconds * 1_000_000_000 + stamp.nanos == expected
