"""Saved-data replay tests. No camera, robot, or listening Viser server is started."""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import av
import numpy as np
import pytest
import viser

from manimux.collection.yam.data.command_resampling import load_command_rate_replay
from manimux.collection.yam.data.replay import ReplayCamera, load_joint_replay
from manimux.viewer import dashboard
from manimux.viewer.replay import CollectionReplayViewer, ReplayVideo
from manimux.viewer.robots.yam import YamAdapter

ORIGIN = 1_789_446_235_000_000_000


def _write_video(path, count):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv420p"
        for index in range(count):
            rgb = np.full((24, 32, 3), index * 3, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture
def episode(tmp_path):
    meta = {
        "schema_version": 1, "arm_names": ["left"], "num_arm_joints": 6,
        "control_hz": 30, "num_frames": 61, "cameras": [{"name": "top"}],
    }
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    (tmp_path / "write_complete.flag").touch()
    native = tmp_path / "native_joints"
    native.mkdir()
    info = {
        "joint_names": [f"joint{i}" for i in range(1, 7)],
        "file": "follower_left.jsonl", "samples": [401] * 6,
    }
    native_meta = {"schema_version": 1, "complete": True, "start_timestamp_ns": ORIGIN,
                   "streams": {"follower_left": info}}
    (native / "metadata.json").write_text(json.dumps(native_meta))
    with (native / info["file"]).open("w") as handle:
        for index in range(401):
            for joint in range(6):
                handle.write(json.dumps({
                    "joint_index": joint, "timestamp_ns": ORIGIN + index * 5_000_000,
                    "position_rad": index / 200 + joint / 10, "motor_error": "0x1",
                }) + "\n")
    ts = ORIGIN + np.rint(np.arange(61) / 30 * 1e9).astype(np.int64)
    np.save(tmp_path / "left-feedback-timestamp-ns.npy", ts)
    np.save(tmp_path / "controller-left-timestamp-ns.npy", ts)
    np.save(tmp_path / "left-gripper_pos.npy", np.linspace(0, 1, 61)[:, None])
    np.save(tmp_path / "action-left-gripper.npy", np.linspace(1, 0, 61)[:, None])
    q = np.arange(61)[:, None] / 30 + np.arange(6)[None, :] / 10
    np.save(tmp_path / "action-left-joint.npy", q)
    # First image deliberately arrives after the first robot sample.
    camera_ts = ts / 1e6 + 150
    camera_ts[6] = camera_ts[5]
    np.save(tmp_path / "top-timestamp.npy", camera_ts)
    _write_video(tmp_path / "top-images-rgb.mp4", 61)
    return tmp_path


def test_same_low_rate_input_hold_and_linear(episode):
    replay = load_joint_replay(episode)
    t = replay.time_s
    np.testing.assert_allclose(np.diff(t), 0.01)
    low_t = np.arange(61) / 30
    # Independently calculate the latest native packet for each 30 Hz tick.
    native_t = np.arange(401) / 200
    low_q = native_t[np.searchsorted(native_t, low_t, side="right") - 1]
    expected_hold = low_q[np.searchsorted(low_t, t, side="right") - 1]
    expected_linear = np.interp(t, low_t, low_q)
    for joint in range(6):
        # The direct reference must bypass the lower-rate samples entirely.
        expected_native = native_t[np.searchsorted(native_t, t, side="right") - 1]
        np.testing.assert_allclose(replay.native["left"][:, joint], expected_native + joint / 10)
        np.testing.assert_allclose(replay.held["left"][:, joint], expected_hold + joint / 10)
        np.testing.assert_allclose(replay.linear["left"][:, joint], expected_linear + joint / 10)
    np.testing.assert_array_equal(replay.held["left"][:, 6], replay.linear["left"][:, 6])
    np.testing.assert_array_equal(replay.native["left"][:, 6], replay.linear["left"][:, 6])
    assert not np.allclose(replay.native["left"][:, 0], replay.held["left"][:, 0])
    assert not np.allclose(replay.native["left"][:, 0], replay.linear["left"][:, 0])
    np.testing.assert_array_equal(replay.timestamps_ns, ORIGIN + np.arange(201) * 10_000_000)


def test_command_uses_recorded_times_instead_of_nominal_fps(episode):
    ts = np.load(episode / "controller-left-timestamp-ns.npy")
    ts[1:-1] += 4_000_000
    np.save(episode / "controller-left-timestamp-ns.npy", ts)
    replay = load_joint_replay(episode, "command")
    assert not replay.native  # No original high-rate command was recorded.
    q = np.load(episode / "action-left-joint.npy")
    frame = 7  # 70 ms, before the shifted third command at 70.667 ms.
    assert replay.held["left"][frame, 0] == q[1, 0]
    expected = np.interp(replay.time_s, (ts - ORIGIN) / 1e9, q[:, 0])
    np.testing.assert_allclose(replay.linear["left"][:, 0], expected)


def test_feedback_endpoints_are_shared_samples_without_changing_trajectory(episode):
    ts = np.load(episode / "left-feedback-timestamp-ns.npy")
    ts[0] += 13_000_000
    ts[-1] -= 7_000_000
    np.save(episode / "left-feedback-timestamp-ns.npy", ts)
    replay = load_joint_replay(episode)
    # Gripper support cuts across both grids: [0.013, 1.993]. Use the shared
    # sample knots [0.1, 1.9], retaining every intermediate 100 Hz tick.
    np.testing.assert_array_equal(replay.time_s, np.arange(10, 191) / 100)
    for method in (replay.held, replay.linear):
        np.testing.assert_array_equal(method["left"][[0, -1]], replay.native["left"][[0, -1]])
    native_t = np.arange(401) / 200
    low_t = np.arange(61) / 30
    low_q = native_t[np.searchsorted(native_t, low_t, side="right") - 1]
    np.testing.assert_allclose(replay.linear["left"][:, 0], np.interp(replay.time_s, low_t, low_q))


def test_camera_uses_host_timestamp_not_ordinal_over_fps(episode):
    replay = load_joint_replay(episode)
    camera = replay.cameras["top"]
    assert camera.frame_at(ORIGIN) == -1
    assert camera.frame_at(int(camera.timestamps_ns[0])) == 0
    assert camera.frame_at(int(camera.timestamps_ns[5])) == 6
    assert camera.frame_at(int(camera.timestamps_ns[5]) - 1) == 4
    assert camera.frame_at(int(camera.timestamps_ns[-1]) + 100_000_000) == 60


def test_missing_native_feedback_stays_a_gap(episode):
    path = episode / "native_joints/follower_left.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    # Mark actual packets invalid; do not collapse or compress the time axis.
    for row in rows:
        if 0.9 < (row["timestamp_ns"] - ORIGIN) / 1e9 < 1.1:
            row["motor_error"] = "0x2"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    replay = load_joint_replay(episode)
    assert len(replay.time_s) == 201
    assert np.all(np.isnan(replay.linear["left"][100, :6]))
    assert np.all(np.isnan(replay.held["left"][100, :6]))
    assert np.all(np.isnan(replay.native["left"][100, :6]))
    assert np.all(np.isfinite(replay.linear["left"][50]))


def test_unfinished_episode_and_native_truncation_are_rejected(episode):
    complete = episode / "write_complete.flag"
    complete.unlink()
    with pytest.raises(ValueError, match="not finalized"):
        load_joint_replay(episode)
    complete.touch()
    path = episode / "native_joints/follower_left.jsonl"
    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="sample count mismatch"):
        load_joint_replay(episode)


def test_video_random_seek_preserves_exact_decoded_ordinals_and_bounded_cache(episode):
    path = episode / "top-images-rgb.mp4"
    with av.open(str(path)) as container:
        expected = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    video = ReplayVideo(path, len(expected), cache_size=3)
    try:
        for index in (0, 1, 33, 60, 10, 9, 40, 39, 0):
            np.testing.assert_array_equal(video.frame(index), expected[index])
            assert len(video.cache) <= 3
    finally:
        video.close()
    with pytest.raises(ValueError, match="video frames !="):
        ReplayVideo(path, 62)


@pytest.fixture
def socket_free_viser():
    """Use real Viser GUI/scene/URDF APIs, replacing only network transport."""
    loop = asyncio.new_event_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
        server = viser.ViserServer.__new__(viser.ViserServer)
        server._websock_server = MagicMock()
        server._websock_server.atomic.side_effect = nullcontext
        server._initial_camera = SimpleNamespace()
        server._client_lock = threading.Lock()
        server._connected_clients = {}
        server.scene = viser.SceneApi(server, executor, loop)
        server.gui = viser.GuiApi(server, executor, loop)
        server.stop = MagicMock()
        yield server
    loop.close()


def test_real_viser_controls_mesh_updates_video_and_clock_without_sockets(
    episode, socket_free_viser, monkeypatch,
):
    def forbidden_socket(*args, **kwargs):
        raise AssertionError("Replay test attempted to open a network/CAN socket")

    monkeypatch.setattr(socket, "socket", forbidden_socket)
    data = load_joint_replay(episode)
    viewer = CollectionReplayViewer(data, YamAdapter(), server=socket_free_viser)
    try:
        assert set(viewer.robot_handles) == {
            ("native", "left"), ("linear", "left"), ("held", "left"),
        }
        assert not viewer.image.visible  # Camera has not captured anything yet.
        viewer.seek(55)
        viewer.tick()
        assert viewer.image.visible
        assert viewer.frame_slider.value == 55
        assert viewer._last_image == ("top", data.cameras["top"].frame_at(data.timestamps_ns[55]))
        for (method, arm), handle in viewer.robot_handles.items():
            np.testing.assert_allclose(
                handle._urdf.cfg,
                viewer.robot.visual_configuration(arm, getattr(data, method)[arm][55]),
            )
        event = SimpleNamespace(client=object(), target=viewer.next)
        for callback in viewer.next._impl.update_cb:
            callback(event)
        viewer.tick()
        assert viewer.frame_slider.value == 56
        # Playback advances by elapsed wall time, not by the number of render calls.
        viewer._playing = True
        viewer._anchor_frame, viewer._anchor_time, viewer._speed = 56, 10.0, 0.25
        viewer.tick(11.0)
        assert viewer.frame_slider.value == 81
        viewer.tick(12.0)
        assert viewer.frame_slider.value == 106
        assert viewer._playing  # Programmatic slider changes must not pause playback.
        viewer.tick(100.0)
        assert not viewer._playing
        assert viewer.frame_slider.value == len(data.time_s) - 1
        viewer.seek(100)
        data.linear["left"][100, 0] = np.nan
        viewer.tick()
        assert not viewer.roots[("linear", "left")].visible
        assert viewer.roots[("held", "left")].visible
    finally:
        viewer.close()
    socket_free_viser.stop.assert_called_once()


@pytest.mark.parametrize("source", ["command", "feedback"])
@pytest.mark.parametrize("custom_rates", [False, True])
def test_replay_cli_branches_before_live_viewer(episode, monkeypatch, custom_rates, source):
    def forbidden(*args, **kwargs):
        raise AssertionError("Live PolicyViewer must not be constructed for saved replay")

    called = {}

    def capture(path, robot, **kwargs):
        called.update(path=path, robot=robot.name, **kwargs)

    monkeypatch.setattr(dashboard, "PolicyViewer", forbidden)
    monkeypatch.setattr("manimux.viewer.replay.serve_collection_replay", capture)
    arguments = [
        "manimux-viewer", "--replay-episode", str(episode), "--replay-source", source,
        "--replay-camera", "top", "--port", "8087", "--host", "127.0.0.1",
    ]
    if custom_rates:
        arguments += ["--replay-target-hz", "60", "--replay-low-hz", "10", "30"]
    monkeypatch.setattr(sys, "argv", arguments)
    dashboard.main()
    extra = {"target_hz": 60.0, "low_rates_hz": (10.0, 30.0)} if custom_rates else {}
    assert called == {"path": episode, "robot": "yam", "source": source, "camera": "top",
                      "port": 8087, "host": "127.0.0.1", **extra}


def test_command_viewer_does_not_invent_native_command_reference(episode, socket_free_viser):
    data = load_joint_replay(episode, "command")
    viewer = CollectionReplayViewer(data, YamAdapter(), server=socket_free_viser)
    try:
        assert set(viewer.robot_handles) == {("linear", "left"), ("held", "left")}
        assert "没有原生 100 Hz command" in viewer.source_note.content
    finally:
        viewer.close()


def test_overlay_shares_base_transforms_opacity_and_endpoint_poses(episode, socket_free_viser):
    data = load_joint_replay(episode)
    viewer = CollectionReplayViewer(data, YamAdapter(), server=socket_free_viser)
    try:
        assert viewer.layout.value == "叠加"
        for method in ("native", "linear"):
            np.testing.assert_array_equal(viewer.method_roots[method].position, (0, 0, 0))
            assert viewer.method_roots[method].visible
        assert not viewer.method_roots["held"].visible
        assert viewer.environment_roots["native"].visible
        assert not viewer.environment_roots["linear"].visible
        np.testing.assert_array_equal(
            viewer.roots[("native", "left")].position, viewer.roots[("linear", "left")].position,
        )
        np.testing.assert_array_equal(
            viewer.roots[("native", "left")].wxyz, viewer.roots[("linear", "left")].wxyz,
        )
        for frame in (0, len(data.time_s) - 1):
            viewer.seek(frame)
            viewer.tick()
            np.testing.assert_array_equal(
                viewer.robot_handles[("native", "left")]._urdf.cfg,
                viewer.robot_handles[("linear", "left")]._urdf.cfg,
            )
        viewer.opacity.value = 0.5
        viewer.tick()  # A paused frame must update when appearance changes.
        for (method, _), handle in viewer.robot_handles.items():
            assert handle._meshes
            expected_opacity = 0.5 if method == "linear" else None
            assert all(m.opacity == expected_opacity for m in handle._meshes)
            assert all(m.side == "front" for m in handle._meshes)
            assert all(m.wireframe == (method == "linear") for m in handle._meshes)
        # A paused replay must respond immediately to either visibility switch.
        viewer.show_reference.value = False
        viewer.tick()
        assert not viewer.roots[("native", "left")].visible
        assert viewer.roots[("linear", "left")].visible
        viewer.show_reference.value = True
        viewer.show_candidate.value = False
        viewer.tick()
        assert viewer.roots[("native", "left")].visible
        assert not viewer.roots[("linear", "left")].visible
        viewer.show_candidate.value = True
        viewer.ghost.value = "30 Hz 保持"
        viewer.tick()
        assert viewer.method_roots["held"].visible
        assert not viewer.method_roots["linear"].visible
        assert all(m.opacity == 0.5 for m in viewer.robot_handles[("held", "left")]._meshes)
        viewer.layout.value = "并排"
        viewer.tick()
        assert all(root.visible for root in viewer.method_roots.values())
        assert len({tuple(root.position) for root in viewer.method_roots.values()}) == 3
        for handle in viewer.robot_handles.values():
            assert all(m.opacity is None for m in handle._meshes)
            assert all(m.side == "front" for m in handle._meshes)
            assert not any(m.wireframe for m in handle._meshes)
        viewer.layout.value = "叠加"
        viewer.tick()
        assert all(tuple(root.position) == (0, 0, 0) for root in viewer.method_roots.values())
        for button, expected in ((viewer.first, 0), (viewer.last, len(data.time_s) - 1)):
            for callback in button._impl.update_cb:
                callback(SimpleNamespace(client=object(), target=button))
            viewer.tick()
            assert viewer.frame_slider.value == expected
    finally:
        viewer.close()


def test_camera_duplicate_timestamps_do_not_create_new_capture_times():
    camera = ReplayCamera(None, np.array([10, 10, 20], dtype=np.int64))
    assert camera.frame_at(10) == 1
    assert camera.frame_at(19) == 1


@pytest.fixture
def rate_episode(episode):
    """60 Hz targets with two brief arm pulses and an off-knot gripper event."""
    meta = json.loads((episode / "metadata.json").read_text())
    meta.update(schema_version=2, control_hz=60, num_frames=121)
    (episode / "metadata.json").write_text(json.dumps(meta))
    ts = ORIGIN + np.rint(np.arange(121) / 60 * 1e9).astype(np.int64)
    q = np.zeros((121, 7))
    q[:, 0] = np.arange(121) * 0.001
    q[1, 0] = 1.0  # missed by both 30 and 10 Hz
    q[2, 1] = 1.0  # retained by 30 Hz, missed by 10 Hz
    q[:, 6] = (np.arange(121) < 7).astype(float)
    _write_rate_commands(episode, ts, q)
    # A wrong action-* loader would now draw a very different trajectory.
    np.save(episode / "action-left-joint.npy", np.full((121, 6), 2.0))
    return episode


def _write_rate_commands(episode, ts, q):
    np.save(episode / "controller-left-timestamp-ns.npy", ts)
    np.save(episode / "controller-left-joint.npy", q[:, :6])
    rows = [{"unix_ns": int(t), "command": {"left_arm": values.tolist()}}
            for t, values in zip(ts, q, strict=True)]
    (episode / "manimux-control.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    meta = json.loads((episode / "metadata.json").read_text())
    meta["num_frames"] = len(ts)
    (episode / "metadata.json").write_text(json.dumps(meta))


def test_60hz_comparison_preserves_commands_endpoints_and_gripper_events(rate_episode):
    data = load_command_rate_replay(rate_episode)
    ref = data.native["left"]
    ten = data.resampled["resample_0"]["left"]
    thirty = data.resampled["resample_1"]["left"]
    assert data.target_hz == 60
    assert data.source_rates_hz["left"] == pytest.approx(60)
    np.testing.assert_allclose(np.diff(data.time_s), 1 / 60)
    assert ref[1, 0] == 1.0
    assert ten[1, 0] == pytest.approx(0.001)
    assert thirty[1, 0] == pytest.approx(0.001)
    assert ten[2, 1] == 0.0
    assert thirty[2, 1] == 1.0
    assert thirty[1, 1] == 0.5  # offline interpolation deliberately uses a future knot
    for candidate in (ten, thirty):
        np.testing.assert_array_equal(candidate[[0, -1]], ref[[0, -1]])
        np.testing.assert_array_equal(candidate[:, 6], ref[:, 6])
        assert candidate[6, 6] == 1 and candidate[7, 6] == 0
    assert len(data.cameras["top"].timestamps_ns) == 61
    assert len(ref) == 121


def test_60hz_reference_uses_actual_submission_time_and_reports_actual_rate(rate_episode):
    lines = (rate_episode / "manimux-control.jsonl").read_text().splitlines()
    rows = [json.loads(s) for s in lines]
    q = np.array([r["command"]["left_arm"] for r in rows])
    ts = ORIGIN + np.arange(121) * 20_000_000  # nominal 60, actual 50 Hz
    _write_rate_commands(rate_episode, ts, q)
    data = load_command_rate_replay(rate_episode)
    assert data.source_rates_hz["left"] == pytest.approx(50)
    assert "50.00" in data.source_note
    assert data.native["left"][1, 0] == 0.0  # 16.67 ms precedes the pulse submitted at 20 ms
    assert data.native["left"][2, 0] == 1.0


def test_60hz_comparison_does_not_interpolate_over_source_gaps(rate_episode):
    ts = np.load(rate_episode / "controller-left-timestamp-ns.npy")
    lines = (rate_episode / "manimux-control.jsonl").read_text().splitlines()
    rows = [json.loads(s) for s in lines]
    q = np.array([r["command"]["left_arm"] for r in rows])
    keep = ~((ts > ORIGIN + 800_000_000) & (ts < ORIGIN + 1_000_000_000))
    _write_rate_commands(rate_episode, ts[keep], q[keep])
    data = load_command_rate_replay(rate_episode)
    assert len(data.time_s) == 121
    for groups in data.trajectories.values():
        assert np.isnan(groups["left"][54, :6]).all()
        assert np.isfinite(groups["left"][60]).all()


@pytest.mark.parametrize("target,lower", [(0, (10,)), (float("nan"), (10,)),
                                           (60, (60,)), (60, (10, 10)),
                                           (60, (7,)), (60, ())])
def test_invalid_command_comparison_rates_are_rejected(rate_episode, target, lower):
    with pytest.raises(ValueError):
        load_command_rate_replay(rate_episode, target_hz=target, low_rates_hz=lower)


def test_command_comparison_accepts_nominal_100hz_and_uses_command_only(rate_episode):
    meta_path = rate_episode / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["control_hz"] = 100  # Requested collection Hz is not measured command Hz.
    meta_path.write_text(json.dumps(meta))
    ts = np.load(rate_episode / "controller-left-timestamp-ns.npy")
    # Make every other potential source visibly different from the real command.
    np.save(rate_episode / "left-joint_pos.npy", np.full((len(ts), 6), 9.0))
    data = load_command_rate_replay(rate_episode)
    assert data.source == "command"
    assert data.target_hz == 60
    assert data.source_rates_hz["left"] == pytest.approx(60)
    assert data.native["left"][1, 0] == 1.0
    assert data.resampled["resample_0"]["left"][1, 0] == pytest.approx(0.001)
    np.testing.assert_array_equal(data.source_samples["left"][0], ts)
    assert "采集设置 100 Hz" in data.source_note


def test_command_comparison_rejects_trace_mismatch(rate_episode):
    q = np.load(rate_episode / "controller-left-joint.npy")
    q[10, 0] += 0.1
    np.save(rate_episode / "controller-left-joint.npy", q)
    with pytest.raises(ValueError, match="disagree"):
        load_command_rate_replay(rate_episode)


def test_viser_switches_both_60hz_candidates_without_hardware(rate_episode, socket_free_viser):
    data = load_command_rate_replay(rate_episode)
    viewer = CollectionReplayViewer(data, YamAdapter(), server=socket_free_viser)
    try:
        assert set(viewer.robot_handles) == {
            ("native", "left"), ("resample_0", "left"), ("resample_1", "left")
        }
        assert viewer.frame_slider.label == "60 Hz 帧索引"
        assert "16.67 ms" in viewer.next.label
        for label, method in [("10 → 60 Hz 插值", "resample_0"),
                              ("30 → 60 Hz 插值", "resample_1")]:
            viewer.ghost.value = label
            viewer.seek(2)
            viewer.tick()
            assert viewer.method_roots[method].visible
            assert label in viewer.comparison.content
            np.testing.assert_allclose(
                viewer.robot_handles[(method, "left")]._urdf.cfg,
                viewer.robot.visual_configuration("left", data.resampled[method]["left"][2]),
            )
            for end in [0, len(data.time_s) - 1]:
                viewer.seek(end)
                viewer.tick()
                np.testing.assert_array_equal(
                    viewer.robot_handles[("native", "left")]._urdf.cfg,
                    viewer.robot_handles[(method, "left")]._urdf.cfg,
                )
        viewer.seek(0)
        viewer._playing = True
        viewer._anchor_frame, viewer._anchor_time, viewer._speed = 0, 10.0, 0.5
        viewer.tick(11.0)
        assert viewer.frame_slider.value == 30
    finally:
        viewer.close()


def test_custom_replay_cli_requires_saved_command_inputs_before_live_setup(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["viewer", "--replay-target-hz", "60"])
    with pytest.raises(SystemExit, match="Custom replay requires"):
        dashboard.main()


@pytest.fixture
def achieved_episode(rate_episode):
    meta = json.loads((rate_episode / "metadata.json").read_text())
    meta.update(control_hz=100, arm_names=["left", "right"])
    (rate_episode / "metadata.json").write_text(json.dumps(meta))
    ts = ORIGIN + np.arange(121) * 20_000_000  # actual50, config100, compare60
    for arm in meta["arm_names"]:
        q = np.tile(np.arange(121)[:, None] * 0.001, (1, 6))
        q[5, 2] = 1.0
        np.save(rate_episode / f"{arm}-joint_pos.npy", q)
        np.save(rate_episode / f"{arm}-feedback-timestamp-ns.npy", ts)
        np.save(rate_episode / f"{arm}-gripper_pos.npy", np.linspace(0.2, 0.8, 121)[:, None])
    return rate_episode


def test_achieved_uses_measured_joints_actual_times_and_same_endpoints(achieved_episode):
    from manimux.collection.yam.data.feedback_resampling import load_feedback_rate_replay
    from manimux.collection.yam.data.replay_report import joint_metrics

    data = load_feedback_rate_replay(achieved_episode)
    assert data.source == "feedback"
    assert data.source_rates_hz == {"left": 50, "right": 50}
    assert data.native["left"][1, 0] == 0  # at16.67ms: latest feedback is0ms
    assert data.native["left"][2, 0] == 0.001  # at33.33ms: source20ms
    assert data.native["left"][6, 2] == 1.0  # measured pulse at100ms
    assert len(data.source_samples["left"][0]) == 121
    for groups in data.resampled.values():
        for arm in data.native:
            np.testing.assert_array_equal(groups[arm][[0,-1]], data.native[arm][[0,-1]])
            np.testing.assert_array_equal(groups[arm][:,6], data.native[arm][:,6])
    metrics = joint_metrics(data)
    assert len(metrics) == 24
    assert all(row["valid_samples"] == len(data.time_s) for row in metrics)


def test_achieved_viser_shows_all_three_and_both_arms(achieved_episode, socket_free_viser):
    from manimux.collection.yam.data.feedback_resampling import load_feedback_rate_replay

    data = load_feedback_rate_replay(achieved_episode)
    viewer = CollectionReplayViewer(data, YamAdapter(), server=socket_free_viser)
    try:
        assert len(viewer.robot_handles) == 6
        assert viewer.layout.value == "三组叠加"
        assert all(root.visible for root in viewer.method_roots.values())
        assert "right J6" in viewer.joint_values.content
        for method in ("resample_0", "resample_1"):
            assert all(m.wireframe for m in viewer.robot_handles[(method, "left")]._meshes)
        viewer.layout.value = "并排"
        viewer.seek(6)
        viewer.tick()
        assert all(root.visible for root in viewer.method_roots.values())
        assert len({tuple(root.position) for root in viewer.method_roots.values()}) == 3
    finally:
        viewer.close()
