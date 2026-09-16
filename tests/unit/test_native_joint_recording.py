import json
import socket
import struct
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from manimux.collection.yam.data.joint_rate_analysis import compare_joint, error_metrics
from manimux.collection.yam.data.native_joints import NativeJointRecording
from manimux.robots.yam.native import NativeJointReader, NativeJointSource


class FakeSocket:
    def __init__(self):
        self.closed = False
        self.options = []

    def setsockopt(self, *args):
        self.options.append(args)

    def bind(self, address):
        self.address = address

    def settimeout(self, value):
        self.timeout = value

    def close(self):
        self.closed = True


def test_native_reader_receives_kernel_timestamps_and_calibrates_without_sending(monkeypatch):
    pytest.importorskip("can")
    fake = FakeSocket()
    monkeypatch.setattr(socket, "socket", lambda *args: fake)
    source = NativeJointSource(
        "follower_left", "test_can", [1], [17], ["test"], np.array([12.0]),
        np.array([-1.0]), np.array([12.4]), np.array([25.0]),
        lambda *args, **kwargs: SimpleNamespace(
            position=-12.4, velocity=2.0, torque=0.3, error_code="0x1"
        ),
    )
    reader = NativeJointReader(source)
    frame = struct.pack("=IB3x8s", 17, 8, bytes(8))
    ancillary = [(socket.SOL_SOCKET, 35, struct.pack("@ll", 123, 456)),
                 (socket.SOL_SOCKET, 40, struct.pack("=I", 2))]
    fake.recvmsg = lambda *args: (frame, ancillary, 0, None)
    sample = reader.read()
    assert sample["timestamp_ns"] == 123_000_000_456
    assert sample["position_rad"] == pytest.approx(-0.6)
    assert sample["velocity_rad_s"] == -2
    assert reader.dropped_frames == 2
    # Stationary but newly received feedback is a new sample; never dedupe by position.
    again = reader.decode(frame, 124_000_000_000, 0)
    assert again["sequence"] == 2
    assert again["position_rad"] == pytest.approx(-0.6)
    assert reader.decode(frame, 125_000_000_000, socket.MSG_DONTROUTE) is None
    fake.recvmsg = lambda *args: (frame, [], 0, None)
    with pytest.raises(RuntimeError, match="kernel receive timestamp"):
        reader.read()
    reader.close()
    assert fake.closed


class FakeSource:
    """A test-only native stream, paced independently of the collection tick."""

    def __init__(self, stream="follower_left", fail=False, dropped=0):
        self.stream = self.channel = stream
        self.motor_ids = [1, 2, 3, 4, 5, 6]
        self.closed = False
        self.fail = fail
        self.dropped_frames = dropped
        self.sequence = 0
        self.index = 0
        self.seen = threading.Event()

    def metadata(self):
        return {"stream": self.stream, "channel": self.channel,
                "joint_names": [f"joint{i + 1}" for i in range(6)]}

    def open(self):
        if self.fail:
            raise OSError("test CAN unavailable")
        return self

    def read(self):
        time.sleep(0.0005)
        index = self.index
        self.index = (index + 1) % 6
        self.sequence += 1
        self.seen.set()
        return {"joint_index": index, "sequence": self.sequence, "timestamp_ns": time.time_ns(),
                "position_rad": float(index), "velocity_rad_s": 0.0, "effort_nm": 0.0,
                "motor_error": "0x1"}

    def close(self):
        self.closed = True


def test_native_source_contract_with_installed_i2rt_parser(monkeypatch):
    sdk = pytest.importorskip("i2rt.motor_drivers.dm_driver")
    from i2rt.motor_drivers.utils import ReceiveMode

    parser = sdk.DMSingleMotorCanInterface.__new__(sdk.DMSingleMotorCanInterface)
    parser.receive_mode = ReceiveMode.p16
    chain = SimpleNamespace(
        running=True, channel="test_can", state_lock=threading.RLock(),
        motor_list=[(i + 1, "DM4340") for i in range(6)],
        motor_offset=np.full(6, 0.2), motor_direction=np.ones(6),
        absolute_positions=np.full(6, 0.1), motor_interface=parser,
    )
    source = NativeJointSource.from_robot(SimpleNamespace(motor_chain=chain), "leader_left")
    assert source.feedback_ids == list(range(17, 23))
    assert source.metadata()["joint_names"] == [f"joint{i + 1}" for i in range(6)]
    monkeypatch.setattr(socket, "socket", lambda *args: FakeSocket())
    reader = source.open()
    payload = bytes.fromhex("11805e7ff8121c1b")
    sample = reader.decode(struct.pack("=IB3x8s", 17, 8, payload), 100, 0)
    assert sample["position_rad"] == pytest.approx(0x805E * 25 / 65535 - 12.5 - 0.2)
    assert sample["motor_error"] == "0x1"
    reader.close()


def test_native_streams_save_independently_and_mark_socket_drops(tmp_path):
    sources = [FakeSource(), FakeSource("leader_left", dropped=1)]
    capture = NativeJointRecording(sources)
    capture.start(tmp_path)
    for source in sources:
        assert source.seen.wait(1)
    time.sleep(0.06)
    meta = capture.stop()
    assert not meta["complete"]
    for source in sources:
        assert source.closed
        rows = [json.loads(row) for row in
                (capture.path / f"{source.stream}.jsonl").read_text().splitlines()]
        assert len(rows) > 6
        assert all(meta["start_timestamp_ns"] <= row["timestamp_ns"] <= meta["end_timestamp_ns"]
                   for row in rows)
    assert meta["streams"]["leader_left"]["dropped_frames"] == 1


def test_native_open_failure_closes_other_listeners(tmp_path):
    good = FakeSource()
    capture = NativeJointRecording([good, FakeSource("leader_left", fail=True)])
    with pytest.raises(OSError, match="unavailable"):
        capture.start(tmp_path)
    assert good.closed
    assert not capture._threads


def test_gui_native_option_records_sidecars_without_changing_camera_rate(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from manimux.collection.yam.config import CameraConfig, build_station_config
    from manimux.collection.yam.gui.server import create_app

    cfg = build_station_config("configs/collection/yam/station.yaml")
    cfg.record_achieved = True  # this test explicitly exercises feedback recording
    cfg.save_root = str(tmp_path / "episodes")
    cfg.cameras = [CameraConfig("top", "mock", "top", width=64, height=48)]
    app = create_app(cfg, mock=True)
    sources = [FakeSource(name) for name in
               ("follower_left", "follower_right", "leader_left", "leader_right")]
    monkeypatch.setattr(app.state.session, "_native_joint_sources", lambda: sources)
    monkeypatch.setenv("YAM_ABC_VIDEO_ENCODER", "libx264")
    with TestClient(app) as client:
        r = client.post("/api/collect/recording-options", json={"record_native_joints": True})
        assert r.status_code == 200
        assert r.json()["record_native_joints"]
        assert client.post("/api/collect/start-teleop").status_code == 200
        assert client.post("/api/collect/start-recording", json={
            "task_name": "rate_test", "include_eepose": False,
        }).status_code == 200
        assert client.get("/api/collect/status").json()["record_native_joints"]
        assert client.post("/api/collect/recording-options", json={
            "record_native_joints": False,
        }).status_code == 409
        time.sleep(0.25)
        result = client.post("/api/collect/stop-recording")
        assert result.status_code == 200, result.text
        episode = Path(result.json()["path"])
        metadata = json.loads((episode / "metadata.json").read_text())
        native = json.loads((episode / "native_joints/metadata.json").read_text())
        assert native["complete"]
        assert metadata["control_hz"] == cfg.control_hz
        assert metadata["cameras"][0]["fps"] == 30
        assert metadata["extra"]["native_joints"]["complete"]
        for source in sources:
            assert source.closed
            assert min(native["streams"][source.stream]["samples"]) > result.json()["frames"]
        assert (episode / "write_complete.flag").exists()


def test_native_recording_abort_joins_writers_before_removing_episode(tmp_path):
    from manimux.collection.yam.config import StationConfig
    from manimux.collection.yam.data.recorder import EpisodeRecorder

    source = FakeSource()
    recorder = EpisodeRecorder(tmp_path, StationConfig(), [], ["left"],
                               native_sources_factory=lambda: [source])
    path = recorder.start("test", record_native_joints=True)
    assert source.seen.wait(1)
    recorder.abort()
    assert source.closed
    assert not path.exists()


def test_reconstruction_constant_zero_error_and_high_frequency_error():
    ts = np.arange(601, dtype=np.int64) * 5_000_000  # real 200 Hz source, 3 seconds
    constant = compare_joint(ts, np.full(len(ts), 0.3))
    for candidate in list(constant["curves"].values())[1:]:
        assert error_metrics(constant["curves"]["native_to_target"], candidate)["rmse_deg"] < 1e-10
    wave = compare_joint(ts, 0.1 * np.sin(2 * np.pi * 20 * ts / 1e9), cutoff_hz=0)
    metric = error_metrics(*wave["curves"].values())
    assert metric["rmse_deg"] > 2
    assert wave["native_hz_mean"] == 200


def test_reconstruction_excludes_gaps_endpoints_and_invalid_motor_samples():
    ts = np.arange(201, dtype=np.int64) * 5_000_000
    selected = (ts < 400_000_000) | (ts > 600_000_000)
    positions = ts[selected] / 1e9
    positions[10] = np.nan
    result = compare_joint(ts[selected], positions, phase_ms=7)
    for candidate in result["curves"].values():
        assert np.isnan(candidate[(result["time_s"] > .4) & (result["time_s"] < .6)]).all()
    assert np.isnan(result["curves"]["low_to_target_linear"][0])
    assert np.isnan(result["curves"]["low_to_target_linear"][-1])
    assert result["native_max_gap_ms"] >= 200
    with pytest.raises(ValueError, match="increase strictly"):
        compare_joint([1, 1, 2], [0, 1, 2])


def test_plotting_exports_per_joint_metrics_and_curves(tmp_path):
    pytest.importorskip("matplotlib")
    from manimux.collection.yam.data.joint_rate_analysis import analyze_episode

    native = tmp_path / "native_joints"
    native.mkdir()
    origin = 1_800_000_000_000_000_000
    meta = {"schema_version": 1, "complete": True, "start_timestamp_ns": origin,
            "end_timestamp_ns": origin + 2_000_000_000,
            "streams": {"leader_left": {
                "file": "leader_left.jsonl", "joint_names": ["joint1"],
            }}}
    (native / "metadata.json").write_text(json.dumps(meta))
    with (native / "leader_left.jsonl").open("w") as handle:
        for i in range(401):
            handle.write(json.dumps({
                "joint_index": 0, "timestamp_ns": origin + i * 5_000_000,
                "position_rad": 0.1 * np.sin(i * .2), "motor_error": "0x1",
            }) + "\n")
    result = analyze_episode(tmp_path)
    assert len(result["metrics"]) == 2
    assert not result["unavailable_joints"]
    for name in ("leader_left.png", "leader_left.svg", "metrics.csv", "leader_left-joint1.csv"):
        assert (tmp_path / "joint_rate_analysis" / name).stat().st_size > 100
