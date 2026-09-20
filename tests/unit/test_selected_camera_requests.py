import json
import select
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from manimux.embodiments.sensor.camera_server.client import (
    CameraClient,
    CameraClientError,
    CameraSubscriber,
)
from manimux.servers.camera.server import CameraServer, _build_cameras_from_config


@pytest.fixture
def running_camera_service():
    """Exercise the real service and sockets with synthetic RGB, without devices."""
    code = """
import json
import time
from types import SimpleNamespace
import numpy as np
import zmq
from manimux.servers.camera.server import CameraServer
image = np.broadcast_to(np.array([11, 22, 33], dtype=np.uint8), (4, 5, 3)).copy()
camera = SimpleNamespace(read_with_timestamp=lambda: (image, None, time.time()),
                         close=lambda: None)
server = CameraServer({'left_wrist': camera, 'right_wrist': camera},
                      rep_endpoint='tcp://127.0.0.1:0',
                      pub_endpoint='tcp://127.0.0.1:0', pub_period_sec=0.02)
server.bind()
print(json.dumps({'rep': server._rep.getsockopt(zmq.LAST_ENDPOINT).decode(),
                  'pub': server._pub.getsockopt(zmq.LAST_ENDPOINT).decode()}), flush=True)
try:
    server.run()
except KeyboardInterrupt:
    pass
"""
    process = subprocess.Popen(
        [sys.executable, "-B", "-u", "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert select.select([process.stdout], [], [], 5)[0], "camera service startup timed out"
        line = process.stdout.readline()
        assert line, process.stderr.read()
        yield json.loads(line)
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
        assert process.returncode == 0


def test_service_round_trip_preserves_rgb_and_timestamped_umi_frames(running_camera_service):
    from manimux.clock import SystemClock
    from manimux.embodiments.sensor.camera_server.timestamped import TimestampedCameraSensor

    endpoints = running_camera_service
    client = CameraClient(endpoints["rep"], request_timeout_ms=1000)
    subscriber = CameraSubscriber(endpoints["pub"])
    sensor = TimestampedCameraSensor(
        {
            "options": {
                "endpoint": endpoints["pub"],
                "camera_names": ["left_wrist", "right_wrist"],
                "output_names": {
                    "left_wrist": "left_wrist_camera",
                    "right_wrist": "right_wrist_camera",
                },
                "max_frame_age_sec": 0.5,
            },
        },
        SystemClock(),
    )
    try:
        assert client.ping()
        images = client.get_obs(["right_wrist"])
        assert set(images) == {"right_wrist"}
        np.testing.assert_array_equal(images["right_wrist"][0, 0], [11, 22, 33])
        with pytest.raises(CameraClientError, match="Unknown cameras"):
            client.get_obs(["missing"])
        assert client.ping()  # A request error must not poison the REQ socket.
        sensor.start()
        deadline = time.monotonic() + 3
        bundle, frames = None, {}
        while time.monotonic() < deadline and (bundle is None or not frames):
            bundle = subscriber.try_recv_bundle() or bundle
            frames = sensor.read()
            time.sleep(0.005)
        assert bundle is not None and set(bundle["frames"]) == {"left_wrist", "right_wrist"}
        assert all(0 <= time.time() - t < 0.5 for t in bundle["timestamps"].values())
        assert set(frames) == {"left_wrist_camera", "right_wrist_camera"}
        for frame in frames.values():
            assert frame.data.dtype == np.uint8 and frame.data.shape == (4, 5, 3)
            np.testing.assert_array_equal(frame.data[0, 0], [11, 22, 33])
            assert 0 <= time.monotonic_ns() - frame.capture_monotonic_ns < 500_000_000
            assert frame.sequence > 0
    finally:
        sensor.close()
        subscriber.close()
        client.close()


def test_duplicate_service_exits_before_camera_access(running_camera_service):
    # Placeholder device serials would fail if the duplicate service opened hardware.
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "manimux.servers.camera.server",
            "--experiment",
            "manimux/configs/experiments/pass_ball/tianji_taccap_umi_dp.yaml",
            "--local",
            "manimux/configs/local/tianji_taccap.example.yaml",
            "--rep-endpoint",
            running_camera_service["rep"],
            "--pub-endpoint",
            running_camera_service["pub"],
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 2
    assert "No cameras were opened or reset" in result.stderr


def test_unselected_broken_camera_does_not_block_request():
    image = np.zeros((4, 5, 3), dtype=np.uint8)

    def broken():
        raise RuntimeError("disconnected")

    server = CameraServer(
        {
            "selected": SimpleNamespace(
                read=lambda: (image, None), _latest_frame_timestamp=time.time()
            ),
            "other": SimpleNamespace(read=broken),
        }
    )
    reply = server._snapshot(["selected"])
    assert list(reply["frames"]) == ["selected"]
    with pytest.raises(RuntimeError, match="disconnected"):
        server._snapshot(["other"])
    with pytest.raises(ValueError, match="Unknown cameras"):
        server._snapshot(["missing"])


def client_with_reply(monkeypatch, timestamps):
    client = CameraClient.__new__(CameraClient)
    client.max_frame_age_sec = 0.15

    def request(cmd, names):
        assert cmd == "obs" and names == ["selected"]
        return {
            "frames": {name: np.zeros((4, 5, 3), np.uint8) for name in timestamps},
            "timestamps": timestamps,
        }

    monkeypatch.setattr(client, "_request", request)
    return client


def test_old_server_extra_stale_frames_are_filtered(monkeypatch):
    client = client_with_reply(monkeypatch, {"selected": time.time(), "other": time.time() - 30})
    assert list(client.get_obs(["selected"])) == ["selected"]


@pytest.mark.parametrize("timestamp", [0, None, float("nan"), float("inf"), "invalid"])
def test_selected_timestamp_must_be_valid(monkeypatch, timestamp):
    client = client_with_reply(monkeypatch, {"selected": timestamp})
    with pytest.raises(CameraClientError, match="Invalid capture timestamp"):
        client.get_obs(["selected"])


def test_selected_stale_frame_is_rejected(monkeypatch):
    client = client_with_reply(monkeypatch, {"selected": time.time() - 1})
    with pytest.raises(CameraClientError, match="Stale frame"):
        client.get_obs(["selected"])


def test_selected_snapshot_keeps_atomic_capture_timestamp():
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    captured_at = time.time() - 0.01

    def unexpected_read():
        raise AssertionError("unselected or legacy camera read")

    server = CameraServer(
        {
            "selected": SimpleNamespace(
                read=unexpected_read,
                read_with_timestamp=lambda: (image, None, captured_at),
                _latest_frame_timestamp=captured_at + 0.01,
            ),
            "other": SimpleNamespace(read_with_timestamp=unexpected_read),
        }
    )
    reply = server._snapshot(["selected"])
    assert list(reply["frames"]) == ["selected"]
    assert reply["frames"]["selected"] is image
    assert reply["timestamps"] == {"selected": captured_at}


@pytest.mark.parametrize("kinds", [("orbbec",), ("realsense", "orbbec", "taccap")])
def test_camera_factory_preserves_orbbec_and_taccap(monkeypatch, tmp_path, kinds):
    discovery = []

    def factory(kind):
        def open_camera(device_id=None, **options):
            return SimpleNamespace(
                kind=kind,
                device_id=device_id or options.pop("camera_serial", None),
                options=options,
                close=lambda: None,
                start=lambda: None,
            )

        return open_camera

    monkeypatch.setitem(
        sys.modules,
        "manimux.embodiments.sensor.realsense",
        SimpleNamespace(
            RealSenseSensor=factory("realsense"),
            get_device_ids=lambda: discovery.append("realsense") or ["rs-serial"],
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "manimux.embodiments.sensor.orbbec",
        SimpleNamespace(OrbbecCamera=factory("orbbec")),
    )
    monkeypatch.setitem(
        sys.modules,
        "manimux.embodiments.sensor.taccap",
        SimpleNamespace(TacCapCamera=factory("taccap")),
    )
    specs = {
        kind: {"type": kind, "camera_serial": "taccap-serial"}
        if kind == "taccap"
        else {
            "type": kind,
            "device_id": f"{kind}-serial",
            "enable_depth": False,
            "width": 640,
            "height": 480,
            "fps": 30,
            "flip": True,
        }
        for kind in kinds
    }
    config = tmp_path / "cameras.yaml"
    config.write_text(yaml.safe_dump({"sensors": {"cameras": specs}}), encoding="utf-8")
    cameras = _build_cameras_from_config(config)
    assert set(cameras) == set(kinds)
    assert discovery == (["realsense"] if "realsense" in kinds else [])
    for kind, camera in cameras.items():
        assert camera.kind == kind
        assert camera.device_id == f"{kind}-serial"
        assert camera.options["fps"] == 30
        if kind != "taccap":
            assert camera.options["enable_depth"] is False
            assert camera.options["flip"] is True
