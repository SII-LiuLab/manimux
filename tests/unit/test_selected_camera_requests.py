import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from manimux.sensors.camera_server.client import CameraClient, CameraClientError
from manimux.sensors.camera_server.server import CameraServer, _build_cameras_from_config


def test_unselected_broken_camera_does_not_block_request():
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    def broken():
        raise RuntimeError("disconnected")
    server = CameraServer({
        "selected": SimpleNamespace(
            read=lambda: (image, None), _latest_frame_timestamp=time.time()
        ),
        "other": SimpleNamespace(read=broken),
    })
    reply = server._snapshot(["selected"])
    assert list(reply["frames"]) == ["selected"]
    with pytest.raises(RuntimeError, match="disconnected"):
        server._snapshot(["other"])
    with pytest.raises(ValueError, match="Unknown cameras"):
        server._snapshot(["missing"])


def client_with_reply(monkeypatch, timestamps):
    client = CameraClient.__new__(CameraClient)
    client.max_frame_age_sec = .15
    def request(cmd, names):
        assert cmd == "obs" and names == ["selected"]
        return {"frames": {name: np.zeros((4, 5, 3), np.uint8) for name in timestamps},
                "timestamps": timestamps}
    monkeypatch.setattr(client, "_request", request)
    return client


def test_old_server_extra_stale_frames_are_filtered(monkeypatch):
    client = client_with_reply(monkeypatch, {"selected": time.time(), "other": time.time()-30})
    assert list(client.get_obs(["selected"])) == ["selected"]


@pytest.mark.parametrize("timestamp", [0, None, float("nan"), float("inf"), "invalid"])
def test_selected_timestamp_must_be_valid(monkeypatch, timestamp):
    client = client_with_reply(monkeypatch, {"selected": timestamp})
    with pytest.raises(CameraClientError, match="Invalid capture timestamp"):
        client.get_obs(["selected"])


def test_selected_stale_frame_is_rejected(monkeypatch):
    client = client_with_reply(monkeypatch, {"selected": time.time()-1})
    with pytest.raises(CameraClientError, match="Stale frame"):
        client.get_obs(["selected"])


def test_selected_snapshot_keeps_atomic_capture_timestamp():
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    captured_at = time.time() - 0.01

    def unexpected_read():
        raise AssertionError("unselected or legacy camera read")

    server = CameraServer({
        "selected": SimpleNamespace(
            read=unexpected_read,
            read_with_timestamp=lambda: (image, None, captured_at),
            _latest_frame_timestamp=captured_at + 0.01,
        ),
        "other": SimpleNamespace(read_with_timestamp=unexpected_read),
    })
    reply = server._snapshot(["selected"])
    assert list(reply["frames"]) == ["selected"]
    assert reply["frames"]["selected"] is image
    assert reply["timestamps"] == {"selected": captured_at}


@pytest.mark.parametrize("kinds", [("orbbec",), ("realsense", "orbbec", "taccap")])
def test_camera_factory_preserves_orbbec_and_taccap(monkeypatch, tmp_path, kinds):
    discovery = []

    def factory(kind):
        def open_camera(device_id, **options):
            return SimpleNamespace(
                kind=kind, device_id=device_id, options=options, close=lambda: None
            )

        return open_camera

    monkeypatch.setitem(sys.modules, "manimux.sensors.realsense", SimpleNamespace(
        RealSenseCamera=factory("realsense"),
        get_device_ids=lambda: discovery.append("realsense") or ["rs-serial"],
    ))
    monkeypatch.setitem(sys.modules, "manimux.sensors.orbbec", SimpleNamespace(
        OrbbecCamera=factory("orbbec"),
    ))
    monkeypatch.setitem(sys.modules, "manimux.sensors.taccap", SimpleNamespace(
        TacCapCamera=factory("taccap"),
    ))
    specs = {
        kind: {"type": kind, "camera_serial": "taccap-serial"}
        if kind == "taccap"
        else {
            "type": kind, "device_id": f"{kind}-serial", "enable_depth": False,
            "width": 640, "height": 480, "fps": 30, "flip": True,
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
