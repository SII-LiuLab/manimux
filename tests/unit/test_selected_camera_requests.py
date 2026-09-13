import time
from types import SimpleNamespace

import numpy as np
import pytest

from manimux.sensors.camera_server.client import CameraClient, CameraClientError
from manimux.sensors.camera_server.server import CameraServer


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
