"""TacCap camera backend and camera-server type selection, with a fake TacCap SDK."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from manimux.sensors.camera_server import server as camera_server
from manimux.sensors.taccap import TacCapCamera, find_camera_device

REPO = Path(__file__).resolve().parents[2]


class FakeCamera:
    instances: list[FakeCamera] = []

    def __init__(self, device: str, width: int, height: int, fps: float, use_mjpg: bool) -> None:
        self.device, self.size, self.use_mjpg = device, (height, width), use_mjpg
        self.stopped = False
        self._thread: threading.Thread | None = None
        FakeCamera.instances.append(self)

    def start(self, callback) -> None:  # type: ignore[no-untyped-def]
        def emit() -> None:
            index = 0
            while not self.stopped:
                image = np.zeros((*self.size, 3), dtype=np.uint8)
                image[..., 0] = 255  # blue channel in BGR
                callback(SimpleNamespace(image=image, frame_index=index))
                index += 1
                time.sleep(0.01)

        self._thread = threading.Thread(target=emit, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.stopped = True
        if self._thread is not None:
            self._thread.join(timeout=1.0)


@pytest.fixture
def by_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for serial in ("XCA28Z0041s", "XCA28Z0042s"):
        for index in (0, 1):
            (tmp_path / f"usb-Xense_imx385_{serial}-video-index{index}").touch()
    FakeCamera.instances.clear()
    monkeypatch.setitem(
        sys.modules,
        "xense.taccap",
        SimpleNamespace(Camera=FakeCamera),
    )
    return tmp_path


def test_camera_is_resolved_by_serial(by_id: Path) -> None:
    assert find_camera_device("XCA28Z0042s", by_id).name.endswith("XCA28Z0042s-video-index0")
    with pytest.raises(RuntimeError, match="found 0; connected TacCap cameras"):
        find_camera_device("XCA-missing", by_id)


def test_camera_serves_rgb_frames_and_detects_stalls(by_id: Path) -> None:
    camera = TacCapCamera("XCA28Z0041s", by_id_root=by_id, max_frame_age_sec=0.2)
    try:
        image, depth = camera.read()
        assert depth is None and image.shape == (480, 640, 3)
        assert image[0, 0].tolist() == [0, 0, 255]  # BGR from the SDK becomes RGB
        assert FakeCamera.instances[0].use_mjpg
        FakeCamera.instances[0].stop()
        time.sleep(0.3)
        with pytest.raises(RuntimeError, match="stale"):
            camera.read()
    finally:
        camera.close()


def test_server_builds_cameras_by_type(by_id: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(camera_server, "V4L_BY_ID", by_id, raising=False)
    config = REPO / "configs/robots/tianji/cameras.yaml"
    cameras = camera_server._build_cameras_from_config(config, by_id_root=by_id)
    try:
        assert set(cameras) == {"left_wrist", "right_wrist"}
        assert all(isinstance(camera, TacCapCamera) for camera in cameras.values())
    finally:
        for camera in cameras.values():
            camera.close()


def test_server_rejects_unknown_types_and_keys(by_id: Path, tmp_path: Path) -> None:
    bad_type = tmp_path / "bad_type.yaml"
    bad_type.write_text(yaml.safe_dump({"sensors": {"cameras": {"c": {"type": "gopro"}}}}))
    with pytest.raises(ValueError, match="unknown camera type 'gopro'"):
        camera_server._build_cameras_from_config(bad_type, by_id_root=by_id)
    bad_key = tmp_path / "bad_key.yaml"
    bad_key.write_text(
        yaml.safe_dump(
            {
                "sensors": {
                    "cameras": {
                        "c": {"type": "taccap", "camera_serial": "XCA28Z0041s", "device_id": "123"}
                    }
                }
            }  # fmt: skip
        )
    )
    with pytest.raises(ValueError, match="device_id"):
        camera_server._build_cameras_from_config(bad_key, by_id_root=by_id)
    assert FakeCamera.instances == []
