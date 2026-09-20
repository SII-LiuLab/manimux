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

from manimux.embodiments.sensor.taccap import TacCapCamera, find_camera_device
from manimux.server.sensor.taccap import server as camera_server


def test_sensor_defers_opening_and_preserves_capture_metadata(by_id):
    from manimux.embodiments.sensor.base import SensorBase
    from manimux.embodiments.sensor.taccap import TacCapSensor

    sensor = TacCapSensor(name="wrist", camera_serial="XCA28Z0041s", by_id_root=by_id)
    assert isinstance(sensor, SensorBase)
    assert not FakeCamera.instances
    with pytest.raises(RuntimeError, match="not started"):
        sensor.read()
    try:
        sensor.start()
        sensor.start()
        assert len(FakeCamera.instances) == 1
        FakeCamera.instances[0].stop()
        first = sensor.read()
        second = sensor.read()
        assert first.name == "wrist"
        assert first.capture_monotonic_ns == second.capture_monotonic_ns
        assert first.sequence == second.sequence
        assert first.data[0, 0].tolist() == [0, 0, 255]
        first.data[:] = 0
        assert sensor.read().data[0, 0].tolist() == [0, 0, 255]
    finally:
        sensor.close()
        sensor.close()


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


def test_sensor_retains_camera_when_start_and_cleanup_fail(by_id, monkeypatch):
    from manimux.embodiments.sensor.taccap import TacCapSensor

    def fail_start(self, callback):
        raise RuntimeError("start failed")

    def fail_stop(self):
        raise RuntimeError("stop failed")

    real_start, real_stop = FakeCamera.start, FakeCamera.stop
    monkeypatch.setattr(FakeCamera, "start", fail_start)
    monkeypatch.setattr(FakeCamera, "stop", fail_stop)
    sensor = TacCapSensor(name="wrist", camera_serial="XCA28Z0041s", by_id_root=by_id)
    with pytest.raises(ExceptionGroup, match="startup and cleanup failed"):
        sensor.start()
    assert sensor._camera._camera is FakeCamera.instances[0]
    with pytest.raises(RuntimeError, match="cleanup incomplete"):
        sensor.start()
    with pytest.raises(RuntimeError, match="not started"):
        sensor.read()
    monkeypatch.setattr(FakeCamera, "stop", real_stop)
    sensor.close()
    assert sensor._camera is None
    monkeypatch.setattr(FakeCamera, "start", real_start)
    try:
        sensor.start()
        assert sensor.read().sequence >= 0
    finally:
        sensor.close()


def test_sensor_close_failure_can_be_retried(by_id, monkeypatch):
    from manimux.embodiments.sensor.taccap import TacCapSensor

    sensor = TacCapSensor(name="wrist", camera_serial="XCA28Z0041s", by_id_root=by_id)
    sensor.start()
    camera = FakeCamera.instances[0]
    real_stop = camera.stop

    def fail_stop():
        raise RuntimeError("stop failed")

    monkeypatch.setattr(camera, "stop", fail_stop)
    with pytest.raises(RuntimeError, match="stop failed"):
        sensor.close()
    assert sensor._camera._camera is camera
    monkeypatch.setattr(camera, "stop", real_stop)
    sensor.close()
    assert camera.stopped and sensor._camera is None


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


def test_server_preserves_frame_timestamp_when_capture_advances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from manimux.embodiments.sensor.taccap import sensor as taccap_module

    camera = object.__new__(TacCapCamera)
    camera._camera_serial = "test"
    camera._max_frame_age_sec = 2.0
    camera._latest_color_image = np.zeros((2, 2, 3), dtype=np.uint8)
    camera._latest_frame_timestamp = 10.0
    original = camera._latest_color_image
    class AdvancingCapture:
        def __enter__(self) -> AdvancingCapture:
            return self

        def __exit__(self, *args: object) -> None:
            # Capture completes immediately after the reader releases its lock.
            camera._latest_color_image = np.ones((2, 2, 3), dtype=np.uint8)
            camera._latest_frame_timestamp = 11.0

    camera._frame_lock = AdvancingCapture()
    monkeypatch.setattr(taccap_module.time, "time", lambda: 11.0)
    snapshot = camera_server.CameraServer({"wrist": camera})._snapshot()
    np.testing.assert_array_equal(snapshot["frames"]["wrist"], original)
    assert snapshot["timestamps"]["wrist"] == 10.0
    assert camera._latest_frame_timestamp == 11.0


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
