"""Exercise the actual Gemini component with fake UVC I/O, without devices."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from manimux.embodiments.sensor import SensorBase, build_camera
from manimux.embodiments.sensor.orbbec import sensor as orbbec


@pytest.fixture
def uvc(monkeypatch):
    import cv2

    opened = []
    rgb = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)

    class Capture:
        def __init__(self, node, backend):
            self.node = node
            self.settings = {}
            self.read_once = False
            self.released = False
            self.after_first = threading.Event()
            opened.append(self)

        def isOpened(self):
            return True

        def set(self, key, value):
            self.settings[key] = value

        def read(self):
            if not self.read_once:
                self.read_once = True
                return True, rgb[:, :, ::-1].copy()
            self.after_first.set()
            return False, None

        def release(self):
            self.released = True

    discoveries = []
    monkeypatch.setattr(cv2, "VideoCapture", Capture)
    monkeypatch.setattr(
        orbbec, "discover_orbbec",
        lambda: discoveries.append(True) or [{"serial": "gemini", "node": "/dev/fake"}],
    )
    return opened, rgb, discoveries, Capture


def test_lifecycle_rgb_cached_metadata_and_restart(uvc):
    opened, rgb, discoveries, _ = uvc
    now = [1_000_000_000]
    clock = SimpleNamespace(now_ns=lambda: now[0])
    camera = build_camera("front", {
        "type": "orbbec", "device_id": "gemini", "width": 2, "height": 1, "flip": True,
    }, clock)
    assert isinstance(camera, SensorBase)
    camera.close()
    assert not opened and not discoveries
    with pytest.raises(RuntimeError, match="unavailable"):
        camera.read()
    try:
        camera.start()
        camera.start()
        assert len(opened) == 1 and opened[0].node == "/dev/fake"
        assert opened[0].after_first.wait(1)
        first = camera.read()
        now[0] += 10_000_000
        second = camera.read()
        np.testing.assert_array_equal(first.data, rgb[:, ::-1])
        assert first.name == "front"
        assert first.sequence == second.sequence == 1
        assert first.capture_monotonic_ns == second.capture_monotonic_ns == 1_000_000_000
        first.data[:] = 0
        np.testing.assert_array_equal(camera.read().data, second.data)
        now[0] += 1_000_000_000
        with pytest.raises(RuntimeError, match="stale"):
            camera.read()
    finally:
        camera.close()
    camera.close()
    assert opened[0].released
    camera.start()
    try:
        assert camera.read().sequence == 2
    finally:
        camera.close()
    assert len(opened) == 2 and opened[1].released


@pytest.mark.parametrize("failure", ["open", "settings"])
def test_failed_start_releases_device_and_can_retry(uvc, monkeypatch, failure):
    opened, _, _, capture = uvc
    camera = orbbec.OrbbecSensor(name="front", camera_serial="gemini", width=2, height=1)
    with monkeypatch.context() as patch:
        if failure == "open":
            patch.setattr(capture, "isOpened", lambda self: False)
        else:
            def fail(*args):
                raise RuntimeError("settings failed")
            patch.setattr(capture, "set", fail)
        with pytest.raises(RuntimeError):
            camera.start()
    assert opened[0].released
    camera.close()
    camera.start()
    try:
        assert camera.read().data.shape == (1, 2, 3)
    finally:
        camera.close()
