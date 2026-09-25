"""Gemini 305/335 UVC RGB capture, selected by USB serial and interface 04."""

from __future__ import annotations

import threading

from manimux.clock import Clock, SystemClock
from manimux.embodiments.sensor.base import SensorBase
from manimux.types import SensorFrame


def discover_orbbec() -> list[dict[str, str]]:
    import pyudev

    cameras = []
    for dev in pyudev.Context().list_devices(subsystem="video4linux"):
        if (dev.get("ID_VENDOR_ID", "").lower() == "2bc5"
                and dev.get("ID_MODEL_ID", "").lower() in {"0840", "0800"}
                and dev.get("ID_USB_INTERFACE_NUM") == "04"
                and ":capture:" in dev.get("ID_V4L_CAPABILITIES", "")
                and dev.get("ID_SERIAL_SHORT") and dev.device_node):
            cameras.append({"serial": dev["ID_SERIAL_SHORT"], "node": dev.device_node})
    return sorted(cameras, key=lambda camera: camera["serial"])


class OrbbecSensor(SensorBase):
    def __init__(self, *, name: str, camera_serial: str | None = None,
                 clock: Clock | None = None, width: int = 640, height: int = 480,
                 fps: int = 30, max_frame_age_sec: float = 0.30,
                 flip: bool = False, enable_depth: bool = False) -> None:
        if enable_depth:
            raise ValueError("Gemini UVC driver supports RGB only")
        if min(width, height, fps, max_frame_age_sec) <= 0:
            raise ValueError("Camera dimensions, FPS and maximum age must be positive")
        self.name = name
        self.camera_serial = camera_serial
        self.clock = clock or SystemClock()
        self._fps = fps
        self._max_age = max_frame_age_sec
        self._flip = flip
        self._shape = (height, width, 3)
        self._frame: SensorFrame | None = None
        self._sequence = 0
        self._error = "No captured frame"
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._cap = None
        self._thread = None

    def start(self) -> None:
        if self._cap is not None:
            return
        import cv2

        if not self.camera_serial:
            raise ValueError(f"Camera {self.name!r} needs camera_serial")
        matches = [camera for camera in discover_orbbec()
                   if camera["serial"] == self.camera_serial]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one Gemini RGB interface for {self.camera_serial}, got {matches}"
            )
        self._stop.clear()
        self._ready.clear()
        self._frame = None
        self._cap = cv2.VideoCapture(matches[0]["node"], cv2.CAP_V4L2)
        try:
            if not self._cap.isOpened():
                raise RuntimeError(f"Could not open Gemini {self.camera_serial}")
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._shape[1])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._shape[0])
            self._cap.set(cv2.CAP_PROP_FPS, self._fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            self._thread = threading.Thread(
                target=self._capture, name=f"gemini-{self.camera_serial}", daemon=True
            )
            self._thread.start()
            if not self._ready.wait(5.0):
                raise RuntimeError(f"Gemini {self.camera_serial} produced no frame: {self._error}")
            self.read()
        except BaseException:
            self.close()
            raise

    def _capture(self) -> None:
        import cv2

        while not self._stop.is_set():
            ok, bgr = self._cap.read()
            if not ok or bgr is None or bgr.shape != self._shape:
                self._error = "Capture failed or image dimensions differ from configuration"
                self._stop.wait(0.02)
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if self._flip:
                rgb = cv2.rotate(rgb, cv2.ROTATE_180)
            with self._lock:
                self._sequence += 1
                self._frame = SensorFrame(self.name, rgb, self.clock.now_ns(), self._sequence)
                self._error = ""
            self._ready.set()

    def read(self) -> SensorFrame:
        with self._lock:
            if self._frame is None:
                raise RuntimeError(f"Gemini frame unavailable: {self._error}")
            frame = self._frame
            age = (self.clock.now_ns() - frame.capture_monotonic_ns) / 1e9
            if age > self._max_age:
                raise RuntimeError(f"Gemini frame unavailable/stale ({age:.3f}s): {self._error}")
            return SensorFrame(self.name, frame.data.copy(), frame.capture_monotonic_ns, frame.sequence)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                raise RuntimeError(f"Gemini {self.camera_serial} capture thread did not stop")
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._frame = None
