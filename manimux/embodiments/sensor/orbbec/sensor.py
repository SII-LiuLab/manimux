"""Gemini 305/335 UVC RGB capture, selected by USB serial and interface 04."""

from __future__ import annotations

import threading
import time


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


class OrbbecCamera:
    def __init__(self, device_id: str, *, width: int = 640, height: int = 480,
                 fps: int = 30, max_frame_age_sec: float = 0.30,
                 flip: bool = False, enable_depth: bool = False) -> None:
        import cv2

        if enable_depth:
            raise ValueError("Gemini UVC driver supports RGB only")
        if min(width, height, fps, max_frame_age_sec) <= 0:
            raise ValueError("Camera dimensions, FPS and maximum age must be positive")
        matches = [camera for camera in discover_orbbec() if camera["serial"] == device_id]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one Gemini RGB interface for {device_id}, got {matches}")
        self._max_age = max_frame_age_sec
        self._flip = flip
        self._shape = (height, width, 3)
        self._latest_frame_timestamp = 0.0
        self._image = None
        self._error = "No captured frame"
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._cap = cv2.VideoCapture(matches[0]["node"], cv2.CAP_V4L2)
        self._thread = None
        try:
            if not self._cap.isOpened():
                raise RuntimeError(f"Could not open Gemini {device_id}")
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._cap.set(cv2.CAP_PROP_FPS, fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            self._thread = threading.Thread(
                target=self._capture, name=f"gemini-{device_id}", daemon=True
            )
            self._thread.start()
            if not self._ready.wait(5.0):
                raise RuntimeError(f"Gemini {device_id} produced no frame: {self._error}")
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
                self._image = rgb
                self._latest_frame_timestamp = time.time()
                self._error = ""
            self._ready.set()

    def read(self):
        with self._lock:
            age = time.time() - self._latest_frame_timestamp
            if self._image is None or age > self._max_age:
                raise RuntimeError(f"Gemini frame unavailable/stale ({age:.3f}s): {self._error}")
            return self._image.copy(), None

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._cap.release()
