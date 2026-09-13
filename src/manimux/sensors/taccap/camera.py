"""TacCap wrist camera, served through the camera server like a RealSense camera.

The camera is a UVC device (IMX385 sensor) streaming MJPG. It is found by its
serial number under ``/dev/v4l/by-id`` and captured with the TacCap SDK's C++
``Camera``, whose callback delivers BGR frames; they are stored as RGB, as in
CalibWrist's ``TacCapCameraPair``. Only the V4L device is opened, never the
gripper's serial link, so the camera server and the robot driver can run side
by side. Timestamps are host receipt times, not exposure times.
"""

from __future__ import annotations

import importlib
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

V4L_BY_ID = Path("/dev/v4l/by-id")


def find_camera_device(camera_serial: str, by_id_root: Path | str = V4L_BY_ID) -> Path:
    """Resolve the capture node of the TacCap camera with ``camera_serial``."""

    root = Path(by_id_root)
    matches = sorted(root.glob(f"*imx385*{camera_serial}*video-index0"))
    if len(matches) != 1:
        available = sorted(path.name for path in root.glob("*imx385*video-index0"))
        raise RuntimeError(
            f"expected one TacCap camera with serial {camera_serial!r} under {root}, "
            f"found {len(matches)}; connected TacCap cameras: {available or 'none'}"
        )
    return matches[0]


class TacCapCamera:
    def __repr__(self) -> str:
        return (
            f"TacCapCamera(serial={self._camera_serial}, "
            f"color={self._width}x{self._height}@{self._fps})"
        )

    def __init__(
        self,
        camera_serial: str,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        max_frame_age_sec: float = 0.30,
        startup_timeout_sec: float = 3.0,
        by_id_root: Path | str = V4L_BY_ID,
    ) -> None:
        self._camera_serial = str(camera_serial)
        self._width, self._height, self._fps = int(width), int(height), int(fps)
        if self._width <= 0 or self._height <= 0 or self._fps <= 0:
            raise ValueError("TacCap width, height, and fps must be positive")
        self._max_frame_age_sec = float(max_frame_age_sec)
        if self._max_frame_age_sec <= 0:
            raise ValueError("TacCap max_frame_age_sec must be positive")
        self._frame_lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._latest_color_image: np.ndarray | None = None
        self._latest_frame_timestamp: float | None = None
        self._latest_frame_index: int | None = None

        self.device = find_camera_device(self._camera_serial, by_id_root)
        taccap = importlib.import_module("xense.taccap")
        self._camera: Any = taccap.Camera(
            str(self.device), self._width, self._height, float(self._fps), True
        )
        self._camera.start(self._receive)
        if not self._frame_ready.wait(timeout=float(startup_timeout_sec)):
            self.close()
            raise RuntimeError(
                f"TacCap camera {self._camera_serial} ({self.device}) produced no frame "
                f"within {startup_timeout_sec:.1f}s"
            )

    def _receive(self, frame: Any) -> None:
        rgb = np.asarray(frame.image)[..., ::-1].copy()  # CameraFrame.image is BGR8
        with self._frame_lock:
            self._latest_color_image = rgb
            self._latest_frame_timestamp = time.time()
            self._latest_frame_index = int(frame.frame_index)
        self._frame_ready.set()

    def read(self, img_size: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        """Return the latest RGB frame; TacCap cameras have no depth stream."""

        if img_size is not None:
            raise ValueError("TacCapCamera serves frames at the configured size only")
        with self._frame_lock:
            image, timestamp = self._latest_color_image, self._latest_frame_timestamp
        if image is None or timestamp is None:
            raise RuntimeError(f"TacCap camera {self._camera_serial} frame is unavailable.")
        age = time.time() - timestamp
        if age > self._max_frame_age_sec:
            raise RuntimeError(
                f"TacCap camera {self._camera_serial} frame is stale ({age:.3f}s old); "
                "camera may be stalled."
            )
        return image, None

    def close(self) -> None:
        camera, self._camera = getattr(self, "_camera", None), None
        if camera is not None:
            camera.stop()
