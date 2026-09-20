"""TacCap wrist camera served through the TacCap camera server.

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

from manimux.clock import Clock, SystemClock
from manimux.embodiments.sensor.base import SensorBase
from manimux.types import SensorFrame

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
        clock: Clock | None = None,
        autostart: bool = True,
    ) -> None:
        self._camera_serial = str(camera_serial)
        self._clock = clock if clock is not None else SystemClock()
        self._width, self._height, self._fps = int(width), int(height), int(fps)
        if self._width <= 0 or self._height <= 0 or self._fps <= 0:
            raise ValueError("TacCap width, height, and fps must be positive")
        self._max_frame_age_sec = float(max_frame_age_sec)
        if not np.isfinite(self._max_frame_age_sec) or self._max_frame_age_sec <= 0:
            raise ValueError("TacCap max_frame_age_sec must be positive")
        self._frame_lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._latest_color_image: np.ndarray | None = None
        self._latest_frame_timestamp: float | None = None
        self._latest_frame_index: int | None = None
        self._latest_frame_monotonic_ns: int | None = None

        self._startup_timeout_sec = float(startup_timeout_sec)
        if not np.isfinite(self._startup_timeout_sec) or self._startup_timeout_sec <= 0:
            raise ValueError("TacCap startup_timeout_sec must be finite and positive")
        self._by_id_root = by_id_root
        self._camera: Any = None
        self._started = False
        if autostart:
            self.start()

    def start(self) -> None:
        if self._started:
            return
        if self._camera is not None:
            raise RuntimeError("camera cleanup incomplete; call close before restarting")
        self.device = find_camera_device(self._camera_serial, self._by_id_root)
        taccap = importlib.import_module("xense.taccap")
        self._camera = taccap.Camera(
            str(self.device), self._width, self._height, float(self._fps), True
        )
        self._frame_ready.clear()
        try:
            self._camera.start(self._receive)
            if not self._frame_ready.wait(timeout=self._startup_timeout_sec):
                raise RuntimeError(
                    f"TacCap camera {self._camera_serial} ({self.device}) produced no frame "
                    f"within {self._startup_timeout_sec:.1f}s"
                )
            self._started = True
        except Exception as error:
            try:
                self.close()
            except Exception as cleanup_error:
                raise ExceptionGroup(
                    "camera startup and cleanup failed", [error, cleanup_error]
                ) from None
            raise

    def _receive(self, frame: Any) -> None:
        rgb = np.asarray(frame.image)[..., ::-1].copy()  # CameraFrame.image is BGR8
        with self._frame_lock:
            self._latest_color_image = rgb
            self._latest_frame_timestamp = time.time()
            self._latest_frame_index = int(frame.frame_index)
            self._latest_frame_monotonic_ns = self._clock.now_ns()
        self._frame_ready.set()

    def read(self, img_size: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        """Return the latest RGB frame; TacCap cameras have no depth stream."""

        image, depth, _timestamp = self.read_with_timestamp(img_size)
        return image, depth

    def read_with_timestamp(
        self,
        img_size: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, float]:
        """Return RGB and its host receipt timestamp from the same capture."""

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
        return image, None, timestamp

    def close(self) -> None:
        self._started = False
        camera = getattr(self, "_camera", None)
        if camera is not None:
            camera.stop()
            self._camera = None
        with self._frame_lock:
            self._latest_color_image = None
            self._latest_frame_timestamp = None
            self._latest_frame_monotonic_ns = None
            self._latest_frame_index = None
        self._frame_ready.clear()

    def read_frame(self, name: str) -> SensorFrame:
        """Runtime snapshot with paired RGB, host receipt time and SDK sequence."""
        with self._frame_lock:
            image = self._latest_color_image
            timestamp = self._latest_frame_monotonic_ns
            sequence = self._latest_frame_index
        if image is None or timestamp is None or sequence is None:
            raise RuntimeError(f"TacCap camera {self._camera_serial} frame is unavailable")
        age_ns = self._clock.now_ns() - timestamp
        if not 0 <= age_ns <= self._max_frame_age_sec * 1e9:
            raise RuntimeError(f"TacCap camera {self._camera_serial} frame is stale")
        return SensorFrame(name, image.copy(), timestamp, sequence)


class TacCapSensor(SensorBase):
    """Deferred-start runtime interface around the existing camera-server backend.

    Both use the official SDK and RGB convention. Construction only saves
    settings; start opens the V4L camera, never the gripper serial connection.
    """

    def __init__(
        self, *, name: str, camera_serial: str | None = None, clock: Clock | None = None, **options
    ):
        self.name = name
        self._serial = camera_serial
        self._clock = clock if clock is not None else SystemClock()
        self._options = dict(options)
        self._camera: TacCapCamera | None = None

    def start(self) -> None:
        # A missing binding must not select an arbitrary physical camera.
        if not self._serial:
            raise ValueError("camera_serial is required to start this sensor")
        if self._camera is None:
            self._camera = TacCapCamera(
                self._serial, clock=self._clock, autostart=False, **self._options
            )
        self._camera.start()

    def read(self) -> SensorFrame:
        if self._camera is None or not self._camera._started:
            raise RuntimeError("TacCap sensor is not started")
        return self._camera.read_frame(self.name)

    def close(self) -> None:
        if self._camera is not None:
            self._camera.close()
            self._camera = None
