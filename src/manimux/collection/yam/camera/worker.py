"""CameraWorker: run a CameraDriver in a background thread."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from .interface import CameraDriver, CameraFrame, CameraMode


class CameraWorker:
    """Wrap a ``CameraDriver`` so ``read()`` returns the latest frame without
    blocking. Exposes the driver's name/role/mode/image_keys so the recorder and
    GUI treat it like a driver.

    An optional ``on_frame`` callback is invoked with ``(name, frame)`` on every
    capture, so previews update at the camera's full frame rate independently of
    the control loop (the loop can be stopped and previews still stream)."""

    def __init__(
        self,
        driver: CameraDriver,
        on_frame: Callable[[str, CameraFrame], None] | None = None,
    ):
        self._driver = driver
        self._on_frame = on_frame
        self.name: str = driver.name
        self.role: str = driver.role
        self.mode: CameraMode = driver.mode
        self._latest: CameraFrame | None = None
        self._lock = threading.Lock()
        self._first = threading.Event()
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_error: str | None = None
        self._last_frame_at: float | None = None
        self._closed = False
        self._frame_listeners: list[Callable[[str, CameraFrame], None]] = []

    def add_frame_listener(self, callback: Callable[[str, CameraFrame], None]) -> None:
        with self._lock:
            if callback not in self._frame_listeners:
                self._frame_listeners.append(callback)

    def remove_frame_listener(self, callback: Callable[[str, CameraFrame], None]) -> None:
        with self._lock:
            if callback in self._frame_listeners:
                self._frame_listeners.remove(callback)

    def image_keys(self) -> list[str]:
        return self._driver.image_keys()

    def start(self, warmup_timeout: float = 5.0) -> None:
        """Start capturing and block until the first frame arrives, so callers are
        guaranteed a valid frame afterwards. Raises if none arrives in time."""
        if self._running:
            return
        if self._closed or (self._thread is not None and self._thread.is_alive()):
            raise RuntimeError(f"camera {self.name!r} must finish closing before reopening")
        self._first.clear()
        self._last_error = None
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._first.wait(warmup_timeout):
            self.stop()
            detail = f"; last capture error: {self._last_error}" if self._last_error else ""
            raise RuntimeError(
                f"camera {self.name!r} produced no frames within {warmup_timeout}s{detail}"
            )

    def _run(self) -> None:
        while self._running:
            try:
                frame = self._driver.read()
            except Exception as exc:
                # Keep the last good frame; a transient read error shouldn't kill
                # capture. A camera that never produces trips the warmup timeout.
                with self._lock:
                    self._last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(0.01)
                continue
            if not self._running:
                break
            with self._lock:
                self._latest = frame
                self._last_frame_at = time.monotonic()
                self._last_error = None
                listeners = tuple(self._frame_listeners)
            for listener in listeners:
                try:
                    listener(self.name, frame)
                except Exception:
                    logging.exception("camera %r recording callback failed", self.name)
            if self._on_frame is not None:
                try:
                    self._on_frame(self.name, frame)
                except Exception:
                    logging.exception("camera %r preview callback failed", self.name)
            self._first.set()

    def read(self) -> CameraFrame | None:
        """Latest captured frame (never None once started + warmed up)."""
        with self._lock:
            return self._latest

    def preview_error(self, max_age_s: float = 2.0) -> str | None:
        """Preview health from capture progress, without probing USB or changing reads."""
        with self._lock:
            if not self._running:
                return "capture stopped"
            if self._last_frame_at is None:
                return self._last_error or "waiting for the first frame"
            if time.monotonic() - self._last_frame_at > max_age_s:
                return self._last_error or f"no new frame for more than {max_age_s:g}s"
        return None

    def stop(self, join_timeout: float = 6.0) -> bool:
        # Join long enough for an in-flight blocking read to return (RealSense
        # wait_for_frames can take ~5s) before stopping the driver — overlapping
        # driver.stop() with a live read can crash librealsense. If the reader is
        # still stuck after that, skip driver.stop() and leak rather than crash.
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)
            if self._thread is not None and self._thread.is_alive():
                logging.warning(
                    "camera %r reader still running after %.0fs; skipping driver.stop()",
                    self.name, join_timeout,
                )
                # Retain the handle: a later retry must not close the driver or
                # open another capture while this read is still in flight.
                return False
            self._thread = None
        if not self._closed:
            self._driver.stop()
            self._closed = True
        return True
