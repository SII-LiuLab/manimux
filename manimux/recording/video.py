from __future__ import annotations

import json
import queue
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from manimux.types import SensorFrame, UInt8Array


@dataclass(frozen=True, slots=True)
class VideoRecordingSummary:
    enabled: bool
    frames_written: int
    dropped_bundles: int
    error: str | None


@dataclass(frozen=True, slots=True)
class _QueuedFrame:
    camera_name: str
    image: UInt8Array
    capture_monotonic_ns: int


class AsyncVideoRecorder:
    """Best-effort per-camera MP4 recording outside the control thread."""

    def __init__(
        self,
        episode_dir: Path,
        *,
        fps: float,
        codec: str,
        queue_size: int,
    ) -> None:
        self._enabled = fps > 0
        self._video_dir = episode_dir / "videos"
        self._fps = fps
        self._codec = codec
        self._period_ns = 0 if not self._enabled else round(1_000_000_000 / fps)
        self._next_sample_ns: int | None = None
        self._queue: queue.Queue[list[_QueuedFrame] | None] = queue.Queue(maxsize=queue_size)
        self._dropped_bundles = 0
        self._frames_written = 0
        self._timestamps: dict[str, list[int]] = {}
        self._error: str | None = None
        self._thread: threading.Thread | None = None
        if self._enabled:
            self._video_dir.mkdir(parents=True, exist_ok=True)

    def _start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="manimux-video-recorder",
            daemon=True,
        )
        self._thread.start()

    @staticmethod
    def _safe_camera_name(name: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-.")
        return normalized or "camera"

    def submit(self, frames: dict[str, SensorFrame]) -> None:
        if not self._enabled or not frames:
            return
        self._start()
        capture_ns = min(frame.capture_monotonic_ns for frame in frames.values())
        if self._next_sample_ns is not None and capture_ns < self._next_sample_ns:
            return
        if self._next_sample_ns is None:
            self._next_sample_ns = capture_ns
        while self._next_sample_ns <= capture_ns:
            self._next_sample_ns += self._period_ns
        bundle = [
            _QueuedFrame(
                camera_name=name,
                image=np.asarray(frame.data, dtype=np.uint8).copy(),
                capture_monotonic_ns=frame.capture_monotonic_ns,
            )
            for name, frame in sorted(frames.items())
        ]
        try:
            self._queue.put_nowait(bundle)
        except queue.Full:
            self._dropped_bundles += 1

    def _run(self) -> None:
        writers: dict[str, Any] = {}
        try:
            import cv2

            cv2_module: Any = cv2
            video_writer_fourcc = cv2_module.VideoWriter_fourcc
            while True:
                bundle = self._queue.get()
                if bundle is None:
                    break
                for frame in bundle:
                    height, width = frame.image.shape[:2]
                    writer = writers.get(frame.camera_name)
                    if writer is None:
                        path = self._video_dir / f"{self._safe_camera_name(frame.camera_name)}.mp4"
                        writer = cv2.VideoWriter(
                            str(path),
                            video_writer_fourcc(*self._codec),
                            self._fps,
                            (width, height),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"video writer did not open for {path}")
                        writers[frame.camera_name] = writer
                    writer.write(cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR))
                    self._timestamps.setdefault(frame.camera_name, []).append(
                        frame.capture_monotonic_ns
                    )
                    self._frames_written += 1
        except Exception as exc:  # noqa: BLE001 - recording failure must not stop robot control
            self._error = f"{type(exc).__name__}: {exc}"
        finally:
            for writer in writers.values():
                writer.release()

    def close(self) -> VideoRecordingSummary:
        if not self._enabled:
            return VideoRecordingSummary(False, 0, 0, None)
        if self._thread is not None:
            while self._thread.is_alive():
                try:
                    self._queue.put(None, timeout=0.1)
                    break
                except queue.Full:
                    continue
            self._thread.join()
        index = {
            "schema": "manimux-video-v1",
            "fps": self._fps,
            "codec": self._codec,
            "frames_written": self._frames_written,
            "dropped_bundles": self._dropped_bundles,
            "error": self._error,
            "cameras": {
                name: {
                    "file": f"{self._safe_camera_name(name)}.mp4",
                    "capture_monotonic_ns": timestamps,
                }
                for name, timestamps in sorted(self._timestamps.items())
            },
        }
        with (self._video_dir / "index.json").open("w", encoding="utf-8") as handle:
            json.dump(index, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return VideoRecordingSummary(
            True,
            self._frames_written,
            self._dropped_bundles,
            self._error,
        )
