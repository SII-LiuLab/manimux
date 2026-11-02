"""Shared RealSense capture implementation; network services read a background cache and collection workers capture directly."""

import threading
import time
from dataclasses import dataclass

import numpy as np

from manimux.clock import SystemClock
from manimux.embodiments.sensor.base import SensorBase
from manimux.types import SensorFrame


def get_device_ids():
    """Enumerate serial numbers without resetting or opening devices already in use."""
    import pyrealsense2 as rs

    return [d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()]


@dataclass(frozen=True)
class Capture:
    """One RGB/depth capture; timestamps are host reception times and sequence comes from the RGB frame."""

    image: np.ndarray
    depth: np.ndarray | None
    depth_scale: float | None
    unix_s: float
    monotonic_ns: int
    sequence: int
    metadata: dict | None = None
    camera_meta: dict | None = None


def read_frame_metadata(color):
    """Read supported RGB metadata; units follow librealsense rs_frame.h.

    Sensor/frame timestamps use the device clock, not Unix host time.
    TIME_OF_ARRIVAL is host Unix milliseconds (integer-truncated in SDK 2.58.3).
    Unsupported fields remain None.
    """
    import pyrealsense2 as rs

    result = {
        "sdk_timestamp_ms": float(color.get_timestamp()),
        "timestamp_domain": str(color.get_frame_timestamp_domain()),
    }
    for key, value in (
        ("actual_exposure_us", rs.frame_metadata_value.actual_exposure),
        ("frame_timestamp_us", rs.frame_metadata_value.frame_timestamp),
        ("sensor_timestamp_us", rs.frame_metadata_value.sensor_timestamp),
        ("sdk_arrival_unix_ms", rs.frame_metadata_value.time_of_arrival),
    ):
        result[key] = (
            int(color.get_frame_metadata(value)) if color.supports_frame_metadata(value) else None
        )
    return result


class RealSenseSensor(SensorBase):
    """Construction neither imports the SDK nor opens USB; start/close explicitly manage one pipeline.

    read returns the RGB SensorFrame used by runtime. capture also retains optional raw
    uint16 depth and its meters-per-unit scale for collection. Configuration explicitly selects depth alignment.
    """

    def __init__(
        self,
        *,
        name,
        camera_serial=None,
        clock=None,
        width=640,
        height=480,
        fps=30,
        enable_depth=False,
        align_depth=False,
        flip=False,
        exposure_us=None,
        white_balance=None,
        warmup_frames=0,
        max_frame_age_sec=0.30,
        read_timeout_ms=1200,
        background=True,
        collect_metadata=False,
    ):
        self.name = name
        self.camera_serial = camera_serial
        self.clock = clock if clock is not None else SystemClock()
        self.width, self.height, self.fps = width, height, fps
        self.enable_depth, self.align_depth = enable_depth, align_depth
        self.flip = flip
        self.exposure_us, self.white_balance = exposure_us, white_balance
        self.warmup_frames = warmup_frames
        self.max_frame_age_sec = max_frame_age_sec
        self.read_timeout_ms = read_timeout_ms
        self.background = background
        self.collect_metadata = collect_metadata
        self._pipeline = None
        self._alignment = None
        self._depth_scale = None
        self._lock = threading.RLock()
        self._frame_lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._stop_event = threading.Event()
        self._capture_thread = None
        self._latest_frame = None
        self._capture_error = None

    def start(self):
        with self._lock:
            if self._pipeline is not None:
                return
            import pyrealsense2 as rs

            pipeline = rs.pipeline()
            config = rs.config()
            # Local or collection station configuration supplies device identity; never select the first enumerated device.
            config.enable_device(self.camera_serial)
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
            if self.enable_depth:
                config.enable_stream(
                    rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
                )
            profile = pipeline.start(config)
            try:
                device = profile.get_device()
                self._depth_scale = (
                    device.first_depth_sensor().get_depth_scale() if self.enable_depth else None
                )
                if self.exposure_us is not None or self.white_balance is not None:
                    # Retain the collection station's D405 exposure settings; propagate SDK setting failures.
                    sensor = device.query_sensors()[0]
                    if self.exposure_us is not None:
                        sensor.set_option(rs.option.enable_auto_exposure, 0)
                        sensor.set_option(rs.option.exposure, self.exposure_us)
                    if self.white_balance is not None:
                        sensor.set_option(rs.option.enable_auto_white_balance, 0)
                        sensor.set_option(rs.option.white_balance, self.white_balance)
                self._alignment = (
                    rs.align(rs.stream.color) if self.enable_depth and self.align_depth else None
                )
                for _ in range(self.warmup_frames):
                    pipeline.wait_for_frames(timeout_ms=self.read_timeout_ms)
            except BaseException:
                # Release this pipeline if startup fails before handing it to the caller.
                pipeline.stop()
                raise
            self._pipeline = pipeline
            with self._frame_lock:
                self._latest_frame = None
                self._capture_error = None
                self._frame_ready.clear()
            self._stop_event.clear()
            if self.background:
                self._capture_thread = threading.Thread(
                    target=self._capture_loop, name=f"realsense_{self.name}", daemon=True
                )
                self._capture_thread.start()

    def capture(self):
        """Block for the next frame; do not replace capture time with the worker's read time."""
        with self._lock:
            frames = self._pipeline.wait_for_frames(timeout_ms=self.read_timeout_ms)
            if self._alignment is not None:
                frames = self._alignment.process(frames)
            color = frames.get_color_frame()
            image = np.asanyarray(color.get_data()).copy()
            depth = (
                np.asanyarray(frames.get_depth_frame().get_data()).copy()
                if self.enable_depth
                else None
            )
            unix_s, monotonic_ns = time.time(), self.clock.now_ns()
            # Keep the host receipt timestamp before optional diagnostic work.
            metadata = read_frame_metadata(color) if self.collect_metadata else None
            camera_meta = None
            if self.enable_depth and self.align_depth:
                intr = color.profile.as_video_stream_profile().get_intrinsics()
                camera_meta = {
                    "height": intr.height,
                    "width": intr.width,
                    "intrinsic_K": [
                        [intr.fx, 0.0, intr.ppx],
                        [0.0, intr.fy, intr.ppy],
                        [0.0, 0.0, 1.0],
                    ],
                    "distortion_model": str(intr.model).split(".")[-1],
                    "distortion_coeffs": list(intr.coeffs),
                    "depth_aligned_to_color": bool(self.align_depth),
                    "depth_unit": "metres",
                    "flipped": bool(self.flip),
                }
            if self.flip:
                image = image[::-1, ::-1].copy()
                if depth is not None:
                    depth = depth[::-1, ::-1].copy()
            return Capture(
                image,
                depth,
                self._depth_scale,
                unix_s,
                monotonic_ns,
                int(color.get_frame_number()),
                metadata,
                camera_meta,
            )

    def _capture_loop(self):
        while not self._stop_event.is_set():
            try:
                frame = self.capture()
            except Exception as exc:
                # Pass the original SDK exception to the reader instead of silently restarting the device.
                with self._frame_lock:
                    self._capture_error = exc
                    self._frame_ready.set()
                break
            with self._frame_lock:
                self._latest_frame = frame
                self._frame_ready.set()

    def _read_capture(self):
        if not self.background:
            return self.capture()
        # Retain first-frame waiting and stalled-frame handling; request threads do not wait for the next USB capture.
        if not self._frame_ready.wait(timeout=1.5):
            raise TimeoutError("Timed out waiting for a RealSense frame")
        with self._frame_lock:
            frame, error = self._latest_frame, self._capture_error
        if error is not None:
            raise error
        if (self.clock.now_ns() - frame.monotonic_ns) / 1e9 > self.max_frame_age_sec:
            raise RuntimeError("RealSense frame is stale; camera may be stalled")
        return frame

    def read(self):
        frame = self._read_capture()
        return SensorFrame(self.name, frame.image, frame.monotonic_ns, frame.sequence)

    def read_rgbd(self):
        frame = self._read_capture()
        if frame.depth is None or frame.depth_scale is None or not self.align_depth:
            raise RuntimeError("RGB-D snapshots require enable_depth=true and align_depth=true")
        if self.flip:
            raise ValueError("RGB-D snapshots require flip=false to retain camera intrinsics")
        return SensorFrame(
            self.name,
            frame.image,
            frame.monotonic_ns,
            frame.sequence,
            depth=frame.depth.astype(np.float64) * frame.depth_scale,
            camera_meta=dict(frame.camera_meta),
        )

    def read_capture(self):
        """Return a coherent latest RGB/depth/host-time/metadata diagnostic snapshot."""
        return self._read_capture()

    def read_with_timestamp(self):
        """RGB/depth/Unix-time interface for camera serving, sharing the same capture path as collection."""
        frame = self._read_capture()
        depth = frame.depth[..., None] if frame.depth is not None else None
        return frame.image, depth, frame.unix_s

    def close(self):
        self._stop_event.set()
        # Wait for reading to exit, then release USB under the pipeline lock to avoid a join/capture deadlock.
        if self._capture_thread is not None:
            self._capture_thread.join()
            self._capture_thread = None
        with self._lock:
            if self._pipeline is not None:
                self._pipeline.stop()
                self._pipeline = None
