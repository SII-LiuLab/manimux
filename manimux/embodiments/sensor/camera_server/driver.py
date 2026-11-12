"""Camera-server sensor plugin used by the generic ManiMux runtime."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

from manimux.clock import Clock
from manimux.embodiments.sensor import SensorBase
from manimux.embodiments.sensor.camera_server.client import CameraClient
from manimux.types import SensorFrame


class CameraServerSensorDriver(SensorBase):
    """Read one coherent multi-camera bundle with one ZMQ request."""

    def __init__(self, config: dict, clock: Clock) -> None:
        endpoint = config["options"].get("endpoint", "tcp://127.0.0.1:5555")
        camera_names = config["options"].get(
            "camera_names",
            ["d405_left", "d405_front", "d405_right"],
        )
        if not isinstance(endpoint, str) or not endpoint:
            raise ValueError("sensor.options.endpoint must be a non-empty string")
        if (
            not isinstance(camera_names, Sequence)
            or isinstance(camera_names, str)
            or not all(isinstance(name, str) for name in camera_names)
        ):
            raise ValueError("sensor.options.camera_names must be a list of strings")
        self._endpoint = endpoint
        self._camera_names = tuple(camera_names)
        self._request_timeout_ms = int(config["options"].get("request_timeout_ms", 500))
        max_age = config["options"].get("max_frame_age_sec", 0.5)
        self._max_frame_age_sec = None if max_age is None else float(max_age)
        self._clock = clock
        self._client: CameraClient | None = None
        self._frames: dict[str, SensorFrame] = {}
        self._unix_offset_ns = 0
        tolerance = float(config["options"].get("clock_jump_tolerance_s", 0.02))
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("clock_jump_tolerance_s must be positive and finite")
        self._clock_jump_ns = round(tolerance * 1e9)

    def start(self) -> None:
        if self._client is not None:
            return
        client = CameraClient(
            self._endpoint,
            request_timeout_ms=self._request_timeout_ms,
            max_frame_age_sec=self._max_frame_age_sec,
        )
        try:
            if not client.ping():
                raise RuntimeError(f"camera server did not answer ping at {self._endpoint}")
        except BaseException:
            client.close()
            raise
        self._client = client
        self._unix_offset_ns = self._clock.now_ns() - time.time_ns()
        self._frames = {}

    def read(self) -> dict[str, SensorFrame]:
        if self._client is None:
            raise RuntimeError("camera-server sensor is not started")
        bundle = self._client.get_bundle(camera_names=list(self._camera_names))
        images, timestamps = bundle["frames"], bundle["timestamps"]
        missing = [name for name in self._camera_names if name not in images]
        if missing:
            raise RuntimeError(f"camera server response is missing cameras: {missing}")
        offset_now = self._clock.now_ns() - time.time_ns()
        if abs(offset_now - self._unix_offset_ns) > self._clock_jump_ns:
            raise RuntimeError("Wall clock changed relative to sensor clock; restart camera reader")
        frames = {}
        for name in self._camera_names:
            # The existing wire protocol supplies Unix capture time, not a frame
            # counter. Use that stable timestamp as the source-frame identity.
            sequence = round(timestamps[name] * 1e9)
            previous = self._frames.get(name)
            if previous is not None and sequence < previous.sequence:
                raise RuntimeError(f"Camera timestamp moved backwards: {name}")
            frames[name] = (
                previous if previous is not None and sequence == previous.sequence
                else SensorFrame(name, images[name], sequence + self._unix_offset_ns, sequence)
            )
        self._frames = frames
        return dict(frames)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        self._frames = {}


def build_sensor(config: dict, clock: Clock) -> CameraServerSensorDriver:
    return CameraServerSensorDriver(config, clock)
