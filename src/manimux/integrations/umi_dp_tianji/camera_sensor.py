"""Preserve capture timestamps from the existing camera-server PUB protocol."""

from __future__ import annotations

import time

import numpy as np

from manimux.server.sensor.taccap.client import CameraSubscriber
from manimux.types import SensorFrame


class TimestampedCameraSensor:
    def __init__(self, config, clock):
        self.clock = clock
        self.endpoint = str(config["options"].get("endpoint", "tcp://127.0.0.1:5556"))
        self.names = tuple(config["options"].get("camera_names", ("left_wrist", "right_wrist")))
        self.output_names = config["options"].get("output_names", {})
        self.max_age_ns = int(float(config["options"].get("max_frame_age_sec", 0.15)) * 1e9)
        self.jump_ns = int(float(config["options"].get("clock_jump_tolerance_s", 0.02)) * 1e9)
        self.startup_ns = int(float(config["options"].get("startup_timeout_s", 5)) * 1e9)
        if not self.names or min(self.max_age_ns, self.jump_ns, self.startup_ns) <= 0:
            raise ValueError("Camera names and positive timing bounds are required")
        self.client = None
        self.frames = {}

    def start(self):
        self.client = CameraSubscriber(self.endpoint)
        self.started_ns = self.clock.now_ns()
        self.offset_ns = self.started_ns - time.time_ns()
        self.frames = {}

    def read(self):
        if self.client is None:
            raise RuntimeError("Timestamped camera sensor is not started")
        now = self.clock.now_ns()
        if abs((now - time.time_ns()) - self.offset_ns) > self.jump_ns:
            raise RuntimeError(
                "Wall clock jumped relative to monotonic time; restart camera history"
            )
        bundle = self.client.try_recv_bundle()
        if bundle is not None:
            timestamps = bundle.get("timestamps", {})
            images = bundle.get("frames", {})
            for name in self.names:
                timestamp = float(timestamps.get(name, 0))
                if name not in images or not np.isfinite(timestamp) or timestamp <= 0:
                    raise ValueError(f"Camera {name} requires an image and capture timestamp")
                sequence = round(timestamp * 1e9)
                capture_ns = sequence + self.offset_ns
                old = self.frames.get(name)
                if old is not None and sequence < old.sequence:
                    raise RuntimeError(f"Camera timestamp moved backwards: {name}")
                if old is None or sequence != old.sequence:
                    self.frames[name] = SensorFrame(
                        self.output_names.get(name, name), images[name], capture_ns, sequence
                    )
        if not self.frames:
            if now - self.started_ns > self.startup_ns:
                raise RuntimeError("Camera PUB stream did not produce timestamped frames")
            return {}
        for name, frame in self.frames.items():
            age = now - frame.capture_monotonic_ns
            if age < -self.jump_ns or age > self.max_age_ns:
                raise RuntimeError(f"Camera {name} capture timestamp is stale or in the future")
        # A cached frame retains object identity, RGB, capture time and sequence.
        # Logical names were assigned once, when the source frame arrived.
        return {frame.name: frame for frame in self.frames.values()}

    def close(self):
        if self.client is not None:
            self.client.close()
            self.client = None
        self.frames.clear()


def build_sensor(config, clock):
    return TimestampedCameraSensor(config, clock)
