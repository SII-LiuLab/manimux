"""RGB-D reader for the shared camera server; never opens a camera device."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from manimux.clock import Clock
from manimux.embodiments.sensor.base import SensorBase
from manimux.embodiments.sensor.camera_server.client import CameraClient
from manimux.types import SensorFrame


class RGBDSensor(SensorBase):
    def __init__(self, config: dict[str, Any], clock: Clock) -> None:
        self.options = config["options"]
        self.clock = clock
        self.client: CameraClient | None = None

    def start(self) -> None:
        self.client = CameraClient(
            self.options["endpoint"],
            request_timeout_ms=self.options.get("request_timeout_ms", 500),
        )

    def read(self) -> dict[str, SensorFrame]:
        assert self.client is not None
        names = self.options["camera_names"]
        bundle = self.client.get_rgbd_bundle(names)
        offset = time.time() - self.clock.now_ns() / 1e9
        result = {}
        for name in names:
            rgb = bundle["frames"][name]
            depth = bundle["depths"][name]
            meta = bundle["camera_meta"][name]
            if depth is None or not isinstance(meta, dict):
                raise ValueError(f"{name}: camera must provide real aligned depth and metadata")
            timestamp = float(bundle["timestamps"][name])
            age = time.time() - timestamp
            if not np.isfinite(age) or not -0.05 <= age <= self.options.get(
                "max_frame_age_sec", 0.5
            ):
                raise ValueError(f"{name}: RGB-D capture timestamp is stale or invalid")
            result[name] = SensorFrame(
                name,
                rgb,
                int((timestamp - offset) * 1e9),
                bundle["sequences"][name],
                depth=depth,
                camera_meta=meta,
            )
        return result

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None


def build_sensor(config: dict[str, Any], clock: Clock) -> RGBDSensor:
    return RGBDSensor(config, clock)
