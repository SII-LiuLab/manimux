"""RGB frames from one AsyncSim snapshot, with original capture metadata."""

from __future__ import annotations

from typing import Any

import numpy as np

from manimux.embodiments.sensor.base import SensorBase
from manimux.types import SensorFrame


class AsyncSimSensor(SensorBase):
    def __init__(self, client: Any, streams: dict[str, str], *, env_index: int = 0) -> None:
        self.client = client
        self.streams = dict(streams)
        self.env_index = env_index
        if not self.streams or len(set(self.streams.values())) != len(self.streams):
            raise ValueError("camera names and streams must be unique and non-empty")
        self._started = False

    def start(self) -> None:
        self._started = True

    def use_snapshot(self, snapshot: dict[str, Any]) -> dict[str, SensorFrame]:
        if not self._started:
            raise RuntimeError("sensor has not been started")
        result = {}
        anchor = snapshot["clock_anchor"]
        for name, stream in self.streams.items():
            packet = snapshot["packets"][stream]["packet"]
            if packet is None:
                continue
            if packet["episode_id"] != self.client.episode_id:
                raise RuntimeError("sensor packet belongs to another episode")
            payload = packet["payload"]
            image = payload.get(self.env_index, payload.get(str(self.env_index)))
            if image is None:
                raise ValueError(f"camera {stream!r} is missing the configured environment")
            image = np.asarray(image)
            if image.dtype != np.uint8:
                raise ValueError(f"camera {stream!r} must be RGB uint8")
            capture_ns = packet.get("wall_ts_ns") or anchor["monotonic_ns"] + round(
                (packet["capture_ts"] - anchor["sim_ts"]) * 1_000_000_000
            )
            result[name] = SensorFrame(name, image, capture_ns, packet["seq"])
        return result

    def read(self) -> dict[str, SensorFrame]:
        return self.use_snapshot(self.client.read_snapshot(list(self.streams.values())))

    def close(self) -> None:
        self._started = False
