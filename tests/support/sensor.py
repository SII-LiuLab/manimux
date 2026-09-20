from __future__ import annotations

import numpy as np

from manimux.clock import Clock
from manimux.embodiments.sensor import SensorBase
from manimux.types import SensorFrame


class SensorDouble(SensorBase):
    """Deterministic RGB source for tests, with no production plugin registration."""

    def __init__(self, name: str, width: int, height: int, clock: Clock) -> None:
        self._name = name
        self._width = width
        self._height = height
        self._clock = clock
        self._sequence = 0
        self._started = False

    def start(self) -> None:
        self._started = True

    def read(self) -> SensorFrame:
        if not self._started:
            raise RuntimeError("mock camera is not started")
        self._sequence += 1
        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        frame[..., 0] = self._sequence % 255
        frame[..., 1] = np.arange(self._width, dtype=np.uint8)[None, :]
        frame[..., 2] = np.arange(self._height, dtype=np.uint8)[:, None]
        return SensorFrame(
            name=self._name,
            data=frame,
            capture_monotonic_ns=self._clock.now_ns(),
            sequence=self._sequence,
        )

    def close(self) -> None:
        self._started = False


def build_sensor(config, clock):
    return SensorDouble(config["name"], config["width"], config["height"], clock)
