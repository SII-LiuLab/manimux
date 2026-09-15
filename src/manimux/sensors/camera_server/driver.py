"""Camera-server sensor plugin used by the generic ManiMux runtime."""

from __future__ import annotations

from collections.abc import Sequence

from manimux.clock import Clock
from manimux.config import SensorConfig
from manimux.sensors.camera_server.client import CameraClient
from manimux.types import SensorFrame


class CameraServerSensorDriver:
    """Read one coherent multi-camera bundle with one ZMQ request."""

    def __init__(self, config: SensorConfig, clock: Clock) -> None:
        endpoint = config.options.get("endpoint", "tcp://127.0.0.1:5555")
        camera_names = config.options.get(
            "camera_names",
            ["left_camera", "front_camera", "right_camera"],
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
        self._request_timeout_ms = int(config.options.get("request_timeout_ms", 500))
        max_age = config.options.get("max_frame_age_sec", 0.5)
        self._max_frame_age_sec = None if max_age is None else float(max_age)
        self._clock = clock
        self._client: CameraClient | None = None
        self._sequence = 0

    def start(self) -> None:
        if self._client is not None:
            return
        client = CameraClient(
            self._endpoint,
            request_timeout_ms=self._request_timeout_ms,
            max_frame_age_sec=self._max_frame_age_sec,
        )
        if not client.ping():
            client.close()
            raise RuntimeError(f"camera server did not answer ping at {self._endpoint}")
        self._client = client

    def read(self) -> dict[str, SensorFrame]:
        if self._client is None:
            raise RuntimeError("camera-server sensor is not started")
        images = self._client.get_obs(camera_names=list(self._camera_names))
        missing = [name for name in self._camera_names if name not in images]
        if missing:
            raise RuntimeError(f"camera server response is missing cameras: {missing}")
        self._sequence += 1
        received_ns = self._clock.now_ns()
        return {
            name: SensorFrame(
                name=name,
                data=images[name],
                capture_monotonic_ns=received_ns,
                sequence=self._sequence,
            )
            for name in self._camera_names
        }

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


def build_sensor(config: SensorConfig, clock: Clock) -> CameraServerSensorDriver:
    return CameraServerSensorDriver(config, clock)
