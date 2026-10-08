"""Device-independent, single-owner polling of runtime sensor sources."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Sequence

from manimux.clock import Clock
from manimux.embodiments.sensor.base import SensorBase
from manimux.types import SensorFrame


def sensor_reading_parameters(**options) -> dict:
    values = {
        "mode": "inline",
        "poll_hz": 100.0,
        "max_frame_age_s": 0.5,
        "startup_timeout_s": 5.0,
        "shutdown_timeout_s": 2.0,
    }
    unknown = options.keys() - values.keys()
    if unknown:
        raise ValueError(f"Unknown sensor_reading fields: {sorted(unknown)}")
    values.update(options)
    if values["mode"] not in {"inline", "background"}:
        raise ValueError("sensor_reading.mode must be inline or background")
    for name in values.keys() - {"mode"}:
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"sensor_reading.{name} must be a positive finite number")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"sensor_reading.{name} must be a positive finite number")
    return values


class SensorReader:
    """Read complete RGB bundles inline or from one background polling thread.

    The polling thread owns start/read/close, including transport sockets. Only
    complete, independently owned snapshots cross the lock. It never touches the
    robot, policy, timeline or scheduler clock. Sources must bound blocking I/O.
    """

    def __init__(self, sensors: Sequence[SensorBase], clock: Clock, **options) -> None:
        self._sensors = list(sensors)
        self._clock = clock
        self._options = sensor_reading_parameters(**options)
        self.mode = self._options["mode"]
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._owned: list[SensorBase] = []
        self._frames: dict[str, SensorFrame] = {}
        self._error: BaseException | None = None
        self._cleanup_error: BaseException | None = None

    def start(self) -> None:
        if self._started:
            raise RuntimeError("Sensor reader is already started")
        self._started = True
        self._stop.clear()
        self._ready.clear()
        self._frames = {}
        self._error = None
        self._cleanup_error = None
        if not self._sensors:
            return
        if self.mode == "inline":
            for sensor in self._sensors:
                self._owned.append(sensor)
                sensor.start()
                sensor.read()
            return
        self._thread = threading.Thread(
            target=self._run, name="manimux-sensor-reader", daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(self._options["startup_timeout_s"]):
            self._stop.set()
            raise TimeoutError("Sensor reader did not produce its first complete snapshot")
        self.read()  # Surface startup errors before connecting the robot.

    def _read_sources(self) -> dict[str, SensorFrame]:
        frames: dict[str, SensorFrame] = {}
        incomplete = False
        for sensor in self._sensors:
            reading = sensor.read()
            batch = {reading.name: reading} if isinstance(reading, SensorFrame) else reading
            if not isinstance(batch, dict):
                raise TypeError("Sensor.read() must return a SensorFrame or named frame bundle")
            incomplete |= not batch
            overlap = frames.keys() & batch.keys()
            if overlap:
                raise RuntimeError(f"duplicate sensor frames: {sorted(overlap)}")
            for name, frame in batch.items():
                if not isinstance(frame, SensorFrame) or name != frame.name:
                    raise ValueError("Sensor bundle keys must match SensorFrame names")
            frames.update(batch)
        return {} if incomplete else frames

    def _run(self) -> None:
        try:
            for sensor in self._sensors:
                if self._stop.is_set():
                    return
                self._owned.append(sensor)
                sensor.start()
            period = 1.0 / self._options["poll_hz"]
            previous: dict[str, SensorFrame] = {}
            while not self._stop.is_set():
                started = time.monotonic()
                frames = self._read_sources()
                if previous and frames.keys() != previous.keys():
                    raise RuntimeError("Sensor snapshot lost or changed its frame names")
                snapshot = {}
                for name, frame in frames.items():
                    old = previous.get(name)
                    if old is not None and (
                        frame.capture_monotonic_ns < old.capture_monotonic_ns
                        or frame.sequence < old.sequence
                    ):
                        raise RuntimeError(f"Sensor capture metadata moved backwards: {name}")
                    if old is not None and (
                        frame.capture_monotonic_ns == old.capture_monotonic_ns
                        and frame.sequence == old.sequence
                    ):
                        snapshot[name] = old
                    else:
                        # Copy before another source read can reuse its image buffer.
                        image = frame.data.copy()
                        image.setflags(write=False)
                        snapshot[name] = SensorFrame(
                            name, image, frame.capture_monotonic_ns, frame.sequence,
                        )
                if snapshot:
                    with self._lock:
                        self._frames = snapshot
                    previous = snapshot
                    self._ready.set()
                self._stop.wait(max(0.0, period - (time.monotonic() - started)))
        except BaseException as exc:
            with self._lock:
                self._error = exc
            self._ready.set()
        finally:
            try:
                self._close_sources()
            except BaseException as exc:
                with self._lock:
                    self._cleanup_error = exc
                    if self._error is None:
                        self._error = exc
                self._ready.set()

    def read(self) -> dict[str, SensorFrame]:
        if not self._started or self._stop.is_set():
            raise RuntimeError("Sensor reader is not running")
        if self.mode == "inline":
            return self._read_sources()
        with self._lock:
            error, frames = self._error, dict(self._frames)
        if error is not None:
            raise RuntimeError(f"Sensor reader failed: {type(error).__name__}: {error}") from error
        if self._sensors and not frames:
            raise RuntimeError("Sensor reader has no complete snapshot")
        now = self._clock.now_ns()
        max_age = self._options["max_frame_age_s"] * 1e9
        for name, frame in frames.items():
            age = now - frame.capture_monotonic_ns
            if age < 0 or age > max_age:
                raise RuntimeError(
                    f"Sensor frame is stale or in the future: {name} ({age / 1e6:.1f} ms)"
                )
        return frames

    def _close_sources(self) -> None:
        errors = []
        while self._owned:
            try:
                self._owned.pop().close()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise BaseExceptionGroup("sensor cleanup failed", errors)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(self._options["shutdown_timeout_s"])
            if self._thread.is_alive():
                # Never close a socket/device underneath the thread using it.
                raise TimeoutError("Sensor reader is still in source I/O; cleanup remains with it")
            self._thread = None
        else:
            self._close_sources()
        self._started = False
        self._frames = {}
        if self._cleanup_error is not None:
            error, self._cleanup_error = self._cleanup_error, None
            raise error
