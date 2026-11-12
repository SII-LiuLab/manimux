"""ZMQ-based camera server.

Hosts camera components in a long-lived process so eval clients can pull
the latest frames on demand without paying for pipeline startup, fighting the
policy loop for camera I/O, or coupling robot-control timing to camera I/O.

Sockets
-------
REP  ``tcp://127.0.0.1:5555``  (default)
    Pull semantics. Client sends a pickled request dict; server replies with a
    pickled response dict. Used by the policy for on-demand obs.

PUB  ``tcp://127.0.0.1:5556``  (default, optional)
    Push semantics. Server publishes the latest obs every ``pub_period_sec``;
    a server whose cameras are all TacCap publishes as soon as any camera has a
    new frame instead. Used by the cv2 live preview and by timestamped runtime
    sensors.

Request protocol
----------------
    {"cmd": "obs"}   ->  {"ok": True, "frames": {cam_name: np.ndarray (H,W,3) uint8 RGB},
                          "timestamps": {cam_name: float}}
    {"cmd": "ping"}  ->  {"ok": True, "pong": True}

Errors come back as ``{"ok": False, "error": str}``. The server keeps running
across any single bad request.

CLI
---
    manimux-camera-server --config <cameras.yaml>
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import pickle
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

import zmq

from manimux.cli import (
    bind_station,
    read_camera_recipe,
    read_experiment,
    read_yaml,
    resolve_local_path,
)
from manimux.embodiments.sensor import SensorBase, build_camera
from manimux.embodiments.sensor.taccap.sensor import TacCapSensor
from manimux.types import SensorFrame

logger = logging.getLogger("camera_server")

DEFAULT_REP_ENDPOINT = "tcp://127.0.0.1:5555"
DEFAULT_PUB_ENDPOINT = "tcp://127.0.0.1:5556"
DEFAULT_PUB_PERIOD_SEC = 1.0 / 30.0
# TacCap-only servers check for new frames at this interval instead of the PUB period.
NEW_FRAME_POLL_SEC = 0.001
DEFAULT_HEARTBEAT_SEC = 10.0


def camera_config(experiment: dict) -> dict:
    """Resolve camera components and per-stream overrides without loading a robot."""
    overrides = experiment.get("robot", {}).get("options", {}).get("component_hardware", {})
    cameras = {}
    for stream_name, camera in experiment["camera_server"]["cameras"].items():
        name = camera["component"]
        component = read_yaml(camera["config"])
        cameras[stream_name] = {
            "implementation": component["implementation"],
            **component.get("options", {}),
            **component.get("hardware", {}),
            **camera.get("options", {}),
            **overrides.get(name, {}),
        }
    return {"sensors": {"cameras": cameras}}


class CameraServer:
    """Owns the configured cameras and serves their latest frames over ZMQ."""

    def __init__(
        self,
        cameras: dict[str, SensorBase],
        rep_endpoint: str = DEFAULT_REP_ENDPOINT,
        pub_endpoint: str | None = None,
        pub_period_sec: float = DEFAULT_PUB_PERIOD_SEC,
        heartbeat_sec: float = DEFAULT_HEARTBEAT_SEC,
    ) -> None:
        self.cameras = cameras
        self.rep_endpoint = rep_endpoint
        self.pub_endpoint = pub_endpoint
        self.pub_period_sec = float(pub_period_sec)
        self.heartbeat_sec = float(heartbeat_sec)

        self._ctx = zmq.Context.instance()
        self._rep: zmq.Socket | None = None
        self._pub: zmq.Socket | None = None

        self._stop_event = threading.Event()
        self._pub_thread: threading.Thread | None = None

        self._req_total = 0
        self._req_window = 0
        self._last_heartbeat = time.time()
        # Keep the existing Unix-seconds wire format using the host capture clock.
        self._unix_offset_s = time.time() - time.monotonic_ns() / 1e9

    # ------------------------------------------------------------------
    # Frame sourcing
    # ------------------------------------------------------------------

    def _read_frames(self, camera_names: list[str] | None = None) -> dict[str, SensorFrame]:
        """Read the latest frame from each named camera, or from every camera."""
        names = list(self.cameras) if camera_names is None else camera_names
        if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
            raise ValueError("camera_names must be a nonempty list of names")
        missing = set(names) - self.cameras.keys()
        if missing:
            raise ValueError(f"Unknown cameras: {sorted(missing)}")
        frames: dict[str, SensorFrame] = {}
        for name in names:
            frame = self.cameras[name].read()
            if not isinstance(frame, SensorFrame):
                raise TypeError(f"Camera {name!r} must return one SensorFrame")
            frames[name] = frame
        return frames

    def _response(self, frames: dict[str, SensorFrame]) -> dict[str, Any]:
        return {
            "ok": True,
            "frames": {name: frame.data for name, frame in frames.items()},
            "timestamps": {
                name: frame.capture_monotonic_ns / 1e9 + self._unix_offset_s
                for name, frame in frames.items()
            },
        }

    def _snapshot(self, camera_names: list[str] | None = None) -> dict[str, Any]:
        """Snapshot the latest color frame from every camera (RGB uint8)."""
        return self._response(self._read_frames(camera_names))

    # ------------------------------------------------------------------
    # Request handling
    # ------------------------------------------------------------------

    def _handle_request(self) -> None:
        assert self._rep is not None
        raw = self._rep.recv()
        try:
            req = pickle.loads(raw)
            cmd = (req or {}).get("cmd", "obs")
        except Exception as exc:  # noqa: BLE001 — surface to client, stay alive
            self._rep.send(pickle.dumps({"ok": False, "error": f"bad request: {exc!r}"}))
            return

        try:
            if cmd == "obs":
                resp = self._snapshot(req.get("camera_names"))
            elif cmd == "ping":
                resp = {"ok": True, "pong": True}
            else:
                resp = {"ok": False, "error": f"unknown cmd: {cmd!r}"}
        except Exception as exc:  # noqa: BLE001 — keep server alive
            logger.exception("Request failed (cmd=%r)", cmd)
            resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        self._rep.send(pickle.dumps(resp), copy=False)
        self._req_total += 1
        self._req_window += 1

    def _pub_loop(self) -> None:
        assert self._pub is not None
        # Every TacCap frame carries its own receipt time, so publish each new
        # frame immediately; other camera types keep the fixed-period stream.
        if self.cameras and all(isinstance(cam, TacCapSensor) for cam in self.cameras.values()):
            self._pub_new_frames()
            return
        next_tick = time.time()
        while not self._stop_event.is_set():
            now = time.time()
            if now < next_tick:
                # Tiny sleep granularity so shutdown is snappy.
                time.sleep(min(0.01, next_tick - now))
                continue
            next_tick = now + self.pub_period_sec
            try:
                resp = self._snapshot()
                self._pub.send(pickle.dumps(resp), copy=False)
            except Exception as exc:  # noqa: BLE001 — pub is best-effort
                logger.warning("PUB tick failed: %s", exc)

    def _pub_new_frames(self) -> None:
        """Publish the bundle whenever any camera's latest capture time changes."""
        assert self._pub is not None
        published = None
        while not self._stop_event.is_set():
            # Capture times on the monotonic clock that the published frames carry.
            stamps = tuple(cam.latest_frame_ns() for cam in self.cameras.values())
            if None in stamps or stamps == published:
                time.sleep(NEW_FRAME_POLL_SEC)
                continue
            try:
                frames = self._read_frames()
                self._pub.send(pickle.dumps(self._response(frames)), copy=False)
                # A frame that arrived after polling is already in this bundle.
                published = tuple(frames[name].capture_monotonic_ns for name in self.cameras)
            except Exception as exc:  # noqa: BLE001 — pub is best-effort
                # Retry only after another frame arrives, not every poll.
                published = stamps
                logger.warning("PUB new frame failed: %s", exc)

    def _maybe_heartbeat(self) -> None:
        now = time.time()
        elapsed = now - self._last_heartbeat
        if elapsed < self.heartbeat_sec:
            return
        hz = self._req_window / elapsed if elapsed > 0 else 0.0
        logger.info(
            "alive: total_requests=%d window=%d (%.1f req/s) cameras=%d",
            self._req_total,
            self._req_window,
            hz,
            len(self.cameras),
        )
        self._req_window = 0
        self._last_heartbeat = now

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def bind(self) -> None:
        """Reserve endpoints before opening hardware, including on duplicate starts."""
        if self._rep is not None:
            return
        self._rep = self._ctx.socket(zmq.REP)
        self._rep.bind(self.rep_endpoint)
        logger.info("REP bound on %s", self.rep_endpoint)

        if self.pub_endpoint:
            self._pub = self._ctx.socket(zmq.PUB)
            self._pub.bind(self.pub_endpoint)
            logger.info(
                "PUB bound on %s (period=%.3fs)",
                self.pub_endpoint,
                self.pub_period_sec,
            )

    def run(self) -> None:
        self.bind()
        if self._pub is not None:
            self._pub_thread = threading.Thread(
                target=self._pub_loop,
                name="camera_server_pub",
                daemon=True,
            )
            self._pub_thread.start()

        poller = zmq.Poller()
        poller.register(self._rep, zmq.POLLIN)
        try:
            while not self._stop_event.is_set():
                # 100 ms tick keeps heartbeats responsive and shutdown snappy.
                socks = dict(poller.poll(timeout=100))
                if self._rep in socks:
                    self._handle_request()
                self._maybe_heartbeat()
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        if self._pub_thread is not None:
            self._pub_thread.join(timeout=2.0)
        for sock in (self._rep, self._pub):
            if sock is not None:
                with contextlib.suppress(Exception):
                    sock.close(linger=0)
        for cam in self.cameras.values():
            with contextlib.suppress(Exception):
                cam.close()
        logger.info("Camera server stopped.")


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------


def _build_cameras_from_config(cfg_path: Path, local: Path | None = None) -> dict[str, SensorBase]:
    """Use the same camera recipes and station bindings as experiment startup."""
    config = bind_station(
        {"camera_server": read_camera_recipe(cfg_path)}, resolve_local_path(cfg_path, local)
    )
    return _build_cameras(camera_config(config))


def _build_cameras(cfg: dict) -> dict[str, SensorBase]:
    """Construct every component before starting any device; clean up failed starts."""
    cameras = {
        name: build_camera(name, spec)
        for name, spec in cfg["sensors"]["cameras"].items()
    }
    try:
        for name, camera in cameras.items():
            logger.info("Starting camera %s", name)
            camera.start()
    except BaseException:
        for camera in cameras.values():
            try:
                camera.close()
            except Exception:
                logger.exception("Camera cleanup failed after startup error")
        raise
    return cameras


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ManiMux multi-camera server (ZMQ).")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--config",
        type=Path,
        help="Camera recipe with cameras.<stream>.config, component and options",
    )
    source.add_argument("--experiment", type=Path, help="Experiment with named camera components")
    parser.add_argument(
        "--local", type=Path,
        help="Station bindings (default: manimux/configs/local/station.yaml)",
    )
    parser.add_argument("--rep-endpoint")
    parser.add_argument(
        "--pub-endpoint",
        help="ZMQ PUB endpoint. Pass empty string to disable the PUB stream.",
    )
    parser.add_argument("--pub-period-sec", type=float, default=DEFAULT_PUB_PERIOD_SEC)
    parser.add_argument("--heartbeat-sec", type=float, default=DEFAULT_HEARTBEAT_SEC)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    local = resolve_local_path(args.experiment or args.config, args.local)
    if args.experiment is not None:
        experiment = read_experiment(args.experiment, local=local)
    else:
        experiment = bind_station({"camera_server": read_camera_recipe(args.config)}, local)
    resolved_cameras = camera_config(experiment)
    service = experiment["camera_server"]
    # Explicit CLI addresses override the station's shared service bindings.
    args.rep_endpoint = args.rep_endpoint or service.get("rep_endpoint", DEFAULT_REP_ENDPOINT)
    if args.pub_endpoint is None:
        args.pub_endpoint = service.get("pub_endpoint", DEFAULT_PUB_ENDPOINT)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    server = CameraServer(
        cameras={},
        rep_endpoint=args.rep_endpoint,
        pub_endpoint=(args.pub_endpoint or None),
        pub_period_sec=args.pub_period_sec,
        heartbeat_sec=args.heartbeat_sec,
    )

    def _handle(signum, _frame):
        logger.info("Signal %d received; shutting down.", signum)
        server.shutdown()
        sys.exit(0)

    try:
        try:
            server.bind()
        except zmq.ZMQError as exc:
            if exc.errno != zmq.EADDRINUSE:
                raise
            logger.error(
                "Camera endpoint already in use (%s / %s). Reuse the running camera "
                "service, or stop it before restarting. No cameras were opened or reset.",
                args.rep_endpoint,
                args.pub_endpoint,
            )
            return 2
        signal.signal(signal.SIGINT, _handle)
        signal.signal(signal.SIGTERM, _handle)
        server.cameras = _build_cameras(resolved_cameras)
        server.run()
    finally:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
