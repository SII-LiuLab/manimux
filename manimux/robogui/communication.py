"""Policy- and robot-independent messages sent to the robogui."""

from __future__ import annotations

import base64
import contextlib
import io
import logging
import math
from collections import deque
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import zmq

PROTOCOL_VERSION = 2


def _groups(values, *, ndim):
    if not isinstance(values, Mapping) or not values:
        raise ValueError("groups must be a non-empty mapping of named configurations")
    result = {}
    horizon = None
    for name, value in values.items():
        array = np.array(value, dtype=np.float64, copy=True)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("group names must be non-empty strings")
        if array.ndim != ndim or not array.size or not np.isfinite(array).all():
            raise ValueError(f"{name}: expected non-empty finite {ndim}D group data")
        if ndim == 2:
            if horizon is not None and len(array) != horizon:
                raise ValueError("action groups must share one horizon")
            horizon = len(array)
        result[name] = array
    return result


@dataclass(slots=True)
class PolicyPlan:
    """One decoded joint-position plan, retaining runtime group names."""

    policy: str
    instruction: str
    groups: dict[str, np.ndarray]
    action_dt: float
    inference_ms: float
    chunk_id: int
    robot: str = ""
    action_space: str = "joint_position"
    metadata: dict[str, Any] = field(default_factory=dict)
    start_index: int = 0

    def __post_init__(self) -> None:
        self.groups = _groups(self.groups, ndim=2)
        if not np.isfinite(self.action_dt) or self.action_dt <= 0:
            raise ValueError("action_dt must be finite and positive")
        if not np.isfinite(self.inference_ms) or self.inference_ms < 0:
            raise ValueError("inference_ms must be finite and non-negative")
        if self.start_index < 0 or self.start_index > len(next(iter(self.groups.values()))):
            raise ValueError("start_index must be within the action horizon")

    def to_wire(self) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "kind": "plan",
            "robot": self.robot,
            "policy": self.policy,
            "instruction": self.instruction,
            "groups": {name: values.tolist() for name, values in self.groups.items()},
            "action_space": self.action_space,
            "action_dt": float(self.action_dt),
            "inference_ms": float(self.inference_ms),
            "chunk_id": int(self.chunk_id),
            "start_index": int(self.start_index),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class RobotSnapshot:
    """Latest achieved joint state, camera images, and execution cursor."""

    groups: dict[str, np.ndarray]
    cameras: dict[str, np.ndarray]
    step: int
    max_steps: int
    chunk_index: int = 0
    connected: bool = True
    robot: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    active_chunk_id: int | None = None
    timestamp_ns: int | None = None
    sequence: int | None = None
    camera_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.groups = _groups(self.groups, ndim=1)

    def to_wire(self, jpeg_quality: int = 75) -> dict[str, Any]:
        from PIL import Image

        encoded: dict[str, str] = {}
        for name, frame in self.cameras.items():
            rgb = np.asarray(frame, dtype=np.uint8)
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError(f"camera {name!r} must have shape (height, width, 3)")
            buffer = io.BytesIO()
            Image.fromarray(rgb).save(buffer, format="JPEG", quality=jpeg_quality)
            encoded[name] = base64.b64encode(buffer.getvalue()).decode("ascii")
        return {
            "protocol_version": PROTOCOL_VERSION,
            "kind": "state",
            "robot": self.robot,
            "groups": {name: values.tolist() for name, values in self.groups.items()},
            "timestamp_ns": self.timestamp_ns,
            "sequence": self.sequence,
            "camera_metadata": self.camera_metadata,
            "cameras_jpeg": encoded,
            "step": int(self.step),
            "max_steps": int(self.max_steps),
            "chunk_index": int(self.chunk_index),
            "active_chunk_id": self.active_chunk_id,
            "connected": bool(self.connected),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class RuntimeEvent:
    """Policy-executor lifecycle or asynchronous-inference status event."""

    event: str
    robot: str = ""
    policy: str = ""
    step: int = 0
    chunk_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "kind": "event",
            "event": self.event,
            "robot": self.robot,
            "policy": self.policy,
            "step": int(self.step),
            "chunk_id": self.chunk_id,
            "metadata": self.metadata,
        }


class RoboGUIPublisher:
    """Non-blocking publisher used by the robot/policy process."""

    def __init__(self, endpoint: str = "tcp://127.0.0.1:5568") -> None:
        self._context: zmq.Context[zmq.Socket[bytes]] = zmq.Context.instance()
        self._socket: zmq.Socket[bytes] = self._context.socket(zmq.PUSH)
        self._socket.setsockopt(zmq.SNDHWM, 2)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(endpoint)

    def publish(self, message: Any) -> None:
        payload = message.to_wire() if hasattr(message, "to_wire") else message
        with contextlib.suppress(zmq.Again):
            self._socket.send_json(payload, flags=zmq.NOBLOCK)

    def close(self) -> None:
        self._socket.close(linger=0)


class _RoboGUIInbox:
    """Coalesce adjacent state samples without crossing plan/event boundaries."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._messages: deque[dict[str, Any]] = deque()

    @staticmethod
    def _identity(message: dict[str, Any]) -> tuple:
        metadata = message.get("metadata") or {}
        return (message.get("robot"), metadata.get("service_id"),
                metadata.get("episode_id"), metadata.get("episode_dir"))

    def put(self, message: dict[str, Any]) -> None:
        message = dict(message)
        message["_robogui_received_ns"] = time.monotonic_ns()
        with self._lock:
            if (self._messages and message.get("kind") == "state"
                    and self._messages[-1].get("kind") == "state"
                    and self._identity(self._messages[-1]) == self._identity(message)):
                previous = self._messages.pop()
                # Images arrive less often than joint samples. Preserve the newest
                # image per camera when a newer, image-free state replaces a sample.
                for key in ("cameras_jpeg", "camera_metadata"):
                    message[key] = {**previous.get(key, {}), **message.get(key, {})}
            self._messages.append(message)

    def take(self) -> list[dict[str, Any]]:
        with self._lock:
            messages = list(self._messages)
            self._messages.clear()
        return messages


class RoboGUIReceiver:
    """Receive independently of rendering; display current state at a bounded rate."""

    def __init__(
        self, endpoint: str, callback: Callable[[dict[str, Any]], None],
        *, render_hz: float = 30.0,
    ) -> None:
        if not math.isfinite(render_hz) or render_hz <= 0:
            raise ValueError("render_hz must be finite and positive")
        self._endpoint = endpoint
        self._callback = callback
        self._period = 1.0 / render_hz
        self._inbox = _RoboGUIInbox()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._startup_error: Exception | None = None
        self._thread = threading.Thread(target=self._run, name="robogui-bridge", daemon=True)
        self._render_thread = threading.Thread(
            target=self._render, name="robogui-render", daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(5.0):
            self.close()
            raise TimeoutError("RoboGUI receiver did not start")
        if self._startup_error is not None:
            self.close()
            raise self._startup_error
        self._render_thread.start()

    def _run(self) -> None:
        # The receive thread owns its ZeroMQ socket for its entire lifetime.
        socket = zmq.Context.instance().socket(zmq.PULL)
        socket.setsockopt(zmq.RCVHWM, 4)
        socket.setsockopt(zmq.LINGER, 0)
        try:
            socket.bind(self._endpoint)
        except Exception as exc:
            self._startup_error = exc
            self._ready.set()
            socket.close(linger=0)
            return
        self._ready.set()
        try:
            while not self._stop.is_set():
                if socket.poll(50, zmq.POLLIN):
                    message = socket.recv_json()
                    if isinstance(message, dict):
                        self._inbox.put(message)
        finally:
            socket.close(linger=0)

    def _render(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            for message in self._inbox.take():
                if self._stop.is_set():
                    break
                try:
                    self._callback(message)
                except Exception:
                    logging.getLogger(__name__).exception("RoboGUI display update failed")
            # Never catch up by replaying missed display ticks.
            self._stop.wait(max(0.0, self._period - (time.monotonic() - started)))

    def close(self) -> None:
        self._stop.set()
        for thread in (self._thread, self._render_thread):
            if thread.ident is not None and thread is not threading.current_thread():
                thread.join(timeout=2.0)


class ControlServer:
    """Serve dashboard controls to one or more policy executors."""

    def __init__(self, endpoint: str, callback: Callable[[], dict[str, Any]]) -> None:
        self._context: zmq.Context[zmq.Socket[bytes]] = zmq.Context.instance()
        self._socket: zmq.Socket[bytes] = self._context.socket(zmq.REP)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(endpoint)
        self._callback = callback
        self._running = True
        self._thread = threading.Thread(target=self._run, name="robogui-controls", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        while self._running:
            if self._socket in dict(poller.poll(100)):
                self._socket.recv()
                self._socket.send_json(self._callback())

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self._socket.close(linger=0)


class ControlClient:
    """Best-effort polling client that fails closed when the robogui is absent."""

    def __init__(
        self, endpoint: str = "tcp://127.0.0.1:5569", *, timeout_ms: int = 20,
        reply_timeout_ms: int | None = None,
    ) -> None:
        self._context: zmq.Context[zmq.Socket[bytes]] = zmq.Context.instance()
        self._endpoint = endpoint
        self._timeout_ms = timeout_ms
        self._reply_timeout_ms = timeout_ms if reply_timeout_ms is None else reply_timeout_ms
        self._pending_since: float | None = None
        self.last_status = "not_ready"
        self.last_reply_request_started_at: float | None = None
        self.last_reply_rtt_ms: float | None = None
        self._socket: zmq.Socket[bytes] | None = None
        self._connect()

    def _connect(self) -> None:
        self._pending_since = None
        if self._socket is not None:
            self._socket.close(linger=0)
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self._endpoint)

    def poll_state(self) -> dict[str, Any] | None:
        """Return only received controls; absence of a reply is not a Pause command."""
        assert self._socket is not None
        try:
            if self._pending_since is None:
                self._socket.send(b"state", flags=zmq.NOBLOCK)
                self._pending_since = time.monotonic()
            if self._socket.poll(self._timeout_ms, zmq.POLLIN):
                state = self._socket.recv_json()
                self.last_reply_request_started_at = self._pending_since
                self.last_reply_rtt_ms = (time.monotonic() - self._pending_since) * 1000
                self._pending_since = None
                if not isinstance(state, dict) or not isinstance(state.get("paused"), bool):
                    self.last_status = "invalid_reply"
                    return None
                self.last_status = "ok"
                return state
            self.last_status = "timeout"
            # Keep the REQ alive for a bounded late reply, including one-shot events.
            if (time.monotonic() - self._pending_since) * 1000 >= self._reply_timeout_ms:
                self._connect()
            return None
        except ValueError:
            self.last_status = "invalid_reply"
        except zmq.ZMQError:
            self.last_status = "transport_error"
        self._connect()
        return None

    def poll(self) -> dict[str, Any]:
        """Keep the session-preparation API fail-closed when no reply arrives."""
        state = self.poll_state()
        if state is not None:
            return state
        return {
            "paused": True,
            "home_requested": False,
            "finish_requested": False,
        }

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
