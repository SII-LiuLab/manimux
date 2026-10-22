"""Request-scoped evidence captured before wire/action decoding and rejection."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import time
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

_current: ContextVar[ResponseLog | None] = ContextVar("policy_response_log", default=None)


def _safe(value):
    if is_dataclass(value) and not isinstance(value, type):
        return _safe(asdict(value))
    if isinstance(value, dict):
        return {
            str(k): "[redacted]"
            if any(
                s in str(k).lower()
                for s in ("authorization", "api_key", "token", "password", "secret")
            )
            else _safe(v)
            for k, v in value.items()
        }
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": _safe(value.tolist()),
        }
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, bytes):
        return {"encoding": "base64", "data": base64.b64encode(value).decode("ascii")}
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"type": type(value).__qualname__, "repr": repr(value)}


class ResponseLog:
    def __init__(self, config, request):
        root = config.get("options", {}).get("response_log_dir")
        self.path = response_log_path(root, request.session_id, request.request_seq)
        self.config, self.request = config, request
        self.count = 0
        self.started_ns = 0 if self.path is None else time.monotonic_ns()

    def write(self, name, payload):
        if self.path is not None:
            self.path.mkdir(parents=True, exist_ok=True)
            (self.path / name).write_text(json.dumps(_safe(payload), allow_nan=False) + "\n")

    def __enter__(self):
        if self.path is None:
            self.token = _current.set(self)
            return self
        request = self.request
        self.write(
            "request.json",
            {
                "session_id": request.session_id,
                "request_seq": request.request_seq,
                "observation_time_ns": request.observation_time_ns,
                "prompt": request.instruction,
                "worker": self.config.get("worker"),
                "expected_backend": self.config.get("expected_backend"),
                "started_ns": self.started_ns,
                "state": request.observation.state.groups,
                "cameras": {
                    name: {
                        "sha256": hashlib.sha256(frame.data.tobytes()).hexdigest(),
                        "capture_monotonic_ns": frame.capture_monotonic_ns,
                        "shape": list(frame.data.shape),
                    }
                    for name, frame in request.observation.frames.items()
                },
            },
        )
        self.token = _current.set(self)
        return self

    def capture(self, payload, stage):
        if self.path is None:
            return
        self.count += 1
        self.write(
            f"response-{self.count:02d}.json",
            {
                "stage": stage,
                "received_ns": time.monotonic_ns(),
                "response": payload,
            },
        )

    def __exit__(self, kind, error, traceback):
        try:
            if self.path is None:
                return
            self.write(
                "outcome.json",
                {
                    "finished_ns": time.monotonic_ns(),
                    "responses": self.count,
                    "error": None if error is None else f"{kind.__name__}:{error}",
                },
            )
        finally:
            _current.reset(self.token)


def capture_response(payload, *, stage="provider"):
    """Called immediately after receive, before inspecting provider fields."""
    log = _current.get()
    if log is not None:
        log.capture(payload, stage)


def response_log_path(root, session_id, request_seq):
    return (
        None
        if root is None
        else Path(root) / hashlib.sha256(session_id.encode()).hexdigest()[:16] / str(request_seq)
    )
