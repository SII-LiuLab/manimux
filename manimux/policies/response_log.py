"""Experiment records linking provider replies to requests and rejected actions."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import time
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

from manimux.policies.base import DEFAULT_PROVIDER_KEY_ENV

_current: ContextVar[ResponseLog | None] = ContextVar("policy_response_log", default=None)


def _redact(text: str, secret: str) -> str:
    return text.replace(secret, "[redacted]") if secret else text


def _safe(value: object, secret: str = "") -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _safe(asdict(value), secret)
    if isinstance(value, dict):
        return {
            _redact(str(k), secret): "[redacted]"
            if any(
                s in str(k).lower()
                for s in ("authorization", "api_key", "token", "password", "secret")
            )
            else _safe(v, secret)
            for k, v in value.items()
        }
    if isinstance(value, np.ndarray):
        return {
            "dtype": _redact(str(value.dtype), secret),
            "shape": list(value.shape),
            "values": _safe(value.tolist(), secret),
        }
    if isinstance(value, np.generic):
        return _safe(value.item(), secret)
    if isinstance(value, (list, tuple)):
        return [_safe(v, secret) for v in value]
    if isinstance(value, bytes):
        if secret:
            value = value.replace(os.fsencode(secret), b"[redacted]")
        return {"encoding": "base64", "data": base64.b64encode(value).decode("ascii")}
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, str):
        return _redact(value, secret)
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return {"type": type(value).__qualname__, "repr": _redact(repr(value), secret)}


class ResponseLog:
    def __init__(self, config, request, *, api_key_env: str | None = None):
        options = config.get("options", {})
        root = options.get("response_log_dir")
        default_env = DEFAULT_PROVIDER_KEY_ENV.get(config.get("worker"), "")
        self._key_env = api_key_env if api_key_env is not None else options.get(
            "api_key_env", default_env
        )
        self.path = response_log_path(root, request.session_id, request.request_seq)
        self.config, self.request = config, request
        self.count = 0
        self.started_ns = 0 if self.path is None else time.monotonic_ns()

    def write(self, name: str, payload: object) -> None:
        if self.path is not None:
            self.path.mkdir(parents=True, exist_ok=True)
            secret = os.environ.get(self._key_env, "") if isinstance(self._key_env, str) else ""
            text = json.dumps(_safe(payload, secret), allow_nan=False)
            (self.path / name).write_text(text + "\n")

    def __enter__(self):
        if self.path is None:
            self.token = _current.set(self)
            return self
        request = self.request
        sampling = getattr(request, "sampling_request", None) or request
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
                "model_state": getattr(request, "model_state", None),
                "sampling": {
                    name: getattr(sampling, name)
                    for name in (
                        "action_condition", "condition_weights", "rtc_beta",
                        "action_condition_space",
                    )
                    if hasattr(sampling, name)
                },
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

    def capture(self, payload: object, stage: str, metadata: object = None) -> None:
        if self.path is None:
            return
        self.count += 1
        self.write(
            f"response-{self.count:02d}.json",
            {
                "stage": stage,
                "received_ns": time.monotonic_ns(),
                "response": payload,
                "metadata": metadata,
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


def capture_response(payload: object, *, stage: str = "provider", metadata: object = None) -> None:
    """Called immediately after receive, before inspecting provider fields."""
    log = _current.get()
    if log is not None:
        log.capture(payload, stage, metadata)


def response_log_path(root, session_id, request_seq):
    return (
        None
        if root is None
        else Path(root) / hashlib.sha256(session_id.encode()).hexdigest()[:16] / str(request_seq)
    )
