"""Persist provider responses before action validation, without credentials."""

import hashlib
import json
import math
import time
import uuid
from pathlib import Path

import numpy as np


def _json_value(value, secret):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist(), secret)
    if isinstance(value, np.generic):
        return _json_value(value.item(), secret)
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    if isinstance(value, dict):
        return {_json_value(str(k), secret): _json_value(v, secret) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v, secret) for v in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise TypeError(f"Unsupported provider response diagnostic type: {type(value).__name__}")


def save_response(
    root, request, *, checkpoint, observation, raw, timing, secret, error=None, provider="dream"
):
    """An atomic record per call; warmup and rollout sequences share no filenames."""
    session = hashlib.sha256(request.session_id.encode()).hexdigest()[:16]
    folder = Path(root).expanduser().resolve() / session
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{time.time_ns()}-seq{request.request_seq}-{uuid.uuid4().hex[:8]}.json"
    record = {
        "schema": f"manimux.{provider}-response.v1",
        "stage": "transport_error" if error else "received_before_validation",
        "nonfinite_encoding": "Non-finite numeric values use NaN/Infinity/-Infinity strings.",
        "checkpoint": checkpoint,
        "session_id": request.session_id,
        "request_seq": request.request_seq,
        "observation_time_ns": request.observation_time_ns,
        "deadline_ns": request.deadline_ns,
        "observation": observation,
        "camera_frames": {
            name: {
                "shape": list(frame.data.shape),
                "capture_monotonic_ns": frame.capture_monotonic_ns,
                "sequence": frame.sequence,
                "sha256": hashlib.sha256(frame.data.tobytes()).hexdigest(),
            }
            for name, frame in request.observation.frames.items()
        },
        "timing": timing,
        "transport_error": error,
        "raw_response": raw,
    }
    payload = json.dumps(_json_value(record, secret), allow_nan=False, indent=2)
    temporary = path.with_suffix(".partial")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
