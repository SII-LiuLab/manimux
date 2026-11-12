"""Lossless, bounded recording of adapter-prepared requests (no pickle or GPU)."""

from __future__ import annotations

import json
import logging
import queue
import threading
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np

from manimux.types import InferenceRequest

logger = logging.getLogger(__name__)
SCHEMA = "manimux-inference-input-v1"


def _snapshot(request: InferenceRequest) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    identities: dict[int, str] = {}

    def encode(value):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise TypeError("object arrays are not supported in input snapshots")
            key = identities.get(id(value))
            if key is None:
                key = f"array_{len(arrays):04d}"
                identities[id(value)] = key
                arrays[key] = value.copy()
            return {"kind": "array", "key": key}
        if isinstance(value, np.generic):
            return encode(value.item())
        if is_dataclass(value) and not isinstance(value, type):
            return {
                "kind": "dataclass",
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
                "fields": {
                    field.name: encode(getattr(value, field.name)) for field in fields(value)
                },
            }
        if isinstance(value, dict):
            return {"kind": "mapping", "items": [[encode(k), encode(v)] for k, v in value.items()]}
        if isinstance(value, (list, tuple)):
            return {"kind": type(value).__name__, "items": [encode(item) for item in value]}
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        raise TypeError(f"unsupported input snapshot value: {type(value).__name__}")

    metadata = {
        "schema": SCHEMA,
        "boundary": "adapter_prepared_worker_submission",
        "request": encode(request),
    }
    arrays["__metadata__"] = np.frombuffer(
        json.dumps(metadata, ensure_ascii=False, allow_nan=False).encode("utf-8"), dtype=np.uint8
    )
    return arrays


def load_input_snapshot(path: str | Path) -> dict:
    """Load arrays and plain fields safely; never instantiate recorded Python classes.

    Dataclasses become dictionaries. Their type names remain in the NPZ metadata.
    The returned ``request`` contains observation, instruction and sampler fields.
    """
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(archive["__metadata__"].tobytes().decode("utf-8"))
        if metadata["schema"] != SCHEMA:
            raise ValueError(f"unsupported input snapshot schema: {metadata['schema']}")

        def decode(value):
            if not isinstance(value, dict):
                return value
            kind = value["kind"]
            if kind == "array":
                return archive[value["key"]]
            if kind == "dataclass":
                return {name: decode(item) for name, item in value["fields"].items()}
            if kind == "mapping":
                return {decode(k): decode(v) for k, v in value["items"]}
            if kind in ("list", "tuple"):
                items = [decode(item) for item in value["items"]]
                return tuple(items) if kind == "tuple" else items
            raise ValueError(f"unsupported input snapshot node: {kind}")

        return {**metadata, "request": decode(metadata["request"])}


class AsyncInputRecorder:
    """Copy selected inputs at submission; write on a bounded background queue.

    Never wait for disk in submit(). A submitted request can be coalesced by the
    worker; join request_seq to runtime response events before calling it inferred.
    """

    def __init__(self, episode_dir: Path, *, enabled: bool = False, queue_size: int = 2):
        if type(enabled) is not bool or type(queue_size) is not int or queue_size < 1:
            raise ValueError("input recording requires a bool and a positive integer queue size")
        self.enabled = enabled
        self._directory = episode_dir / "inference_inputs"
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._thread: threading.Thread | None = None
        self._closed = False
        self._submitted = 0
        self._saved = 0
        self._dropped = 0
        self._error: str | None = None

    def submit(self, request: InferenceRequest) -> dict:
        if not self.enabled:
            return {"status": "disabled"}
        self._submitted += 1
        if self._closed or self._error is not None:
            self._dropped += 1
            return {"status": "unavailable", "error": self._error or "recorder closed"}
        if self._queue.full():
            self._dropped += 1
            return {"status": "dropped_queue_full"}
        try:
            arrays = _snapshot(request)
            filename = f"request-{request.request_seq:06d}.npz"
            entry = {
                "request_seq": request.request_seq,
                "observation_time_ns": request.observation_time_ns,
                "file": filename,
            }
            self._queue.put_nowait((entry, arrays))
            if self._thread is None:
                thread = threading.Thread(target=self._run, name="inference-input-recorder")
                thread.start()
                self._thread = thread
            return {"status": "queued", "file": f"inference_inputs/{filename}"}
        except queue.Full:
            self._dropped += 1
            return {"status": "dropped_queue_full"}
        except Exception as exc:
            self._dropped += 1
            self._error = f"{type(exc).__name__}: {exc}"
            logger.exception("Inference input recording failed")
            return {"status": "error", "error": self._error}

    def _run(self) -> None:
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            with (self._directory / "index.jsonl").open("a", encoding="utf-8") as index:
                while True:
                    item = self._queue.get()
                    if item is None:
                        break
                    entry, arrays = item
                    target = self._directory / entry["file"]
                    temporary = target.with_suffix(".npz.tmp")
                    with temporary.open("wb") as stream:
                        # Uncompressed NPZ preserves RGB/numeric values without video loss.
                        np.savez(stream, **arrays)
                    temporary.replace(target)
                    index.write(json.dumps(entry) + "\n")
                    index.flush()
                    self._saved += 1
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            logger.exception("Inference input writer failed")

    def close(self) -> dict:
        if not self._closed:
            self._closed = True
            if self._thread is not None:
                while self._thread.is_alive():
                    try:
                        self._queue.put(None, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                self._thread.join()
            # Release any buffered arrays after a writer failure.
            while not self._queue.empty():
                self._queue.get_nowait()
        return {
            "enabled": self.enabled,
            "submitted_requests": self._submitted,
            "saved_requests": self._saved,
            "dropped_requests": self._dropped,
            "unsaved_requests": self._submitted - self._saved,
            "error": self._error,
        }
