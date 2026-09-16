"""Optional native-rate joint sidecars; standard camera/action episodes stay intact."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path


class NativeJointRecording:
    def __init__(self, sources):
        self.sources = list(sources)
        names = [source.stream for source in self.sources]
        channels = [source.channel for source in self.sources]
        if not names or len(set(names)) != len(names) or len(set(channels)) != len(channels):
            raise ValueError("native joint recording requires distinct streams and CAN channels")
        if any(not name.replace("_", "").isalnum() for name in names):
            raise ValueError("invalid native joint stream name")
        self._readers = []
        self._files = []
        self._threads = []
        self._stop = threading.Event()
        self._end_ns = None
        self._start_ns = 0
        self._stats = {}

    def start(self, episode: Path):
        self.path = Path(episode) / "native_joints"
        self.path.mkdir()
        try:
            for source in self.sources:
                self._readers.append(source.open())
                self._files.append((self.path / f"{source.stream}.jsonl").open("w"))
                self._stats[source.stream] = {
                    **source.metadata(), "file": f"{source.stream}.jsonl",
                    "samples": [0] * len(source.motor_ids), "error": None,
                    "dropped_frames": 0, "invalid_feedback": 0,
                }
            self._start_ns = time.time_ns()
            self._write_metadata(complete=False)
            for source, reader, handle in zip(
                self.sources, self._readers, self._files, strict=True
            ):
                thread = threading.Thread(
                    target=self._capture, args=(source, reader, handle), daemon=True,
                    name=f"native-joints-{source.stream}",
                )
                thread.start()
                self._threads.append(thread)
        except BaseException:
            self.request_stop()
            for thread in self._threads:
                thread.join(2)
            for reader in self._readers:
                reader.close()
            for handle in self._files:
                handle.close()
            raise

    def _capture(self, source, reader, handle):
        stats = self._stats[source.stream]
        previous = [None] * len(source.motor_ids)
        try:
            while True:
                sample = reader.read()
                if sample is None:
                    if self._stop.is_set():
                        break
                    continue
                timestamp = sample["timestamp_ns"]
                if self._end_ns is not None and timestamp > self._end_ns:
                    break
                if timestamp < self._start_ns:
                    continue
                index = sample["joint_index"]
                if previous[index] is not None and timestamp <= previous[index]:
                    raise RuntimeError("kernel timestamps moved backwards or repeated")
                previous[index] = timestamp
                handle.write(json.dumps(sample, allow_nan=False) + "\n")
                stats["samples"][index] += 1
                stats["invalid_feedback"] += sample["motor_error"] != "0x1"
        except Exception as exc:
            stats["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            stats["dropped_frames"] = reader.dropped_frames
            reader.close()
            handle.close()

    def request_stop(self):
        if not self._stop.is_set():
            self._end_ns = time.time_ns()
            self._stop.set()

    def _write_metadata(self, *, complete):
        data = {
            "schema_version": 1, "complete": complete,
            "timestamp_domain": "Unix nanoseconds from host kernel CAN receive timestamps",
            "timestamp_scope": "per-motor feedback reception, not simultaneous arm sampling",
            "start_timestamp_ns": self._start_ns, "end_timestamp_ns": self._end_ns,
            "streams": self._stats,
        }
        (self.path / "metadata.json").write_text(json.dumps(data, indent=2) + "\n")
        return data

    def stop(self):
        self.request_stop()
        for thread in self._threads:
            thread.join(2)
        if any(thread.is_alive() for thread in self._threads):
            raise RuntimeError("native joint writer did not stop; recording remains incomplete")
        complete = all(
            not row["error"] and not row["dropped_frames"] and not row["invalid_feedback"]
            and all(row["samples"]) for row in self._stats.values()
        )
        return self._write_metadata(complete=complete)
