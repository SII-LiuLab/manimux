"""Synchronous AsyncSim wire client owned by the simulation runtime."""

from __future__ import annotations

import itertools
from typing import Any


class AsyncSimClient:
    def __init__(self, endpoint: str, *, timeout_s: float = 10.0) -> None:
        if not endpoint.startswith(("ws://", "wss://")) or timeout_s <= 0:
            raise ValueError("AsyncSim requires a WebSocket endpoint and positive timeout")
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.episode_id: str | None = None
        self._connection: Any = None
        self._connection_context: Any = None
        self._ids = itertools.count(1)

    def connect(self) -> None:
        if self._connection is not None:
            return
        try:
            from websockets.sync.client import connect
        except ImportError as exc:
            raise RuntimeError("AsyncSim requires the xpolicylab WebSocket extra") from exc
        context = connect(self.endpoint, open_timeout=self.timeout_s, max_size=64 * 1024 * 1024)
        self._connection = context.__enter__()
        self._connection_context = context

    def request(self, op: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._connection is None:
            raise RuntimeError("AsyncSim client is not connected")
        import msgpack
        import msgpack_numpy

        request_id = f"manimux-{next(self._ids):08d}"
        wire = {"request_id": request_id, "op": op, "payload": payload or {}, "version": "v1"}
        self._connection.send(msgpack.packb(wire, use_bin_type=True, default=msgpack_numpy.encode))
        response = msgpack.unpackb(
            self._connection.recv(timeout=self.timeout_s),
            raw=False, strict_map_key=False, object_hook=msgpack_numpy.decode,
        )
        if response.get("version") != "v1" or response.get("request_id") != request_id:
            raise RuntimeError("AsyncSim response version or request id mismatch")
        if not response.get("ok"):
            error = response.get("error") or {}
            raise RuntimeError(f"AsyncSim {error.get('code', 'error')}: {error.get('message', '')}")
        return response["payload"]

    def reset(self, *, seed: int | None = None) -> dict[str, Any]:
        result = self.request("reset", {} if seed is None else {"seed": seed})
        self.episode_id = result["episode_id"]
        return result

    def subscribe(self, streams: list[str]) -> None:
        self.request("subscribe", {"streams": streams})

    def read_snapshot(self, streams: list[str]) -> dict[str, Any]:
        result = self.request("read_snapshot", {"streams": streams})
        if result["episode_id"] != self.episode_id:
            raise RuntimeError("AsyncSim snapshot belongs to another episode")
        return result

    def submit_command(self, command: dict[str, Any]) -> dict[str, Any]:
        return self.request("submit_command", {"command": command})

    def health(self) -> dict[str, Any]:
        return self.request("health")

    def result(self) -> dict[str, Any]:
        return self.request("get_result")

    def metrics(self) -> dict[str, Any]:
        return self.request("get_metrics")

    def close(self) -> None:
        connection = self._connection
        if connection is not None:
            try:
                self.request("close")
            except (OSError, RuntimeError, TimeoutError):
                pass
            finally:
                self._connection = None
                context, self._connection_context = self._connection_context, None
                context.__exit__(None, None, None)
        self.episode_id = None
