"""Synchronous AsyncSim wire client owned by the simulation runtime."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any


class AsyncSimRemoteError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"AsyncSim {code}: {message}")


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
        self.inference_pause_token: str | None = None
        self._resume_on_command = False

    def connect(self) -> None:
        if self._connection is not None:
            return
        try:
            from websockets.sync.client import connect
        except ImportError as exc:
            raise RuntimeError("AsyncSim requires the xpolicylab WebSocket extra") from exc
        context = connect(
            self.endpoint, open_timeout=self.timeout_s, max_size=64 * 1024 * 1024,
            ping_timeout=max(20.0, self.timeout_s),
        )
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
            raise AsyncSimRemoteError(error.get("code", "error"), error.get("message", ""))
        return response["payload"]

    def reset(
        self, *, seed: int | None = None, recording_directory: str | Path | None = None,
    ) -> dict[str, Any]:
        payload = {} if seed is None else {"seed": seed}
        if recording_directory is not None:
            payload["recording_directory"] = str(Path(recording_directory).resolve())
        result = self.request("reset", payload)
        self.episode_id = result["episode_id"]
        self.inference_pause_token = None
        self._resume_on_command = False
        return result

    def subscribe(self, streams: list[str]) -> None:
        self.request("subscribe", {"streams": streams})

    def attach(self, episode_id: str) -> None:
        result = self.request("attach", {"episode_id": episode_id})
        if result.get("episode_id") != episode_id:
            raise RuntimeError("AsyncSim attach returned another episode")
        self.episode_id = episode_id

    def read_snapshot(self, streams: list[str]) -> dict[str, Any]:
        result = self.request("read_snapshot", {"streams": streams})
        if result["episode_id"] != self.episode_id:
            raise RuntimeError("AsyncSim snapshot belongs to another episode")
        return result

    def submit_command(self, command: dict[str, Any]) -> dict[str, Any]:
        if self.inference_pause_token is not None:
            if not self._resume_on_command:
                raise RuntimeError("cannot submit control while inference is paused")
            return self.finish_inference(command=command)
        return self.request("submit_command", {"command": command})

    def submit_action_chunk(self, chunk: dict[str, Any]) -> dict[str, Any]:
        if self.inference_pause_token is not None:
            if not self._resume_on_command:
                raise RuntimeError("cannot submit action chunk while inference is paused")
            return self.finish_inference(chunk=chunk)
        return self.request("submit_action_chunk", {"chunk":chunk})

    def policy_snapshot(self, streams: list[str], *, mode: str, request_seq: int,
                        timeout_s: float) -> dict[str, Any]:
        result = self.request("policy_snapshot", {
            "streams": streams, "mode": mode, "request_seq": request_seq, "timeout_s": timeout_s,
        })
        if result["episode_id"] != self.episode_id:
            raise RuntimeError("policy snapshot belongs to another episode")
        self.inference_pause_token = result["ablation"].get("pause_token")
        self._resume_on_command = False
        return result

    def resume_with_next_command(self) -> None:
        self._resume_on_command = True

    def finish_inference(self, *, command: dict | None = None, chunk: dict | None = None,
                         reason: str = "command_ready") -> dict:
        if self.inference_pause_token is None:
            return {}
        result = self.request("finish_inference", {
            "token": self.inference_pause_token, "command": command, "chunk":chunk, "reason": reason,
        })
        self.inference_pause_token = None
        self._resume_on_command = False
        return result

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
            except Exception:
                # Closing a broken WebSocket is best effort; keep the original error.
                pass
            finally:
                self._connection = None
                context, self._connection_context = self._connection_context, None
                try:
                    context.__exit__(None, None, None)
                except Exception:
                    pass
        self.episode_id = None
        self.inference_pause_token = None
        self._resume_on_command = False
