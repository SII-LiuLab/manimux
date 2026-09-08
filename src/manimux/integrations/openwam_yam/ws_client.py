"""Thin synchronous client for OpenWAM's official JSON WebSocket protocol."""

from __future__ import annotations

import inspect
import json
from typing import Any


class OpenWAMProtocolError(RuntimeError):
    """The OpenWAM server returned a malformed or error response."""


class OpenWAMWsClient:
    """One exclusive persistent connection to an OpenWAM policy server.

    OpenWAM exposes one action per ``obs`` message while buffering a generated
    chunk internally. ``infer_chunk`` drains exactly one configured server
    horizon so ManiMux, rather than the remote service, owns execution timing.
    The server must therefore be dedicated to this client and launched with an
    inference horizon equal to ManiMux's policy horizon.
    """

    def __init__(
        self,
        url: str,
        *,
        connect_timeout_s: float = 30.0,
        request_timeout_s: float = 300.0,
    ) -> None:
        self._url = url
        self._connect_timeout_s = connect_timeout_s
        self._request_timeout_s = request_timeout_s
        self._conn: Any | None = None
        self._last_step = 0

    def connect(self) -> None:
        if self._conn is not None:
            raise OpenWAMProtocolError("client is already connected")
        try:
            from websockets.sync.client import connect
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                'the OpenWAM bridge requires websockets; install the "openwam" extra'
            ) from exc

        kwargs: dict[str, object] = {
            "open_timeout": self._connect_timeout_s,
            "max_size": None,
            "compression": None,
            "ping_interval": None,
            # websockets>=15 otherwise consults HTTP(S)_PROXY even for loopback.
            "proxy": None,
        }
        supported = set(inspect.signature(connect).parameters)
        connect_untyped: Any = connect
        self._conn = connect_untyped(
            self._url,
            **{key: value for key, value in kwargs.items() if key in supported},
        )

    def _roundtrip(self, payload: dict[str, object]) -> dict[str, object]:
        if self._conn is None:
            raise OpenWAMProtocolError("client is not connected")
        self._conn.send(json.dumps(payload, separators=(",", ":")))
        try:
            raw = self._conn.recv(timeout=self._request_timeout_s)
        except TypeError:  # pragma: no cover - websockets<13 compatibility
            raw = self._conn.recv()
        if not isinstance(raw, str):
            raise OpenWAMProtocolError("OpenWAM must reply with a JSON text frame")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OpenWAMProtocolError("OpenWAM returned invalid JSON") from exc
        if not isinstance(body, dict):
            raise OpenWAMProtocolError("OpenWAM response must be a JSON object")
        if body.get("type") == "error":
            raise OpenWAMProtocolError(
                f"OpenWAM error {body.get('code', 'unknown')}: {body.get('message', '')}"
            )
        return body

    def ping(self) -> dict[str, object]:
        body = self._roundtrip({"type": "ping"})
        if body.get("type") != "pong":
            raise OpenWAMProtocolError(f"expected pong, got {body.get('type')!r}")
        return body

    def reset(self) -> None:
        body = self._roundtrip({"type": "reset"})
        if body.get("type") != "reset_ack":
            raise OpenWAMProtocolError(f"expected reset_ack, got {body.get('type')!r}")
        self._last_step = 0

    def infer_chunk(
        self,
        observation: dict[str, object],
        *,
        horizon_steps: int,
        response_mode: str = "chunk",
    ) -> list[list[float]]:
        if horizon_steps <= 1:
            raise ValueError("OpenWAM chunk horizon must be greater than one")
        if response_mode == "chunk":
            return self._request_chunk(observation, horizon_steps=horizon_steps)
        if response_mode != "stream":
            raise ValueError("OpenWAM response_mode must be 'chunk' or 'stream'")
        return self._drain_stream(observation, horizon_steps=horizon_steps)

    def _request_chunk(
        self,
        observation: dict[str, object],
        *,
        horizon_steps: int,
    ) -> list[list[float]]:
        body = self._roundtrip({"type": "obs", **observation})
        if body.get("type") != "action":
            raise OpenWAMProtocolError(f"expected action, got {body.get('type')!r}")
        self._accept_step(body)
        raw = body.get("action")
        if not isinstance(raw, list) or len(raw) != horizon_steps:
            raise OpenWAMProtocolError(
                f"OpenWAM chunk response must contain {horizon_steps} action rows"
            )
        actions: list[list[float]] = []
        for index, row in enumerate(raw):
            if not isinstance(row, list) or not row:
                raise OpenWAMProtocolError(f"OpenWAM action row {index} is not a flat list")
            try:
                actions.append([float(value) for value in row])
            except (TypeError, ValueError) as exc:
                raise OpenWAMProtocolError(
                    f"OpenWAM action row {index} contains a non-number"
                ) from exc
        return actions

    def _drain_stream(
        self,
        observation: dict[str, object],
        *,
        horizon_steps: int,
    ) -> list[list[float]]:
        actions: list[list[float]] = []
        for _ in range(horizon_steps):
            body = self._roundtrip({"type": "obs", **observation})
            if body.get("type") != "action":
                raise OpenWAMProtocolError(
                    f"expected action, got {body.get('type')!r}"
                )
            self._accept_step(body)
            action = body.get("action")
            if not isinstance(action, list) or not action:
                raise OpenWAMProtocolError("OpenWAM action response has no flat action")
            try:
                values = [float(value) for value in action]
            except (TypeError, ValueError) as exc:
                raise OpenWAMProtocolError("OpenWAM action contains a non-number") from exc
            actions.append(values)
        return actions

    def _accept_step(self, body: dict[str, object]) -> None:
        step = body.get("step")
        if not isinstance(step, int) or isinstance(step, bool):
            raise OpenWAMProtocolError("OpenWAM action response has no integer step")
        if step != self._last_step + 1:
            raise OpenWAMProtocolError(
                "OpenWAM server is not exclusive or was reset out of band: "
                f"expected step {self._last_step + 1}, got {step}"
            )
        self._last_step = step

    def close(self) -> None:
        connection, self._conn = self._conn, None
        if connection is not None:
            connection.close()
