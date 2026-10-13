"""Minimal synchronous client for an XPolicyLab WebSocket policy server.

We deliberately do not import XPolicyLab's own ``client_server`` package. The
entire reason its model runs in a separate environment is that ManiMux never
takes on that environment's dependencies, and vendoring the client would drag
the coupling back in through the front door. The wire format is small enough to
speak directly.

One frame is a msgpack map (numpy arrays encoded by ``msgpack_numpy``)::

    message_type, message_id, evaluation_id, action_case_id,
    trial_id, repeat_index, step, sent_at, payload

Only three request/response pairs are needed for inference:

===============  ==================  ==========================================
request          reply               payload
===============  ==================  ==========================================
``hello``        ``hello_ack``       ``{}``
``reset``        ``reset_result``    ``{"trial_id": ...}``
``infer``        ``infer_result``    ``{"observation": ..., "sampling": ...}``
===============  ==================  ==========================================

The result of a ``call`` is read back from ``payload["result"]``.

Third-party imports are lazy so that this module -- and therefore the adapter
that lives beside it -- can be imported for unit tests in an environment that
has neither ``msgpack`` nor ``websockets`` installed.
"""

from __future__ import annotations

import time
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from types import TracebackType
from typing import Any

HELLO = "hello"
HELLO_ACK = "hello_ack"
RESET = "reset"
RESET_RESULT = "reset_result"
INFER = "infer"
INFER_RESULT = "infer_result"
ERROR = "error"

_EXPECTED_REPLY = {
    HELLO: HELLO_ACK,
    RESET: RESET_RESULT,
    INFER: INFER_RESULT,
}

# Three uncompressed 480x640x3 frames are ~2.8 MB, well past the 1 MiB default
# that ``websockets`` enforces on inbound messages.
_NO_FRAME_SIZE_LIMIT = None


class XPolicyLabProtocolError(RuntimeError):
    """The policy server replied with an error frame or an unusable frame."""


class XPolicyLabTimeoutError(TimeoutError):
    """The policy server did not answer a request before its deadline."""


_CODEC: tuple[Any, Any] | None = None


def _codec() -> tuple[Any, Any]:
    """Import and cache ``msgpack`` and ``msgpack_numpy`` on first use."""

    global _CODEC
    if _CODEC is None:
        try:
            import msgpack
            import msgpack_numpy
        except ImportError as exc:  # pragma: no cover - depends on the venv
            raise ImportError(
                "the XPolicyLab bridge requires msgpack and msgpack-numpy; "
                'install them with the "xpolicylab" extra'
            ) from exc
        _CODEC = (msgpack, msgpack_numpy)
    return _CODEC


def pack_frame(frame: dict[str, Any]) -> bytes:
    msgpack, msgpack_numpy = _codec()
    return msgpack.packb(frame, default=msgpack_numpy.encode, use_bin_type=True)


def unpack_frame(raw: bytes | str) -> dict[str, Any]:
    if isinstance(raw, str):
        raise XPolicyLabProtocolError("expected a binary frame, got text")
    msgpack, msgpack_numpy = _codec()
    decoded = msgpack.unpackb(
        raw,
        object_hook=msgpack_numpy.decode,
        raw=False,
        strict_map_key=False,
    )
    if not isinstance(decoded, dict):
        raise XPolicyLabProtocolError("a frame must decode to a map")
    return decoded


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class XPolicyLabWsClient:
    """Single-threaded RPC and streaming I/O for one policy server connection.

    Ordinary requests block for their reply; streaming separates upload and
    polling. The ManiMux worker drives both from one thread, so no locking is
    needed, but this object is not safe to share between threads.
    """

    def __init__(
        self,
        *,
        url: str,
        evaluation_id: str,
        trial_id: str,
        action_case_id: str | None = None,
        repeat_index: int | None = None,
        connect_timeout_s: float = 30.0,
        request_timeout_s: float = 10.0,
    ) -> None:
        self._url = normalize_url(url)
        self._evaluation_id = evaluation_id
        self._trial_id = trial_id
        self._action_case_id = action_case_id
        self._repeat_index = repeat_index
        self._connect_timeout_s = connect_timeout_s
        self._request_timeout_s = request_timeout_s
        self._conn: Any | None = None
        self._step = 0
        self._sampling_modes = frozenset({"default"})
        self.multi_inflight = False
        self._backend_metadata: dict[str, object] = {}
        self._pending_requests: set[str] = set()

    @property
    def url(self) -> str:
        return self._url

    @property
    def step(self) -> int:
        return self._step

    @property
    def sampling_modes(self) -> frozenset[str]:
        return self._sampling_modes

    @property
    def backend_metadata(self) -> dict[str, object]:
        return dict(self._backend_metadata)

    def connect(self) -> None:
        """Open the socket and complete the ``hello`` handshake."""

        if self._conn is not None:
            raise XPolicyLabProtocolError("client is already connected")
        try:
            from websockets.sync.client import connect as ws_connect
        except ImportError as exc:  # pragma: no cover - depends on the venv
            raise ImportError(
                'the XPolicyLab bridge requires websockets; install it with the "xpolicylab" extra'
            ) from exc
        self._conn = ws_connect(
            self._url,
            open_timeout=self._connect_timeout_s,
            max_size=_NO_FRAME_SIZE_LIMIT,
        )
        try:
            # Finish terminates this rollout's transport worker. Its disconnect
            # must invalidate queued server work instead of retaining it for retry.
            reply = self.request(
                HELLO, {"cancel_pending_on_disconnect": True},
                timeout_s=self._connect_timeout_s,
            )
            payload = reply.get("payload")
            capabilities = payload.get("capabilities") if isinstance(payload, dict) else None
            modes = capabilities.get("sampling_modes") if isinstance(capabilities, dict) else None
            if isinstance(modes, list) and modes and all(isinstance(mode, str) for mode in modes):
                self._sampling_modes = frozenset(modes)
            self.multi_inflight = (
                isinstance(capabilities, dict) and capabilities.get("multi_inflight") is True
            )
            if isinstance(payload, dict):
                model_metadata = payload.get("model_metadata")
                self._backend_metadata = {
                    "server": payload.get("server"),
                    "server_instance_id": payload.get("server_instance_id"),
                    "server_revision": payload.get("server_revision"),
                    "model": model_metadata if isinstance(model_metadata, dict) else {},
                }
        except Exception:
            self.close()
            raise

    def reset(self) -> None:
        # A local timeout does not cancel server work. Receive its completion
        # before RESET so even preprocessing outside the model lock is fenced.
        self.drain()
        self.request(RESET, {"trial_id": self._trial_id})
        self._step = 0

    def drain(self) -> None:
        """Wait at most one request timeout for already-sent calls to finish.

        Late replies are discarded, never retried. Failure leaves their IDs
        pending so a later reset can continue draining the same connection.
        """
        if not self._pending_requests:
            return
        conn = self._conn
        if conn is None:
            raise XPolicyLabProtocolError("cannot drain a disconnected client")
        deadline = time.monotonic() + self._request_timeout_s
        while self._pending_requests:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise XPolicyLabTimeoutError(
                    f"draining policy requests timed out after {self._request_timeout_s:.3f}s"
                )
            reply = unpack_frame(conn.recv(timeout=remaining))
            self._pending_requests.discard(reply.get("message_id"))

    def infer(
        self,
        observation: dict[str, Any],
        *,
        sampling: dict[str, Any],
        timeout_s: float | None = None,
    ) -> Any:
        """Send one complete inference request and return its action chunk."""

        reply = self.request(
            INFER,
            {"observation": observation, "sampling": sampling},
            timeout_s=timeout_s,
        )
        self._step += 1
        payload = reply.get("payload")
        if not isinstance(payload, dict) or "actions" not in payload:
            raise XPolicyLabProtocolError("infer reply has no actions")
        if (
            sampling.get("mode") in {"aac", "paint", "autohorizon"}
            or "action_semantics" in payload
        ):
            return dict(payload)
        return payload["actions"]

    def request(
        self,
        message_type: str,
        payload: dict[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Send one frame and block until its matching reply arrives."""

        request_id = self._send(message_type, payload)
        return self._await_reply(
            self._conn,
            request_id=request_id,
            message_type=message_type,
            timeout_s=self._request_timeout_s if timeout_s is None else timeout_s,
        )

    def submit_infer(
        self, observation: dict[str, Any], *, sampling: dict[str, Any] | None = None,
    ) -> str:
        """Upload an observation without waiting for model execution."""
        if not self.multi_inflight:
            raise XPolicyLabProtocolError("server does not support multi_inflight")
        request_id = self._send(INFER, {
            "observation": observation,
            "sampling": {"mode": "default"} if sampling is None else sampling,
            "latest_only": True,
        })
        self._step += 1
        return request_id

    def poll_infer(self) -> dict[str, Any] | None:
        """Receive one available result or superseded acknowledgement."""
        if self._conn is None:
            raise XPolicyLabProtocolError("client is not connected")
        try:
            reply = unpack_frame(self._conn.recv(timeout=0))
        except TimeoutError:
            return None
        self._pending_requests.discard(reply.get("message_id"))
        return reply

    def _send(self, message_type: str, payload: dict[str, Any]) -> str:

        conn = self._conn
        if conn is None:
            raise XPolicyLabProtocolError("client is not connected")
        request_id = str(uuid.uuid4())
        conn.send(
            pack_frame(
                {
                    "message_type": message_type,
                    "message_id": request_id,
                    "evaluation_id": self._evaluation_id,
                    "action_case_id": self._action_case_id,
                    "trial_id": self._trial_id,
                    "repeat_index": self._repeat_index,
                    "step": self._step,
                    "sent_at": _utc_now_iso(),
                    "payload": payload,
                }
            )
        )
        self._pending_requests.add(request_id)
        return request_id

    def _await_reply(
        self,
        conn: Any,
        *,
        request_id: str,
        message_type: str,
        timeout_s: float,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise XPolicyLabTimeoutError(f"{message_type} timed out after {timeout_s:.3f}s")
            reply = unpack_frame(conn.recv(timeout=remaining))
            self._pending_requests.discard(reply.get("message_id"))
            # Heartbeat acks and replies to an earlier request we already gave
            # up on share the socket; drain them instead of mismatching.
            if reply.get("message_id") != request_id:
                continue
            kind = reply.get("message_type")
            if kind == ERROR:
                payload = reply.get("payload")
                detail = payload if isinstance(payload, dict) else {}
                raise XPolicyLabProtocolError(
                    f"{message_type} failed: {detail.get('message', detail) or 'unknown error'}"
                )
            expected = _EXPECTED_REPLY.get(message_type)
            if expected is not None and kind != expected:
                raise XPolicyLabProtocolError(
                    f"{message_type} expected a {expected} reply, got {kind!r}"
                )
            return reply

    def close(self) -> None:
        conn, self._conn = self._conn, None
        if conn is None:
            return
        # The socket may already be gone; closing is best-effort by design.
        with suppress(Exception):
            conn.close()

    def __enter__(self) -> XPolicyLabWsClient:
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def normalize_url(server: str) -> str:
    """Accept ``host:port``, ``http://...`` and ``ws://...`` spellings alike."""

    url = server.strip().rstrip("/")
    if not url:
        raise ValueError("XPolicyLab server must be a non-empty string")
    if url.startswith("http://"):
        url = "ws://" + url[len("http://") :]
    elif url.startswith("https://"):
        url = "wss://" + url[len("https://") :]
    elif "://" not in url:
        url = "ws://" + url
    if not url.startswith(("ws://", "wss://")):
        raise ValueError(f"XPolicyLab server must be a ws:// or wss:// URL, got {server!r}")
    return url
