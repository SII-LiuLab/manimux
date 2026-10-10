"""Latest-observation scheduling around XPolicyLab's unchanged policy server."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field

from client_server.ws.model_server import PolicyServer, PolicyServerConfig, _InferenceConnection
from client_server.ws.protocol.exceptions import ErrorCode, WsError
from client_server.ws.protocol.messages import MessageType
from client_server.ws.protocol.schemas import Frame


@dataclass
class _PendingInference:
    frame: Frame
    generation: int | None
    connection: _InferenceConnection | None
    response: asyncio.Future[Frame | None]


@dataclass
class _Stream:
    latest: _PendingInference | None = None
    runner: asyncio.Task[None] | None = None


@dataclass
class StreamingPolicyServer(PolicyServer):
    """Coalesce only latest_only requests; inherit transport and model execution."""

    _streams: dict[int, _Stream] = field(default_factory=dict, init=False, repr=False)

    async def _dispatch_frame(
        self, frame: Frame, *, inference_generation: int | None = None,
        inference_connection: _InferenceConnection | None = None,
    ) -> Frame | None:
        if frame.message_type != MessageType.INFER or frame.payload.get("latest_only") is not True:
            response = await super()._dispatch_frame(
                frame, inference_generation=inference_generation,
                inference_connection=inference_connection,
            )
            if frame.message_type == MessageType.HELLO and response is not None:
                response.payload["capabilities"]["multi_inflight"] = True
            return response

        sampling = frame.payload.get("sampling") or {"mode": "default"}
        if not isinstance(sampling, Mapping) or sampling.get("mode") not in {"default", "rtc"}:
            raise WsError(ErrorCode.INVALID_FRAME, "latest_only requires default or RTC sampling")

        # The upstream execution wrapper owns duplicate IDs, caching and reset
        # generations. Each connection adds just one replaceable waiting input.
        key = id(inference_connection)
        stream = self._streams.get(key)
        if stream is None:
            stream = self._streams[key] = _Stream()
        previous = stream.latest
        if previous is not None and not previous.response.done():
            previous.response.set_result(self._reply(
                previous.frame, MessageType.INFER_RESULT, {"superseded": True},
            ))
        response = asyncio.get_running_loop().create_future()
        stream.latest = _PendingInference(
            frame, inference_generation, inference_connection, response,
        )
        if stream.runner is None:
            stream.runner = asyncio.create_task(self._infer_latest(key, stream))
        return await response

    async def _infer_latest(self, key: int, stream: _Stream) -> None:
        request = None
        try:
            while stream.latest is not None:
                request, stream.latest = stream.latest, None
                try:
                    if request.connection is not None and request.connection.disconnected:
                        raise WsError(
                            ErrorCode.INFER_FAILED, "inference cancelled by client disconnect",
                        )
                    result = await super()._dispatch_frame(
                        request.frame, inference_generation=request.generation,
                        inference_connection=request.connection,
                    )
                except Exception as exc:
                    if not request.response.done():
                        request.response.set_exception(exc)
                else:
                    if not request.response.done():
                        request.response.set_result(result)
                # The upstream handler delivers this reply independently while
                # the next inference starts with the newest waiting input.
                request = None
        finally:
            for pending in (request, stream.latest):
                if pending is not None and not pending.response.done():
                    pending.response.cancel()
            self._streams.pop(key, None)


def serve(config: dict) -> None:
    """Use the existing model loader with ManiMux's scheduling extension."""
    from XPolicyLab import setup_policy_server

    if config.get("protocol", "ws") != "ws":
        setup_policy_server.main(config)
        return
    model_class = setup_policy_server.eval_function_decorator(
        f"XPolicyLab.policy.{config.get('policy_name')}.model", "Model",
    )
    server = StreamingPolicyServer(
        model_class(config),
        PolicyServerConfig(
            host=config.get("host", "0.0.0.0"),
            port=int(config.get("port")),
            ws_ping_interval_s=config.get("ws_ping_interval_s", 20.0),
            ws_ping_timeout_s=config.get("ws_ping_timeout_s", 20.0),
            model_metadata=setup_policy_server._deployment_model_metadata(config),
        ),
    )
    try:
        asyncio.run(server.serve_forever())
    except KeyboardInterrupt:
        print("\nShutting down websocket policy server...")
