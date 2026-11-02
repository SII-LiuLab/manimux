"""HTTP transport for the runtime's bounded policy-call mailbox."""

from __future__ import annotations

import argparse
import base64
import hmac
import io
import json
import logging
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from PIL import Image

from manimux.runtime.dispatch import PolicyDispatch
from manimux.types import ObservationSnapshot


def encode_observation(snapshot: ObservationSnapshot) -> dict[str, Any]:
    frames = {}
    for name, frame in snapshot.frames.items():
        buffer = io.BytesIO()
        Image.fromarray(frame.data).save(buffer, format="PNG")
        frames[name] = {
            "encoding": "png",
            "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "capture_monotonic_ns": frame.capture_monotonic_ns,
            "sequence": frame.sequence,
        }
    return {
        "state": {name: values.tolist() for name, values in snapshot.state.groups.items()},
        "monotonic_ns": snapshot.state.monotonic_ns,
        "sequence": snapshot.state.sequence,
        "frames": frames,
    }


class DispatchServer:
    """Run network requests on worker threads; only EdgeRuntime touches hardware."""

    def __init__(
        self,
        dispatch: PolicyDispatch,
        *,
        host: str = "127.0.0.1",
        port: int = 8088,
        token: str | None = None,
    ) -> None:
        if host not in {"127.0.0.1", "localhost"} and not token:
            raise ValueError("a non-loopback bind requires MANIMUX_DISPATCH_TOKEN")

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def do_GET(self) -> None:
                self._handle()

            def do_POST(self) -> None:
                self._handle()

            def _handle(self) -> None:
                status = 200
                try:
                    if token and not hmac.compare_digest(
                        self.headers.get("Authorization", ""),
                        f"Bearer {token}",
                    ):
                        self._reply(401, {"error": "unauthorized"})
                        return
                    target = urlsplit(self.path)
                    if self.command == "GET":
                        if target.path == "/status":
                            result = dispatch.status()
                        elif target.path == "/observation":
                            request_ids = parse_qs(target.query).get("request_id", [])
                            request_id = request_ids[0] if request_ids else None
                            result = {
                                "session_id": dispatch.session_id,
                                "observation": encode_observation(dispatch.observation(request_id)),
                            }
                        elif target.path.startswith("/requests/"):
                            result = dispatch.result(unquote(target.path[len("/requests/") :]))
                        else:
                            self._reply(404, {"error": "unknown endpoint"})
                            return
                    else:
                        size = int(self.headers.get("Content-Length", "0"))
                        if not 0 < size <= 65536:
                            raise ValueError("request body must contain 1..65536 bytes")
                        body = json.loads(self.rfile.read(size))
                        if not isinstance(body, dict):
                            raise ValueError("request body must be a JSON object")
                        methods: dict[str, Callable[..., dict[str, Any]]] = {
                            "/execute": dispatch.submit,
                            "/motion": dispatch.submit_motion,
                            "/cancel": dispatch.cancel,
                            "/finish": dispatch.finish,
                            "/agent-decision": dispatch.report_agent_decision,
                        }
                        if target.path not in methods:
                            self._reply(404, {"error": "unknown endpoint"})
                            return
                        result = methods[target.path](**body)
                except KeyError:
                    status, result = 404, {"error": "unknown request_id"}
                except (ValueError, TypeError) as exc:
                    status, result = 400, {"error": str(exc)}
                except RuntimeError as exc:
                    status, result = 409, {"error": str(exc)}
                except Exception:
                    logging.getLogger(__name__).exception("dispatch request failed")
                    status, result = 500, {"error": "internal dispatch error"}
                self._reply(status, result)

            def _reply(self, status: int, result: dict[str, Any]) -> None:
                body = json.dumps(result, allow_nan=False).encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def setup(self) -> None:
                super().setup()
                self.connection.settimeout(5.0)

        self.http = ThreadingHTTPServer((host, port), Handler)
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return self.http.server_port

    def __enter__(self) -> DispatchServer:
        self.thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.http.shutdown()
        self.http.server_close()
        self.thread.join()


def main() -> int:
    parser = argparse.ArgumentParser(description="Retired standalone policy-dispatch launcher")
    parser.error(
        "Use manimux-rpent serve --config <agent.yaml>. Motion tools and "
        "execute_policy share that runtime; the recipe selects the experiment and motion profile."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
