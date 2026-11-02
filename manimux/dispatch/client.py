"""Dependency-light client usable from RPent's independent Python environment."""

from __future__ import annotations

import json
import math
import time
import uuid
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import ProxyHandler, Request, build_opener

_TERMINAL = {"completed", "cancelled", "failed"}


class DispatchClient:
    def __init__(
        self, url: str, *, token: str | None = None, request_timeout_s: float = 5.0
    ) -> None:
        if not math.isfinite(request_timeout_s) or request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive and finite")
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = request_timeout_s
        # Private robot-service traffic must not go through an ambient HTTP proxy.
        self._opener = build_opener(ProxyHandler({}))
        self.session_id: str | None = None
        status = self.status()
        if status.get("protocol_version") != 1:
            raise ValueError("unsupported ManiMux dispatch protocol")
        self.session_id = status["session_id"]

    def _request(self, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = Request(
            self.url + path,
            headers=headers,
            data=None if body is None else json.dumps(body, allow_nan=False).encode(),
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                result: dict[str, Any] = json.load(response)
        except HTTPError as exc:
            error = json.loads(exc.read()).get("error", str(exc))
            raise RuntimeError(f"ManiMux dispatch HTTP {exc.code}: {error}") from exc
        if self.session_id is not None and result.get("session_id") != self.session_id:
            raise RuntimeError("ManiMux runtime restarted; construct a new client")
        return result

    def status(self) -> dict[str, Any]:
        return self._request("/status")

    def observe(self, request_id: str | None = None) -> dict[str, Any]:
        suffix = "" if request_id is None else "?request_id=" + quote(request_id, safe="")
        observation: dict[str, Any] = self._request("/observation" + suffix)["observation"]
        return observation

    def report_agent_decision(self, **decision: Any) -> dict[str, Any]:
        """Publish planner telemetry without changing runtime controls."""
        return self._request("/agent-decision", {"session_id": self.session_id, **decision})

    def submit(
        self,
        prompt: str,
        *,
        chunks: int,
        use_length: int,
        timeout_s: float = 120.0,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "/execute",
            {
                "session_id": self.session_id,
                "request_id": request_id or uuid.uuid4().hex,
                "prompt": prompt,
                "chunks": chunks,
                "use_length": use_length,
                "timeout_s": timeout_s,
            },
        )

    def result(self, request_id: str) -> dict[str, Any]:
        return self._request("/requests/" + quote(request_id, safe=""))

    def submit_motion(
        self,
        tool: str,
        arguments: dict[str, Any],
        *,
        timeout_s: float = 120.0,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "/motion",
            {
                "session_id": self.session_id,
                "request_id": request_id or uuid.uuid4().hex,
                "tool": tool,
                "arguments": arguments,
                "timeout_s": timeout_s,
            },
        )

    def motion(
        self,
        tool: str,
        arguments: dict[str, Any],
        *,
        timeout_s: float = 120.0,
        check_cancelled: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        return self._execute_call(
            lambda request_id: self.submit_motion(
                tool, arguments, timeout_s=timeout_s, request_id=request_id
            ),
            timeout_s=timeout_s,
            check_cancelled=check_cancelled,
        )

    def cancel(self, request_id: str) -> dict[str, Any]:
        return self._request("/cancel", {"session_id": self.session_id, "request_id": request_id})

    def finish(self) -> dict[str, Any]:
        """Explicitly end the rollout; ordinary client/toolkit cleanup does not do this."""
        return self._request("/finish", {"session_id": self.session_id})

    def execute(
        self,
        prompt: str,
        *,
        chunks: int,
        use_length: int,
        timeout_s: float = 120.0,
        check_cancelled: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        return self._execute_call(
            lambda request_id: self.submit(
                prompt,
                chunks=chunks,
                use_length=use_length,
                timeout_s=timeout_s,
                request_id=request_id,
            ),
            timeout_s=timeout_s,
            check_cancelled=check_cancelled,
        )

    def _execute_call(
        self,
        submit: Callable[[str], dict[str, Any]],
        *,
        timeout_s: float,
        check_cancelled: Callable[[], None] | None,
    ) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        submitted = False
        try:
            # Retain the ID even if the submission reply is lost.
            submitted = True
            result = submit(request_id)
            deadline = time.monotonic() + timeout_s + self.timeout + 2.0
            while result["state"] not in _TERMINAL:
                if check_cancelled is not None:
                    check_cancelled()
                if time.monotonic() > deadline:
                    raise TimeoutError(f"dispatch result timed out: {request_id}")
                time.sleep(0.05)
                result = self.result(request_id)
            result["observation"] = self.observe(request_id)
            return result
        except BaseException:
            if submitted:
                try:
                    self.cancel(request_id)
                    deadline = time.monotonic() + self.timeout
                    while time.monotonic() < deadline:
                        if self.result(request_id)["state"] in _TERMINAL:
                            break
                        time.sleep(0.05)
                except Exception:
                    # The server-side deadline still bounds execution after connection loss.
                    pass
            raise
