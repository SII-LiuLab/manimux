"""Runtime control mailbox; network I/O belongs exclusively to its worker thread."""

from __future__ import annotations

import logging
import math
import threading
import time

from .communication import ControlClient

logger = logging.getLogger(__name__)


def control_parameters(**options) -> dict:
    values = {"poll_hz": 100.0, "timeout_s": 0.02, "max_age_s": 0.05}
    unknown = options.keys() - values.keys()
    if unknown:
        raise ValueError(f"Unknown robogui control options: {sorted(unknown)}")
    values.update(options)
    for name, value in values.items():
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"robogui.control.{name} must be finite and positive")
        values[name] = float(value)
    return values


class RuntimeControlClient:
    """Read a fresh local snapshot and consume latched controls without network waits."""

    def __init__(self, endpoint: str = "tcp://127.0.0.1:5569", **options) -> None:
        config = control_parameters(**options)
        self._endpoint = endpoint
        self._period = 1.0 / config["poll_hz"]
        self._timeout_ms = max(1, math.ceil(config["timeout_s"] * 1000))
        self._max_age = config["max_age_s"]
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._updated_at: float | None = None
        self._paused = True
        self._pause_pending = False
        self._home = False
        self._finish = False
        self._finished = False
        self._finish_home: bool | None = None
        self._transport_status = "not_ready"
        self._timeouts = 0
        self._errors = 0
        self._reply_rtt_ms: float | None = None
        self._server_timing: dict = {}
        self._thread = threading.Thread(
            target=self._run, name="runtime-robogui-controls", daemon=True,
        )
        self._thread.start()

    def _update(
        self, state: dict, *, sampled_at: float | None = None,
        reply_rtt_ms: float | None = None,
    ) -> None:
        # Only literal False enables execution; missing/malformed values fail closed.
        paused = state.get("paused") is not False
        with self._lock:
            self._paused = paused
            self._pause_pending |= paused
            self._home |= state.get("home_requested") is True
            if state.get("finish_requested") is True:
                self._finish = True
                self._finished = True
                value = state.get("finish_home")
                self._finish_home = value if isinstance(value, bool) else None
            # Conservatively date the sample to request time, not a delayed receipt.
            self._updated_at = time.monotonic() if sampled_at is None else sampled_at
            self._transport_status = "ok"
            self._reply_rtt_ms = reply_rtt_ms
            self._server_timing = dict(state.get("control_timing") or {})

    def _missing_reply(self, status: str) -> None:
        with self._lock:
            self._transport_status = status
            self._timeouts += status == "timeout"
            self._errors += status != "timeout"

    def poll(self) -> dict:
        with self._lock:
            age = None if self._updated_at is None else time.monotonic() - self._updated_at
            reason = (
                "finish" if self._finished else
                "control_thread_stopped" if self._stop.is_set() else
                "home" if self._home else
                "not_ready" if age is None else
                "stale_control" if age > self._max_age else
                "robogui_pause" if self._paused or self._pause_pending else
                "running"
            )
            state = {
                "paused": reason != "running",
                "home_requested": self._home and not self._finished,
                "finish_requested": self._finish,
                "finish_home": self._finish_home if self._finish else None,
                "diagnostics": {
                    "reason": reason, "transport_status": self._transport_status,
                    "control_age_ms": None if age is None else age * 1000,
                    "timeout_count": self._timeouts, "error_count": self._errors,
                    "reply_rtt_ms": self._reply_rtt_ms,
                    "server_timing": dict(self._server_timing),
                },
            }
            self._pause_pending = self._home = self._finish = False
            self._finish_home = None
            return state

    def _run(self) -> None:
        client = None
        try:
            # Creation, requests, reconnects and close all stay on the same thread.
            client = ControlClient(
                self._endpoint, timeout_ms=self._timeout_ms,
                reply_timeout_ms=max(self._timeout_ms, math.ceil(self._max_age * 1000)),
            )
            while not self._stop.is_set():
                started = time.monotonic()
                state = client.poll_state()
                if state is None:
                    self._missing_reply(client.last_status)
                else:
                    self._update(
                        state, sampled_at=client.last_reply_request_started_at,
                        reply_rtt_ms=client.last_reply_rtt_ms,
                    )
                if state is not None and state.get("finish_requested") is True:
                    break
                self._stop.wait(max(0.0, self._period - (time.monotonic() - started)))
        except Exception:
            logger.exception("RoboGUI control communication stopped; runtime will remain paused")
        finally:
            self._stop.set()
            if client is not None:
                client.close()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self._timeout_ms / 1000 + 1.0)
        if self._thread.is_alive():
            raise RuntimeError("RoboGUI control thread did not stop")
