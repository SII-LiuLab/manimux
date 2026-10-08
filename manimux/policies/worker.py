from __future__ import annotations

import multiprocessing as mp
import queue
import time
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass
from multiprocessing.queues import Queue
from typing import Any

from manimux.policies import build_policy_model
from manimux.policies.capabilities import PolicyCapabilities
from manimux.types import InferenceRequest, InferenceResponse


@dataclass(frozen=True)
class _PolicyResetRequest:
    session_id: str
    reset_seq: int


@dataclass(frozen=True)
class PolicyResetResult:
    """Acknowledgement emitted only after the backend's reset call returns."""

    session_id: str
    reset_seq: int
    finished_time_ns: int
    error: str | None = None


def _put_latest(target: Queue[Any], item: object) -> None:
    try:
        target.put_nowait(item)
        return
    except queue.Full:
        pass
    with suppress(queue.Empty):
        target.get_nowait()
    target.put_nowait(item)


def _worker_main(
    request_queue: Queue[Any],
    response_queue: Queue[Any],
    startup_queue: Queue[Any],
    reset_queue: Queue[Any],
    session_id: str,
    config_data: dict[str, object],
) -> None:
    model = None
    try:
        config = config_data
        model = build_policy_model(config)
        model.reset(session_id)
    except Exception as exc:
        _put_latest(startup_queue, ("error", f"{type(exc).__name__}:{exc}"))
        return
    try:
        capability_method = getattr(model, "capabilities", None)
        capabilities = capability_method() if callable(capability_method) else PolicyCapabilities()
        if not isinstance(capabilities, PolicyCapabilities):
            raise TypeError("model capabilities have an invalid type")
    except Exception as exc:
        _put_latest(startup_queue, ("error", f"capability_error:{type(exc).__name__}:{exc}"))
        return
    _put_latest(startup_queue, ("ready", capabilities))
    try:
        while True:
            request = request_queue.get()
            if request is None:
                break
            if isinstance(request, _PolicyResetRequest):
                error = None
                try:
                    model.reset(request.session_id)
                    session_id = request.session_id
                except Exception as exc:
                    error = f"reset_error:{type(exc).__name__}:{exc}"
                _put_latest(
                    reset_queue,
                    PolicyResetResult(
                        session_id=request.session_id,
                        reset_seq=request.reset_seq,
                        finished_time_ns=time.monotonic_ns(),
                        error=error,
                    ),
                )
                continue
            if not isinstance(request, InferenceRequest):
                continue
            started_ns = time.monotonic_ns()
            if started_ns > request.deadline_ns:
                response = InferenceResponse(
                    session_id=session_id,
                    request_seq=request.request_seq,
                    finished_time_ns=started_ns,
                    inference_ms=0.0,
                    raw_action=None,
                    observation_time_ns=request.observation_time_ns,
                    error="deadline_exceeded_before_start",
                )
                _put_latest(response_queue, response)
                continue
            try:
                action = model.infer(request)
                finished_ns = time.monotonic_ns()
                response = InferenceResponse(
                    session_id=session_id,
                    request_seq=request.request_seq,
                    finished_time_ns=finished_ns,
                    inference_ms=(finished_ns - started_ns) / 1_000_000,
                    raw_action=action,
                    observation_time_ns=request.observation_time_ns,
                )
            except Exception as exc:  # worker boundary must report model failures
                finished_ns = time.monotonic_ns()
                response = InferenceResponse(
                    session_id=session_id,
                    request_seq=request.request_seq,
                    finished_time_ns=finished_ns,
                    inference_ms=(finished_ns - started_ns) / 1_000_000,
                    raw_action=None,
                    observation_time_ns=request.observation_time_ns,
                    error=f"model_error:{type(exc).__name__}:{exc}",
                )
            _put_latest(response_queue, response)
    finally:
        if model is not None:
            model.close()


class PolicyWorkerClient:
    """One local model process with latest-wins bounded request/response queues."""

    def __init__(self, config: dict, session_id: str) -> None:
        context = mp.get_context("spawn")
        self._request_queue: Queue[Any] = context.Queue(maxsize=1)
        self._response_queue: Queue[Any] = context.Queue(maxsize=1)
        self._startup_queue: Queue[Any] = context.Queue(maxsize=1)
        self._reset_queue: Queue[Any] = context.Queue(maxsize=1)
        self._startup_timeout_s = config["startup_timeout_s"]
        self._process = context.Process(
            target=_worker_main,
            args=(
                self._request_queue,
                self._response_queue,
                self._startup_queue,
                self._reset_queue,
                session_id,
                deepcopy(config),
            ),
            name="manimux-policy-worker",
            daemon=True,
        )
        self._started = False
        self._stopping = False
        self._capabilities = PolicyCapabilities()
        self._reset_seq = 0
        self._pending_reset_seq: int | None = None
        self._reset_error: str | None = None

    def start(self) -> None:
        if self._started:
            return
        self._process.start()
        self._started = True
        try:
            status, detail = self._startup_queue.get(timeout=self._startup_timeout_s)
        except queue.Empty as exc:
            self.close()
            raise RuntimeError(
                f"policy worker did not become ready within {self._startup_timeout_s:.1f}s"
            ) from exc
        if status != "ready":
            self.close()
            raise RuntimeError(f"policy worker failed during startup: {detail}")
        if not isinstance(detail, PolicyCapabilities):
            self.close()
            raise RuntimeError("policy worker returned invalid capabilities")
        self._capabilities = detail

    def submit_latest(self, request: InferenceRequest) -> None:
        if self._stopping:
            raise RuntimeError("policy worker is stopping")
        if not self._started or not self._process.is_alive():
            raise RuntimeError("policy worker is not running")
        if self._pending_reset_seq is not None:
            raise RuntimeError("policy worker reset is pending")
        if self._reset_error is not None:
            raise RuntimeError(f"policy worker reset failed: {self._reset_error}")
        _put_latest(self._request_queue, request)

    def submit_reset(self, session_id: str) -> int:
        """Queue a reset without waiting for inference or backend communication.

        The caller drains its outstanding inference first, then waits for
        ``poll_reset`` before submitting further inference. ``queue.Full`` means
        a queued request has not been consumed yet; it can be retried later.
        Unlike inference submissions, a reset never replaces pending work.
        """
        if self._stopping:
            raise RuntimeError("policy worker is stopping")
        if not self._started or not self._process.is_alive():
            raise RuntimeError("policy worker is not running")
        if self._pending_reset_seq is not None:
            raise RuntimeError("policy worker reset is already pending")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("policy worker reset requires a session ID")
        reset_seq = self._reset_seq + 1
        self._request_queue.put_nowait(_PolicyResetRequest(session_id, reset_seq))
        self._reset_seq = reset_seq
        self._pending_reset_seq = reset_seq
        return reset_seq

    def poll_reset(self) -> PolicyResetResult | None:
        """Poll the reset acknowledgement; an error keeps inference disabled."""
        if not self._started or self._stopping:
            return None
        try:
            result = self._reset_queue.get_nowait()
        except queue.Empty:
            return None
        if not isinstance(result, PolicyResetResult):
            raise TypeError("policy worker returned an unexpected reset message")
        if result.reset_seq != self._pending_reset_seq:
            raise RuntimeError("policy worker returned an unexpected reset sequence")
        self._pending_reset_seq = None
        self._reset_error = result.error
        return result

    def poll(self) -> InferenceResponse | None:
        if not self._started or self._stopping:
            return None
        try:
            response = self._response_queue.get_nowait()
        except queue.Empty:
            return None
        if not isinstance(response, InferenceResponse):
            raise TypeError("policy worker returned an unexpected message")
        return response

    @property
    def is_alive(self) -> bool:
        return self._started and not self._stopping and self._process.is_alive()

    @property
    def capabilities(self) -> PolicyCapabilities:
        return self._capabilities

    def request_stop(self) -> None:
        """Stop this transport worker immediately without waiting on inference.

        The worker owns no robot resources. Closing its process also disconnects
        its model socket, invalidating queued work for opted-in WS sessions.
        An already-running remote model call may finish; its output is discarded.
        ``close`` still joins the process and releases the IPC queues afterwards.
        """
        if not self._started or self._stopping:
            return
        self._stopping = True
        if self._process.is_alive():
            self._process.terminate()

    def close(self) -> None:
        if not self._started:
            return
        if not self._stopping:
            with suppress(OSError, ValueError):
                _put_latest(self._request_queue, None)
        self._process.join(timeout=2.0)
        if self._process.is_alive():
            if self._stopping:
                self._process.kill()
            else:
                self._process.terminate()
            self._process.join(timeout=1.0)
        self._request_queue.close()
        self._response_queue.close()
        self._startup_queue.close()
        self._reset_queue.close()
        self._started = False
