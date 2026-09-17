from __future__ import annotations

import multiprocessing as mp
import queue
import time
from contextlib import suppress
from copy import deepcopy
from multiprocessing.queues import Queue
from typing import Any

from manimux.policies import build_policy_model
from manimux.policies.capabilities import PolicyCapabilities
from manimux.types import InferenceRequest, InferenceResponse


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
        self._startup_timeout_s = config["startup_timeout_s"]
        self._process = context.Process(
            target=_worker_main,
            args=(
                self._request_queue,
                self._response_queue,
                self._startup_queue,
                session_id,
                deepcopy(config),
            ),
            name="manimux-policy-worker",
            daemon=True,
        )
        self._started = False
        self._capabilities = PolicyCapabilities()

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
        if not self._started or not self._process.is_alive():
            raise RuntimeError("policy worker is not running")
        _put_latest(self._request_queue, request)

    def poll(self) -> InferenceResponse | None:
        if not self._started:
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
        return self._started and self._process.is_alive()

    @property
    def capabilities(self) -> PolicyCapabilities:
        return self._capabilities

    def close(self) -> None:
        if not self._started:
            return
        with suppress(OSError, ValueError):
            _put_latest(self._request_queue, None)
        self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
        self._request_queue.close()
        self._response_queue.close()
        self._startup_queue.close()
        self._started = False
