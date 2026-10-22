"""Exercise the real worker response boundary and shared preview admission."""

import queue
from dataclasses import replace
from types import SimpleNamespace

import pytest

from manimux.failures import describe_failure
from manimux.policies import worker
from manimux.runtime.preview import PreviewPlanner
from manimux.types import InferenceRequest


@pytest.mark.parametrize("hold_steps", [0, 16, 28])
@pytest.mark.parametrize("wire_seq", [-1, 0])
@pytest.mark.parametrize(
    "finished_ns,admission_ns,expected",
    [
        (3_000_000_000, 3_000_000_000, None),
        (6_000_000_001, 6_000_000_001, "inference_deadline_exceeded"),
        (3_000_000_000, 6_000_000_001, "deadline_exceeded_before_decode"),
    ],
)
def test_worker_response_keeps_deadline_for_admission(
    monkeypatch, hold_steps, wire_seq, finished_ns, admission_ns, expected
):
    model = SimpleNamespace(
        reset=lambda _: None, infer=lambda _: {"action": "recorded"},
        close=lambda: None,
    )
    monkeypatch.setattr(worker, "build_policy_model", lambda _: model)
    times = iter([1_000_000_000, finished_ns])
    monkeypatch.setattr(worker.time, "monotonic_ns", lambda: next(times))
    request = InferenceRequest("session", wire_seq, 1_000_000_000, 6_000_000_000, None)
    requests, responses, startup, resets = (queue.Queue() for _ in range(4))
    requests.put(request)
    requests.put(None)
    worker._worker_main(requests, responses, startup, resets, "session", {})
    response = responses.get_nowait()
    assert response.deadline_ns == request.deadline_ns
    assert response.observation_time_ns == request.observation_time_ns
    assert response.error is None
    # Warmup maps negative wire IDs back to its private planning sequence.
    response = replace(response, request_seq=0)
    planner = PreviewPlanner.__new__(PreviewPlanner)
    planner._multi_inflight = False
    planner._min_execute_steps = hold_steps
    planner._session_id = "session"
    planner._timeline = SimpleNamespace(accepted_request_seq=-1)
    planner._clock = SimpleNamespace(now_ns=lambda: admission_ns)
    planner.last_seq = 0
    planner.last_deadline_ns = request.deadline_ns
    if hold_steps == 0 and expected == "deadline_exceeded_before_decode":
        expected = None  # Legacy single-flight checks completion time only.
    assert planner.rejection_reason(response) == expected


@pytest.mark.parametrize("error", [
    "inference_deadline_exceeded",
    "deadline_exceeded_before_decode",
    "deadline_exceeded_before_start",
])
def test_deadline_failure_is_explicit_without_claiming_network_cause(error):
    description = describe_failure(error)
    assert description["reason"] == error
    assert description["title"] == "Inference response rejected by deadline check"
