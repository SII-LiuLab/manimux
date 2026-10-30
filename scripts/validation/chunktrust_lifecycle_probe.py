#!/usr/bin/env python3
"""CPU checks of production warmup, capability and dispatch boundaries.

All observations, responses, latencies and sinks are synthetic. No model, socket,
robot SDK or hardware is started. The existing production components run unchanged.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import unittest
from collections import deque
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from chunktrust_wire_probe import payload, synthetic_identity  # noqa: E402

from manimux.cli import load_config  # noqa: E402
from manimux.policies.capabilities import PolicyCapabilities  # noqa: E402
from manimux.policies.xpolicylab.codec import GroupLayout, decode_policy_actions  # noqa: E402
from manimux.policy_adapter import build_policy_adapter  # noqa: E402
from manimux.runtime.edge import EdgeRuntime  # noqa: E402
from manimux.runtime.executors import DirectExecutor  # noqa: E402
from manimux.runtime.inference import (  # noqa: E402
    build_inference_strategy,
    decode_strategy_action,
)
from manimux.runtime.safety import SafetyGuard  # noqa: E402
from manimux.runtime.warmup import PolicyWarmup  # noqa: E402
from manimux.types import (  # noqa: E402
    ActionContext,
    InferenceResponse,
    ObservationSnapshot,
    RobotCommand,
    RobotState,
    SensorFrame,
)

EXAMPLES = ROOT / "manimux/configs/experiments/example-yam-pi05/put_bottles"


def configuration(*, rtc=False):
    config = load_config(EXAMPLES / ("chunktrust_ahs_rtc.yaml" if rtc else "chunktrust_ahs.yaml"))
    config["policy"]["expected_backend"] = synthetic_identity(rtc=rtc)
    config["policy"]["timeout_s"] = 30.0
    config["inference"]["max_plan_age_s"] = 30.0
    return config


def capabilities(*, rtc=False):
    return PolicyCapabilities(
        sampling_modes=frozenset({"default", "rtc"} if rtc else {"default"}),
        backend_metadata=synthetic_identity(rtc=rtc),
    )


class Clock:
    ns = 10_000_000_000

    def now_ns(self):
        return self.ns


def observation(config, clock):
    frames = {
        name: SensorFrame(name, np.zeros((8, 8, 3), dtype=np.uint8), clock.ns, 1)
        for name in config["policy"]["adapter"]["camera_map"].values()
    }
    groups = {name: np.zeros(width) for name, width in config["robot"]["group_dims"].items()}
    return ObservationSnapshot(RobotState(groups, clock.ns, 1), frames)


class Worker:
    """Queued CPU replies with deliberately long cold calls and a real RESET fence."""

    is_alive = True

    def __init__(self, clock, *, rtc, bad_ack=False):
        self.clock = clock
        self.capabilities = capabilities(rtc=rtc)
        self.requests = []
        self.seen_branches = set()
        self.queue = deque()
        self.reset_ack = None
        self.reset_count = 0
        self.bad_ack = bad_ack

    def submit_latest(self, request):
        assert request.request_seq < 0, "warmup transport IDs must be negative"
        branch = getattr(request, "action_condition", None) is not None
        delay = 900_000_000 if branch not in self.seen_branches else 50_000_000
        self.seen_branches.add(branch)
        self.requests.append(request)
        ready = self.clock.ns + delay
        response = InferenceResponse(
            request.session_id, request.request_seq, ready, delay / 1e6,
            decode_policy_actions(
                deepcopy(payload()), format="joint", layouts=(
                    GroupLayout("left_arm", "left", 6, 1),
                    GroupLayout("right_arm", "right", 6, 1),
                ),
            ), observation_time_ns=request.observation_time_ns,
            deadline_ns=request.deadline_ns,
        )
        self.queue.append((ready, response))

    def poll(self):
        return self.queue.popleft()[1] if self.queue and self.queue[0][0] <= self.clock.ns else None

    def submit_reset(self, session_id):
        self.reset_count += 1
        self.reset_ack = SimpleNamespace(
            session_id=session_id,
            reset_seq=self.reset_count + int(self.bad_ack),
            error=None,
        )
        return self.reset_count

    def poll_reset(self):
        ack, self.reset_ack = self.reset_ack, None
        return ack


def warmup_fixture(*, rtc=False, bad_ack=False):
    config = configuration(rtc=rtc)
    strategy = build_inference_strategy(config)
    clock = Clock()
    worker = Worker(clock, rtc=rtc, bad_ack=bad_ack)
    warmup = PolicyWarmup(
        config, worker=worker, strategy=strategy,
        adapter=build_policy_adapter(config["robot"], config["policy"]),
        session_id="synthetic-lifecycle", clock=clock,
        executor=DirectExecutor(control_dt_s=0.01), kinematics=None,
        safety=SafetyGuard(config["robot"]["group_dims"], 4.0),
    )
    return config, strategy, clock, worker, warmup


class LifecycleChecks(unittest.TestCase):
    def run_warmup(self, *, rtc=False, bad_ack=False):
        config, strategy, clock, worker, warmup = warmup_fixture(rtc=rtc, bad_ack=bad_ack)
        formal = build_inference_strategy(config)
        before = formal.state_dict()
        warmup.activate()
        for _ in range(3000):
            warmup.advance(paused=not warmup.ready, snapshot=observation(config, clock))
            if warmup.phase in {"complete", "error"}:
                break
            clock.ns += 10_000_000
        self.assertEqual(formal.state_dict(), before)
        self.assertEqual(worker.reset_count, 1, warmup.metadata())
        self.assertTrue(warmup.ready, warmup.metadata())
        self.assertTrue(warmup.latency_ns)
        self.assertLess(max(warmup.latency_ns), 100_000_000)
        for branch in warmup.first_inference_ms:
            self.assertGreaterEqual(warmup.first_inference_ms[branch], 900.0)
        if bad_ack:
            self.assertEqual(warmup.phase, "error")
            self.assertIn("reset acknowledgement", warmup.error)
        else:
            self.assertEqual(warmup.phase, "complete", warmup.metadata())
            self.assertEqual(strategy.state_dict()["history_revision"], 0)
            self.assertEqual(strategy.state_dict()["selector"]["revision"], 0)
        if rtc:
            conditioned = [r for r in worker.requests if r.action_condition is not None]
            self.assertGreaterEqual(len(conditioned), 2)
            self.assertTrue(all(r.condition_weights.shape == (50,) for r in conditioned))
            formal.seed_warmup(latency_ns=list(warmup.latency_ns))
            self.assertLessEqual(max(formal.state_dict()["delay_policy_steps"]), 3)
            formal.reset()
            self.assertLessEqual(max(formal.state_dict()["delay_policy_steps"]), 3)
        return warmup.metadata()

    def test_ahs_warmup_reset_and_formal_isolation(self):
        self.run_warmup()

    def test_rtc_both_cold_branches_excluded_from_online_delay(self):
        self.run_warmup(rtc=True)

    def test_rejected_first_conditioned_trace_recovers_and_resets(self):
        config, strategy, clock, worker, warmup = warmup_fixture(rtc=True)
        submit = worker.submit_latest
        rejected_sequences = []

        def inject_once(request):
            submit(request)
            if request.action_condition is not None and not rejected_sequences:
                ready, response = worker.queue[-1]
                raw = deepcopy(response.raw_action)
                raw["denoise_trace"]["v"] = raw["denoise_trace"]["v"].astype(np.float64)
                worker.queue[-1] = (ready, replace(response, raw_action=raw))
                rejected_sequences.append(request.request_seq)

        with patch.object(worker, "submit_latest", side_effect=inject_once):
            warmup.activate()
            for _ in range(3000):
                warmup.advance(paused=not warmup.ready, snapshot=observation(config, clock))
                if warmup.phase in {"complete", "error"}:
                    break
                clock.ns += 10_000_000
        self.assertEqual(len(rejected_sequences), 1)
        failures = [fields for name, fields in warmup.take_events()
                    if name == "warmup_inference_failed"]
        self.assertEqual(len(failures), 1)
        self.assertIn("denoise_trace.v", failures[0]["error"])
        self.assertIn("float32", failures[0]["error"])
        self.assertEqual(warmup.phase, "complete", warmup.metadata())
        self.assertTrue(warmup.ready)
        self.assertGreaterEqual(sum(r.action_condition is not None for r in worker.requests), 3)
        self.assertEqual(worker.reset_count, 1)
        self.assertEqual(strategy.state_dict()["selector"]["revision"], 0)
        self.assertEqual(strategy.state_dict()["history_revision"], 0)
        self.assertEqual(strategy.state_dict()["pending_requests"], [])

    def test_formal_first_accepted_latency_is_not_excluded(self):
        config, strategy, clock, worker, warmup = warmup_fixture(rtc=True)
        # No activate/begin_warmup: exercise an ordinary formal strategy through
        # real decoding, selection and timeline acceptance on a disconnected planner.
        strategy.validate_capabilities(worker.capabilities)
        planner = warmup.planner
        snapshot = observation(config, clock)
        planner.initialize(snapshot.state)
        dt = int(config["policy"]["action_dt_s"] * 1e9)
        strategy.seed_warmup(latency_ns=[dt])
        submission = planner.submission(snapshot, in_flight=False)
        self.assertIsNotNone(submission)
        request = submission.request
        planner.submitted(request, clock.ns)
        clock.ns += 4 * dt
        raw = decode_policy_actions(deepcopy(payload()), format="joint", layouts=(
            GroupLayout("left_arm", "left", 6, 1), GroupLayout("right_arm", "right", 6, 1),
        ))
        response = InferenceResponse(
            request.session_id, request.request_seq, clock.ns, 4 * dt / 1e6, raw,
            observation_time_ns=request.observation_time_ns, deadline_ns=request.deadline_ns,
        )
        decoded = decode_strategy_action(strategy, planner._adapter, raw, ActionContext(
            response.request_seq, response.observation_time_ns, response.finished_time_ns,
        ))
        _, result, fields = planner.accept(decoded, response, snapshot.state)
        self.assertTrue(result.accepted, result.reason)
        self.assertEqual(fields["measured_delay"], 4)
        self.assertEqual(strategy.state_dict()["delay_policy_steps"], [1, 4])
        self.assertEqual(strategy.state_dict()["selector"]["revision"], 1)

    def test_reset_ack_identity_required(self):
        self.run_warmup(bad_ack=True)

    def test_capability_gate_and_startup_order(self):
        for rtc in (False, True):
            config = configuration(rtc=rtc)
            runtime = EdgeRuntime.__new__(EdgeRuntime)
            runtime._config = config
            runtime._strategy = build_inference_strategy(config)
            runtime._worker = SimpleNamespace(capabilities=capabilities(rtc=rtc))
            runtime._validate_policy_capabilities()
            metadata = synthetic_identity(rtc=rtc)
            metadata["model"].pop("denoise_trace")
            runtime._worker.capabilities = PolicyCapabilities(
                sampling_modes=capabilities(rtc=rtc).sampling_modes,
                backend_metadata=metadata,
            )
            with self.assertRaises(ValueError):
                runtime._validate_policy_capabilities()
        # Current published trace server is default-only. RTC must reject it.
        runtime._worker.capabilities = capabilities(rtc=False)
        with self.assertRaisesRegex(RuntimeError, "sampling modes"):
            runtime._validate_policy_capabilities()
        # Static startup-order assertion complements the executed validation hooks.
        tree = ast.parse((ROOT / "manimux/runtime/edge.py").read_text())
        run = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run")
        calls = [(n.lineno, ast.unparse(n.func)) for n in ast.walk(run) if isinstance(n, ast.Call)]
        first_gate = min(
            line for line, name in calls if name == "self._validate_policy_capabilities"
        )
        connect = min(line for line, name in calls if name == "self._robot.connect")
        self.assertLess(first_gate, connect)

    def test_dispatch_success_failure_and_validated_only(self):
        groups = {"left_arm": np.zeros(7), "right_arm": np.zeros(7)}
        state = RobotState(deepcopy(groups), 10_000_000_000, 1)
        command = RobotCommand(deepcopy(groups), state.monotonic_ns, "selected")
        for outcome in (
            "success", "validated_only", "dispatch_failure", "safety_failure", "legacy",
        ):
            with self.subTest(outcome=outcome):
                seen = []
                runtime = EdgeRuntime.__new__(EdgeRuntime)
                runtime._config = {"robot": {"options": {"execute": outcome != "validated_only"}}}
                runtime._strategy = (
                    object() if outcome == "legacy"
                    else SimpleNamespace(on_command_feedback=seen.append)
                )
                runtime._timeline = SimpleNamespace(accepted_request_seq=0)
                runtime._safety = SafetyGuard({name: 7 for name in groups}, 4.0)
                runtime._robot = Mock(kinematics=SimpleNamespace(models={}))
                runtime._robot.command_status.return_value = {"partially_submitted": True}
                recorder = Mock()
                candidate = deepcopy(command)
                if outcome == "dispatch_failure":
                    runtime._robot.send_command.side_effect = RuntimeError(
                        "synthetic partial dispatch"
                    )
                if outcome == "safety_failure":
                    candidate.groups["left_arm"][0] = 5.0
                if outcome.endswith("failure"):
                    with self.assertRaises((ValueError, RuntimeError)):
                        runtime._send_command(candidate, state, recorder, step=0,
                                              reference_plan_id="selected")
                    self.assertFalse(seen)
                    recorder.event.assert_called_once()
                else:
                    runtime._send_command(candidate, state, recorder, step=0,
                                          reference_plan_id="selected")
                    runtime._robot.send_command.assert_called_once()
                    if outcome != "legacy":
                        self.assertEqual(seen[0].source, outcome if outcome == "validated_only"
                                         else "submitted")
                        candidate.groups["left_arm"][0] = 1.0
                        self.assertEqual(seen[0].command.groups["left_arm"][0], 0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with patch("socket.socket", side_effect=AssertionError("no sockets in CPU lifecycle probe")):
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(LifecycleChecks)
        )
    report = {
        "passed": result.wasSuccessful(), "tests": result.testsRun,
        "failures": [str(test) for test, _ in result.failures + result.errors],
        "evidence": "synthetic CPU production-component lifecycle checks",
        "model_inference": False, "hardware": False,
        "startup_order": "static AST order plus executed capability hook",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
