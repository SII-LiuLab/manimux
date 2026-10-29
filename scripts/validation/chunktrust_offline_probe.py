#!/usr/bin/env python3
"""Exercise ChunkTrust's public integration with synthetic data and no hardware.

Uses the real loader, wire codecs, both XPolicyLab client paths, JointAdapter,
strategy, timeline, DirectExecutor, command submission and preview/reset fence.
The transport is an in-memory byte queue: no socket, model or SDK is constructed.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import socket
import subprocess
import sys
import time
import unittest
from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass, replace
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import unquote, urlsplit

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).with_name("chunktrust_fixtures")
sys.path.insert(0, str(ROOT))

from manimux.cli import load_config  # noqa: E402
from manimux.policies.response_log import ResponseLog  # noqa: E402
from manimux.policies.xpolicylab.client import XPolicyLabWsPolicyModel  # noqa: E402
from manimux.policies.xpolicylab.ws_client import (  # noqa: E402
    XPolicyLabWsClient,
    pack_frame,
    unpack_frame,
)
from manimux.policy_adapter import build_policy_adapter  # noqa: E402
from manimux.runtime.edge import EdgeRuntime  # noqa: E402
from manimux.runtime.executors import DirectExecutor  # noqa: E402
from manimux.runtime.feedback import notify_command_feedback  # noqa: E402
from manimux.runtime.inference import (  # noqa: E402
    build_inference_strategy,
    decode_strategy_action,
)
from manimux.runtime.preview import PreviewPlanner  # noqa: E402
from manimux.runtime.safety import SafetyGuard  # noqa: E402
from manimux.runtime.warmup import PolicyWarmup  # noqa: E402
from manimux.types import (  # noqa: E402
    ActionContext,
    InferenceRequest,
    InferenceResponse,
    ObservationSnapshot,
    RobotCommand,
    RobotState,
    SensorFrame,
)

SELECTING_YAML = ROOT / "manimux/configs/experiments/example-yam-pi05/put_bottles/chunktrust.yaml"
SERIAL_YAML = SELECTING_YAML.with_name("serial.yaml")
GROUPS = ("left_arm", "right_arm")


def json_value(value):
    if is_dataclass(value) and not isinstance(value, type):
        return json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + "\n")


def git_revision(directory):
    if not (directory / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=directory,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def chunktrust_provenance():
    import chunktrust

    distribution = metadata.distribution("chunktrust")
    direct_text = distribution.read_text("direct_url.json")
    direct = None if direct_text is None else json.loads(direct_text)
    package = Path(chunktrust.__file__).resolve().parent
    source = None
    if direct is not None and urlsplit(direct.get("url", "")).scheme == "file":
        source = Path(unquote(urlsplit(direct["url"]).path))
    return {
        "version": distribution.version,
        "direct_url": direct,
        "editable": bool((direct or {}).get("dir_info", {}).get("editable", False)),
        "vcs_commit_id": (direct or {}).get("vcs_info", {}).get("commit_id"),
        "source_revision": None if source is None else git_revision(source),
        "package_source_sha256": {
            str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(package.rglob("*.py"))
        },
    }


def synthetic_arrays():
    """Inputs match the committed, independently captured numerical reference."""
    fixture = json.loads((FIXTURES / "synthetic_reference.json").read_text())
    generator = fixture["generator"]
    rng = np.random.default_rng(generator["seed"])
    velocity = rng.normal(0, generator["trace_std"], generator["shape"]).astype(np.float32)
    index = np.arange(50, dtype=np.float32)[:, None]
    dimension = np.arange(14, dtype=np.float32)[None, :]
    actions = (0.06 * np.sin(index * 0.08 + dimension * 0.17)).astype(np.float32)
    actions[:, [6, 13]] = 0.4 + 0.02 * np.sin(index * 0.08)
    return actions, velocity, fixture["expected"]


def synthetic_payload():
    actions, velocity, _ = synthetic_arrays()
    steps = []
    for row in actions:
        steps.append(
            {
                "left_arm_joint_state": row[:6],
                "left_ee_joint_state": row[6:7],
                "right_arm_joint_state": row[7:13],
                "right_ee_joint_state": row[13:14],
            }
        )
    return {"actions": steps, "denoise_trace": {"v": velocity}}


@dataclass
class ProbeClock:
    ns: int = 10_000_000_000

    def now_ns(self):
        return self.ns

    def sleep_until_ns(self, target_ns):
        self.ns = max(self.ns, target_ns)


def snapshot(config, now_ns):
    groups = {name: np.zeros(width) for name, width in config["robot"]["group_dims"].items()}
    for row in groups.values():
        row[-1] = 0.4
    frames = {
        name: SensorFrame(name, np.full((8, 8, 3), 40 + 30 * index, np.uint8), now_ns, index)
        for index, name in enumerate(config["policy"]["adapter"]["camera_map"].values())
    }
    return ObservationSnapshot(RobotState(groups, now_ns, 1), frames)


class MemoryConnection:
    """Byte-level protocol double, never a network connection or model server."""

    def __init__(self, identity, directory, payload=None):
        self.identity = deepcopy(identity)
        self.directory = directory
        self.payload = synthetic_payload() if payload is None else payload
        self.replies = deque()
        self.requests = []
        self.closed = False

    def send(self, wire):
        request = unpack_frame(wire)
        self.requests.append(request)
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / f"request_{len(self.requests):03}.msgpack").write_bytes(wire)
        kind = request["message_type"]
        if kind == "hello":
            payload = {
                "capabilities": {"sampling_modes": ["default"], "multi_inflight": True},
                "server": self.identity["server"],
                "model_metadata": deepcopy(self.identity["model"]),
            }
            reply_kind = "hello_ack"
        elif kind == "reset":
            payload, reply_kind = {"ok": True}, "reset_result"
        elif kind == "infer":
            if request["payload"]["sampling"] != {"mode": "default"}:
                raise AssertionError("trace deployments must request ordinary default sampling")
            payload, reply_kind = deepcopy(self.payload), "infer_result"
        else:
            raise AssertionError(f"unexpected offline RPC {kind}")
        reply = pack_frame(
            {
                "message_type": reply_kind,
                "message_id": request["message_id"],
                "payload": payload,
            }
        )
        (self.directory / f"response_{len(self.requests):03}.msgpack").write_bytes(reply)
        self.replies.append(reply)

    def recv(self, timeout=None):
        if not self.replies:
            raise TimeoutError("in-memory response queue is empty")
        return self.replies.popleft()

    def close(self):
        self.closed = True


class DisconnectedSink:
    """Submission recorder with no robot, SDK, CAN object or socket."""

    kinematics = SimpleNamespace(models={})

    def __init__(self, *, fail=False):
        self.fail = fail
        self.commands = []

    def send_command(self, command):
        if self.fail:
            raise RuntimeError("injected disconnected-sink dispatch failure")
        self.commands.append(deepcopy(command))


class ProbeContext:
    def __init__(self, output):
        self.output = output
        self.config = load_config(SELECTING_YAML)
        # A private station's diagnostic directory is never used by this probe.
        self.config["policy"]["options"]["response_log_dir"] = str(output / "provider_responses")
        self.config["robot"]["control_hz"] = 100.0
        self.results = []

    def client(self, directory, *, payload=None):
        wire = XPolicyLabWsClient(
            url="ws://127.0.0.1:1", evaluation_id="offline", trial_id="offline"
        )
        connection = MemoryConnection(self.config["policy"]["expected_backend"], directory, payload)
        with patch("websockets.sync.client.connect", return_value=connection):
            wire.connect()
        wire.reset()
        model = XPolicyLabWsPolicyModel(self.config["policy"])
        model._client, model._session_id = wire, "offline"
        return model, connection

    def planner(self, *, config=None, strategy=None, clock=None):
        config = self.config if config is None else config
        strategy = build_inference_strategy(config) if strategy is None else strategy
        clock = ProbeClock() if clock is None else clock
        adapter = build_policy_adapter(config["robot"], config["policy"])
        planner = PreviewPlanner(
            config,
            strategy=strategy,
            adapter=adapter,
            executor=DirectExecutor(control_dt_s=0.01),
            safety=SafetyGuard(config["robot"]["group_dims"], 4.0),
            kinematics=None,
            decoder=None,
            session_id="offline",
            clock=clock,
        )
        measured = snapshot(config, clock.now_ns())
        planner.initialize(measured.state)
        return planner, strategy, adapter, clock, measured


class ImmediateWorker:
    """The production warmup state machine sees queued client results and RESET ACKs."""

    is_alive = True

    def __init__(self, model, clock):
        self.model, self.clock = model, clock
        self.responses = deque()
        self.reset_ack = None
        self.reset_count = 0

    def submit_latest(self, request):
        raw = self.model.infer(request)
        self.responses.append(
            InferenceResponse(
                request.session_id,
                request.request_seq,
                self.clock.now_ns(),
                0.0,
                raw,
                observation_time_ns=request.observation_time_ns,
                deadline_ns=request.deadline_ns,
            )
        )

    def poll(self):
        return self.responses.popleft() if self.responses else None

    def submit_reset(self, session_id):
        self.reset_count += 1
        self.model.reset(session_id)
        self.reset_ack = SimpleNamespace(
            session_id=session_id, reset_seq=self.reset_count, error=None
        )
        return self.reset_count

    def poll_reset(self):
        result, self.reset_ack = self.reset_ack, None
        return result


class OfflineChecks(unittest.TestCase):
    context: ProbeContext

    def setUp(self):
        self.directory = self.context.output / "cases" / self._testMethodName
        self.directory.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.addCleanup(self._record)

    def _record(self):
        write_json(
            self.directory / "case.json",
            {
                "name": self._testMethodName,
                "elapsed_s": time.monotonic() - self.started,
                "evidence": "synthetic CPU integration; no network, model or hardware",
            },
        )

    def decoded(self, *, payload=None, seq=0, clock=None):
        model, connection = self.context.client(self.directory / f"wire_{seq}", payload=payload)
        self.addCleanup(model.close)
        clock = ProbeClock() if clock is None else clock
        obs = snapshot(self.context.config, clock.now_ns())
        request = InferenceRequest(
            "offline", seq, clock.now_ns(), clock.now_ns() + 10**10, obs, "Synthetic probe"
        )
        log_config = deepcopy(self.context.config["policy"])
        log_config["options"]["response_log_dir"] = str(self.directory / "provider_responses")
        with ResponseLog(log_config, request) as journal:
            raw = model.infer(request)
        connection.response_journal = journal.path
        response = InferenceResponse(
            "offline",
            seq,
            clock.now_ns(),
            0.0,
            raw,
            observation_time_ns=clock.now_ns(),
            deadline_ns=request.deadline_ns,
        )
        return raw, response, obs, model, connection

    def accept(self, planner, adapter, response, measured):
        context = ActionContext(
            response.request_seq, response.observation_time_ns, response.finished_time_ns
        )
        chunk = decode_strategy_action(planner._strategy, adapter, response.raw_action, context)
        return planner.accept(chunk, response, measured.state)

    def test_frozen_numerical_replay_and_non_candidate_k(self):
        from chunktrust._pi_ahs import Selector

        recorded = json.loads((FIXTURES / "recorded_decision_scores.json").read_text())
        replayed = []
        for scores, candidates, expected in zip(
            recorded["ahs_replan_scores"],
            recorded["ahs_replan_candidates"],
            recorded["selected_K"],
            strict=True,
        ):
            probability = Selector._normalize_score_distribution(
                np.asarray(scores), recorded["temperature"]
            )
            actual = int(round(float(np.dot(probability, candidates))))
            self.assertEqual(actual, expected)
            replayed.append(actual)
        planner, strategy, adapter, clock, measured = self.context.planner()
        raw, response, _, _, _ = self.decoded(clock=clock)
        chunk, result, _ = self.accept(planner, adapter, response, measured)
        self.assertTrue(result.accepted)
        actions, velocity, expected = synthetic_arrays()
        self.assertEqual(chunk.horizon_steps, expected["selected_k"])
        self.assertNotIn(
            chunk.horizon_steps, self.context.config["inference"]["chunktrust"]["candidates"]
        )
        info = chunk.metadata["chunktrust"]
        np.testing.assert_allclose(
            info["evidence"]["q_mix"], expected["q_mix"], atol=2e-6, rtol=2e-6
        )
        np.testing.assert_allclose(
            info["evidence"]["z_intra"], expected["z_intra"], atol=2e-6, rtol=2e-6
        )
        from chunktrust import AHS, AHSConfig

        replay = AHS(AHSConfig(**info["selector_state_before"]["config"]))
        replay.load_state_dict(info["selector_state_before"])
        replay_k, _ = replay.select(
            velocity[:, :, :14],
            actions,
            np.asarray(info["history_targets"], np.float32).reshape(-1, 14),
        )
        self.assertEqual(replay_k, chunk.horizon_steps)
        self.assertEqual(json_value(replay.state_dict()), info["selector_state_after"])
        self.assertEqual(strategy.state_dict()["history"].shape, (0, 14))
        np.savez_compressed(self.directory / "inputs.npz", actions=actions, velocity=velocity)
        write_json(self.directory / "selection.json", {"recorded_score_replay": replayed, **info})
        write_json(self.directory / "accepted_state.json", strategy.state_dict())
        self.assertEqual(raw["denoise_trace"]["v"].shape, (10, 50, 32))

    def test_both_client_paths_keep_trace_and_default_rpc(self):
        raw, response, measured, model, connection = self.decoded()
        original = synthetic_payload()["denoise_trace"]["v"]
        np.testing.assert_array_equal(raw["denoise_trace"]["v"], original)
        self.assertEqual(raw["denoise_trace"]["v"].dtype, np.float32)
        request = InferenceRequest(
            "offline", 1, measured.state.monotonic_ns, 10**20, measured, "Streaming transport only"
        )
        self.assertTrue(model.submit(request))
        streamed = model.poll()
        self.assertIsNotNone(streamed)
        self.assertIsNone(streamed.error)
        np.testing.assert_array_equal(streamed.raw_action["denoise_trace"]["v"], original)
        for request in connection.requests:
            if request["message_type"] == "infer":
                self.assertEqual(request["payload"]["sampling"], {"mode": "default"})
        write_json(self.directory / "parsed_outputs.json", {"blocking": raw, "streaming": streamed})

    def test_bad_trace_is_persisted_then_rejected_without_selector_update(self):
        variants = {
            "missing": None,
            "wrong_dtype": "dtype",
            "wrong_shape": "shape",
            "wrong_steps": "steps",
            "nonfinite": "nan",
        }
        for case_index, (label, mutation) in enumerate(variants.items(), start=100):
            with self.subTest(label=label):
                payload = synthetic_payload()
                if mutation is None:
                    payload.pop("denoise_trace")
                elif mutation == "dtype":
                    payload["denoise_trace"]["v"] = payload["denoise_trace"]["v"].astype(np.float64)
                elif mutation == "shape":
                    payload["denoise_trace"]["v"] = payload["denoise_trace"]["v"][:, :-1]
                elif mutation == "steps":
                    payload["denoise_trace"]["v"] = payload["denoise_trace"]["v"][:-1]
                else:
                    payload["denoise_trace"]["v"][0, 0, 0] = np.nan
                planner, strategy, adapter, clock, measured = self.context.planner()
                before = json_value(strategy.state_dict()["selector"])
                _, response, _, _, connection = self.decoded(
                    payload=payload, seq=case_index, clock=clock
                )
                journal_path = connection.response_journal / "response-01.json"
                self.assertTrue(journal_path.is_file())
                journal = json.loads(journal_path.read_text())
                self.assertEqual(journal["stage"], "provider_wire")
                self.assertEqual(journal["response"]["encoding"], "base64")
                captured = unpack_frame(base64.b64decode(journal["response"]["data"]))
                if mutation is None:
                    self.assertNotIn("denoise_trace", captured["payload"])
                else:
                    captured_velocity = captured["payload"]["denoise_trace"]["v"]
                    original_velocity = payload["denoise_trace"]["v"]
                    self.assertEqual(captured_velocity.dtype, original_velocity.dtype)
                    np.testing.assert_array_equal(captured_velocity, original_velocity)
                with self.assertRaisesRegex(ValueError, "denoise_trace"):
                    self.accept(planner, adapter, response, measured)
                strategy.on_response_rejected(response)
                self.assertEqual(before, json_value(strategy.state_dict()["selector"]))
                self.assertEqual(strategy.state_dict()["pending"], [])
                write_json(self.directory / f"rejected_{label}.json", strategy.state_dict())

    def test_schema_and_unsupported_config_fail_closed(self):
        mutations = {
            "missing_contract": lambda c: c["policy"]["expected_backend"]["model"].pop(
                "denoise_trace"
            ),
            "wrong_schema": lambda c: c["policy"]["expected_backend"]["model"][
                "denoise_trace"
            ].update(schema_version=2),
            "guided_velocity": lambda c: c["policy"]["expected_backend"]["model"][
                "denoise_trace"
            ].update(velocity_kind="guided_update"),
            "boolean_indices": lambda c: c["policy"]["expected_backend"]["model"][
                "denoise_trace"
            ].update(valid_action_indices=[False, *range(1, 14)]),
            "duplicate_indices": lambda c: c["policy"]["expected_backend"]["model"][
                "denoise_trace"
            ].update(valid_action_indices=[0, *range(13)]),
            "out_of_range_indices": lambda c: c["policy"]["expected_backend"]["model"][
                "denoise_trace"
            ].update(valid_action_indices=[*range(13), 32]),
            "nonfinite_temperature": lambda c: c["inference"]["chunktrust"].update(
                temperature=float("nan")
            ),
            "async": lambda c: c["inference"].update(inference_schedule="single_inflight"),
            "blend": lambda c: c["inference"].update(blend_policy_steps=2),
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label):
                config = deepcopy(self.context.config)
                mutation(config)
                with self.assertRaises((TypeError, ValueError)):
                    build_inference_strategy(config)
        edge = EdgeRuntime.__new__(EdgeRuntime)
        edge._strategy = build_inference_strategy(self.context.config)
        edge._config = self.context.config
        model, _ = self.context.client(self.directory / "hello")
        self.addCleanup(model.close)
        edge._worker = SimpleNamespace(capabilities=model.capabilities())
        edge._validate_policy_capabilities()
        bad = deepcopy(model.capabilities().backend_metadata)
        bad["model"]["denoise_trace"]["enabled"] = False
        edge._worker = SimpleNamespace(
            capabilities=replace(model.capabilities(), backend_metadata=bad)
        )
        with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            edge._validate_policy_capabilities()
        bad_steps = deepcopy(model.capabilities().backend_metadata)
        bad_steps["model"]["num_steps"] += 1
        edge._worker = SimpleNamespace(
            capabilities=replace(model.capabilities(), backend_metadata=bad_steps)
        )
        with self.assertRaisesRegex(RuntimeError, "num_steps"):
            edge._validate_policy_capabilities()

    def test_rejection_duplicate_and_late_reply_are_transactional(self):
        planner, strategy, adapter, clock, measured = self.context.planner()
        _, response, _, _, _ = self.decoded(clock=clock)
        chunk = adapter.decode_action(
            response.raw_action, ActionContext(0, clock.now_ns(), clock.now_ns())
        )
        before = json_value(strategy.state_dict()["selector"])
        first = strategy.prepare_chunk(chunk=chunk, response=response, now_ns=clock.now_ns())
        second = strategy.prepare_chunk(chunk=chunk, response=response, now_ns=clock.now_ns())
        self.assertEqual(first.horizon_steps, second.horizon_steps)
        self.assertEqual(before, json_value(strategy.state_dict()["selector"]))
        strategy.on_response_rejected(replace(response, error="injected rejection"))
        self.assertEqual(before, json_value(strategy.state_dict()["selector"]))
        late = replace(response, deadline_ns=clock.now_ns() - 1)
        planner.last_seq, planner.last_deadline_ns = 0, clock.now_ns() - 1
        self.assertIsNotNone(planner.rejection_reason(late))
        strategy.on_response_rejected(late)
        old = replace(response, observation_time_ns=clock.now_ns() - 10**10)
        old_chunk = adapter.decode_action(
            old.raw_action,
            ActionContext(old.request_seq, old.observation_time_ns, old.finished_time_ns),
        )
        _, rejection, _ = planner.accept(old_chunk, old, measured.state)
        self.assertFalse(rejection.accepted)
        strategy.on_response_rejected(old)
        self.assertEqual(before, json_value(strategy.state_dict()["selector"]))
        selected, result, _ = self.accept(planner, adapter, response, measured)
        self.assertTrue(result.accepted)
        accepted = json_value(strategy.state_dict()["selector"])
        with self.assertRaisesRegex(ValueError, "acceptance"):
            strategy.on_plan_accepted(
                chunk=selected, result=result, response=response, now_ns=clock.now_ns()
            )
        self.assertEqual(accepted, json_value(strategy.state_dict()["selector"]))
        write_json(
            self.directory / "transaction_states.json", {"before": before, "accepted": accepted}
        )

    def test_preview_policy30_control100_deduplicates_and_holds_k_intervals(self):
        planner, strategy, adapter, clock, measured = self.context.planner()
        _, response, _, _, _ = self.decoded(clock=clock)
        chunk, result, _ = self.accept(planner, adapter, response, measured)
        self.assertTrue(result.accepted)
        origin = clock.now_ns()
        end = origin + chunk.horizon_steps * chunk.dt_ns
        self.assertEqual(planner._timeline.remaining_ns(origin), end - origin)
        for ns in range(origin, end, 10_000_000):
            clock.ns = ns
            planner.tick(measured.state, loop_started_ns=ns)
        state = strategy.state_dict()
        self.assertEqual(state["history_sources"]["preview"], chunk.horizon_steps)
        self.assertEqual(state["history_sources"]["submitted"], 0)
        self.assertEqual(len(state["history"]), chunk.horizon_steps)
        self.assertGreater(planner.steps, chunk.horizon_steps)
        clock.ns = end
        planner.tick(measured.state, loop_started_ns=end)
        self.assertEqual(strategy.state_dict()["history_sources"], state["history_sources"])
        write_json(
            self.directory / "preview_history.json", {"control_ticks": planner.steps, **state}
        )

    def test_real_dispatch_boundary_feedback_success_failure_and_reset(self):
        planner, strategy, adapter, clock, measured = self.context.planner()
        _, response, _, _, _ = self.decoded(clock=clock)
        chunk, result, _ = self.accept(planner, adapter, response, measured)
        self.assertTrue(result.accepted)
        edge = EdgeRuntime.__new__(EdgeRuntime)
        edge._config, edge._strategy = self.context.config, strategy
        edge._timeline, edge._safety = planner._timeline, planner.safety
        edge._robot = DisconnectedSink()
        events = []
        recorder = SimpleNamespace(event=lambda kind, **fields: events.append((kind, fields)))
        reference = planner._timeline.reference_horizon(
            now_ns=clock.now_ns(), dt_ns=10_000_000, horizon_steps=2
        )
        command = planner.execution.step(
            clock.now_ns(), measured.state, reference, last_command=planner.last_command
        ).command
        edge._send_command(
            command, measured.state, recorder, step=0, reference_plan_id=chunk.plan_id
        )
        self.assertEqual(strategy.state_dict()["history_sources"]["submitted"], 1)
        saved = strategy.state_dict()["history"].copy()
        command.groups["left_arm"][:] = 0.0
        np.testing.assert_array_equal(strategy.state_dict()["history"], saved)
        next_command = RobotCommand(
            {name: values[1].copy() for name, values in chunk.groups.items()},
            clock.now_ns() + chunk.dt_ns,
            chunk.plan_id,
        )
        edge._robot.fail = True
        with (
            self.assertLogs("manimux.runtime.edge", level="ERROR"),
            self.assertRaisesRegex(RuntimeError, "dispatch failure"),
        ):
            edge._send_command(
                next_command, measured.state, recorder, step=1, reference_plan_id=chunk.plan_id
            )
        self.assertEqual(strategy.state_dict()["history_sources"]["submitted"], 1)
        edge._robot.fail = False
        invalid_groups = deepcopy(next_command.groups)
        invalid_groups["left_arm"][0] = 10.0
        invalid = replace(next_command, groups=invalid_groups)
        with (
            self.assertLogs("manimux.runtime.edge", level="ERROR"),
            self.assertRaises(ValueError),
        ):
            edge._send_command(
                invalid, measured.state, recorder, step=2, reference_plan_id=chunk.plan_id
            )
        self.assertEqual(len(edge._robot.commands), 1)
        notify_command_feedback(
            strategy, command=next_command, reference_plan_id=None, source="submitted"
        )
        notify_command_feedback(
            strategy, command=next_command, reference_plan_id=chunk.plan_id, source="validated_only"
        )
        self.assertEqual(strategy.state_dict()["history_sources"]["submitted"], 1)
        # The production Pause/Home lifecycle invokes this same public reset contract.
        for operation in ("Pause", "Home", "episode_reset"):
            strategy.reset()
            self.assertEqual(
                strategy.state_dict()["history_sources"], {"submitted": 0, "preview": 0}
            )
            self.assertEqual(strategy.state_dict()["pending"], [])
            self.assertIsNone(strategy.state_dict()["active"])
            write_json(self.directory / f"after_{operation}.json", strategy.state_dict())
        write_json(
            self.directory / "dispatch.json",
            {"submitted": edge._robot.commands, "rejections": events},
        )

    def test_real_policy_warmup_two_previews_and_start_reset_fence(self):
        planner, formal, adapter, clock, measured = self.context.planner()
        preview = build_inference_strategy(self.context.config)
        formal_before = json_value(formal.state_dict())
        initial_preview = json_value(preview.state_dict()["selector"])
        model, connection = self.context.client(self.directory / "warmup_wire")
        self.addCleanup(model.close)
        worker = ImmediateWorker(model, clock)
        warmup = PolicyWarmup(
            self.context.config,
            worker=worker,
            strategy=preview,
            adapter=adapter,
            session_id="offline",
            clock=clock,
            executor=DirectExecutor(control_dt_s=0.01),
            kinematics=None,
            safety=SafetyGuard(self.context.config["robot"]["group_dims"], 4.0),
        )
        warmup.activate()
        for _ in range(1000):
            observed = snapshot(self.context.config, clock.now_ns())
            warmup.advance(paused=True, snapshot=observed)
            self.assertNotEqual(warmup.phase, "error", warmup.error)
            if warmup.completed_requests >= 2:
                break
            clock.ns += 10_000_000
        self.assertGreaterEqual(warmup.completed_requests, 2)
        self.assertGreater(preview.state_dict()["history_sources"]["preview"], 0)
        self.assertEqual(formal_before, json_value(formal.state_dict()))
        write_json(
            self.directory / "before_start.json",
            {"preview": preview.state_dict(), "warmup": warmup.metadata()},
        )
        for _ in range(100):
            warmup.advance(paused=False, snapshot=snapshot(self.context.config, clock.now_ns()))
            self.assertNotEqual(warmup.phase, "error", warmup.error)
            if warmup.complete:
                break
            clock.ns += 10_000_000
        self.assertTrue(warmup.complete)
        self.assertEqual(worker.reset_count, 1)
        self.assertEqual(initial_preview, json_value(preview.state_dict()["selector"]))
        self.assertEqual(preview.state_dict()["history_sources"], {"submitted": 0, "preview": 0})
        self.assertTrue(any(request["message_type"] == "reset" for request in connection.requests))
        write_json(
            self.directory / "after_start.json",
            {"preview": preview.state_dict(), "warmup": warmup.metadata()},
        )

    def test_existing_serial_keeps_full_chunk_without_trace(self):
        config = load_config(SERIAL_YAML)
        self.assertIsNone(config["inference"]["strategy"])
        config["robot"]["control_hz"] = 100.0
        planner, strategy, adapter, clock, measured = self.context.planner(config=config)
        payload = synthetic_payload()
        payload.pop("denoise_trace")
        _, response, _, _, _ = self.decoded(payload=payload, clock=clock)
        chunk, result, _ = self.accept(planner, adapter, response, measured)
        self.assertTrue(result.accepted)
        self.assertEqual(chunk.horizon_steps, 50)
        self.assertNotIn("chunktrust", chunk.metadata)
        self.assertEqual(planner._timeline.remaining_ns(clock.now_ns()), 50 * chunk.dt_ns)
        planner.tick(measured.state, loop_started_ns=clock.now_ns())
        self.assertEqual(planner.steps, 1)
        write_json(
            self.directory / "serial.json",
            {"selected_steps": chunk.horizon_steps, "metadata": chunk.metadata},
        )


class ReportResult(unittest.TextTestResult):
    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            test.context.results.append(
                {
                    "case": test._testMethodName,
                    "subtest": str(subtest),
                    "status": "failed",
                    "error": self._exc_info_to_string(err, test),
                }
            )

    def addSuccess(self, test):
        super().addSuccess(test)
        test.context.results.append({"case": test._testMethodName, "status": "passed"})

    def addFailure(self, test, err):
        super().addFailure(test, err)
        test.context.results.append(
            {
                "case": test._testMethodName,
                "status": "failed",
                "error": self._exc_info_to_string(err, test),
            }
        )

    def addError(self, test, err):
        super().addError(test, err)
        test.context.results.append(
            {
                "case": test._testMethodName,
                "status": "error",
                "error": self._exc_info_to_string(err, test),
            }
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="Artifact directory outside this checkout"
    )
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error("--output must be outside the repository checkout")
    output.mkdir(parents=True, exist_ok=True)
    context = ProbeContext(output)
    OfflineChecks.context = context
    import websockets.sync.client  # noqa: F401 - load before the no-socket guard

    versions = {}
    for package in (
        "manimux",
        "chunktrust",
        "numpy",
        "msgpack",
        "msgpack-numpy",
        "websockets",
        "PyYAML",
    ):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "source checkout"
    write_json(output / "resolved_config.json", context.config)
    runner = unittest.TextTestRunner(verbosity=2, resultclass=ReportResult)
    with patch.object(
        socket, "socket", side_effect=AssertionError("network sockets are forbidden by this probe")
    ):
        result = runner.run(unittest.defaultTestLoader.loadTestsFromTestCase(OfflineChecks))
    source_paths = [
        Path(__file__).relative_to(ROOT),
        Path("manimux/policies/xpolicylab/ws_client.py"),
        Path("manimux/policies/xpolicylab/client.py"),
        Path("manimux/runtime/chunktrust.py"),
        Path("manimux/runtime/feedback.py"),
        Path("manimux/runtime/edge.py"),
        Path("manimux/runtime/preview.py"),
        Path("manimux/runtime/warmup.py"),
        Path("manimux/runtime/timeline.py"),
    ]
    report = {
        "success": result.wasSuccessful(),
        "tests_run": result.testsRun,
        "cases": context.results,
        "python": sys.version,
        "dependencies": versions,
        "chunktrust_declared_requirements": metadata.requires("chunktrust"),
        "chunktrust_installation": chunktrust_provenance(),
        "source_revision": git_revision(ROOT),
        "source_sha256": {
            str(path): hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in source_paths
        },
        "selecting_yaml": str(SELECTING_YAML.relative_to(ROOT)),
        "fixtures": {
            "synthetic_reference": json.loads((FIXTURES / "synthetic_reference.json").read_text()),
            "recorded_score_source": json.loads(
                (FIXTURES / "recorded_decision_scores.json").read_text()
            )["release_reference"],
        },
        "provenance": {
            "observations": "synthetic state and 8x8 RGB arrays",
            "provider": "in-memory XPolicyLab binary frames; no model",
            "command_sink": "disconnected recorder; no hardware driver",
            "network_sockets": "forbidden throughout checks",
        },
        "not_verified": [
            "real checkpoint numerical equivalence or model throughput",
            "GPU inference, camera images or connected controller behavior",
            "CAN transmission, measured arrival, physical safety or task success",
            "full EdgeRuntime lifecycle: Pause/Home checks cover the strategy reset contract",
            "real-robot velocity replay: released recording contains selection scores only",
        ],
    }
    write_json(output / "report.json", report)
    print(f"Report: {output / 'report.json'}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
