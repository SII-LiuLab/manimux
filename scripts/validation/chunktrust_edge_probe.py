#!/usr/bin/env python3
"""Run the complete Edge loop against synthetic observations and a disconnected sink.

The production loader, adapter, strategy, timeline, executor, safety, feedback and
EpisodeRecorder run normally. A deterministic in-memory worker supplies actions and
trace. Injected selection time is clock pressure, not a model/CPU benchmark.
No real model, socket, camera or robot driver is constructed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import unittest
from collections import deque
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chunktrust_runtime_probe import (  # noqa: E402
    EXPERIMENTS,
    Clock,
    Harness,
    synthetic_capabilities,
    synthetic_raw,
    write_json,
)

from manimux.cli import load_config  # noqa: E402
from manimux.policies.capabilities import PolicyCapabilities  # noqa: E402
from manimux.runtime.bid import BidBackwardStrategy  # noqa: E402
from manimux.runtime.edge import EdgeRuntime  # noqa: E402
from manimux.runtime.inference import build_inference_strategy  # noqa: E402
from manimux.types import InferenceResponse  # noqa: E402
from scripts.validation.starvla_fixtures.robot import RobotDouble  # noqa: E402


class DisconnectedRobot(RobotDouble):
    """Complete the old numerical plant's metadata API without device components."""

    def command_bounds(self):
        return {}

    def runtime_metadata(self):
        return {"fixture": "disconnected_chunktrust_edge_probe"}

    def command_status(self):
        return {}

    def sent_command_snapshots(self):
        return {}


def build_robot(config, clock):
    return DisconnectedRobot(config["group_dims"], clock)


def run_case(
    output,
    *,
    legacy=False,
    pressure_steps=0,
    anchor_pressure_steps=0,
    missing_trace=False,
    bid=False,
    max_plan_age_s=None,
    edge_type=EdgeRuntime,
):
    fixture = Harness(output / "inputs", legacy=legacy or bid)
    config = deepcopy(fixture.config)
    if bid:
        settings = yaml.safe_load(
            (ROOT / "manimux/configs/inference/yam_bid_single_inflight.yaml").read_text()
        )
        settings["bid"]["num_samples"] = 2
        selecting = output / "inputs/bid.yaml"
        selecting.write_text(
            yaml.safe_dump(
                {
                    "_base_": [str(EXPERIMENTS / "serial.yaml")],
                    "robot": {"control_hz": 100.0},
                    "inference": settings,
                }
            )
        )
        config = load_config(selecting)
    if max_plan_age_s is not None:
        config["inference"]["max_plan_age_s"] = max_plan_age_s
    config["robot"]["type"] = "scripts.validation.chunktrust_edge_probe:build_robot"
    config["robot"]["options"] = {
        "execute": True,
        "move_to_start_on_connect": False,
        "home_on_close": False,
    }
    config["sensors"] = [
        {
            "name": name,
            "driver": "scripts.validation.starvla_fixtures.sensor:build_sensor",
            "width": 8,
            "height": 8,
        }
        for name in config["policy"]["adapter"]["camera_map"].values()
    ]
    config["robogui"]["enabled"] = False
    config["recording"]["enabled"] = False
    config["recording"]["video_fps"] = 0.0
    config["run"]["warmup_before_start"] = False
    config["run"]["max_control_steps"] = 12 if missing_trace or max_plan_age_s is not None else 80
    config["run"]["sensor_reading"]["mode"] = "inline"
    clock = Clock()
    strategy = build_inference_strategy(config)
    accepted = []
    commands = []
    selections = []
    selecting = False
    initial_selector = None
    ahs = not legacy and not bid
    if bid and not isinstance(strategy, BidBackwardStrategy):
        raise AssertionError("BID control must use the production selector")

    class Worker:
        def __init__(self, policy, session, **kwargs):
            self.session = session
            self.queue = deque()
            self.is_alive = True
            self.capabilities = synthetic_capabilities(config, rtc=False)
            if bid:
                self.capabilities = PolicyCapabilities(
                    frozenset({"default", "aac"}), self.capabilities.backend_metadata
                )

        def start(self):
            pass

        def request_stop(self):
            self.is_alive = False

        def close(self):
            self.is_alive = False

        def submit_latest(self, request):
            nonlocal initial_selector
            if ahs and initial_selector is None:
                # Capture after the production Start/reset fence, before any decision.
                initial_selector = deepcopy(strategy.state_dict()["selector"])
            raw = synthetic_raw(config["policy_server"]["num_steps"])
            if missing_trace:
                del raw["denoise_trace"]
            if bid:
                candidates = []
                for index in range(config["inference"]["bid"]["num_samples"]):
                    candidate = deepcopy(raw)
                    del candidate["denoise_trace"]
                    for values in candidate["actions"].values():
                        values += 0.002 * index + 0.0001 * request.request_seq
                    candidates.append(candidate)
                raw = {"bid_candidates": candidates}
            self.queue.append(
                InferenceResponse(
                    self.session,
                    request.request_seq,
                    clock.ns,
                    0.0,
                    raw,
                    observation_time_ns=request.observation_time_ns,
                    deadline_ns=request.deadline_ns,
                )
            )

        def poll(self):
            return self.queue.popleft() if self.queue else None

    with ExitStack() as stack:
        stack.enter_context(patch("socket.socket", side_effect=AssertionError("sockets forbidden")))
        stack.enter_context(
            patch.object(sys.modules[edge_type.__module__], "PolicyWorkerClient", Worker)
        )
        if not legacy:
            select = strategy.select_chunk

            def delayed_selection(*, chunk, now_ns):
                nonlocal selecting
                selected = select(chunk=chunk, now_ns=now_ns)
                selections.append(
                    {
                        "request_seq": chunk.request_seq,
                        "snapshot_ns": now_ns,
                        "proposal": deepcopy(selected.metadata.get("chunktrust")),
                        "bid": deepcopy(selected.metadata.get("bid")),
                    }
                )
                clock.ns += pressure_steps * fixture.dt
                selecting = True
                return selected

            stack.enter_context(patch.object(strategy, "select_chunk", delayed_selection))
        runtime = edge_type(config, output / "runtime", clock=clock, strategy=strategy)
        anchor = runtime._execution.reference_from_joints

        def delayed_anchor(groups):
            nonlocal selecting
            result = anchor(groups)
            if selecting:
                clock.ns += anchor_pressure_steps * fixture.dt
                selecting = False
            return result

        stack.enter_context(
            patch.object(runtime._execution, "reference_from_joints", delayed_anchor)
        )
        accept = strategy.on_plan_accepted

        def observe_accept(**kwargs):
            fields = accept(**kwargs)
            active = runtime._timeline.active_horizon()
            accepted.append(
                {
                    "plan_id": kwargs["chunk"].plan_id,
                    "request_seq": kwargs["chunk"].request_seq,
                    "actual_ns": clock.ns,
                    "accepted_ns": kwargs["now_ns"],
                    "timeline_start_ns": active.start_time_ns,
                    "observation_ns": kwargs["chunk"].observation_time_ns,
                    "observation_to_commit_ms": kwargs["chunk"].metadata[
                        "observation_to_commit_ms"
                    ],
                    "source_offset_steps": kwargs["chunk"].source_offset_steps,
                    "time_trimmed_steps": kwargs["result"].time_trimmed_steps,
                    "bid": deepcopy(kwargs["chunk"].metadata.get("bid")),
                    "proposal": deepcopy(kwargs["chunk"].metadata.get("chunktrust")),
                    **fields,
                }
            )
            return fields

        stack.enter_context(patch.object(strategy, "on_plan_accepted", observe_accept))
        send = runtime._robot.send_command

        def capture(command):
            commands.append(
                {
                    "actual_ns": clock.ns,
                    "command_ns": command.monotonic_ns,
                    "plan_id": command.plan_id,
                    "groups": {name: values.tolist() for name, values in command.groups.items()},
                }
            )
            return send(command)

        stack.enter_context(patch.object(runtime._robot, "send_command", capture))
        result = runtime.run()
    events = [
        json.loads(line)
        for path in (output / "runtime").rglob("events.jsonl")
        for line in path.read_text().splitlines()
    ]
    report = {
        "algorithm": config["inference"]["algorithm"],
        "pressure_policy_intervals": pressure_steps,
        "anchor_pressure_policy_intervals": anchor_pressure_steps,
        "completed": result.run_completed,
        "accepted": accepted,
        "commands": commands,
        "selections": selections,
        "policy_dt_ns": fixture.dt,
        "history_rows": strategy.state_dict()["history_rows"] if ahs else [],
        "selector_revision": strategy.state_dict()["selector"]["revision"] if ahs else None,
        "initial_selector": initial_selector,
        "final_selector": strategy.state_dict()["selector"] if ahs else None,
        "pending_requests": strategy.state_dict()["pending_requests"] if ahs else [],
        "rejections": [event for event in events if event["kind"] == "plan_rejected"],
        "recorded_accepts": [event for event in events if event["kind"] == "plan_accepted"],
    }
    write_json(output / "result.json", report)
    return report


class EdgeChecks(unittest.TestCase):
    output: Path
    baseline_edge = None

    def check_ahs(self, pressure, anchor_pressure=0):
        result = run_case(
            self.output / f"ahs-pressure{pressure}-anchor{anchor_pressure}",
            pressure_steps=pressure,
            anchor_pressure_steps=anchor_pressure,
        )
        self.assertTrue(result["completed"])
        self.assertGreaterEqual(len(result["accepted"]), 2)
        self.assertEqual(len(result["recorded_accepts"]), len(result["accepted"]))
        rows = result["history_rows"]
        first_rows = {}
        for row in rows:
            first_rows.setdefault(row["plan_id"], row["source_row"])
        self.assertTrue(first_rows)
        self.assertEqual(set(first_rows.values()), {0})
        for accept, recorded in zip(result["accepted"], result["recorded_accepts"], strict=True):
            self.assertEqual(accept["actual_ns"], accept["accepted_ns"])
            self.assertEqual(accept["actual_ns"], accept["timeline_start_ns"])
            expected_ms = (accept["actual_ns"] - accept["observation_ns"]) / 1e6
            self.assertEqual(accept["observation_to_commit_ms"], expected_ms)
            self.assertEqual(recorded["observation_to_commit_ms"], expected_ms)
            selection = next(
                item
                for item in result["selections"]
                if item["request_seq"] == accept["request_seq"]
            )
            self.assertEqual(accept["proposal"], selection["proposal"])
            self.assertEqual(accept["proposal"]["handoff_time_ns"], selection["snapshot_ns"])
        self.assertEqual(result["selector_revision"], len(result["accepted"]))
        self.assertFalse(result["rejections"])

    def test_ahs_without_selection_pressure(self):
        self.check_ahs(0)

    def test_ahs_after_three_intervals_of_selection(self):
        self.check_ahs(3)

    def test_ahs_clock_includes_anchor_conversion(self):
        self.check_ahs(0, anchor_pressure=3)

    def test_ordinary_serial_path(self):
        result = run_case(self.output / "ordinary-serial", legacy=True)
        self.assertTrue(result["completed"])
        self.assertTrue(result["accepted"])
        self.assertFalse(result["rejections"])
        commands = [command for command in result["commands"] if command["plan_id"] is not None]
        self.assertTrue(commands)
        self.assertEqual(commands[0]["actual_ns"], result["accepted"][0]["actual_ns"])
        self.assertEqual(len(result["commands"]), 80)

    def test_missing_trace_never_advances_selector(self):
        result = run_case(self.output / "missing-trace", missing_trace=True)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["selector_revision"], 0)
        self.assertEqual(result["history_rows"], [])
        self.assertTrue(result["rejections"])
        self.assertTrue(all(command["plan_id"] is None for command in result["commands"]))

    def test_selection_finishes_after_max_plan_age(self):
        result = run_case(self.output / "selection-expired", pressure_steps=3, max_plan_age_s=0.05)
        self.assertTrue(result["completed"])
        self.assertTrue(result["selections"])
        self.assertFalse(result["accepted"])
        self.assertTrue(result["rejections"])
        self.assertTrue(all(item["reason"] == "plan_too_old" for item in result["rejections"]))
        self.assertEqual(result["final_selector"], result["initial_selector"])
        self.assertEqual(result["history_rows"], [])
        self.assertEqual(result["pending_requests"], [])
        self.assertTrue(all(command["plan_id"] is None for command in result["commands"]))

    def test_bid_retains_the_selection_clock(self):
        result = run_case(self.output / "bid-pressure", bid=True, pressure_steps=3)
        self.assertTrue(result["completed"])
        self.assertGreaterEqual(len(result["accepted"]), 2)
        self.assertFalse(result["rejections"])
        self.assertTrue(any(item["bid"]["overlap_steps"] for item in result["accepted"]))
        for accepted, selected in zip(result["accepted"], result["selections"], strict=True):
            self.assertEqual(accepted["accepted_ns"], selected["snapshot_ns"])
            self.assertEqual(accepted["bid"], selected["bid"])
            self.assertEqual(
                accepted["actual_ns"] - accepted["accepted_ns"], 3 * result["policy_dt_ns"]
            )

    def check_baseline(self, label, **options):
        if self.baseline_edge is None:
            self.skipTest("--baseline-edge not supplied")
        current = run_case(self.output / f"{label}-current", **options)
        baseline = run_case(
            self.output / f"{label}-baseline", edge_type=self.baseline_edge, **options
        )
        self.assertTrue(current["completed"] and baseline["completed"])
        self.assertFalse(current["rejections"] or baseline["rejections"])
        self.assertEqual(len(current["commands"]), 80)
        self.assertEqual(comparison_data(current), comparison_data(baseline))

    def test_ordinary_serial_matches_baseline(self):
        self.check_baseline("serial-comparison", legacy=True)

    def test_bid_matches_baseline_with_selection_pressure(self):
        self.check_baseline("bid-comparison", bid=True, pressure_steps=3, anchor_pressure_steps=1)


def comparison_data(result):
    """Ignore random plan identifiers, retaining commands, selection and commit timing."""
    plans = {}

    def normalize(value):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if key == "plan_id" and item is not None:
                    item = plans.setdefault(item, len(plans))
                result[key] = normalize(item)
            return result
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    return normalize({key: result[key] for key in ("accepted", "commands", "selections")})


def load_baseline(path):
    """Load an explicitly supplied trusted historical Edge module for differential checks."""
    name = "manimux.runtime._chunktrust_edge_baseline"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load baseline Edge module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.EdgeRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--baseline-edge",
        type=Path,
        help="trusted historical edge.py source for serial/BID differential checks",
    )
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.is_relative_to(ROOT):
        parser.error("--output must be outside the repository checkout")
    if output.exists():
        parser.error("--output already exists; choose a new directory")
    output.mkdir(parents=True)
    EdgeChecks.output = output
    baseline_path = (
        None if args.baseline_edge is None else args.baseline_edge.expanduser().resolve()
    )
    if baseline_path is not None:
        EdgeChecks.baseline_edge = load_baseline(baseline_path)
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(EdgeChecks)
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    report = {
        "ok": result.wasSuccessful(),
        "tests": result.testsRun,
        "skipped": [str(test) for test, _ in result.skipped],
        "failures": [str(test) for test, _ in result.failures + result.errors],
        "revision": revision.stdout.strip(),
        "source_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in (
                "manimux/runtime/edge.py",
                "scripts/validation/chunktrust_edge_probe.py",
            )
        },
        "evidence": "complete Edge.run and real EpisodeRecorder, synthetic policy and devices",
        "clock_pressure": "deterministic injected duration, not measured selector latency",
        "model_inference": False,
        "hardware": False,
        "baseline": None
        if baseline_path is None
        else {
            "path": str(baseline_path),
            "sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        },
    }
    write_json(output / "report.json", report)
    print(json.dumps({"ok": report["ok"], "report": str(output / "report.json")}))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
