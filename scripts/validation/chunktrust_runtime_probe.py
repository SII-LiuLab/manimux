#!/usr/bin/env python3
"""Exercise real ChunkTrust runtime components on a disconnected synthetic station.

Uses the production loader, strategy, JointAdapter, PreviewPlanner/Timeline,
DirectExecutor, SafetyGuard and feedback helper. There is no provider, socket,
camera, robot driver, model or GPU. Full Edge.run/warmup are separate checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
import sys
import unittest
from copy import deepcopy
from dataclasses import dataclass, replace
from importlib import metadata
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from manimux.cli import load_config  # noqa: E402
from manimux.policies.capabilities import PolicyCapabilities  # noqa: E402
from manimux.policy_adapter import build_policy_adapter  # noqa: E402
from manimux.runtime.chunktrust.core import ChunkTrustAdapter  # noqa: E402
from manimux.runtime.chunktrust.strategy import json_value  # noqa: E402
from manimux.runtime.executors import DirectExecutor  # noqa: E402
from manimux.runtime.feedback import notify_command_feedback  # noqa: E402
from manimux.runtime.inference import (  # noqa: E402
    InferenceTimingError,
    build_inference_strategy,
    decode_strategy_action,
)
from manimux.runtime.preview import PreviewPlanner  # noqa: E402
from manimux.runtime.safety import SafetyGuard  # noqa: E402
from manimux.runtime.timeline import ActionTimeline, CommitResult  # noqa: E402
from manimux.types import (  # noqa: E402
    ActionContext,
    InferenceResponse,
    ObservationSnapshot,
    RobotCommand,
    RobotState,
    SensorFrame,
    copy_action_chunk,
    copy_group_vector,
)

EXPERIMENTS = ROOT / "manimux/configs/experiments/example-yam-pi05/put_bottles"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_value(value), indent=2, allow_nan=False, default=str) + "\n")


@dataclass
class Clock:
    ns: int = 10_000_000_000

    def now_ns(self):
        return self.ns

    def sleep_until_ns(self, target_ns):
        self.ns = max(self.ns, target_ns)


def synthetic_raw(num_steps=10, seed=51):
    rng = np.random.default_rng(seed)
    velocity = rng.normal(0, 0.2, (num_steps, 50, 32)).astype(np.float32)
    index = np.arange(50, dtype=np.float32)[:, None]
    dimension = np.arange(14, dtype=np.float32)[None]
    actions = (0.03 * np.sin(index * 0.11 + dimension * 0.17)).astype(np.float32)
    actions[:, [6, 13]] = 0.4 + 0.01 * np.sin(index * 0.11)
    return {
        "format": "joint",
        "actions": {"left_arm": actions[:, :7], "right_arm": actions[:, 7:]},
        "denoise_trace": {"v": velocity},
    }


def synthetic_capabilities(config, rtc):
    """An explicit fixture, never evidence of an available real RTC backend."""
    identity = deepcopy(config["policy"]["expected_backend"])
    model = identity["model"]
    model.update(
        observation_profile="yam_native",
        action_type="joint",
        action_dim=32,
        action_horizon=50,
        num_steps=config["policy_server"]["num_steps"],
        inference_seed=config["policy_server"]["inference_seed"],
    )
    if rtc:
        model["rtc_guidance"] = {
            "padding_mask": "valid_action_indices",
            "valid_action_indices": list(range(14)),
        }
    identity["fixture_provenance"] = "synthetic_cpu_runtime_probe"
    return PolicyCapabilities(frozenset({"default", "rtc"} if rtc else {"default"}), identity)


class Sink:
    """A list of submitted commands; no device or I/O object can be attached."""

    def __init__(self):
        self.commands = []
        self.fail = False

    def send_command(self, command):
        if self.fail:
            raise RuntimeError("injected disconnected submission failure")
        self.commands.append(deepcopy(command))


class Harness:
    def __init__(self, output, *, rtc=False, legacy=False, candidates=None):
        self.output = output
        output.mkdir(parents=True, exist_ok=True)
        name = "serial" if legacy else "chunktrust_ahs_rtc" if rtc else "chunktrust_ahs"
        override = {
            "_base_": [str(EXPERIMENTS / f"{name}.yaml")],
            "robot": {"control_hz": 100.0},
        }
        if not legacy:
            override["inference"] = {"chunktrust": {"candidates": candidates or [10]}}
            if rtc:
                override["inference"]["rtc"] = {"initial_delay_policy_steps": 3}
        selecting = output / "selecting.yaml"
        selecting.write_text(yaml.safe_dump(override))
        self.config = load_config(selecting)
        self.strategy = build_inference_strategy(self.config)
        if not legacy:
            self.strategy.validate_capabilities(synthetic_capabilities(self.config, rtc))
        self.clock = Clock()
        self.adapter = build_policy_adapter(self.config["robot"], self.config["policy"])
        self.planner = PreviewPlanner(
            self.config,
            strategy=self.strategy,
            adapter=self.adapter,
            executor=DirectExecutor(control_dt_s=0.01),
            safety=SafetyGuard(self.config["robot"]["group_dims"], 4.0),
            kinematics=None,
            decoder=None,
            session_id="runtime-probe",
            clock=self.clock,
        )
        self.sink = Sink()
        self.planner.initialize(self.snapshot().state)
        self.dt = int(self.config["policy"]["action_dt_s"] * 1e9)
        self.counter = 0
        write_json(output / "resolved-config.json", self.config)

    def snapshot(self):
        groups = {name: np.zeros(dim) for name, dim in self.config["robot"]["group_dims"].items()}
        for row in groups.values():
            row[-1] = 0.4
        frames = {
            name: SensorFrame(name, np.zeros((8, 8, 3), np.uint8), self.clock.ns, 1)
            for name in self.config["policy"]["adapter"]["camera_map"].values()
        }
        return ObservationSnapshot(RobotState(groups, self.clock.ns, 1), frames)

    def request(self, *, required=True, in_flight=False):
        submission = self.planner.submission(self.snapshot(), in_flight=in_flight)
        if submission is None:
            if required:
                raise AssertionError("expected an admitted strategy request")
            return None
        self.planner.submitted(submission.request, self.clock.ns)
        write_json(
            self.output / f"request-{submission.request.request_seq:03}.json",
            {
                "seq": submission.request.request_seq,
                "fields": submission.event_fields,
                "condition": getattr(submission.request, "action_condition", None),
                "weights": getattr(submission.request, "condition_weights", None),
            },
        )
        return submission

    def response(self, submission, *, seed=51):
        request = submission.request
        raw = synthetic_raw(self.config["policy_server"]["num_steps"], seed)
        np.savez_compressed(
            self.output / f"input-{request.request_seq:03}.npz",
            velocity=raw["denoise_trace"]["v"],
            **raw["actions"],
        )
        return InferenceResponse(
            request.session_id,
            request.request_seq,
            self.clock.ns,
            0.0,
            raw,
            observation_time_ns=request.observation_time_ns,
            deadline_ns=request.deadline_ns,
        )

    def decode(self, response):
        return decode_strategy_action(
            self.strategy,
            self.adapter,
            response.raw_action,
            ActionContext(
                response.request_seq, response.observation_time_ns, response.finished_time_ns
            ),
        )

    def accept(self, response):
        decoded = self.decode(response)
        prefix, result, fields = self.planner.accept(decoded, response, self.snapshot().state)
        if not result.accepted:
            self.strategy.on_response_rejected(response)
            raise AssertionError(f"unexpected timeline rejection: {result.reason}")
        self.counter += 1
        write_json(
            self.output / f"accepted-{self.counter:03}.json",
            {
                "prefix": {
                    "source_offset": prefix.source_offset_steps,
                    "groups": prefix.groups,
                    "metadata": prefix.metadata,
                },
                "fields": fields,
            },
        )
        return prefix, fields

    def submit_at(self, ns):
        """Disconnected boundary using production execution, safety and feedback.

        The parent Edge probe separately verifies actual _send_command ordering.
        """
        self.clock.ns = ns
        reference = self.planner._timeline.reference_horizon(
            now_ns=ns,
            dt_ns=10_000_000,
            horizon_steps=self.planner.execution.horizon_steps,
        )
        if reference is None:
            command = RobotCommand(
                copy_group_vector(self.planner.last_command),
                ns,
                self.planner._timeline.active_plan_id,
            )
        else:
            command = self.planner.execution.step(
                ns,
                self.snapshot().state,
                reference,
                last_command=self.planner.last_command,
            ).command
        self.planner.safety.validate_command(command)
        self.sink.send_command(command)
        notify_command_feedback(
            self.strategy,
            command=command,
            reference_plan_id=None if reference is None else reference.plan_id,
            source="submitted",
        )
        self.planner.previous_command, self.planner.last_command = (
            self.planner.last_command,
            copy_group_vector(command.groups),
        )
        return command

    def submit_rows(self, origin, first, end):
        for row in range(first, end):
            self.submit_at(origin + row * self.dt)
        self.clock.ns = origin + end * self.dt


class RuntimeChecks(unittest.TestCase):
    output: Path

    def harness(self, label, **options):
        return Harness(self.output / self._testMethodName / label, **options)

    def test_serial_multiround_k_intervals_and_policy_row_feedback(self):
        h = self.harness("serial")
        first = h.request()
        prefix, _ = h.accept(h.response(first))
        origin = h.clock.ns
        self.assertEqual(prefix.horizon_steps, 10)
        self.assertEqual(h.strategy.state_dict()["history"], [])
        self.assertEqual(h.planner._timeline.remaining_ns(origin), 10 * h.dt)
        held = RobotCommand(copy_group_vector(h.planner.last_command), origin, prefix.plan_id)
        notify_command_feedback(
            h.strategy, command=held, reference_plan_id=prefix.plan_id, source="validated_only"
        )
        notify_command_feedback(
            h.strategy, command=held, reference_plan_id="old-reference", source="submitted"
        )
        self.assertEqual(h.strategy.state_dict()["history"], [])
        self.assertIsNone(h.request(required=False, in_flight=True))
        h.sink.fail = True
        with self.assertRaisesRegex(RuntimeError, "submission failure"):
            h.submit_at(origin)
        self.assertEqual(h.strategy.state_dict()["history"], [])
        h.sink.fail = False
        commands = []
        for ns in range(origin, origin + 10 * h.dt, 10_000_000):
            commands.append(h.submit_at(ns))
        state = h.strategy.state_dict()
        self.assertEqual(state["history_sources"], {"submitted": 10, "preview": 0})
        self.assertEqual([row["row"] for row in state["history_rows"]], list(range(10)))
        self.assertGreater(len(commands), len(state["history"]))
        first_command_per_row = {}
        for command in commands:
            row = (command.monotonic_ns - origin) // h.dt
            first_command_per_row.setdefault(
                row,
                np.concatenate(
                    [
                        command.groups["left_arm"],
                        command.groups["right_arm"],
                    ]
                ).astype(np.float32),
            )
        policy_rows = np.concatenate([prefix.groups[name] for name in ("left_arm", "right_arm")],
                                     axis=1).astype(np.float32)
        np.testing.assert_array_equal(state["history"], policy_rows)
        self.assertFalse(np.allclose(policy_rows, list(first_command_per_row.values())))
        # Actual submitted commands remain interpolated and are not rewritten by AHS.
        np.testing.assert_array_equal(commands[1].groups["left_arm"],
                                      h.sink.commands[1].groups["left_arm"])
        commands[0].groups["left_arm"][:] = 2.0
        self.assertEqual(h.strategy.state_dict()["history"], state["history"])
        h.clock.ns = origin + 10 * h.dt - 1
        self.assertIsNone(h.request(required=False))
        h.clock.ns += 1
        second = h.request()
        next_prefix, _ = h.accept(h.response(second, seed=52))
        self.assertEqual(next_prefix.metadata["chunktrust"]["history_targets"], state["history"])
        self.assertEqual(next_prefix.metadata["chunktrust"]["history_semantics"],
                         "submitted_policy_rows_before_interpolation")
        self.assertEqual(h.strategy.state_dict()["selector"]["revision"], 2)
        write_json(h.output / "ledger.json", h.strategy.state_dict())

    def test_history_and_scores_ignore_control_sampling_phase(self):
        results = []
        for label, period, phase in (("aligned", 10_000_000, 0),
                                     ("offset", 7_000_000, 2_000_000)):
            h = self.harness(label, candidates=[10, 20, 30, 40, 50])
            expected_history = np.empty((0, 14), dtype=np.float32)
            for round_index in range(7):
                submission = h.request()
                response = h.response(submission, seed=51 + round_index)
                prefix, _ = h.accept(response)
                origin = h.clock.ns
                rows = np.concatenate([prefix.groups[name] for name in ("left_arm", "right_arm")],
                                      axis=1).astype(np.float32)
                # Every source interval is submitted, at different control phases.
                for ns in range(origin + phase, origin + prefix.horizon_steps * h.dt, period):
                    h.submit_at(ns)
                expected_history = np.concatenate([expected_history, rows])[-60:]
                np.testing.assert_array_equal(h.strategy.state_dict()["history"], expected_history)
                h.clock.ns = origin + prefix.horizon_steps * h.dt
            self.assertEqual(len(expected_history), 60)
            submission = h.request()
            response = h.response(submission, seed=80)
            prefix, _ = h.accept(response)
            evidence = prefix.metadata["chunktrust"]["evidence"]
            self.assertTrue(np.isfinite(evidence["u_inter_raw"]).all())
            results.append((evidence, prefix.horizon_steps, h.strategy.state_dict()["selector"]))
        # History accumulates across repeated chunks. Inter evidence, selected K,
        # and posterior/RNG state must not depend on the interpolated tick samples.
        self.assertEqual(results[0], results[1])

    def test_rtc_early_and_late_common_suffix_and_bounded_budget(self):
        for label, additionally_consumed, hold_steps in (("early", 0, 0), ("late", 3, 5)):
            h = self.harness(label, rtc=True)
            first = h.request()
            initial, _ = h.accept(h.response(first))
            origin = h.clock.ns
            h.submit_rows(origin, 0, 6)
            self.assertIsNone(h.request(required=False))
            h.submit_rows(origin, 6, 7)
            conditioned = h.request()
            self.assertEqual(conditioned.event_fields["frozen_budget"], 3)
            self.assertEqual(conditioned.request.rtc_beta, 10)
            np.testing.assert_array_equal(conditioned.request.condition_weights[:3], 1)
            old = np.concatenate(
                [initial.groups[name] for name in ("left_arm", "right_arm")], axis=1
            )
            np.testing.assert_allclose(conditioned.request.action_condition[:3], old[7:10])
            self.assertIsNone(h.request(required=False, in_flight=True))
            h.submit_rows(origin, 7, 7 + additionally_consumed)
            if hold_steps:
                before = h.strategy.state_dict()["history_revision"]
                for row in range(10, 10 + hold_steps):
                    h.submit_at(origin + row * h.dt)
                self.assertEqual(h.strategy.state_dict()["history_revision"], before)
            response = h.response(conditioned, seed=53)
            with patch.object(
                h.strategy._selector, "prepare", wraps=h.strategy._selector.prepare
            ) as spy:
                replacement, fields = h.accept(response)
            self.assertEqual(replacement.source_offset_steps, additionally_consumed)
            self.assertEqual(fields["chunktrust_unexecuted_old_budget"], 3 - additionally_consumed)
            expected = response.raw_action["denoise_trace"]["v"][:, additionally_consumed:, :14]
            np.testing.assert_array_equal(spy.call_args.args[0], expected)
            full_actions = np.concatenate(
                [response.raw_action["actions"][name] for name in ("left_arm", "right_arm")], axis=1
            )
            np.testing.assert_array_equal(
                spy.call_args.args[1], full_actions[additionally_consumed:]
            )
            np.testing.assert_array_equal(
                replacement.groups["left_arm"][0], full_actions[additionally_consumed, :7]
            )
            self.assertEqual(
                replacement.metadata["chunktrust"]["available_horizon"], 50 - additionally_consumed
            )
            self.assertEqual(h.planner._timeline.remaining_ns(h.clock.ns), 10 * h.dt)
            h.submit_at(h.clock.ns)
            self.assertEqual(
                h.strategy.state_dict()["history_rows"][-1]["source_row"], additionally_consumed
            )
            np.testing.assert_array_equal(
                h.strategy.state_dict()["history"],
                np.concatenate([old[:7 + additionally_consumed],
                                full_actions[additionally_consumed:additionally_consumed + 1]]),
            )
            # A non-grid control tick in the cropped suffix must still record
            # the original model row, with its source offset applied exactly once.
            command = h.submit_at(h.clock.ns + h.dt + h.dt // 4)
            after = h.strategy.state_dict()
            np.testing.assert_array_equal(after["history"][-1],
                                          full_actions[additionally_consumed + 1])
            self.assertFalse(np.allclose(after["history"][-1][:7], command.groups["left_arm"]))
            self.assertEqual(after["history_rows"][-1]["source_row"], additionally_consumed + 1)
            write_json(h.output / "ledger.json", h.strategy.state_dict())

    def test_reject_duplicate_expired_and_reset_are_transactional(self):
        h = self.harness("transactions")
        first = h.request()
        response = h.response(first)
        decoded = h.decode(response)
        before = h.strategy.state_dict()["selector"]
        proposal = h.strategy.select_chunk(chunk=decoded, now_ns=h.clock.ns)
        repeated = h.strategy.select_chunk(chunk=decoded, now_ns=h.clock.ns)
        self.assertEqual(proposal.metadata, repeated.metadata)
        self.assertEqual(before, h.strategy.state_dict()["selector"])
        h.strategy.on_response_rejected(response)
        self.assertEqual(before, h.strategy.state_dict()["selector"])
        self.assertEqual(h.strategy.state_dict()["pending_requests"], [])
        expired = replace(response, finished_time_ns=response.deadline_ns + 1)
        self.assertEqual(h.planner.rejection_reason(expired), "inference_deadline_exceeded")
        self.assertEqual(
            h.planner.rejection_reason(replace(response, session_id="old-session")),
            "session_mismatch",
        )
        with self.assertRaisesRegex(ValueError, "current-generation"):
            h.decode(response)
        second = h.request()
        accepted_response = h.response(second)
        selected, _ = h.accept(accepted_response)
        accepted = h.strategy.state_dict()["selector"]
        with self.assertRaises(ValueError):
            h.strategy.on_plan_accepted(
                chunk=selected,
                result=type("Result", (), {"accepted": True})(),
                response=accepted_response,
                now_ns=h.clock.ns,
            )
        self.assertEqual(accepted, h.strategy.state_dict()["selector"])
        stale_result = h.planner._timeline.commit(
            selected,
            now_ns=h.clock.ns,
            max_plan_age_ns=None,
            current_command=h.planner.last_command,
            blend_steps=0,
        )
        self.assertEqual(stale_result.reason, "stale_request_seq")
        h.strategy.reset()
        reset = h.strategy.state_dict()
        self.assertEqual(reset["selector"]["revision"], 0)
        self.assertEqual(reset["history"], [])
        self.assertIsNone(reset["active"])
        self.assertEqual(reset["selector"]["config"], before["config"])
        with self.assertRaisesRegex(ValueError, "current-generation"):
            h.decode(accepted_response)
        write_json(h.output / "reset.json", reset)

    def test_proposal_mutations_fail_before_committing_selector_state(self):
        """Both acceptance hooks bind actions, source rows and request identity."""
        cases = (
            "groups",
            "source_offset",
            "dt",
            "metadata",
            "request_seq",
            "plan_id",
            "observation_time",
            "action_space",
            "hold_rows",
            "response_session",
            "response_observation",
            "response_sequence",
            "missing_request",
        )
        for rtc in (False, True):
            for case in cases:
                with self.subTest(rtc=rtc, mutation=case):
                    label = ("rtc-" if rtc else "serial-") + case
                    h = self.harness(label, rtc=rtc)
                    request = h.request()
                    if rtc:
                        h.accept(h.response(request))
                        origin = h.clock.ns
                        h.submit_rows(origin, 0, 7)
                        request = h.request()
                        h.submit_rows(origin, 7, 9)
                    response = h.response(request, seed=72)
                    decoded = h.decode(response)
                    selected = h.strategy.select_chunk(chunk=decoded, now_ns=h.clock.ns)
                    self.assertEqual(selected.source_offset_steps, 2 if rtc else 0)
                    before = deepcopy(h.strategy.state_dict())
                    changed = copy_action_chunk(selected)
                    changed.metadata = deepcopy(selected.metadata)
                    changed_response = response
                    if case == "groups":
                        changed.groups["left_arm"][0, 0] += 0.001
                    elif case == "source_offset":
                        changed.source_offset_steps += 1
                    elif case == "dt":
                        changed.dt_ns += 1
                    elif case == "metadata":
                        changed.metadata["chunktrust"]["selector_state_after"]["config"][
                            "intra_alpha"
                        ] = 7.0
                    elif case == "request_seq":
                        changed.request_seq += 100
                    elif case == "plan_id":
                        changed.plan_id += "-changed"
                    elif case == "observation_time":
                        changed.observation_time_ns += 1
                    elif case == "action_space":
                        changed.action_space = "other"
                    elif case == "hold_rows":
                        changed.hold_from_step = {"left_arm": 0}
                    elif case == "response_session":
                        changed_response = replace(response, session_id="other-session")
                    elif case == "response_observation":
                        changed_response = replace(
                            response, observation_time_ns=response.observation_time_ns + 1
                        )
                    elif case == "response_sequence":
                        changed_response = replace(response, request_seq=response.request_seq + 100)
                    elif case == "missing_request":
                        h.strategy._requests.pop(response.request_seq)
                    with self.assertRaises(ValueError):
                        h.strategy.prepare_chunk(
                            chunk=changed,
                            response=changed_response,
                            now_ns=h.clock.ns,
                        )
                    with self.assertRaises(ValueError):
                        h.strategy.on_plan_accepted(
                            chunk=changed,
                            result=CommitResult(True, "accepted"),
                            response=changed_response,
                            now_ns=h.clock.ns,
                        )
                    # Failed identity/prefix checks must not advance posterior/RNG,
                    # history, active source ownership or the full prediction tail.
                    self.assertEqual(before, h.strategy.state_dict())
                    h.strategy.on_response_rejected(response)
                    self.assertEqual(before["selector"], h.strategy.state_dict()["selector"])
                    self.assertEqual(h.strategy.state_dict()["pending_requests"], [])

    def test_cached_proposals_do_not_share_nested_audit_metadata(self):
        h = self.harness("cached-metadata")
        response = h.response(h.request())
        decoded = h.decode(response)
        selected = h.strategy.select_chunk(chunk=decoded, now_ns=h.clock.ns)
        expected = deepcopy(selected.metadata["chunktrust"])
        selected.metadata["chunktrust"]["selector_state_after"]["config"]["intra_alpha"] = 7.0
        selected.groups["left_arm"][0, 0] += 0.001
        cached = h.strategy.select_chunk(chunk=decoded, now_ns=h.clock.ns)
        self.assertEqual(cached.metadata["chunktrust"], expected)
        cached.metadata["chunktrust"]["selector_state_before"]["config"]["intra_alpha"] = 9.0
        another = h.strategy.select_chunk(chunk=decoded, now_ns=h.clock.ns)
        self.assertEqual(another.metadata["chunktrust"], expected)
        np.testing.assert_array_equal(another.groups["left_arm"], decoded.groups["left_arm"][:10])
        # An untouched copy still travels through the real Timeline acceptance path.
        _, result, _ = h.planner.accept(decoded, response, h.snapshot().state)
        self.assertTrue(result.accepted)
        self.assertEqual(h.strategy.state_dict()["selector"]["revision"], 1)

    def test_preview_isolated_multicall_and_legacy_timeline(self):
        h = self.harness("preview")
        first = h.request()
        prefix, _ = h.accept(h.response(first))
        origin = h.clock.ns
        for ns in range(origin, origin + prefix.horizon_steps * h.dt, 10_000_000):
            h.clock.ns = ns
            h.planner.tick(h.snapshot().state, loop_started_ns=ns)
        h.clock.ns = origin + prefix.horizon_steps * h.dt
        self.assertEqual(
            h.strategy.state_dict()["history_sources"], {"submitted": 0, "preview": 10}
        )
        self.assertEqual(len(h.sink.commands), 0)
        np.testing.assert_array_equal(
            h.strategy.state_dict()["history"],
            np.concatenate([prefix.groups[name] for name in ("left_arm", "right_arm")], axis=1)
            .astype(np.float32),
        )
        second = h.request()
        h.accept(h.response(second, seed=54))
        self.assertEqual(h.strategy.state_dict()["selector"]["revision"], 2)
        h.strategy.reset()
        self.assertEqual(h.strategy.state_dict()["history_sources"], {"submitted": 0, "preview": 0})
        legacy = self.harness("legacy", legacy=True)
        request = legacy.request()
        chunk, _ = legacy.accept(legacy.response(request))
        self.assertEqual(legacy.planner._timeline.remaining_ns(legacy.clock.ns), 50 * legacy.dt)
        legacy.planner.tick(legacy.snapshot().state, loop_started_ns=legacy.clock.ns)
        self.assertFalse(hasattr(legacy.strategy, "on_command_feedback"))
        ordinary_timeline = ActionTimeline(
            legacy.config["robot"]["group_dims"],
            action_start_mode="first_step_when_ready",
            hold_last_step=True,
        )
        suffix = copy_action_chunk(chunk)
        suffix.source_offset_steps = 1
        self.assertEqual(
            ordinary_timeline.commit(
                suffix,
                now_ns=legacy.clock.ns,
                max_plan_age_ns=None,
                current_command=legacy.planner.last_command,
                blend_steps=0,
            ).reason,
            "serial_requires_untrimmed_chunk",
        )

    def test_delay_and_available_suffix_fail_explicitly(self):
        h = self.harness("infeasible", rtc=True, candidates=[49])
        first = h.request()
        h.accept(h.response(first))
        with self.assertRaisesRegex(InferenceTimingError, "candidate suffix"):
            h.request()
        core = ChunkTrustAdapter({"candidates": [10, 20, 30, 40]})
        raw = synthetic_raw()
        actions = np.concatenate(list(raw["actions"].values()), axis=1)
        with self.assertRaisesRegex(ValueError, "suffix|candidate"):
            core.prepare(
                raw["denoise_trace"]["v"][:, 11:, :14],
                actions[11:],
                np.empty((0, 14)),
                request_key="too-short",
                context={"source_offset": 11},
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.is_relative_to(ROOT) or (output / "report.json").exists():
        parser.error("choose a new output directory outside the repository")
    output.mkdir(parents=True, exist_ok=True)
    RuntimeChecks.output = output
    with patch.object(socket, "socket", side_effect=AssertionError("network sockets forbidden")):
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(RuntimeChecks)
        )
    sources = [
        Path(__file__).resolve(),
        *sorted((ROOT / "manimux/runtime/chunktrust").glob("*.py")),
        *[
            ROOT / "manimux/runtime" / name
            for name in (
                "preview.py",
                "planning.py",
                "timeline.py",
                "feedback.py",
                "scheduling.py",
            )
        ],
        ROOT / "manimux/policy_adapter/joint.py",
    ]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    report = {
        "ok": result.wasSuccessful(),
        "test_groups": result.testsRun,
        "failures": [text for _, text in result.failures + result.errors],
        "manimux_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "versions": {"python": sys.version, "numpy": metadata.version("numpy")},
        "evidence": "synthetic CPU runtime components and disconnected submission list",
        "limitations": (
            "no real backend/model/JAX/GPU/sensor/SDK/hardware; Edge.run and PolicyWarmup separate"
        ),
    }
    write_json(output / "report.json", report)
    print(json.dumps({"ok": report["ok"], "report": str(output / "report.json")}))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
