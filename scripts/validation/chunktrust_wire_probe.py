#!/usr/bin/env python3
"""CPU wire/trace checks with synthetic evidence and no socket, SDK or model.

The XPolicyLab codec and RTC validator are real. Its Pi05 get_action_rtc method
is compiled unchanged from the pinned source, with only its downstream policy
replaced by a recording double. This is not a model/sampler correctness test.
"""

from __future__ import annotations

import argparse
import ast
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
from importlib import metadata
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
XPL = ROOT / "third_party/XPolicyLab"
sys.path[:0] = [str(ROOT), str(XPL), str(XPL.parent)]

from client_server.ws.protocol.codec import decode_frame, encode_frame  # noqa: E402
from client_server.ws.protocol.schemas import Frame  # noqa: E402

from manimux.policies.capabilities import PolicyCapabilities  # noqa: E402
from manimux.policies.response_log import ResponseLog  # noqa: E402
from manimux.policies.xpolicylab.client import XPolicyLabWsPolicyModel  # noqa: E402
from manimux.policies.xpolicylab.codec import GroupLayout  # noqa: E402
from manimux.policies.xpolicylab.ws_client import (  # noqa: E402
    XPolicyLabWsClient,
    unpack_frame,
)
from manimux.runtime.chunktrust.trace import (  # noqa: E402
    extract_velocity,
    validate_capabilities,
)
from manimux.runtime.rtc.request import RtcInferenceRequest  # noqa: E402
from manimux.types import ObservationSnapshot, RobotState  # noqa: E402


def synthetic_identity(*, rtc=False):
    """Fixture only: no real backend is claimed to implement RTC+trace."""
    modes = ["default", "rtc"] if rtc else ["default"]
    model = {
        "policy_family": "pi05",
        "action_type": "joint",
        "observation_profile": "yam_native",
        "execution": {"backend": "jax"},
        "action_horizon": 50,
        "action_dim": 32,
        "num_steps": 7,
        "inference_seed": 123,
        "denoise_trace": {
            "enabled": True,
            "schema_version": 1,
            "field": "v",
            "axes": ["denoising_step", "action_step", "model_action"],
            "coordinate_space": "normalized_model",
            "velocity_kind": "raw_network_velocity",
            "model_action_dim": 32,
            "valid_action_indices": list(range(14)),
            "dtype": "float32",
            "supported_modes": modes,
        },
    }
    if rtc:
        model["rtc_guidance"] = {
            "padding_mask": "valid_action_indices",
            "valid_action_indices": list(range(14)),
        }
    return {"server": "synthetic_cpu_probe", "model": model}


def payload():
    actions = np.arange(50 * 14, dtype=np.float32).reshape(50, 14) / 10000
    steps = [
        {
            "left_arm_joint_state": row[:6],
            "left_ee_joint_state": row[6:7],
            "right_arm_joint_state": row[7:13],
            "right_ee_joint_state": row[13:14],
        }
        for row in actions
    ]
    velocity = np.arange(7 * 50 * 32, dtype=np.float32).reshape(7, 50, 32) / 10000
    return {"actions": steps, "denoise_trace": {"v": velocity}}


def runtime_config():
    return {
        "policy": {
            "worker": "xpolicylab_ws",
            "horizon_policy_steps": 50,
            "action_dt_s": 1 / 30,
            "timeout_s": 2.0,
            "startup_timeout_s": 2.0,
            "options": {"action_format": "joint"},
            "adapter": {},
        }
    }


def rtc_request(seq=1):
    now = time.monotonic_ns()
    return RtcInferenceRequest(
        session_id="chunktrust-wire-probe",
        request_seq=seq,
        observation_time_ns=now,
        deadline_ns=now + 2_000_000_000,
        observation=ObservationSnapshot(
            RobotState(
                {
                    "left_arm": np.zeros(7),
                    "right_arm": np.zeros(7),
                },
                now,
                1,
            ),
            {},
        ),
        instruction="synthetic probe",
        action_condition=np.zeros((50, 14)),
        condition_weights=np.r_[np.ones(5), np.zeros(45)],
        rtc_beta=1.25,
    )


class RecordingModel:
    """Actual Pi05 RTC forwarding method, without importing or loading OpenPI."""

    action_horizon = 50
    robot_action_dim = 14
    action_dim = 32

    def __init__(self):
        source = ast.parse((XPL / "policy/Pi_05/model.py").read_text())
        model = next(
            node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "Model"
        )
        method = next(
            node
            for node in model.body
            if isinstance(node, ast.FunctionDef) and node.name == "get_action_rtc"
        )
        namespace = {"Any": Any}
        exec(
            compile(
                ast.Module(body=[method], type_ignores=[]),
                str(XPL / "policy/Pi_05/model.py"),
                "exec",
            ),
            namespace,
        )
        self.rtc_method = namespace["get_action_rtc"]
        self.consumed = []
        self.output = payload()

    def _require_sampling_mode(self, mode):
        if mode != "rtc":
            raise ValueError(mode)

    def get_action(self, **kwargs):
        self.consumed.append(kwargs)
        return deepcopy(self.output)

    def infer(self, sampling):
        if sampling["mode"] == "rtc":
            return self.rtc_method(self, sampling)
        if sampling != {"mode": "default"}:
            raise ValueError("default probe accepts no sampling extras")
        return self.get_action()


class MemoryConnection:
    """Use real XPL Frame parsing/serialization over an in-memory byte queue."""

    def __init__(self, model, output):
        self.model, self.output = model, output
        self.replies, self.requests = deque(), []

    def send(self, wire):
        frame = Frame.from_wire_dict(decode_frame(wire))
        self.requests.append(frame.to_wire_dict())
        kind = frame.message_type.value
        if kind == "hello":
            result, reply = (
                {
                    "server": "synthetic_cpu_probe",
                    "model_metadata": synthetic_identity(rtc=True)["model"],
                    "capabilities": {"sampling_modes": ["default", "rtc"], "multi_inflight": True},
                },
                "hello_ack",
            )
        elif kind == "reset":
            result, reply = {"ok": True}, "reset_result"
        elif kind == "infer":
            result, reply = self.model.infer(frame.payload["sampling"]), "infer_result"
        else:
            raise AssertionError(kind)
        response = Frame.from_wire_dict(
            {
                **frame.to_wire_dict(),
                "message_type": reply,
                "payload": result,
            }
        )
        encoded = encode_frame(response)
        self.output.mkdir(parents=True, exist_ok=True)
        index = len(self.requests)
        (self.output / f"request-{index:03}.msgpack").write_bytes(wire)
        (self.output / f"response-{index:03}.msgpack").write_bytes(encoded)
        self.replies.append(encoded)

    def recv(self, timeout=None):
        if not self.replies:
            raise TimeoutError("empty synthetic byte queue")
        return self.replies.popleft()

    def close(self):
        pass


class WireChecks(unittest.TestCase):
    output: Path

    def test_strict_live_capabilities(self):
        config = runtime_config()
        identity = synthetic_identity()
        cap = PolicyCapabilities(frozenset({"default"}), identity)
        self.assertEqual(validate_capabilities(cap, config).num_steps, 7)
        for field, bad in (
            ("num_steps", True),
            ("inference_seed", False),
            ("observation_profile", []),
            ("action_horizon", 49),
        ):
            broken = deepcopy(identity)
            broken["model"][field] = bad
            with self.assertRaises(ValueError):
                validate_capabilities(PolicyCapabilities(frozenset({"default"}), broken), config)
        expected_config = deepcopy(config)
        expected_config["policy"]["expected_backend"] = {"model": {"num_steps": 10}}
        with self.assertRaisesRegex(ValueError, "num_steps"):
            validate_capabilities(cap, expected_config)
        with self.assertRaisesRegex(ValueError, "default only"):
            validate_capabilities(cap, config, rtc=True)
        for field, bad in (
            ("schema_version", True),
            ("enabled", 1),
            ("model_action_dim", True),
            ("valid_action_indices", [True, *range(1, 14)]),
            ("valid_action_indices", [0] * 14),
            ("valid_action_indices", list(range(1, 15))),
            ("velocity_kind", "guided_update"),
        ):
            with self.subTest(field=field, value=bad):
                broken = deepcopy(identity)
                broken["model"]["denoise_trace"][field] = bad
                with self.assertRaises(ValueError):
                    validate_capabilities(
                        PolicyCapabilities(frozenset({"default"}), broken), config
                    )
        combined = synthetic_identity(rtc=True)
        validate_capabilities(
            PolicyCapabilities(frozenset({"default", "rtc"}), combined), config, rtc=True
        )
        del combined["model"]["rtc_guidance"]
        with self.assertRaisesRegex(ValueError, "rtc_guidance"):
            validate_capabilities(
                PolicyCapabilities(frozenset({"default", "rtc"}), combined), config, rtc=True
            )

    def test_blocking_and_streaming_beta_and_trace(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                config = runtime_config()
                directory = self.output / ("streaming" if streaming else "blocking")
                config["policy"]["options"]["response_log_dir"] = str(directory / "journal")
                model = RecordingModel()
                connection = MemoryConnection(model, directory / "wire")
                client = XPolicyLabWsClient(
                    url="ws://127.0.0.1:1",
                    evaluation_id="probe",
                    trial_id="probe",
                )
                with patch("websockets.sync.client.connect", return_value=connection):
                    client.connect()
                worker = XPolicyLabWsPolicyModel(config["policy"])
                worker._client = client
                worker._session_id = "chunktrust-wire-probe"
                worker._layouts = (
                    GroupLayout("left_arm", "left", 6, 1),
                    GroupLayout("right_arm", "right", 6, 1),
                )
                request = rtc_request()
                profile = validate_capabilities(worker.capabilities(), config, rtc=True)
                with (
                    patch.object(worker, "_encode_observation", return_value={}),
                    ResponseLog(config["policy"], request),
                ):
                    if streaming:
                        self.assertTrue(worker.submit(request))
                        response = worker.poll()
                        self.assertIsNotNone(response)
                        self.assertIsNone(response.error)
                        result = response.raw_action
                    else:
                        result = worker.infer(request)
                self.assertEqual(connection.requests[-1]["payload"]["sampling"]["beta"], 1.25)
                self.assertEqual(model.consumed[-1]["rtc_beta"], 1.25)
                np.testing.assert_array_equal(
                    model.consumed[-1]["condition_weights"],
                    request.condition_weights.astype(np.float32),
                )
                actual = extract_velocity(result, profile)
                np.testing.assert_array_equal(actual, model.output["denoise_trace"]["v"][:, :, :14])
                self.assertEqual(result["actions"]["left_arm"].shape, (50, 7))
                client.close()

    def test_bad_trace_rejection_preserves_provider_wire(self):
        for label in ("missing", "dtype", "shape", "nan"):
            model = RecordingModel()
            if label == "missing":
                model.output["denoise_trace"] = {}
            elif label == "dtype":
                model.output["denoise_trace"]["v"] = np.zeros((7, 50, 32), np.float64)
            elif label == "shape":
                model.output["denoise_trace"]["v"] = np.zeros((8, 50, 32), np.float32)
            else:
                model.output["denoise_trace"]["v"][0, 0, 0] = np.nan
            config = runtime_config()
            config["policy"]["options"]["response_log_dir"] = str(self.output / label / "journal")
            connection = MemoryConnection(model, self.output / label / "wire")
            client = XPolicyLabWsClient(
                url="ws://127.0.0.1:1",
                evaluation_id="probe",
                trial_id="probe",
            )
            with patch("websockets.sync.client.connect", return_value=connection):
                client.connect()
            request = rtc_request()
            profile = validate_capabilities(
                PolicyCapabilities(frozenset({"default"}), synthetic_identity()),
                config,
            )
            with ResponseLog(config["policy"], request):
                result = client.infer({}, sampling={"mode": "default"})
                with self.assertRaisesRegex(ValueError, "denoise_trace.v"):
                    extract_velocity(result, profile)
            journal = list((self.output / label / "journal").rglob("response*.json"))
            self.assertEqual(len(journal), 1)
            saved = json.loads(journal[0].read_text())
            self.assertEqual(saved["stage"], "provider_wire")
            restored = unpack_frame(base64.b64decode(saved["response"]["data"]))["payload"]
            if label == "missing":
                self.assertEqual(restored["denoise_trace"], {})
            else:
                wanted = model.output["denoise_trace"]["v"]
                got = restored["denoise_trace"]["v"]
                self.assertEqual(got.dtype, wanted.dtype)
                self.assertEqual(got.shape, wanted.shape)
                np.testing.assert_array_equal(got, wanted)
            client.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.is_relative_to(ROOT):
        parser.error("--output must be outside the repository checkout")
    if (output / "report.json").exists():
        parser.error("report already exists; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    WireChecks.output = output
    with patch.object(socket, "socket", side_effect=AssertionError("network sockets forbidden")):
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(WireChecks)
        )
    paths = [
        Path(__file__).resolve(),
        XPL / "policy/Pi_05/model.py",
        XPL / "utils/sampling_contracts.py",
        XPL / "client_server/ws/protocol/codec.py",
        XPL / "client_server/ws/protocol/schemas.py",
        ROOT / "manimux/runtime/chunktrust/trace.py",
        ROOT / "manimux/policies/xpolicylab/client.py",
        ROOT / "manimux/policies/xpolicylab/ws_client.py",
    ]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    backend_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=XPL,
        capture_output=True,
        text=True,
        check=False,
    )
    report = {
        "ok": result.wasSuccessful(),
        "test_groups": result.testsRun,
        "failures": [text for _, text in result.failures + result.errors],
        "manimux_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "xpolicylab_revision": (
            backend_revision.stdout.strip() if backend_revision.returncode == 0 else None
        ),
        "python": sys.version,
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
        "versions": {
            name: metadata.version(name)
            for name in (
                "numpy",
                "msgpack",
                "msgpack-numpy",
                "pydantic",
                "websockets",
            )
        },
        "evidence": (
            "synthetic arrays; real XPL codec/validator and extracted Pi05 forwarding method"
        ),
        "limitations": "no model, sampler/JAX/GPU, service, socket, sensor or hardware validation",
        "rtc_identity": "synthetic test capability, not an available RTC+trace backend",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"ok": report["ok"], "report": str(output / "report.json")}))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
