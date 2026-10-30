#!/usr/bin/env python3
"""Check the pinned managed Pi05 trace boundary without weights or sockets.

An isolated child loads the actual managed launcher, pinned Model/Policy, server
loader, codecs and ManiMux validators. The pinned test support supplies NumPy JAX
substitutes. Checkpoint loading, training config and the network are doubles, and
server execution is suppressed. This does not validate JAX, weights or deployment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
import traceback
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
XPL = ROOT / "third_party/XPolicyLab"
EXAMPLES = ROOT / "manimux/configs/experiments/example-yam-pi05/put_bottles"


def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def source_record(path: Path) -> dict:
    path = path.resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_source(expected_revision: str | None) -> dict:
    entry = git("ls-tree", "HEAD", "third_party/XPolicyLab").split()
    if len(entry) != 4 or entry[:2] != ["160000", "commit"]:
        raise ValueError("HEAD does not pin the XPolicyLab submodule")
    head_pin = entry[2]
    actual = git("rev-parse", "HEAD", cwd=XPL)
    expected = expected_revision or head_pin
    if not re.fullmatch("[0-9a-f]{40}", expected):
        raise ValueError("expected revision must be a complete lowercase commit SHA")
    if actual != expected:
        raise ValueError(f"XPolicyLab HEAD {actual} differs from required {expected}")
    if git("status", "--porcelain", "--untracked-files=all", cwd=XPL):
        raise ValueError("XPolicyLab source must be clean, including untracked files")
    url = git("config", "-f", ".gitmodules", "submodule.XPolicyLab.url")
    if url != "https://github.com/Cuzyoung/XPolicyLab.git":
        raise ValueError(f"unexpected managed backend URL: {url}")
    return {
        "manimux_revision": git("rev-parse", "HEAD"),
        "manimux_tracked_changes": git("status", "--porcelain", "--untracked-files=no"),
        "head_gitlink": head_pin,
        "backend_revision": actual,
        "backend_url": url,
        "head_gitlink_matches_checkout": head_pin == actual,
        "mode": "committed_pin" if head_pin == actual else "explicit_candidate_not_in_HEAD",
        "probe": source_record(Path(__file__)),
    }


def immediate(coroutine):
    """Drive a coroutine whose model execution has a synchronous CPU substitute."""
    try:
        coroutine.send(None)
    except StopIteration as result:
        return result.value
    else:
        raise AssertionError("CPU metadata handler unexpectedly requested an event loop")
    finally:
        coroutine.close()


def require_rejection(name, callback, errors=(TypeError, ValueError)) -> dict:
    try:
        callback()
    except errors as exc:
        return {"case": name, "error_type": type(exc).__name__, "message": str(exc)}
    raise AssertionError(f"invalid case was accepted: {name}")


def configurations(output: Path):
    import yaml

    from manimux.cli import load_config

    station = output / "synthetic-station.yaml"
    station.write_text(yaml.safe_dump({"paths": {"checkpoints": str(output / "no-weights")}}))
    ah_path = output / "autohorizon.yaml"
    ah_path.write_text(
        yaml.safe_dump(
            {
                "_base_": str(EXAMPLES / "serial.yaml"),
                "inference": {"algorithm": "autohorizon", "chunk_policy_steps": None},
            }
        )
    )
    paths = {
        "ahs": EXAMPLES / "chunktrust_ahs.yaml",
        "serial": EXAMPLES / "serial.yaml",
        "autohorizon": ah_path,
    }
    configs = {name: load_config(path, local=station) for name, path in paths.items()}
    for name, config in configs.items():
        assert config["policy_server"].get("return_denoise_trace", False) is (name == "ahs")
    return configs


def exercise_boundary(output: Path, report: dict) -> None:
    import numpy as np
    import yaml

    from manimux.policies.capabilities import PolicyCapabilities
    from manimux.policies.xpolicylab.ws_client import unpack_frame
    from manimux.runtime.chunktrust import trace as trace_contract
    from manimux.servers import pi05 as launcher

    # Check resolution before the lightweight support installs package shells to
    # bypass unrelated eager __init__ imports. Those shells do not prove routing.
    launcher._prepare_imports()
    spec = importlib.machinery.PathFinder.find_spec("XPolicyLab", sys.path)
    expected_entries = {(XPL / name).resolve() for name in ("__init__.py", "XPolicyLab.py")}
    if spec is None or Path(str(spec.origin)).resolve() not in expected_entries:
        raise AssertionError("managed import setup did not select the pinned XPolicyLab")
    package = importlib.import_module("XPolicyLab")
    if [Path(path).resolve() for path in package.__path__] != [XPL.resolve()]:
        raise AssertionError("managed XPolicyLab package path is not the pinned checkout")
    openpi = importlib.machinery.PathFinder.find_spec("openpi", sys.path)
    expected_openpi = XPL / "policy/Pi_05/openpi/src/openpi/__init__.py"
    if openpi is None or Path(str(openpi.origin)).resolve() != expected_openpi.resolve():
        raise AssertionError("managed import setup did not select pinned OpenPI")
    report["managed_resolution"] = {
        "XPolicyLab": str(spec.origin),
        "openpi": str(openpi.origin),
        "launcher": source_record(Path(launcher.__file__)),
    }
    assert report["source"]["backend_url"] == trace_contract.TRACE_BACKEND_URL
    assert report["source"]["backend_revision"] == trace_contract.TRACE_BACKEND_REVISION
    deploy = yaml.safe_load((XPL / "policy/Pi_05/deploy.yml").read_text())
    assert deploy["return_denoise_trace"] is False
    configs = configurations(output)
    exporter = load_file(
        "managed_trace_backend_exporter",
        ROOT / "scripts/setup/export_chunktrust_backend.py",
    )
    exported = exporter.export_backend(
        EXAMPLES / "chunktrust_ahs.yaml",
        local=output / "synthetic-station.yaml",
        output=output / "export/backend.yaml",
    )
    assert exported["schema_version"] == 1
    assert exported["backend_source"] == {
        "url": report["source"]["backend_url"],
        "revision": report["source"]["backend_revision"],
    }
    assert exported["backend_source_status"] == (
        "required_dependency_not_checked_out_or_loaded_by_exporter"
    )
    assert exported["validation"] == (
        "configuration_only; no checkpoint, environment, GPU or hardware check"
    )
    assert yaml.safe_load(Path(exported["recipe"]).read_text())["return_denoise_trace"] is True
    report["exporter"] = exported
    support_path = XPL / "tests/pi05_trace/support.py"
    support = load_file("pinned_trace_probe_support", support_path)
    report["sources"] = {"test_support": source_record(support_path)}
    report["config_trace_flags"] = {}
    report["rejections"] = []

    with support.isolated_adapter_modules() as modules:
        policy_module, backend = support.load_policy_with_numpy_backend()
        from manimux.policies.xpolicylab import server as managed

        for name, module in {
            "Model": modules.model,
            "Policy": policy_module,
            "codec": modules.codec,
            "shared_server": modules.server,
        }.items():
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(XPL.resolve()):
                raise AssertionError(f"{name} loaded outside the pinned submodule: {path}")
            report["sources"][name] = source_record(path)
        report["sources"]["managed_server"] = source_record(Path(managed.__file__))
        models, networks, servers = {}, {}, {}
        active = {}

        class Network:
            """Fixed arrays, not a neural sampler or an Euler/JAX implementation."""

            pi05, action_horizon, action_dim = True, 50, 32

            def __init__(self):
                self.calls = []

            def sample_actions(self, rng, observation, **kwargs):
                self.calls.append({"rng": int(rng), "options": dict(kwargs)})
                assert observation.state.shape == (1, 32)
                actions = np.full((1, 50, 32), 2, dtype=np.float32)
                if not kwargs.get("return_denoise_trace", False):
                    return actions
                steps = kwargs["num_steps"]
                velocity = np.arange(steps * 50 * 32, dtype=np.float32).reshape(1, steps, 50, 32)
                return actions, velocity / 1000

        def input_transform(value):
            state = np.asarray(value["observation/state"], dtype=np.float32)
            assert state.shape == (14,)
            return {"state": np.pad(state, (0, 18))}

        def output_transform(value):
            assert "denoise_trace" not in value
            return {"actions": value["actions"][:, :14] * 3 + 100}

        def train_config(_config):
            return SimpleNamespace(
                name=_config["train_config_name"],
                data=modules.config.LeRobotYamDataConfig(),
                model=SimpleNamespace(pi05=True, action_horizon=50, action_dim=32),
            )

        def load_policy(instance, model_cfg):
            name = active["name"]
            assert model_cfg == active["recipe"]
            instance.model_root = Path(model_cfg["model_path"])
            instance.norm_stats_path = Path(model_cfg["norm_stats_path"])
            network = networks[name] = Network()
            models[name] = instance
            return policy_module.Policy(
                network,
                transforms=[input_transform],
                output_transforms=[output_transform],
            )

        real_server = managed.StreamingPolicyServer

        def capture_server(model, config):
            server = real_server(model, config)
            servers[active["name"]] = server
            return server

        def suppress_serve(coroutine):
            coroutine.close()

        async def inline_model_call(method, *args):
            return method(*args)

        with ExitStack() as stack:
            stack.enter_context(patch.object(modules.model, "_resolve_train_config", train_config))
            stack.enter_context(patch.object(modules.model.Model, "get_model", load_policy))
            stack.enter_context(patch.object(managed, "StreamingPolicyServer", capture_server))
            stack.enter_context(patch.object(managed.asyncio, "run", suppress_serve))
            for name, config in configs.items():
                recipe = deepcopy(config["policy_server"])
                active.update(
                    name=name,
                    recipe={k: v for k, v in recipe.items() if k != "autohorizon"},
                )
                # Actual serve() performs the actual dynamic Model import and
                # Model(config) construction. Only network loading/start are replaced.
                managed.serve(recipe)
                model = models[name]
                assert model.return_denoise_trace is (name == "ahs")
                report["config_trace_flags"][name] = {
                    "algorithm": config["inference"]["algorithm"],
                    "recipe_flag": recipe.get("return_denoise_trace", False),
                    "model_flag": model.return_denoise_trace,
                    "model_class": f"{type(model).__module__}.{type(model).__name__}",
                    "config_sources": config["config_sources"],
                    "model_config_sha256": hashlib.sha256(
                        json.dumps(recipe, sort_keys=True, default=str).encode(),
                    ).hexdigest(),
                }
                if name != "ahs":
                    assert "denoise_trace" not in model.runtime_metadata()
            model, server, config = models["ahs"], servers["ahs"], configs["ahs"]
            model.update_obs(support.observation(yam=True))
            result = model.get_action()
            assert len(networks["ahs"].calls) == 1
            assert networks["ahs"].calls[0]["options"]["return_denoise_trace"] is True
            steps = config["policy_server"]["num_steps"]
            expected_velocity = np.arange(steps * 50 * 32, dtype=np.float32).reshape(steps, 50, 32)
            expected_velocity /= 1000
            np.testing.assert_array_equal(result["denoise_trace"]["v"], expected_velocity)
            # Visibly transformed actions and untransformed full-width trace.
            actions = np.stack(
                [
                    np.concatenate(
                        [
                            row[key]
                            for key in (
                                "left_arm_joint_state",
                                "left_ee_joint_state",
                                "right_arm_joint_state",
                                "right_ee_joint_state",
                            )
                        ]
                    )
                    for row in result["actions"]
                ]
            )
            np.testing.assert_array_equal(actions, np.full((50, 14), 106, dtype=np.float32))
            hello_request = modules.schemas.Frame(
                message_type=modules.messages.MessageType.HELLO,
                request_id="managed-probe-hello",
                evaluation_id="synthetic_cpu_probe",
                payload={},
            )
            with patch.object(server, "_call_model_method", inline_model_call):
                hello = immediate(server._dispatch_frame(hello_request))
            hello_wire = modules.codec.encode_frame(hello)
            hello_payload = unpack_frame(hello_wire)["payload"]
            capability = PolicyCapabilities(
                frozenset(hello_payload["capabilities"]["sampling_modes"]),
                {
                    "server": hello_payload["server"],
                    "model": hello_payload["model_metadata"],
                    "server_revision": hello_payload["server_revision"],
                },
            )
            profile = trace_contract.validate_capabilities(capability, config)
            response = modules.schemas.Frame(
                message_type=modules.messages.MessageType.INFER_RESULT,
                request_id="managed-probe-infer",
                evaluation_id="synthetic_cpu_probe",
                payload=result,
            )
            wire = modules.codec.encode_frame(response)
            decoded = unpack_frame(wire)["payload"]
            velocity = trace_contract.extract_velocity(decoded, profile)
            np.testing.assert_array_equal(velocity, expected_velocity[:, :, :14])
            (output / "hello.msgpack").write_bytes(hello_wire)
            (output / "response.msgpack").write_bytes(wire)
            report["trace"] = {
                "full_shape": list(decoded["denoise_trace"]["v"].shape),
                "selected_shape": list(velocity.shape),
                "dtype": str(velocity.dtype),
                "network_calls": len(networks["ahs"].calls),
                "wrapper_calls": len(backend.jit_calls),
                "payload_bytes": len(wire),
                "metadata": hello_payload["model_metadata"],
            }
            for label in ("missing", "dtype", "shape", "nonfinite"):
                broken = deepcopy(decoded)
                if label == "missing":
                    del broken["denoise_trace"]
                elif label == "dtype":
                    broken["denoise_trace"]["v"] = expected_velocity.astype(np.float64)
                elif label == "shape":
                    broken["denoise_trace"]["v"] = expected_velocity[:, :, :14]
                else:
                    broken["denoise_trace"]["v"][0, 0, 0] = np.nan
                report["rejections"].append(
                    require_rejection(
                        label,
                        lambda broken=broken: trace_contract.extract_velocity(broken, profile),
                    )
                )
            report["rejections"].append(
                require_rejection(
                    "rtc_capabilities",
                    lambda: trace_contract.validate_capabilities(capability, config, rtc=True),
                )
            )
            report["rejections"].append(
                require_rejection(
                    "rtc_model_request",
                    lambda: model.get_action_rtc({"mode": "rtc"}),
                )
            )
            assert len(networks["ahs"].calls) == 1
            serial = models["serial"]
            serial.update_obs(support.observation(yam=True))
            assert isinstance(serial.get_action(), list)
            assert not networks["serial"].calls[0]["options"].get("return_denoise_trace", False)
            without_trace = PolicyCapabilities(
                frozenset(serial.sampling_modes()),
                {"server": hello_payload["server"], "model": serial.runtime_metadata()},
            )
            report["rejections"].append(
                require_rejection(
                    "trace_disabled_model",
                    lambda: trace_contract.validate_capabilities(without_trace, config),
                )
            )


def child(output: Path, expected_revision: str | None) -> int:
    report = {
        "ok": False,
        "native_jax": False,
        "checkpoint_inference": False,
        "listening_server": False,
        "hardware": False,
        "evidence": "managed import/config/Model/Policy/codec boundary with a synthetic network",
        "substitutions": [
            "pinned NumPy JAX/Flax and optional dependency stubs",
            "training config and checkpoint loader",
            "network and sentinel model transforms",
            "server event-loop execution",
            "HELLO model-call scheduling",
        ],
    }
    try:
        report["source"] = verify_source(expected_revision)
        sys.path.insert(0, str(ROOT))
        with patch.object(socket, "socket", side_effect=AssertionError("sockets forbidden")):
            exercise_boundary(output, report)
        report["ok"] = True
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        traceback.print_exc()
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["ok"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory outside the checkout",
    )
    parser.add_argument(
        "--expected-revision",
        help="Full candidate SHA before recording its gitlink in HEAD",
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.is_relative_to(ROOT):
        parser.error("output must be outside the source checkout")
    if args.child:
        return child(output, args.expected_revision)
    output.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        "-I",
        str(Path(__file__).resolve()),
        "--child",
        "--output",
        str(output),
    ]
    if args.expected_revision:
        command.extend(["--expected-revision", args.expected_revision])
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}
    }
    result = subprocess.run(command, cwd=output, env=environment, text=True, capture_output=True)
    (output / "child.stdout.log").write_text(result.stdout)
    (output / "child.stderr.log").write_text(result.stderr)
    print(json.dumps({"ok": result.returncode == 0, "report": str(output / "report.json")}))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
