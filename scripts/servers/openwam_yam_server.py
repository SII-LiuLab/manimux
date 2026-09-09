#!/usr/bin/env python3
"""Validate or launch OpenWAM inside the standard XPolicyLab server."""

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO / "configs/openwam/yam/server/finetune.yaml"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--bind-runtime-config", type=Path,
                        help="Write a checkpoint-bound ManiMux config and paired server config; do not serve")
    parser.add_argument(
        "--dummy", action="store_true", help="Protocol debugging only; no model weights"
    )
    args = parser.parse_args()
    for path in (REPO, REPO / "XPolicyLab"):
        sys.path.insert(0, str(path))
    from XPolicyLab.policy.OpenWAM.model import validate_deployment

    config = yaml.safe_load(args.config.read_text())
    if config.get("policy_name") != "OpenWAM" or config.get("protocol") != "ws":
        raise ValueError("Expected the standard XPolicy OpenWAM WebSocket config")
    if args.checkpoint:
        config["ckpt_dir"] = str(args.checkpoint.resolve())
    if args.dummy:
        config["allow_dummy_policy"] = True
    report = validate_deployment(config)
    print(json.dumps(report, indent=2), flush=True)
    if args.bind_runtime_config:
        if config.get("allow_dummy_policy"):
            raise ValueError("Cannot bind deployment identity to a dummy policy")
        output = args.bind_runtime_config.resolve()
        server_output = output.with_name(output.stem + "-server.yaml")
        if output.exists() or server_output.exists():
            raise FileExistsError("Refusing to overwrite bound deployment configs")
        runtime = yaml.safe_load((REPO / "configs/openwam/yam/infra/manimux.yaml").read_text())
        identity_keys = ("checkpoint_path", "checkpoint_file", "checkpoint_sha256",
                         "training_config_sha256", "norm_stats_path", "norm_stats_sha256",
                         "action_horizon")
        identity = {key: report[key] for key in identity_keys}
        runtime["policy"]["expected_backend"]["model"].update(identity)
        runtime["policy"]["options"]["deployment_bound"] = True
        runtime["policy"]["horizon_steps"] = report["action_horizon"]
        host = config.get("host", "127.0.0.1")
        if host in {"0.0.0.0", "::"}:
            host = "127.0.0.1"
        runtime["policy"]["options"]["server"] = f"ws://{host}:{config.get('port', 8500)}"
        config["expected_artifacts"] = identity
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            yaml.safe_dump(runtime, stream, sort_keys=False)
        with server_output.open("x") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
        print(f"Runtime config: {output}\nServer config: {server_output}")
    elif not args.check:
        from XPolicyLab.setup_policy_server import main as serve

        serve(config)


if __name__ == "__main__":
    main()
