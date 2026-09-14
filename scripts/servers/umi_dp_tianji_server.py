#!/usr/bin/env python3
"""Validate/bind UMI_DP artifacts or launch the shared XPolicyLab WebSocket server."""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO / "configs/umi_dp/tianji/server/pass_ball/default.yaml"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--bind-runtime-config", type=Path)
    parser.add_argument("--ik-backend", choices=("analytic", "diff"))
    parser.add_argument("--diff-ik-config", type=Path)
    parser.add_argument(
        "--runtime-template",
        type=Path,
        default=REPO / "configs/umi_dp/tianji/infra/pass_ball/default.yaml",
    )
    args = parser.parse_args()
    if (args.ik_backend or args.diff_ik_config) and not args.bind_runtime_config:
        parser.error("IK choices apply only to --bind-runtime-config")
    for path in (REPO, REPO / "XPolicyLab", REPO / "src"):
        sys.path.insert(0, str(path))
    config = yaml.safe_load(args.config.read_text())
    if config.get("policy_name") != "UMI_DP" or config.get("protocol") != "ws":
        raise ValueError("Expected the standard UMI_DP WebSocket config")
    if args.checkpoint:
        config["checkpoint_path"] = str(args.checkpoint.resolve())
    if args.check or args.bind_runtime_config:
        from XPolicyLab.policy.UMI_DP.artifact_identity import validate_deployment

        report = validate_deployment(config)
        print(json.dumps(report, indent=2), flush=True)
    if args.bind_runtime_config:
        output = args.bind_runtime_config.resolve()
        server_output = output.with_name(output.stem + "-server.yaml")
        if output.exists() or server_output.exists():
            raise FileExistsError("Refusing to overwrite paired deployment configs")
        runtime = yaml.safe_load(args.runtime_template.read_text())
        profile = (args.runtime_template.parent / runtime["control_profile"]).resolve()
        profile_data = yaml.safe_load(profile.read_text())
        if profile_data["action_dt_s"] != report["action_dt_s"]:
            raise ValueError(
                "Checkpoint action_dt_s differs from the shared control profile; "
                "select an explicitly matching profile"
            )
        runtime["control_profile"] = os.path.relpath(profile, output.parent)
        policy = runtime["policy"]
        if args.ik_backend:
            policy["options"]["ik_backend"] = args.ik_backend
        if args.diff_ik_config:
            if policy["options"].get("ik_backend", "analytic") != "diff":
                parser.error("--diff-ik-config requires ik_backend: diff")
            policy["options"]["diff_ik"] = yaml.safe_load(args.diff_ik_config.read_text())
        policy["expected_backend"]["model"].update(report)
        policy["horizon_steps"] = report["action_horizon"]
        policy["options"].update(
            {
                "deployment_bound": True,
                "observation_period_s": report["observation_period_s"],
                "first_action_offset_s": report["first_action_offset_s"],
            }
        )
        host = config.get("host", "127.0.0.1")
        if host in {"0.0.0.0", "::"}:
            host = "127.0.0.1"
        policy["options"]["server"] = f"ws://{host}:{config['port']}"
        config["checkpoint_path"] = report["checkpoint_path"]
        config["expected_artifacts"] = report
        output.parent.mkdir(parents=True, exist_ok=True)
        # Validate full profile merge and delegate RTC constraints before writing.
        import tempfile

        from manimux.config import load_config
        from manimux.integrations.umi_dp_tianji.history import HistoryStrategy
        from manimux.integrations.umi_dp_tianji.ik_config import bind_diff_ik_profile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", dir=output.parent) as stream:
            yaml.safe_dump(runtime, stream, sort_keys=False)
            stream.flush()
            resolved = load_config(stream.name)
            bind_diff_ik_profile(resolved)
            HistoryStrategy(resolved)
            runtime["policy"]["options"] = resolved.policy.options
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
