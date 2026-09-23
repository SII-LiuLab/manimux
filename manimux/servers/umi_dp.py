#!/usr/bin/env python3
"""Validate/bind UMI_DP artifacts or launch the shared XPolicyLab WebSocket server."""

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--config",
        type=Path,
        default=REPO / "manimux/configs/policy/umi_dp/tianji/pass_ball/default.yaml",
    )
    source.add_argument(
        "--experiment", type=Path, help="Shared experiment entry with policy_server"
    )
    parser.add_argument(
        "--local", type=Path, help="Override the default station file for --experiment"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--bind-runtime-config", type=Path)
    parser.add_argument("--ik-backend", choices=("analytic", "diff"))
    parser.add_argument("--diff-ik-config", type=Path)
    args = parser.parse_args()
    if (args.ik_backend or args.diff_ik_config) and not args.bind_runtime_config:
        parser.error("IK choices apply only to --bind-runtime-config")
    if args.local is not None and args.experiment is None:
        parser.error("--local requires --experiment")
    if args.bind_runtime_config is not None and args.experiment is None:
        parser.error("--bind-runtime-config requires --experiment")
    for path in (REPO, REPO / "XPolicyLab"):
        sys.path.insert(0, str(path))
    from manimux.cli import load_config, read_experiment, resolve_local_path

    # Resolve once so the launcher and any exported runtime select the same station.
    local_path = (
        resolve_local_path(args.experiment, args.local) if args.experiment is not None else None
    )
    experiment = (
        read_experiment(args.experiment, local=local_path) if args.experiment is not None else None
    )
    config = (
        experiment["policy_server"]
        if experiment is not None
        else yaml.safe_load(args.config.read_text())
    )
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
        runtime = read_experiment(args.experiment, local=local_path, bind_local=False)
        # The experiment owns action timing; body profiles only supply layout and limits.
        expected_dt = runtime["policy"]["action_dt_s"]
        if expected_dt != report["action_dt_s"]:
            raise ValueError(
                "Checkpoint action_dt_s differs from the experiment; "
                "select explicitly matching policy/control settings"
            )
        # Keep the same assembly when the bound experiment is written elsewhere.
        if runtime.get("robot", {}).get("config") is not None:
            assembly = Path(runtime["robot"]["config"])
            runtime["robot"]["config"] = str(assembly)
        policy = runtime["policy"]
        local_policy_endpoint = experiment is not None and runtime.get("local") is not None
        if args.ik_backend:
            policy["adapter"]["ik_backend"] = args.ik_backend
        if args.diff_ik_config:
            if policy["adapter"].get("ik_backend", "analytic") != "diff":
                parser.error("--diff-ik-config requires ik_backend: diff")
            policy["adapter"]["diff_ik"] = yaml.safe_load(args.diff_ik_config.read_text())
        policy["expected_backend"]["model"].update(report)
        policy["horizon_policy_steps"] = report["action_horizon"]
        host = config.get("host", "127.0.0.1")
        if host in {"0.0.0.0", "::"}:
            host = "127.0.0.1"
        # Preserve the reachable client address instead of replacing it with bind_host.
        if not local_policy_endpoint:
            policy["options"]["server"] = f"ws://{host}:{config['port']}"
        config["checkpoint_path"] = report["checkpoint_path"]
        config["expected_artifacts"] = report
        if experiment is not None:
            runtime["policy_server"] = config
        output.parent.mkdir(parents=True, exist_ok=True)
        # Validate full profile merge and delegate RTC constraints before writing.
        import tempfile

        from manimux.policy_adapter.umi_dp.history import HistoryStrategy
        from manimux.policy_adapter.umi_dp.ik_config import bind_diff_ik_profile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", dir=output.parent) as stream:
            yaml.safe_dump(runtime, stream, sort_keys=False)
            stream.flush()
            resolved = load_config(stream.name)
            bind_diff_ik_profile(resolved)
            HistoryStrategy(resolved)
            runtime["policy"]["options"] = resolved["policy"]["options"]
            runtime["policy"]["adapter"] = resolved["policy"]["adapter"]
            if local_policy_endpoint:
                # Keep addresses in the selected station instead of copying stale values.
                runtime["policy"]["options"].pop("server", None)
        with output.open("x") as stream:
            yaml.safe_dump(runtime, stream, sort_keys=False)
        with server_output.open("x") as stream:
            # This explicit --config artifact is a snapshot; --experiment rereads the station.
            yaml.safe_dump(config, stream, sort_keys=False)
        print(f"Runtime config: {output}\nServer config: {server_output}")
    elif not args.check:
        from XPolicyLab.setup_policy_server import main as serve

        serve(config)


if __name__ == "__main__":
    main()
