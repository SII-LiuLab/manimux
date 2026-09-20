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
        "--config", type=Path, default=REPO / "configs/policy/umi_dp/server.yaml"
    )
    source.add_argument(
        "--experiment", type=Path, help="Shared experiment entry with policy_server"
    )
    parser.add_argument("--local", type=Path, help="Local device, service and checkpoint bindings")
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
    for path in (REPO, REPO / "XPolicyLab", REPO / "src"):
        sys.path.insert(0, str(path))
    from manimux.cli import load_config, read_experiment

    experiment = (
        read_experiment(args.experiment, local=args.local) if args.experiment is not None else None
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
        runtime = read_experiment(args.experiment, local=args.local, bind_local=False)
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
        # local 提供的是客户端可达地址；不能用服务端 bind_host 覆盖它。
        if not local_policy_endpoint:
            policy["options"]["server"] = f"ws://{host}:{config['port']}"
        config["checkpoint_path"] = report["checkpoint_path"]
        config["expected_artifacts"] = report
        if experiment is not None:
            runtime["policy_server"] = config
        output.parent.mkdir(parents=True, exist_ok=True)
        # Validate full profile merge and delegate RTC constraints before writing.
        import tempfile

        from manimux.integrations.umi_dp_tianji.history import HistoryStrategy
        from manimux.integrations.umi_dp_tianji.ik_config import bind_diff_ik_profile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", dir=output.parent) as stream:
            yaml.safe_dump(runtime, stream, sort_keys=False)
            stream.flush()
            resolved = load_config(stream.name)
            bind_diff_ik_profile(resolved)
            HistoryStrategy(resolved)
            runtime["policy"]["options"] = resolved["policy"]["options"]
            if local_policy_endpoint:
                # 地址随工位选择，生成文件不保留会过期的第二份副本。
                runtime["policy"]["options"].pop("server", None)
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
