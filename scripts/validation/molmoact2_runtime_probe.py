#!/usr/bin/env python3
"""Real MolmoAct2 WS server and shared runtime with recorded pixels and a fake plant."""

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import threading
import subprocess

import numpy as np



def run_with_model(model, model_cfg, args):
    """Keep model dependencies in their process; use the existing runtime env."""
    from XPolicyLab.setup_policy_server import _deployment_model_metadata
    from client_server.ws.model_server import PolicyServerConfig
    from manimux.policies.xpolicylab.server import StreamingPolicyServer
    root = Path(__file__).resolve().parents[2]
    loop = asyncio.new_event_loop()
    server = StreamingPolicyServer(model, PolicyServerConfig(
        host="127.0.0.1", port=0, model_metadata=_deployment_model_metadata(model_cfg)))
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    asyncio.run_coroutine_threadsafe(server.start(), loop).result(timeout=20)
    port = server._server.sockets[0].getsockname()[1]
    try:
        command = [str(root / "envs/yam/.venv/bin/python"), str(Path(__file__).resolve()),
                   "--local", str(args.local), "--checkpoint", str(args.checkpoint),
                   "--observation", str(args.observation), "--output", str(args.output),
                   "--seconds", str(args.seconds), "--variants", *args.variants,
                   "--endpoint", f"ws://127.0.0.1:{port}"]
        subprocess.run(command, check=True)
    finally:
        asyncio.run_coroutine_threadsafe(server.stop(), loop).result(timeout=40)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--variants", nargs="+", default=["async", "rtc", "paint", "autohorizon", "bid"])
    parser.add_argument("--endpoint", help="Internal client-only mode in the runtime environment")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    recipes = root / "manimux/configs/experiments/example-yam-molmoact2/coin"
    if args.endpoint is None:
        import yaml
        from XPolicyLab.policy.MolmoAct2.model import Model
        model_cfg = yaml.safe_load((root / "manimux/configs/policy/molmoact2/yam/latest300-step30000.yaml").read_text())
        model_cfg["model_path"] = str(args.checkpoint.resolve())
        model_cfg["num_steps"] = args.steps
        model = Model(model_cfg)
        run_with_model(model, model_cfg, args)
        return
    from manimux.cli import load_config
    from manimux.runtime import build_runtime
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.observation, allow_pickle=False) as saved:
        groups = {name: saved["state__" + name].copy() for name in ("left_arm", "right_arm")}
    report = {"hardware_used": False, "server": args.endpoint,
              "observation": str(args.observation.resolve()), "variants": {}}
    provenance = args.observation.parent / "input-provenance.json"
    if provenance.exists():
        report["input_provenance"] = json.loads(provenance.read_text())
    try:
        for variant in args.variants:
            config = load_config(recipes / f"{variant}.yaml", local=args.local)
            # Replace all physical constructors before building the runtime.
            config["robot"]["type"] = "scripts.validation.dream_api_probe:build_probe_robot"
            folder = args.output / variant
            folder.mkdir(exist_ok=True)
            config["robot"]["options"] = {
                "offline_model": True, "initial_groups": groups,
                "move_to_start_on_connect": False, "home_on_close": False,
                "command_log": str(folder / "commands.npz"),
            }
            config["sensors"] = [
                {"name": name, "driver": "scripts.validation.dream_api_probe:build_recorded_sensor",
                 "observation_npz": str(args.observation.resolve())}
                for name in config["policy"]["adapter"]["camera_map"].values()
            ]
            config["policy"]["options"]["server"] = report["server"]
            config["policy"]["options"]["response_log_dir"] = str(folder / "responses")
            config["robogui"]["enabled"] = False
            config["recording"]["enabled"] = False
            config["run"]["warmup_before_start"] = False
            config["run"]["max_control_steps"] = round(args.seconds * config["robot"]["control_hz"])
            config["run"]["task_prompt"] = "put the coin into the bin"
            try:
                result = build_runtime(config, folder).run()
                entry = {"result": asdict(result)}
                with np.load(folder / "commands.npz", allow_pickle=False) as commands:
                    entry["command_sink"] = {"count": len(commands["monotonic_ns"]),
                        "finite": all(np.isfinite(commands[name]).all() for name in groups)}
                events = [json.loads(line) for line in
                          (Path(result.episode_dir) / "events.jsonl").read_text().splitlines()]
                submitted = [event for event in events if event.get("kind") == "inference_submitted"]
                entry["conditioned_requests"] = sum(event.get("conditioned") is True for event in submitted)
                entry["unconditioned_requests"] = sum(event.get("conditioned") is False for event in submitted)
                entry["timing_events"] = [event for event in events if event.get("kind") in
                                          {"rtc_delay_infeasible", "plan_rejected"}]
                report["variants"][variant] = entry
            except Exception as exc:
                report["variants"][variant] = {"error": f"{type(exc).__name__}: {exc}"}
            (args.output / "report.json").write_text(json.dumps(report, indent=2, default=str))
            print(variant, json.dumps(report["variants"][variant], default=str), flush=True)
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2, default=str))
    if any("error" in entry or not entry["result"]["run_completed"]
           or not entry["command_sink"]["finite"] or entry["command_sink"]["count"] == 0
           for entry in report["variants"].values()):
        raise SystemExit("Disconnected runtime qualification failed; inspect report.json")


if __name__ == "__main__":
    main()
