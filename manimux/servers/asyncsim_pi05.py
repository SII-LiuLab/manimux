"""Launch the ManiMux XPolicyLab Pi_05 server paired with an AsyncSim experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from manimux.cli import read_experiment


REPO_ROOT = Path(__file__).resolve().parents[2]
XPOLICY_ROOT = REPO_ROOT / "XPolicyLab"
OPENPI_SRC = XPOLICY_ROOT / "policy/Pi_05/openpi/src"


def resolve_server(experiment: Path, local: Path) -> dict:
    config = read_experiment(experiment, local=local)
    server = config["policy_server"]
    if server.get("policy_name") != "Pi_05" or server.get("protocol") != "ws":
        raise ValueError("AsyncSim Pi_05 requires the ManiMux XPolicyLab WebSocket recipe")
    model_root = Path(server["model_path"])
    stats_dir = Path(server["norm_stats_path"])
    if not (model_root / "params").is_dir():
        raise FileNotFoundError(f"Pi_05 params not found: {model_root}")
    if not (stats_dir / "norm_stats.json").is_file():
        raise FileNotFoundError(f"Pi_05 norm_stats.json not found: {stats_dir}")
    expected = config["policy"]["expected_backend"]["model"]
    if expected["model_root"] != str(model_root) or expected["norm_stats_path"] != str(stats_dir):
        raise ValueError("policy backend identity does not match resolved checkpoint")
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    server = resolve_server(args.experiment, args.local)
    print(json.dumps({
        key: server[key] for key in (
            "policy_name", "task_name", "train_config_name", "repo_id", "model_path",
            "norm_stats_path", "action_horizon", "num_steps", "host", "port",
        )
    }, indent=2), flush=True)
    if args.check:
        return 0
    for path in (REPO_ROOT, XPOLICY_ROOT, OPENPI_SRC):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import setup_policy_server

    setup_policy_server.main(server)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
