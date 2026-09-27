"""Load a paired recipe and launch XPolicyLab's shared policy server."""

import argparse
import json
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--experiment", type=Path)
    parser.add_argument("--local", type=Path)
    parser.add_argument(
        "--check", action="store_true", help="Print resolved config without loading a model"
    )
    args = parser.parse_args()
    if args.experiment:
        from manimux.cli import read_experiment, resolve_local_path

        config = read_experiment(
            args.experiment, local=resolve_local_path(args.experiment, args.local)
        )["policy_server"]
    else:
        if args.local:
            parser.error("--local requires --experiment")
        config = yaml.safe_load(args.config.read_text())
        config.pop("backend_identity", None)
    if config.get("policy_name") != "starVLA":
        raise ValueError("Expected a starVLA policy recipe")
    if not config.get("checkpoint_path"):
        raise ValueError(
            "Bind paths.checkpoint in the local file or set checkpoint_path in the recipe"
        )
    if args.check:
        print(json.dumps(config, indent=2))
        return
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "XPolicyLab"))
    from XPolicyLab.setup_policy_server import main as serve

    serve(config)


if __name__ == "__main__":
    main()
