"""Launch a config or config list's model in its existing model environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path, help="One runtime config; read its policy_server")
    source.add_argument(
        "--config-list", type=Path, help="Config list; select a shared model recipe"
    )
    parser.add_argument("--experiment-id")
    parser.add_argument("--local", type=Path)
    parser.add_argument(
        "--check", action="store_true", help="Resolve configuration only; no model load"
    )
    args = parser.parse_args(argv)
    if args.experiment_id and args.config_list is None:
        parser.error("--experiment-id requires --config-list")
    if args.config_list is not None:
        from manimux.collections import select_service
        config = select_service(
            args.config_list, "policy_server", local=args.local, experiment=args.experiment_id
        )
    else:
        from manimux.cli import read_experiment, resolve_local_path
        config = read_experiment(args.config, local=resolve_local_path(args.config, args.local))
    if config["policy"]["worker"] != "xpolicylab_ws":
        raise ValueError("This launcher supports XPolicyLab deployments only")
    recipe = config["policy_server"]
    if not recipe.get("policy_name"):
        raise ValueError("policy_server.policy_name is required")
    print(
        json.dumps(
            {"policy_server": recipe, "configuration_only": args.check}, default=str, indent=2
        )
    )
    if args.check:
        return 0
    repository = Path(__file__).resolve().parents[2]
    for path in (repository, repository / "third_party", repository / "third_party" / "XPolicyLab"):
        sys.path.insert(0, str(path))
    if recipe["policy_name"] == "Pi_05":
        from manimux.servers.pi05 import _prepare_imports, _validate_paths

        _prepare_imports()
        _validate_paths(recipe)
    from manimux.policies.xpolicylab.server import serve

    serve(recipe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
