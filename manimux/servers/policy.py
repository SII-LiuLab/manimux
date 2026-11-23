"""Launch a config or config list's model in its existing model environment."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path, help="One runtime config; read its policy_server")
    source.add_argument(
        "--config-list", type=Path, help="Config list; select a shared model recipe"
    )
    source.add_argument("--resolved-recipe", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--parent-pid", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--experiment-id")
    parser.add_argument("--local", type=Path)
    parser.add_argument(
        "--check", action="store_true", help="Resolve configuration only; no model load"
    )
    args = parser.parse_args(argv)
    if args.parent_pid is not None:
        if args.resolved_recipe is None or sys.platform != "linux":
            parser.error("Owned model children require Linux and a resolved recipe")
        import ctypes
        import signal

        # Kernel-owned parent-death signal also covers an abruptly killed router.
        if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "Cannot set model parent-death signal")
        if os.getppid() != args.parent_pid:
            return 1
    if args.experiment_id and args.config_list is None:
        parser.error("--experiment-id requires --config-list")
    if args.config_list is not None and args.experiment_id is None:
        from manimux.servers.model_catalog import ModelCatalogServer, resolve_catalog

        catalog = resolve_catalog(args.config_list, args.local)
        if catalog is not None:
            # Validate every declared artifact before publishing the catalog listener.
            for entry in catalog["routes"].values():
                if entry["recipe"]["policy_name"] == "Pi_05":
                    from manimux.servers.pi05 import _resolved_contract

                    _resolved_contract(catalog["source"], entry["recipe"])
            print(
                json.dumps(
                    {"model_catalog": catalog, "configuration_only": args.check},
                    default=str,
                    indent=2,
                ),
                flush=True,
            )
            if not args.check:
                logging.basicConfig(level=logging.INFO)
                asyncio.run(ModelCatalogServer(catalog).run())
            return 0
    if args.config_list is not None:
        from manimux.collections import select_service

        config = select_service(
            args.config_list, "policy_server", local=args.local, experiment=args.experiment_id
        )
    elif args.resolved_recipe is not None:
        config = {
            "policy": {"worker": "xpolicylab_ws"},
            "policy_server": json.loads(args.resolved_recipe.read_text()),
        }
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
