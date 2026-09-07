#!/usr/bin/env python3
"""Check or launch the XPolicyLab Isaac 0.5 service."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
XPOLICY_ROOT = REPO_ROOT / "XPolicyLab"
DEFAULT_CONFIG = REPO_ROOT / "configs/isaac05/libero/server/base.yaml"


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"server config must be a mapping: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    for path in (REPO_ROOT, XPOLICY_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from XPolicyLab.policy.Isaac_05.model import validate_deployment

    config_path = args.config.expanduser().resolve()
    config = _load_config(config_path)
    report = validate_deployment(config)
    report["server_config"] = str(config_path)
    print("[isaac05-server] resolved setup", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    if args.check:
        return 0 if report["status"] == "ready" else 2
    if report["status"] != "ready":
        raise RuntimeError("Isaac 0.5 deployment is not ready: " + "; ".join(report["errors"]))
    import setup_policy_server

    setup_policy_server.main(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
