#!/usr/bin/env python3
"""Launch the XPolicyLab SAPolicy server for a configured YAM checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
XPOLICY_ROOT = REPO_ROOT / "XPolicyLab"
DEFAULT_CONFIG = REPO_ROOT / "configs/sapolicy/yam/server/abc-bottles.yaml"


def _prepare_imports() -> None:
    for path in (REPO_ROOT, XPOLICY_ROOT):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _load_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"server config must be a mapping: {path}")
    return loaded


def _validate(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("policy_name") != "SAPolicy":
        raise ValueError(f"policy_name must be 'SAPolicy', got {config.get('policy_name')!r}")
    if config.get("protocol", "ws") != "ws":
        raise ValueError("SAPolicy ManiMux path requires protocol: ws")
    dry_run = bool(config.get("dry_run", False))
    model_path = Path(str(config.get("model_path", ""))).expanduser().resolve()
    cfg_file = Path(str(config.get("cfg_file", ""))).expanduser().resolve()
    sapolicy_root = Path(str(config.get("sapolicy_root", ""))).expanduser().resolve()
    if not dry_run:
        if not model_path.exists():
            raise FileNotFoundError(f"model_path not found: {model_path}")
        if not cfg_file.is_file():
            raise FileNotFoundError(
                f"cfg_file not found: {cfg_file}. "
                "Point it at the SpatialAlign Hydra/resolved yaml matching this checkpoint."
            )
        if not sapolicy_root.is_dir():
            raise FileNotFoundError(f"sapolicy_root not found: {sapolicy_root}")
    horizon = int(config.get("action_horizon", 16))
    if horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {horizon}")
    return {
        "contract_status": "ready" if not dry_run else "dry_run",
        "policy_name": "SAPolicy",
        "protocol": "ws",
        "host": config.get("host", "127.0.0.1"),
        "port": config.get("port", 8500),
        "model_path": str(model_path),
        "cfg_file": str(cfg_file),
        "sapolicy_root": str(sapolicy_root),
        "action_type": config.get("action_type", "ee"),
        "output_format": config.get("output_format", "action_dict"),
        "action_horizon": horizon,
        "use_ema": bool(config.get("use_ema", True)),
        "dry_run": dry_run,
        "xpolicylab_root": str(XPOLICY_ROOT),
        "wire_action_dim": 16,
        "action_space": "absolute_ee",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check", action="store_true", help="validate and print resolved setup")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    config = _load_config(config_path)
    _prepare_imports()
    from XPolicyLab.policy.SAPolicy import DEFAULT_SAPOLICY_ROOT
    from XPolicyLab.policy.SAPolicy.assets import resolve_assets

    config.setdefault("sapolicy_root", str(DEFAULT_SAPOLICY_ROOT))
    # ManiMux deployment paths are workspace-relative; standalone XPolicyLab
    # eval uses its normal policy-directory checkpoint resolution instead.
    for key in (
        "model_path",
        "cfg_file",
        "sapolicy_root",
        "workspace",
        "backbone_path",
        "normalizer_path",
    ):
        if config.get(key):
            path = Path(str(config[key])).expanduser()
            config[key] = str((REPO_ROOT / path).resolve() if not path.is_absolute() else path)
    if not config.get("dry_run", False):
        config = resolve_assets(config)
        config.pop("resolved_cfg", None)
    contract = _validate(config)
    print("[sapolicy-yam-server] resolved setup", flush=True)
    print(json.dumps(contract, indent=2), flush=True)

    if args.check:
        return 0

    import setup_policy_server

    setup_policy_server.main(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
