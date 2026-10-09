#!/usr/bin/env python3
"""Launch the XPolicyLab ABC_DiT server for a configured YAM checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
XPOLICY_ROOT = REPO_ROOT / "XPolicyLab"
DEFAULT_CONFIG = REPO_ROOT / "manimux/configs/policy/abc_dit/yam/put-bottles/pretrained-200k.yaml"
MODEL_PYTHON = REPO_ROOT / "envs/abc/.venv/bin/python"
PATH_KEYS = ("model_path", "norm_stats_path", "clip_cache_dir")
# Station service bound by a standalone --config launch (experiments name their own).
STATION_SERVICE = "policy_abc_dit"


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


def _resolve_paths(config: dict[str, Any]) -> dict[str, Any]:
    """Recipe paths are repository-relative unless absolute or home-relative."""
    resolved = dict(config)
    for key in PATH_KEYS:
        if resolved.get(key):
            path = Path(str(resolved[key])).expanduser()
            resolved[key] = str(path if path.is_absolute() else (REPO_ROOT / path).resolve())
    return resolved


def _validate(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    if config.get("policy_name") != "ABC_DiT":
        raise ValueError(f"policy_name must be 'ABC_DiT', got {config.get('policy_name')!r}")
    if config.get("protocol", "ws") != "ws":
        raise ValueError("ABC_DiT ManiMux path requires protocol: ws")
    if config.get("action_type", "joint") != "joint":
        raise ValueError("ABC-DiT emits absolute joint positions; action_type must be joint")
    model_path = Path(str(config.get("model_path", "")))
    if not model_path.is_file():
        raise FileNotFoundError(f"ABC-DiT checkpoint not found: {model_path}")
    if config.get("norm_stats_path") and not Path(config["norm_stats_path"]).is_file():
        raise FileNotFoundError(f"norm_stats_path not found: {config['norm_stats_path']}")
    clip_dir = Path(config.get("clip_cache_dir") or Path.home() / ".cache/clip")
    missing_clip = [
        name for name in ("ViT-B-32.pt", "bpe_simple_vocab_16e6.txt.gz") if not (clip_dir / name).is_file()
    ]
    num_steps = int(config.get("num_steps", 10))
    horizon = int(config.get("action_horizon", 30))
    if num_steps <= 0 or horizon != 30:
        raise ValueError("ABC-DiT XL needs num_steps > 0 and action_horizon: 30")
    return {
        "contract_status": "ready",
        "server_config": str(config_path),
        "model_python": str(MODEL_PYTHON),
        "environment_present": MODEL_PYTHON.is_file(),
        "xpolicylab_root": str(XPOLICY_ROOT),
        "host": config.get("host", "127.0.0.1"),
        "port": config.get("port"),
        "model_path": str(model_path),
        "checkpoint_source": config.get("checkpoint_source"),
        "norm_stats": config.get("norm_stats_path") or "checkpoint",
        "clip_assets": str(clip_dir) if not missing_clip else f"will download {missing_clip} to {clip_dir}",
        "prompt": config.get("prompt"),
        "action_space": "absolute_joint_position",
        "action_horizon": horizon,
        "num_steps": num_steps,
        "inference_seed": config.get("inference_seed", 0),
        "fast_inference": bool(config.get("fast_inference", False)),
        # paint: official ABC action-prefix RTC; rtc: PiGDM inference-time guidance.
        "sampling_modes": ["default", "paint", "rtc"],
        "rtc_prefix_length": int(config.get("rtc_prefix_length", 4)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    source.add_argument("--experiment", type=Path, help="Use the experiment's paired policy_server")
    parser.add_argument(
        "--local",
        type=Path,
        help="Station bindings (default: manimux/configs/local/station.yaml)",
    )
    parser.add_argument("--check", action="store_true", help="validate and print resolved setup")
    args = parser.parse_args(argv)

    _prepare_imports()
    from manimux.cli import bind_station, read_experiment, resolve_local_path

    local = resolve_local_path(args.experiment, args.local)
    config_path = (args.experiment or args.config).expanduser().resolve()
    if args.experiment is not None:
        config = read_experiment(config_path, local=local)["policy_server"]
    else:
        standalone = {"policy": {"service": STATION_SERVICE}, "policy_server": _load_config(config_path)}
        config = bind_station(standalone, local)["policy_server"]
    config = _resolve_paths(config)
    contract = _validate(config_path, config)
    contract["local"] = str(local)
    print("[abc-dit-yam-server] resolved setup", flush=True)
    print(json.dumps(contract, indent=2), flush=True)

    if args.check:
        return 0

    import setup_policy_server

    setup_policy_server.main(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
