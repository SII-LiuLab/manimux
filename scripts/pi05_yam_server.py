#!/usr/bin/env python3
"""Launch the managed XPolicyLab Pi_05 server for a configured YAM checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
XPOLICY_ROOT = REPO_ROOT / "XPolicyLab"
OPENPI_SRC = XPOLICY_ROOT / "policy/Pi_05/openpi/src"
DEFAULT_CONFIG = REPO_ROOT / "configs/pi05/yam/server/finetune.yaml"
MODEL_PYTHON = XPOLICY_ROOT / "policy/Pi_05/openpi/.venv/bin/python"
HARDWARE_VERIFIED_VARIANTS = {
    "pi05_base_with_yam_stats",
    "pi05_yam_finetuned",
    "pi05_yam_pick_red_ball_box_step_1000",
}
YAM_FINETUNED_VARIANTS = HARDWARE_VERIFIED_VARIANTS
GPU_FORWARD_VERIFIED_VARIANTS = YAM_FINETUNED_VARIANTS
UMI_FINETUNED_VARIANTS = {
    "pi05_umi_exchange_ball",
}
TASK_QUALITY_LIMITED_VARIANTS = {
    "pi05_yam_finetuned",
    "pi05_yam_pick_red_ball_box_step_1000",
}


def _prepare_imports() -> None:
    for path in (REPO_ROOT, XPOLICY_ROOT, OPENPI_SRC):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _load_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"server config must be a mapping: {path}")
    return loaded


def _validate_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    if not (XPOLICY_ROOT / ".git").exists():
        raise FileNotFoundError(
            f"XPolicyLab submodule is missing under {XPOLICY_ROOT}; initialize submodules first"
        )
    model_root = Path(str(config["model_path"])).expanduser().resolve()
    stats_dir = Path(str(config["norm_stats_path"])).expanduser().resolve()
    if not (model_root / "params").is_dir():
        raise FileNotFoundError(f"Pi05 params not found under {model_root}")
    if not (stats_dir / "norm_stats.json").is_file():
        raise FileNotFoundError(f"YAM norm_stats.json not found under {stats_dir}")
    return model_root, stats_dir


def _resolved_contract(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    _prepare_imports()
    from openpi.training import config as openpi_config

    train_config = openpi_config.get_config(str(config["train_config_name"]))
    horizon = int(config.get("action_horizon", train_config.model.action_horizon))
    if horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {horizon}")
    num_steps = int(config.get("num_steps", 0))
    if num_steps <= 0:
        raise ValueError("server config num_steps must be positive")
    checkpoint_variant = str(config["checkpoint_variant"])
    environment_present = MODEL_PYTHON.is_file()
    hardware_verified = checkpoint_variant in HARDWARE_VERIFIED_VARIANTS
    gpu_forward_verified = checkpoint_variant in GPU_FORWARD_VERIFIED_VARIANTS
    return {
        "contract_status": "ready",
        "runtime_status": (
            "hardware_verified"
            if environment_present and hardware_verified
            else "gpu_forward_verified_hardware_not_verified"
            if environment_present and gpu_forward_verified
            else "blocked_missing_model_environment"
            if not environment_present
            else "environment_present_gpu_forward_not_verified"
        ),
        "inference_status": (
            "hardware_verified"
            if hardware_verified
            else "gpu_forward_verified"
            if gpu_forward_verified
            else "not_verified"
        ),
        "policy_status": (
            "yam_finetune_task_quality_limited"
            if checkpoint_variant in TASK_QUALITY_LIMITED_VARIANTS
            else "yam_finetune_not_evaluated"
            if checkpoint_variant in YAM_FINETUNED_VARIANTS
            else "umi_finetune_not_evaluated"
            if checkpoint_variant in UMI_FINETUNED_VARIANTS
            else "base_checkpoint_not_yam_finetune"
        ),
        "model_python": str(MODEL_PYTHON),
        "server_config": str(config_path),
        "checkpoint_variant": checkpoint_variant,
        "checkpoint_source": config.get("checkpoint_source"),
        "norm_stats_source": config.get("norm_stats_source"),
        "xpolicylab_root": str(XPOLICY_ROOT),
        "train_config_name": train_config.name,
        "action_space": "absolute_joint_position",
        "action_horizon": horizon,
        "num_steps": num_steps,
        "rtc": "pi_guided_v1",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check", action="store_true", help="validate and print resolved setup")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    config = _load_config(config_path)
    model_root, stats_dir = _validate_paths(config)
    contract = _resolved_contract(config_path, config)
    contract.update(
        model_root=str(model_root),
        norm_stats=str(stats_dir / "norm_stats.json"),
    )
    print("[pi05-yam-server] resolved setup", flush=True)
    print(json.dumps(contract, indent=2), flush=True)

    if args.check:
        return 0

    import setup_policy_server

    setup_policy_server.main(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
