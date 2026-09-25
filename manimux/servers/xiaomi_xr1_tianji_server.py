#!/usr/bin/env python3
"""Validate or serve Xiaomi Robotics 1 pass-ball through XPolicyLab WebSocket."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
XPOLICY_ROOT = REPO_ROOT / "XPolicyLab"
XR1_ROOT = XPOLICY_ROOT / "policy/Xiaomi_Robotics_1/xiaomi_robotics_1/xr1"
DEFAULT_EXPERIMENT = (
    REPO_ROOT
    / "manimux/configs/experiments/pass_ball/xiaomi-xr1/tianji_taccap_xiaomi_xr1_step50000.yaml"
)


def _shape(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    if not value:
        return (0,)
    child = _shape(value[0])
    if any(_shape(item) != child for item in value):
        raise ValueError("normalization arrays must be rectangular")
    return (len(value), *child)


def _resolve_checkpoint(value: object) -> Path:
    path = Path(str(value)).expanduser().resolve()
    if path.is_dir():
        candidates = (
            path / "mp_rank_00_model_states.pt",
            path / "last.ckpt/checkpoint/mp_rank_00_model_states.pt",
        )
        matches = [candidate for candidate in candidates if candidate.is_file()]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected one XR-1 mp_rank_00_model_states.pt under {path}, got {matches}"
            )
        return matches[0]
    return path


def _resolve_stats(config: dict[str, Any], checkpoint: Path) -> Path:
    configured = config.get("norm_stats_path")
    if configured:
        return Path(str(configured)).expanduser().resolve()
    candidate = checkpoint.parent / "training_metadata/normalize.json"
    if candidate.is_file():
        return candidate
    raise ValueError("norm_stats_path is required and was not found beside the checkpoint")


def _validate(config: dict[str, Any]) -> dict[str, Any]:
    required = {
        "policy_name": "Xiaomi_Robotics_1",
        "protocol": "ws",
        "action_type": "ee",
        "output_format": "packed_ee_delta",
        "ego_view_mode": "black",
        "observation_profile": "tianji_taccap_two_wrist_black_ego",
        "action_semantics": "anchor_relative_ee_delta",
    }
    mismatches = {
        name: (config.get(name), value)
        for name, value in required.items()
        if config.get(name) != value
    }
    if mismatches:
        raise ValueError(f"XR-1 Tianji server contract mismatch: {mismatches}")
    if int(config.get("action_length", 0)) != 30:
        raise ValueError("action_length must be 30 for the pass-ball checkpoint")
    if int(config.get("action_horizon", 0)) != 30:
        raise ValueError("action_horizon must be 30 for the pass-ball checkpoint")
    if int(config.get("num_steps", 0)) != 5:
        raise ValueError("num_steps must be 5 for Xiaomi Robotics 1")

    checkpoint_value = config.get("checkpoint_path")
    if not checkpoint_value:
        raise ValueError("checkpoint_path is required")
    checkpoint = _resolve_checkpoint(checkpoint_value)
    if not checkpoint.is_file() or not zipfile.is_zipfile(checkpoint):
        raise FileNotFoundError(f"XR-1 checkpoint is missing or invalid: {checkpoint}")
    with zipfile.ZipFile(checkpoint) as archive:
        tensor_records = [name for name in archive.namelist() if "/data/" in name]
    if not tensor_records:
        raise ValueError(f"XR-1 checkpoint has no tensor records: {checkpoint}")

    stats_path = _resolve_stats(config, checkpoint)
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    expected_shapes = {
        "mean": (30, 60),
        "std": (30, 60),
        "q01": (1, 60),
        "q99": (1, 60),
    }
    actual_shapes = {name: _shape(stats.get(name)) for name in expected_shapes}
    if actual_shapes != expected_shapes:
        raise ValueError(
            f"XR-1 normalization shapes must be {expected_shapes}, got {actual_shapes}"
        )
    for name in expected_shapes:
        flat = (number for row in stats[name] for number in row)
        if not all(isinstance(number, int | float) and math.isfinite(number) for number in flat):
            raise ValueError(f"normalization field {name} must contain finite numbers")

    processor_value = config.get("vlm_processor_path")
    if not isinstance(processor_value, str) or not processor_value.strip():
        raise ValueError("vlm_processor_path must be a local path or HuggingFace repo id")
    processor = Path(processor_value).expanduser()
    processor_status = "huggingface_repo_id"
    if processor.is_absolute() or processor.exists():
        processor = processor.resolve()
        for name in ("config.json", "tokenizer.json", "preprocessor_config.json"):
            if not (processor / name).is_file():
                raise FileNotFoundError(f"XR-1 processor is missing {processor / name}")
        processor_status = "local_files_present"
        processor_value = str(processor)
    elif "/" not in processor_value:
        raise ValueError("a non-local vlm_processor_path must be a HuggingFace repo id")

    config["checkpoint_path"] = str(checkpoint)
    config["norm_stats_path"] = str(stats_path)
    return {
        "contract_status": "ready",
        "inference_status": "not_run",
        "policy_name": config["policy_name"],
        "checkpoint_variant": config.get("checkpoint_variant"),
        "checkpoint": str(checkpoint),
        "checkpoint_tensor_records": len(tensor_records),
        "norm_stats": str(stats_path),
        "normalization_shapes": {name: list(shape) for name, shape in actual_shapes.items()},
        "processor": processor_value,
        "processor_status": processor_status,
        "model_action_shape": [30, 60],
        "runtime_action_shape": [30, 16],
        "ego_view": "synthetic_pure_black_rgb",
        "required_physical_cameras": ["left_wrist", "right_wrist"],
        "model_environment": {
            name: importlib.util.find_spec(name) is not None
            for name in ("torch", "transformers", "deepspeed", "websockets")
        },
        "python": sys.executable,
    }


def _prepare_imports() -> None:
    for path in (REPO_ROOT, XPOLICY_ROOT, XR1_ROOT):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--local", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Checkpoint file or directory containing mp_rank_00_model_states.pt",
    )
    parser.add_argument("--norm-stats", type=Path)
    parser.add_argument("--processor", help="Local processor directory or HuggingFace repo id")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    _prepare_imports()
    from manimux.cli import read_experiment

    experiment = read_experiment(args.experiment, local=args.local)
    config = dict(experiment["policy_server"])
    if args.checkpoint is not None:
        config["checkpoint_path"] = str(args.checkpoint)
    if args.norm_stats is not None:
        config["norm_stats_path"] = str(args.norm_stats)
    if args.processor is not None:
        config["vlm_processor_path"] = args.processor

    report = _validate(config)
    print(json.dumps(report, indent=2), flush=True)
    if args.check:
        return 0

    missing = [
        name
        for name, available in report["model_environment"].items()
        if not available
    ]
    if missing:
        raise RuntimeError(
            "policy environment is missing dependencies: " + ", ".join(missing)
        )
    from XPolicyLab.setup_policy_server import main as serve

    serve(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
