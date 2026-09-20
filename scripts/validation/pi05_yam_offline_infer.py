#!/usr/bin/env python3
"""Run one Pi05-YAM model forward without servers, sensors, or robot hardware."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
XPOLICY_ROOT = REPO_ROOT / "XPolicyLab"
OPENPI_SRC = XPOLICY_ROOT / "policy/Pi_05/openpi/src"
DEFAULT_CONFIG = REPO_ROOT / "manimux/configs/policy/pi05/yam/finetune.yaml"
DEFAULT_INFRA_CONFIG = REPO_ROOT / "manimux/configs/experiments/pick_red_object/yam_pi05_manimux.yaml"


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


def _resolve_path(value: str, *, relative_to: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (relative_to / path).resolve()


def _load_start_state(infra_path: Path) -> np.ndarray:
    infra = _load_config(infra_path)
    robot = infra.get("robot")
    if not isinstance(robot, dict):
        raise ValueError(f"infra config is missing robot mapping: {infra_path}")
    options = robot.get("options")
    if not isinstance(options, dict):
        raise ValueError(f"infra config is missing robot.options: {infra_path}")
    left_path = _resolve_path(str(robot["config"]), relative_to=REPO_ROOT)
    right_path = _resolve_path(str(options["right_config"]), relative_to=REPO_ROOT)
    left = np.asarray(_load_config(left_path)["agent"]["start_joints"], dtype=np.float32)
    right = np.asarray(_load_config(right_path)["agent"]["start_joints"], dtype=np.float32)
    if left.shape != (7,) or right.shape != (7,):
        raise ValueError(
            f"YAM start_joints must be 7D per arm, got {left.shape} and {right.shape}"
        )
    return np.concatenate([left, right])


def _synthetic_observation(
    image_height: int,
    image_width: int,
    start_state: np.ndarray,
) -> dict[str, Any]:
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    return {
        "images": {
            "cam_high": image,
            "cam_left_wrist": image.copy(),
            "cam_right_wrist": image.copy(),
        },
        "state": start_state,
        "instruction": "Pick the red ball up and place it into the box.",
    }


def _pack_action_chunk(action_steps: Any) -> np.ndarray:
    if not isinstance(action_steps, list) or not action_steps:
        raise TypeError("Pi05-YAM adapter must return a non-empty action-step list")
    packed_steps = []
    for step in action_steps:
        packed_steps.append(
            np.concatenate(
                [
                    np.asarray(step["left_arm_joint_state"]),
                    np.asarray(step["left_ee_joint_state"]),
                    np.asarray(step["right_arm_joint_state"]),
                    np.asarray(step["right_ee_joint_state"]),
                ]
            )
        )
    return np.stack(packed_steps).astype(np.float32, copy=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--infra-config", type=Path, default=DEFAULT_INFRA_CONFIG)
    parser.add_argument("--image-height", type=int, default=360)
    parser.add_argument("--image-width", type=int, default=640)
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    infra_path = args.infra_config.expanduser().resolve()
    config = _load_config(config_path)
    start_state = _load_start_state(infra_path)
    _prepare_imports()

    from XPolicyLab.policy.Pi_05.model import Model

    load_started = time.monotonic()
    model = Model(config)
    load_seconds = time.monotonic() - load_started

    model.update_obs(
        _synthetic_observation(args.image_height, args.image_width, start_state)
    )
    infer_started = time.monotonic()
    action_chunk = _pack_action_chunk(model.get_action())
    infer_seconds = time.monotonic() - infer_started

    expected_shape = (int(config["action_horizon"]), 14)
    if action_chunk.shape != expected_shape:
        raise ValueError(f"expected action shape {expected_shape}, got {action_chunk.shape}")
    if not np.isfinite(action_chunk).all():
        raise ValueError("model returned non-finite actions")

    report = {
        "mode": "offline_synthetic_observation",
        "hardware_connected": False,
        "config": str(config_path),
        "infra_config": str(infra_path),
        "model_path": str(Path(config["model_path"]).expanduser().resolve()),
        "norm_stats": str(
            Path(config["norm_stats_path"]).expanduser().resolve() / "norm_stats.json"
        ),
        "action_semantics": "absolute_joint_position",
        "action_shape": list(action_chunk.shape),
        "load_seconds": round(load_seconds, 3),
        "infer_seconds": round(infer_seconds, 3),
        "action_min": float(action_chunk.min()),
        "action_max": float(action_chunk.max()),
        "action_mean": float(action_chunk.mean()),
        "first_action": action_chunk[0].tolist(),
        "initial_state": start_state.tolist(),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
