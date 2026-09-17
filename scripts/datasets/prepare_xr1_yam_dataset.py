#!/usr/bin/env python3
"""Convert native YAM recordings with recorded EE poses to XR-1 JSON format.

XR-1 trains on end-effector targets.  Current YAM recordings persist the exact
forward-kinematics result used at collection time, so this converter consumes
those arrays directly instead of rebuilding MuJoCo kinematics in a training
job.  It also emits the normalization statistics and Hydra data config consumed
by XR-1 training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from manimux.integrations.xr1_yam.codec import (
    ACTION_DIM,
    ACTION_PARTS,
    STATE_DIM,
    rotm2aa_batch,
)
ACTION_LENGTH = 30
REQUIRED_ARRAYS = (
    "left-joint_pos.npy",
    "left-gripper_pos.npy",
    "right-joint_pos.npy",
    "right-gripper_pos.npy",
    "action-left-joint.npy",
    "action-left-gripper.npy",
    "action-right-joint.npy",
    "action-right-gripper.npy",
)
REQUIRED_EE_ARRAYS = tuple(
    f"{prefix}{arm}-ee_{kind}.npy"
    for arm in ("left", "right")
    for prefix in ("", "action-")
    for kind in ("pos", "rotm", "transform")
)
VIDEOS = {
    "ego": "top-images-rgb.mp4",
    "wrist_left": "left-images-rgb.mp4",
    "wrist_right": "right-images-rgb.mp4",
}
STATE_SLOTS = {
    "left": {"joints": slice(0, 6), "gripper": 7},
    "right": {"joints": slice(8, 14), "gripper": 15},
}


def _episodes(root: Path) -> list[Path]:
    episodes = []
    for candidate in sorted(path for path in root.iterdir() if path.is_dir()):
        required = [
            candidate / name
            for name in (*REQUIRED_ARRAYS, *REQUIRED_EE_ARRAYS, *VIDEOS.values())
        ]
        if (candidate / "write_complete.flag").is_file() and all(path.is_file() for path in required):
            episodes.append(candidate)
    return episodes


def _load_episode(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays = {
        name.removesuffix(".npy"): np.load(path / name)
        for name in (*REQUIRED_ARRAYS, *REQUIRED_EE_ARRAYS)
    }
    metadata = json.loads((path / "metadata.json").read_text())
    length = int(metadata["num_frames"])
    lengths = {name: len(value) for name, value in arrays.items()}
    if any(value != length for value in lengths.values()):
        raise ValueError(f"{path}: metadata num_frames={length}, array lengths={lengths}")
    for arm in ("left", "right"):
        for prefix in ("", "action-"):
            joints = arrays[f"{prefix}{arm}-joint" if prefix else f"{arm}-joint_pos"]
            gripper = arrays[f"{prefix}{arm}-gripper" if prefix else f"{arm}-gripper_pos"]
            if joints.shape != (length, 6) or gripper.shape != (length, 1):
                raise ValueError(
                    f"{path}: unexpected {prefix}{arm} shapes {joints.shape}, {gripper.shape}"
                )
            pos = arrays[f"{prefix}{arm}-ee_pos"]
            rotm = arrays[f"{prefix}{arm}-ee_rotm"]
            transform = arrays[f"{prefix}{arm}-ee_transform"]
            if pos.shape != (length, 3) or rotm.shape != (length, 9) or transform.shape != (
                length,
                4,
                4,
            ):
                raise ValueError(
                    f"{path}: unexpected recorded EE shapes for {prefix}{arm}: "
                    f"{pos.shape}, {rotm.shape}, {transform.shape}"
                )
            if not np.isfinite(pos).all() or not np.isfinite(rotm).all() or not np.isfinite(
                transform
            ).all():
                raise ValueError(f"{path}: non-finite recorded EE pose for {prefix}{arm}")
            np.testing.assert_allclose(transform[:, :3, 3], pos, atol=1e-9, rtol=0)
            np.testing.assert_allclose(
                transform[:, :3, :3].reshape(length, 9), rotm, atol=1e-9, rtol=0
            )
    if not metadata.get("extra", {}).get("eepose", {}).get("enabled", False):
        raise ValueError(f"{path}: metadata does not declare recorded EE poses")
    return arrays, metadata


def _episode_payload(
    episode: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    instruction: str,
) -> tuple[dict[str, Any], np.ndarray, dict[int, list[np.ndarray]]]:
    length = int(metadata["num_frames"])
    poses: dict[str, dict[str, np.ndarray]] = {}
    for arm in ("left", "right"):
        poses[arm] = {
            "state": arrays[f"{arm}-ee_transform"],
            "action": arrays[f"action-{arm}-ee_transform"],
        }

    states = np.zeros((length, STATE_DIM), dtype=np.float64)
    action_by_step: dict[int, list[np.ndarray]] = {
        step: [] for step in range(ACTION_LENGTH)
    }
    parts = dict(ACTION_PARTS)
    for arm in ("left", "right"):
        slots = STATE_SLOTS[arm]
        states[:, slots["joints"]] = arrays[f"{arm}-joint_pos"]
        states[:, slots["gripper"]] = arrays[f"{arm}-gripper_pos"][:, 0]

    for frame in range(length):
        for step in range(min(ACTION_LENGTH, length - frame)):
            target = frame + step
            packed = np.zeros(ACTION_DIM, dtype=np.float64)
            for arm in ("left", "right"):
                current = poses[arm]["state"][frame]
                command = poses[arm]["action"][target]
                rotation = current[:3, :3]
                packed[parts[f"{arm}_ee_pos"]] = rotation.T @ (
                    command[:3, 3] - current[:3, 3]
                )
                packed[parts[f"{arm}_ee_aa"]] = rotm2aa_batch(
                    (rotation.T @ command[:3, :3])[None]
                )[0]
                packed[parts[f"{arm}_gripper"]] = (
                    arrays[f"action-{arm}-gripper"][target, 0]
                    - arrays[f"{arm}-gripper_pos"][frame, 0]
                )
            action_by_step[step].append(packed)

    prompt = (
        "The following observations are captured from multiple views.\n"
        "# Ego View\n<image>\n# Left-Wrist View\n<image>\n"
        "# Right-Wrist View\n<image>\nGenerate robot actions for the task:\n"
        f"{instruction}"
    )
    video = lambda name: [{"path": str((episode / name).resolve()), "start": 0, "crop_bbox": None}]
    payload = {
        "trajectory_type": "success",
        "time": episode.name,
        "num_frames": length,
        "instruction": {
            "general": [
                {
                    "images": [
                        "observations.ego",
                        "observations.wrist_left",
                        "observations.wrist_right",
                    ],
                    "conversations": [
                        {"from": "human", "value": prompt},
                        {"from": "gpt", "value": ""},
                    ],
                }
            ]
        },
        "observations": {
            "ego": video(VIDEOS["ego"]),
            "wrist_left": video(VIDEOS["wrist_left"]),
            "wrist_right": video(VIDEOS["wrist_right"]),
        },
        "proprios": {
            "left_ee_pos": poses["left"]["state"][:, :3, 3].tolist(),
            "left_ee_rotm": poses["left"]["state"][:, :3, :3].reshape(length, 9).tolist(),
            "left_arm_joint": arrays["left-joint_pos"].tolist(),
            "left_gripper_pos": arrays["left-gripper_pos"].tolist(),
            "right_ee_pos": poses["right"]["state"][:, :3, 3].tolist(),
            "right_ee_rotm": poses["right"]["state"][:, :3, :3].reshape(length, 9).tolist(),
            "right_arm_joint": arrays["right-joint_pos"].tolist(),
            "right_gripper_pos": arrays["right-gripper_pos"].tolist(),
            "waist_pos": np.zeros((length, 1)).tolist(),
        },
        "actions": {
            "left_ee_pos": poses["left"]["action"][:, :3, 3].tolist(),
            "left_ee_rotm": poses["left"]["action"][:, :3, :3].reshape(length, 9).tolist(),
            "left_gripper_pos": arrays["action-left-gripper"].tolist(),
            "right_ee_pos": poses["right"]["action"][:, :3, 3].tolist(),
            "right_ee_rotm": poses["right"]["action"][:, :3, :3].reshape(length, 9).tolist(),
            "right_gripper_pos": arrays["action-right-gripper"].tolist(),
            "waist_pos": np.zeros((length, 1)).tolist(),
            "base_vel": np.zeros((length, 3)).tolist(),
        },
    }
    return payload, states, action_by_step


def _stats(
    states: list[np.ndarray], actions: dict[int, list[np.ndarray]]
) -> dict[str, Any]:
    all_states = np.concatenate(states)
    mean = np.zeros((ACTION_LENGTH, ACTION_DIM), dtype=np.float64)
    std = np.ones((ACTION_LENGTH, ACTION_DIM), dtype=np.float64)
    active_action = np.zeros(ACTION_DIM, dtype=bool)
    active_action[0:7] = True
    active_action[8:15] = True
    for step in range(ACTION_LENGTH):
        values = np.stack(actions[step])
        mean[step, active_action] = values[:, active_action].mean(axis=0)
        std[step, active_action] = np.maximum(
            values[:, active_action].std(axis=0), 1e-6
        )

    q01 = np.zeros((1, STATE_DIM), dtype=np.float64)
    q99 = np.zeros((1, STATE_DIM), dtype=np.float64)
    active_state = np.zeros(STATE_DIM, dtype=bool)
    for arm in ("left", "right"):
        active_state[STATE_SLOTS[arm]["joints"]] = True
        active_state[STATE_SLOTS[arm]["gripper"]] = True
    q01[0, active_state] = np.quantile(all_states[:, active_state], 0.01, axis=0)
    q99[0, active_state] = np.quantile(all_states[:, active_state], 0.99, axis=0)
    degenerate = q99 <= q01
    q01[degenerate] = 0.0
    q99[degenerate] = 0.0
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "q01": q01.tolist(),
        "q99": q99.tolist(),
        "action_length": ACTION_LENGTH,
    }


def _write_config(path: Path, data_dir: Path, stats: dict[str, Any], batch_size: int) -> None:
    config = {
        "data": {
            "type": "BaseDataModule",
            "params": {
                "type": "json",
                "max_steps": "${trainer.max_steps}",
                "train_datasets": {
                    "batch_size": batch_size,
                    "action_length": ACTION_LENGTH,
                    "paths": [str(data_dir.resolve())],
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "q01": stats["q01"],
                    "q99": stats["q99"],
                },
            },
        }
    }
    path.write_text("# @package _global_\n" + yaml.safe_dump(config, sort_keys=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True, help="one YAM task directory")
    parser.add_argument("--output", type=Path, required=True, help="XR-1 dataset root")
    parser.add_argument(
        "--instruction",
        default="Assemble the screwdriver.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--config-name", default="yam_dataset")
    args = parser.parse_args()

    episodes = _episodes(args.episodes.resolve())
    if not episodes:
        raise SystemExit(f"no complete YAM episodes under {args.episodes}")
    data_dir = args.output.resolve() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    all_states: list[np.ndarray] = []
    all_actions: dict[int, list[np.ndarray]] = {
        step: [] for step in range(ACTION_LENGTH)
    }
    manifest_episodes = []
    total_frames = 0
    for index, episode in enumerate(episodes, 1):
        arrays, metadata = _load_episode(episode)
        payload, states, actions = _episode_payload(episode, arrays, metadata, args.instruction)
        destination = data_dir / f"{episode.name}.json"
        destination.write_text(json.dumps(payload, separators=(",", ":")))
        all_states.append(states)
        for step, values in actions.items():
            all_actions[step].extend(values)
        total_frames += len(states)
        manifest_episodes.append(
            {
                "episode": episode.name,
                "frames": len(states),
                "annotation": destination.name,
            }
        )
        print(f"[{index}/{len(episodes)}] {episode.name}: {len(states)} frames")

    stats = _stats(all_states, all_actions)
    stats["_source"] = (
        f"{len(episodes)} complete YAM episodes and {total_frames} frames under "
        f"{args.episodes.resolve()}"
    )
    stats_path = args.output.resolve() / "norm_stats.json"
    stats_path.write_text(json.dumps(stats))
    config_path = args.output.resolve() / f"{args.config_name}.yaml"
    _write_config(config_path, data_dir, stats, args.batch_size)
    manifest = {
        "schema": "manimux.xr1_yam_dataset.v1",
        "source": str(args.episodes.resolve()),
        "instruction": args.instruction,
        "episodes": len(episodes),
        "frames": total_frames,
        "action_length": ACTION_LENGTH,
        "annotations": manifest_episodes,
        "stats": {"path": stats_path.name},
        "config": {"path": config_path.name},
    }
    (args.output.resolve() / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(episodes)} episodes / {total_frames} frames to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
