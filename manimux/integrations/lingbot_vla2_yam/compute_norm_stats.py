"""Compute LingBot-VLA2 normalization statistics from YAM episodes.

LingBot uses absolute 12-D arm joints and two normalized grippers.  These
statistics only define the YAM feature scale; they do not make the public
foundation checkpoint a YAM post-trained policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REQUIRED = (
    "left-joint_pos.npy",
    "left-gripper_pos.npy",
    "right-joint_pos.npy",
    "right-gripper_pos.npy",
    "action-left-joint.npy",
    "action-left-gripper.npy",
    "action-right-joint.npy",
    "action-right-gripper.npy",
)


def usable_episodes(root: Path) -> list[Path]:
    return [
        candidate
        for candidate in sorted(root.glob("*/*/"))
        if (candidate / "write_complete.flag").is_file()
        and all((candidate / name).is_file() for name in REQUIRED)
    ]


def episode_features(episode: Path) -> dict[str, np.ndarray]:
    arrays = {name[:-4]: np.load(episode / name) for name in REQUIRED}
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError(f"episode arrays have mismatched lengths: {episode}")
    return {
        "observation.state.arm.position": np.concatenate(
            [arrays["left-joint_pos"], arrays["right-joint_pos"]], axis=-1
        ),
        "observation.state.effector.position": np.concatenate(
            [arrays["left-gripper_pos"], arrays["right-gripper_pos"]], axis=-1
        ),
        "action.arm.position": np.concatenate(
            [arrays["action-left-joint"], arrays["action-right-joint"]], axis=-1
        ),
        "action.effector.position": np.concatenate(
            [arrays["action-left-gripper"], arrays["action-right-gripper"]], axis=-1
        ),
    }


def summarize(values: np.ndarray) -> dict[str, list[float]]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError(f"normalization input must be finite [N,D], got {values.shape}")
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q02": np.quantile(values, 0.02, axis=0).tolist(),
        "q98": np.quantile(values, 0.98, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def compute_stats(episodes: list[Path]) -> dict[str, object]:
    if not episodes:
        raise ValueError("at least one complete YAM episode is required")
    by_feature: dict[str, list[np.ndarray]] = {}
    for episode in episodes:
        for feature, values in episode_features(episode).items():
            by_feature.setdefault(feature, []).append(values)
    merged = {key: np.concatenate(values) for key, values in by_feature.items()}
    counts = {len(values) for values in merged.values()}
    if len(counts) != 1:
        raise ValueError(f"feature transition counts differ: {sorted(counts)}")
    return {
        "norm_stats": {key: summarize(values) for key, values in merged.items()},
        "count": counts.pop(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes",
        default="/home/ubuntu/yam-abc-reproduce/data/episodes",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.episodes).expanduser().resolve()
    episodes = usable_episodes(root)
    payload = compute_stats(episodes)
    payload["_source"] = (
        f"computed from {len(episodes)} complete YAM episodes under {root} by "
        "manimux.integrations.lingbot_vla2_yam.compute_norm_stats"
    )
    output = Path(args.out).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"episodes={len(episodes)} transitions={payload['count']}")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
