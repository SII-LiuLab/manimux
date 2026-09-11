#!/usr/bin/env python3
"""Compute Pi05 joint+EE statistics from LeRobot numeric columns, without decoding videos.

Matches YamJointEeInputs followed by DeltaActions: 14D joint/gripper actions plus
12D current-EE-frame translation/rotation-vector targets. All frame anchors are
included, with future indices clamped to each episode's last frame. OpenPI's own
RunningStats and serialization are loaded from the explicitly supplied source.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def action_targets(state, actions, observation_ee, action_ee, anchors, horizon, joint_only=False):
    """Return 14D joint or 26D joint+EEF targets without crossing episodes."""
    indices = np.minimum(anchors[:, None] + np.arange(horizon), len(actions) - 1)
    joints = actions[indices].copy()
    for start in (0, 7):
        joints[..., start : start + 6] -= state[anchors, None, start : start + 6]
    if joint_only:
        return joints.astype(np.float32)
    relative = []
    for offset in (0, 12):
        current = observation_ee[anchors, offset : offset + 12]
        target = action_ee[indices, offset : offset + 12]
        rotation = current[:, 3:].reshape(-1, 3, 3)
        target_rotation = target[..., 3:].reshape(*target.shape[:2], 3, 3)
        position = np.einsum(
            "bij,bhj->bhi", rotation.transpose(0, 2, 1), target[..., :3] - current[:, None, :3]
        )
        relative_rotation = np.einsum(
            "bij,bhjk->bhik", rotation.transpose(0, 2, 1), target_rotation
        )
        rotvec = Rotation.from_matrix(relative_rotation.reshape(-1, 3, 3)).as_rotvec()
        relative.append(np.concatenate([position, rotvec.reshape(*position.shape)], axis=-1))
    return np.concatenate([joints, *relative], axis=-1).astype(np.float32)


def compute(
    dataset: Path, openpi_root: Path, output: Path, horizon: int = 50,
    batch_size: int = 8, joint_only: bool = False,
):
    import pyarrow.parquet as pq

    if horizon < 1 or batch_size < 1:
        raise ValueError("horizon and batch_size must be positive")
    if (output / "norm_stats.json").exists():
        raise FileExistsError(output / "norm_stats.json")
    source = openpi_root / "src/openpi/shared/normalize.py"
    spec = importlib.util.spec_from_file_location("pi05_source_normalize", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    columns = [
        "episode_index",
        "frame_index",
        "observation.state",
        "action",
        "observation.ee_pose",
        "action.ee_pose",
    ]
    if joint_only:
        columns = columns[:4]
    tables = [
        pq.read_table(p, columns=columns).to_pandas()
        for p in sorted((dataset / "data").rglob("*.parquet"))
    ]
    import pandas as pd

    rows = pd.concat(tables, ignore_index=True).sort_values(["episode_index", "frame_index"])
    info = json.loads((dataset / "meta/info.json").read_text())
    assert len(rows) == info["total_frames"]
    stats = {key: module.RunningStats() for key in ("state", "actions")}
    episode_count = 0
    for episode_index, episode in rows.groupby("episode_index", sort=True):
        assert np.array_equal(episode["frame_index"].to_numpy(), np.arange(len(episode)))
        arrays = [
            np.stack(episode[key].to_numpy()).astype(np.float32) for key in columns[2:]
        ]
        state, actions = arrays[:2]
        observation_ee, action_ee = (None, None) if joint_only else arrays[2:]
        assert state.shape == actions.shape == (len(episode), 14)
        if not joint_only:
            assert observation_ee.shape == action_ee.shape == (len(episode), 24)
        for start in range(0, len(episode), batch_size):
            anchors = np.arange(start, min(start + batch_size, len(episode)))
            targets = action_targets(
                state, actions, observation_ee, action_ee, anchors, horizon, joint_only
            )
            if not np.isfinite(targets).all() or not np.isfinite(state[anchors]).all():
                raise ValueError(f"Non-finite values in episode {episode_index}")
            stats["state"].update(state[anchors])
            stats["actions"].update(targets)
        episode_count += 1
        print(f"Stats episode {episode_count}: {len(episode)} anchors", flush=True)
    assert episode_count == info["total_episodes"]
    module.save(output, {key: value.get_statistics() for key, value in stats.items()})
    report = {
        "dataset": str(dataset),
        "episodes": episode_count,
        "frames": len(rows),
        "horizon": horizon,
        "state_dim": 14,
        "action_dim": 14 if joint_only else 26,
        "joint_only": joint_only,
        "state_samples": len(rows),
        "action_samples": len(rows) * horizon,
        "tail": "repeat episode last action",
        "anchors": "all frames; includes partial final batches",
        "normalizer_source": str(source),
    }
    (output / "stats_provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--openpi-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--joint-only", action="store_true", help="Compute 14D joint/gripper targets only")
    args = parser.parse_args()
    compute(args.dataset, args.openpi_root, args.output_dir, args.horizon, args.batch_size, args.joint_only)
