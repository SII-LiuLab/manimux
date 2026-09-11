#!/usr/bin/env python3
"""Read back a prepared LeRobot dataset and verify it against its raw YAM source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def validate(raw: Path, dataset: Path, repo_id: str, output: Path) -> dict:
    import av
    import pyarrow.parquet as pq
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    episodes = sorted(p for p in raw.iterdir() if (p / "write_complete.flag").is_file())
    info = json.loads((dataset / "meta/info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert info["total_episodes"] == len(episodes)
    arrays = [pq.read_table(p).to_pandas() for p in sorted((dataset / "data").rglob("*.parquet"))]
    import pandas as pd

    rows = pd.concat(arrays, ignore_index=True).sort_values(["episode_index", "frame_index"])
    assert len(rows) == info["total_frames"]
    metadata_rows = sum(
        pq.ParquetFile(p).metadata.num_rows for p in (dataset / "meta/episodes").rglob("*.parquet")
    )
    assert metadata_rows == len(episodes), "Episode parquet writer not fully finalized"
    assert info["fps"] == 30
    decoded_video_frames = {}
    for role in ("left", "top", "right"):
        count = 0
        for video in sorted((dataset / "videos" / f"observation.images.{role}_rgb").rglob("*.mp4")):
            with av.open(str(video)) as container:
                count += sum(1 for _ in container.decode(video=0))
        assert count == len(rows), (role, "video frame loss", count, len(rows))
        decoded_video_frames[role] = count
        print(f"Full video decode: {role} {count} frames", flush=True)
    sample_indices = []
    for index, ep in enumerate(episodes):
        episode_rows = rows[rows["episode_index"] == index]
        metadata = json.loads((ep / "metadata.json").read_text())
        assert len(episode_rows) == metadata["num_frames"]
        assert np.array_equal(episode_rows["frame_index"], np.arange(len(episode_rows)))
        fields = {
            "observation.state": [
                f"{a}-{k}" for a in ("left", "right") for k in ("joint_pos", "gripper_pos")
            ],
            "action": [f"action-{a}-{k}" for a in ("left", "right") for k in ("joint", "gripper")],
            "observation.ee_pose": [
                f"{a}-{k}" for a in ("left", "right") for k in ("ee_pos", "ee_rotm")
            ],
            "action.ee_pose": [
                f"action-{a}-{k}" for a in ("left", "right") for k in ("ee_pos", "ee_rotm")
            ],
        }
        for key, sources in fields.items():
            expected = np.concatenate(
                [np.load(ep / f"{name}.npy") for name in sources], axis=1
            ).astype(np.float32)
            actual = np.stack(episode_rows[key].to_numpy())
            np.testing.assert_array_equal(actual, expected)
            assert np.isfinite(actual).all()
        sample_indices.extend(
            [int(episode_rows.iloc[0]["index"]), int(episode_rows.iloc[-1]["index"])]
        )
    loader = LeRobotDataset(
        repo_id=repo_id,
        root=dataset,
        video_backend="pyav",
        delta_timestamps={key: [i / 30 for i in range(50)] for key in ("action", "action.ee_pose")},
    )
    cameras = [f"observation.images.{role}_rgb" for role in ("left", "top", "right")]
    for count, index in enumerate(sample_indices):
        frame = loader[index]
        assert tuple(frame["observation.state"].shape) == (14,)
        assert tuple(frame["action"].shape) == (50, 14)
        assert tuple(frame["observation.ee_pose"].shape) == (24,)
        assert tuple(frame["action.ee_pose"].shape) == (50, 24)
        for key in cameras:
            image = frame[key].numpy()
            assert image.shape == (3, 480, 640)
            assert np.isfinite(image).all() and 0 <= image.min() <= image.max() <= 1
        if count % 10 == 0:
            print(f"Readback {count + 1}/{len(sample_indices)}", flush=True)
    result = {
        "raw": str(raw),
        "dataset": str(dataset),
        "episodes": len(episodes),
        "frames": len(rows),
        "numeric_comparison": "all rows exact float32 match",
        "lerobot_sample_reads": len(sample_indices),
        "decoded_camera_samples": 3 * len(sample_indices),
        "action_horizon": 50,
        "state_dim": 14,
        "joint_action_dim": 14,
        "ee_pose_dim": 24,
        "episode_metadata_rows": metadata_rows,
        "video_frame_counts": decoded_video_frames,
        "status": "passed",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    validate(args.raw, args.dataset, args.repo_id, args.output)
