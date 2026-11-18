"""Generate a small disconnected trial, export it, and check the official reader."""

import argparse
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from manimux.recording.episode import EpisodeRecorder
from manimux.recording.export_lerobot import export_episode
from manimux.types import RobotState, SensorFrame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="new directory for synthetic artifacts")
    root = parser.parse_args().root
    root.mkdir(parents=True, exist_ok=False)
    task = "Pick up the block."
    recorder = EpisodeRecorder(
        root, "synthetic", {"arm": 6}, {"task": task}, video_fps=20, video_queue_size=32
    )
    cameras = ["camera1", "camera2", "camera3"]
    for i in range(12):
        now = 1_000_000_000 + i * 50_000_000
        state = {"arm": np.arange(6) * 0.02 + i * 0.01}
        frames = {}
        for j, camera in enumerate(cameras):
            rgb = np.full((256, 256, 3), [i * 14, 40 + j * 50, 180], dtype=np.uint8)
            frames[camera] = SensorFrame(camera, rgb, now - (j + 1) * 1_000_000, i)
        recorder.record_tick(
            monotonic_ns=now,
            state=RobotState(state, now, i),
            command={"arm": state["arm"] + 0.1},
            scheduled={"arm": state["arm"] + 0.3},
            plan_id=None,
            inference_ms=None,
            frames=frames,
            camera_times_ns={k: v.capture_monotonic_ns for k, v in frames.items()},
        )
    episode = recorder.finish(
        run_completed=True, terminal_reason="fixture", steps=12, wall_time_s=0.6
    )
    output = export_episode(
        episode, root / "v3", group="arm", arm_dofs=5, cameras=cameras, fps=10, max_age_ms=10
    )
    dataset = LeRobotDataset("manimux/episode", root=output, video_backend="pyav")
    assert dataset.meta.info["codebase_version"] == "v3.0"
    assert len(dataset) == 6
    for row, sample in enumerate(dataset):
        i = 2 * row
        expected = np.arange(6) * 0.02 + i * 0.01
        np.testing.assert_allclose(sample["observation.state"], expected, atol=1e-6)
        np.testing.assert_allclose(sample["action"], expected + 0.1, atol=1e-6)
        assert sample["task"] == task
        np.testing.assert_allclose(sample["timestamp"], row / 10, atol=1e-6)
        for j, camera in enumerate(cameras):
            rgb = sample[f"observation.images.{camera}"][:, 128, 128].numpy() * 255
            np.testing.assert_allclose(rgb, [i * 14, 40 + j * 50, 180], atol=12)
    print(f"Verified all {len(dataset)} frames and {len(cameras)} cameras: {output}")


if __name__ == "__main__":
    main()
