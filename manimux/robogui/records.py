"""Read finalized experiment trajectories without opening UI or hardware."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from .robot_view import RobotView

SOURCES = {
    "state": "Measured state",
    "scheduled": "Scheduled reference",
    "command": "Executor command",
}


def load_episode_trajectory(
    episode: Path, source: str, robot: RobotView,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Read tick samples with their original host times; never resample feedback."""
    if episode.name.endswith(".partial") or not (episode / "result.json").is_file():
        raise ValueError("Choose a finalized episode with result.json")
    if source not in SOURCES:
        raise ValueError("Choose state, scheduled or command")
    if not (episode / "data.zarr").is_dir():
        raise ValueError(f"No recorded trajectory at {episode / 'data.zarr'}")
    ticks = zarr.open_group(str(episode / "data.zarr"), mode="r")["ticks"]
    if source not in ticks or "monotonic_ns" not in ticks:
        raise ValueError(f"Episode has no ticks/{source} or ticks/monotonic_ns")
    actions = robot.validate_groups(
        {name: np.asarray(values) for name, values in ticks[source].arrays()}, sequence=True
    )
    stamps = np.asarray(ticks["monotonic_ns"], dtype=np.int64)
    count = len(next(iter(actions.values())))
    if stamps.shape != (count,) or np.any(np.diff(stamps) <= 0):
        raise ValueError("Recorded timestamps must match samples and strictly increase")
    return actions, stamps
