#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class Episode:
    path: Path
    metadata: dict[str, Any]
    boundaries: list[dict[str, Any]]

    @property
    def label(self) -> str:
        return str(
            self.metadata.get("policy_label")
            or self.metadata.get("policy_worker")
            or self.path.parent.name
        )

    @property
    def policy_key(self) -> tuple[str, str, str]:
        return (
            str(self.metadata.get("policy_label", "")),
            str(self.metadata.get("policy_worker", "")),
            str(self.metadata.get("policy_adapter", "")),
        )


def _load_episode(path: Path) -> Episode | None:
    episode_dir = path if path.is_dir() else path.parent
    events_path = episode_dir / "events.jsonl"
    meta_path = episode_dir / "meta.json"
    if not events_path.is_file():
        raise FileNotFoundError(f"missing events.jsonl: {events_path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    boundaries = []
    with events_path.open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("kind") == "plan_boundary":
                boundaries.append(event)
    if not boundaries:
        return None
    return Episode(episode_dir, metadata, boundaries)


def _discover_latest(data_dir: Path, count: int) -> list[Episode]:
    candidates = sorted(
        data_dir.glob("run-*/episode*/events.jsonl"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    selected: list[Episode] = []
    policy_keys: set[tuple[str, str, str]] = set()
    for events_path in candidates:
        episode = _load_episode(events_path)
        if episode is None or episode.policy_key in policy_keys:
            continue
        selected.append(episode)
        policy_keys.add(episode.policy_key)
        if len(selected) == count:
            break
    return list(reversed(selected))


def _flatten(groups: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        [np.asarray(groups[name], dtype=np.float64).reshape(-1) for name in sorted(groups)]
    )


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def _percentiles(values: list[float]) -> str:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return "n/a"
    p50, p95 = np.percentile(finite, [50, 95])
    return f"p50={p50:.5f}, p95={p95:.5f}, max={finite.max():.5f}"


def _negative_fraction(values: list[float]) -> str:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return "n/a"
    return f"{np.mean(finite < 0.0):.1%} ({np.sum(finite < 0.0)}/{finite.size})"


def _positive_fraction(values: list[float]) -> str:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return "n/a"
    return f"{np.mean(finite > 0.0):.1%} ({np.sum(finite > 0.0)}/{finite.size})"


def _summarize(episode: Episode) -> None:
    replan_intervals_s: list[float] = []
    tracking_gaps: list[float] = []
    raw_jumps: list[float] = []
    committed_jumps: list[float] = []
    raw_motion_cosines: list[float] = []
    committed_motion_cosines: list[float] = []
    raw_toward_measured_cosines: list[float] = []
    committed_toward_measured_cosines: list[float] = []
    blend_changes: list[float] = []

    prior_time_ns: int | None = None
    for boundary in episode.boundaries:
        time_ns = int(boundary["monotonic_ns"])
        if prior_time_ns is not None:
            replan_intervals_s.append((time_ns - prior_time_ns) / 1_000_000_000)
        prior_time_ns = time_ns

        previous_command = _flatten(boundary["previous_command"])
        last_command = _flatten(boundary["last_command"])
        measured = _flatten(boundary["measured"])
        raw_first = _flatten(boundary["raw_first"])
        committed_first = _flatten(boundary["committed_first"])

        previous_motion = last_command - previous_command
        toward_measured = measured - last_command
        raw_jump = raw_first - last_command
        committed_jump = committed_first - last_command
        tracking_gaps.append(float(np.max(np.abs(last_command - measured))))
        raw_jumps.append(float(np.max(np.abs(raw_jump))))
        committed_jumps.append(float(np.max(np.abs(committed_jump))))
        blend_changes.append(float(np.max(np.abs(committed_first - raw_first))))
        raw_motion_cosines.append(_cosine(previous_motion, raw_jump))
        committed_motion_cosines.append(_cosine(previous_motion, committed_jump))
        raw_toward_measured_cosines.append(_cosine(toward_measured, raw_jump))
        committed_toward_measured_cosines.append(_cosine(toward_measured, committed_jump))

    metadata = episode.metadata
    print(f"\n=== {episode.label} ===")
    print(f"episode: {episode.path}")
    print(
        "contract: "
        f"worker={metadata.get('policy_worker', 'n/a')}, "
        f"dt={metadata.get('action_dt_s', 'n/a')}s, "
        f"horizon={metadata.get('horizon_steps', 'n/a')}, "
        f"blend={metadata.get('blend_steps', 'n/a')}"
    )
    print(f"plan boundaries: {len(episode.boundaries)}")
    print(f"replan interval [s]: {_percentiles(replan_intervals_s)}")
    print(f"|last command - measured| max joint [rad]: {_percentiles(tracking_gaps)}")
    print(f"|raw first - last command| max joint [rad]: {_percentiles(raw_jumps)}")
    print(f"|committed first - last command| max joint [rad]: {_percentiles(committed_jumps)}")
    print(f"|committed first - raw first| max joint [rad]: {_percentiles(blend_changes)}")
    print(
        "raw first reverses previous command motion: "
        f"{_negative_fraction(raw_motion_cosines)}"
    )
    print(
        "committed first reverses previous command motion: "
        f"{_negative_fraction(committed_motion_cosines)}"
    )
    print(
        "raw first moves toward measured state: "
        f"{_positive_fraction(raw_toward_measured_cosines)}"
    )
    print(
        "committed first moves toward measured state: "
        f"{_positive_fraction(committed_toward_measured_cosines)}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ManiMux plan-boundary motion without starting any service."
    )
    parser.add_argument(
        "--episode",
        action="append",
        type=Path,
        help="episode directory or events.jsonl; repeat to compare explicit episodes",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--latest", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.episode:
        episodes = []
        for path in args.episode:
            episode = _load_episode(path)
            if episode is None:
                raise SystemExit(f"no plan_boundary events: {path}")
            episodes.append(episode)
    else:
        episodes = _discover_latest(args.data_dir, args.latest)
    if not episodes:
        raise SystemExit("no episodes with plan_boundary events found")
    for episode in episodes:
        _summarize(episode)


if __name__ == "__main__":
    main()
