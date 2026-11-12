"""Offline study defaults for the operator's single layout selector."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def reference_task(task: str, available: tuple[str, ...]) -> str:
    """Match a task prompt to a gallery without guessing semantic synonyms."""
    def normalize(value: str) -> str:
        return re.sub(r"[^\w]+", "_", value.casefold()).strip("_")
    matches = [name for name in available if normalize(name) == normalize(task)]
    return matches[0] if len(matches) == 1 else ""


def completed_repeats(
    root: Path | None, task: str, defaults: dict[str, Any],
) -> dict[tuple[str, str], set[int]]:
    """Read finalized, human-scored trials across sessions of this method's output root.

    Failed tasks count; invalid/skipped/unstarted attempts do not. Reference hashes
    keep a replaced layout from inheriting trials from the previous image.
    """
    result: dict[tuple[str, str], set[int]] = {}
    if root is None or not root.is_dir():
        return result
    for path in root.glob("session-*/rollout-*/evaluation/human-label.json"):
        episode = path.parent.parent
        if episode.name.endswith(".partial") or not (episode / "result.json").is_file():
            continue
        try:
            label = json.loads(path.read_text())
            meta = json.loads((episode / "meta.json").read_text())
        except (OSError, ValueError):
            continue
        if label.get("task_result") not in {"success", "failure"}:
            continue
        if not meta.get("experiment_mode") or not meta.get("identity_frozen") or meta.get("task") != task:
            continue
        if any(meta.get(k, "") != defaults.get(k, "") for k in ("experiment_name", "condition")):
            continue
        repeat = meta.get("repeat_id")
        if type(repeat) is not int or repeat < 1:
            continue
        key = (meta.get("layout_id", ""), (meta.get("reference_layout") or {}).get("sha256", ""))
        result.setdefault(key, set()).add(repeat)
    return result


def next_repeat(used: set[int], limit: int | None) -> int:
    repeat = 1
    while repeat in used:
        repeat += 1
    if limit is not None and repeat > limit:
        raise ValueError(f"All {limit} repeats are scored for this layout. Choose another layout.")
    return repeat


def unscored_experiments(root: Path | None, task: str) -> dict[str, Path]:
    """Find completed, started study trials awaiting a human label in this method directory."""
    choices: dict[str, Path] = {}
    if root is None or not root.is_dir():
        return choices
    for path in sorted(root.glob("session-*/rollout-*/meta.json"), reverse=True):
        episode = path.parent
        if episode.name.endswith(".partial") or (episode / "evaluation/human-label.json").exists():
            continue
        try:
            meta = json.loads(path.read_text())
            result = json.loads((episode / "result.json").read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(meta, dict) or not isinstance(result, dict):
            continue
        if (not meta.get("experiment_mode") or not meta.get("identity_frozen")
                or meta.get("task") != task or result.get("incomplete")
                or not isinstance(result.get("steps"), int) or result["steps"] <= 0):
            continue
        label = (
            f"Layout {meta.get('layout_id') or '—'} · Repeat {meta.get('repeat_id') or '—'}"
            f" · {episode.parent.name}/{episode.name}"
        )
        choices[label] = episode
    return choices
