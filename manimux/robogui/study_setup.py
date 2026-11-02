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
    *, history_sessions: tuple[Path, ...] = (),
) -> dict[tuple[str, str], set[int]]:
    """Read finalized, human-scored trials across sessions of this method's output root.

    Failed tasks count; invalid/skipped/unstarted attempts do not. Reference hashes
    keep a replaced layout from inheriting trials from the previous image.
    """
    result: dict[tuple[str, str], set[int]] = {}
    for path in _record_files(root, history_sessions, "evaluation/human-label.json"):
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


def remaining_trials(
    choices: list[dict[str, Any]], task: str, reference_sha: dict[str, str],
) -> list[dict[str, Any]]:
    """Unscored layout/repeat slots for every method of one task in a config list.

    A slot counts as done exactly when Prepare would skip it: the same layout, the
    current reference image and a scored repeat within the template limit.
    """
    rows = []
    for choice in choices:
        study = choice.get("study") or {}
        template = study.get("experiment_template") or {}
        layouts, repeats = template.get("layout_ids") or [], template.get("repeats")
        if choice.get("task") != task or not layouts or not repeats:
            continue
        history = completed_repeats(
            Path(study["output_dir"]), task, study.get("research_defaults", {}),
            history_sessions=tuple(Path(p) for p in study.get("history_sessions", [])),
        )
        missing = [
            (layout, repeat) for layout in layouts for repeat in range(1, repeats + 1)
            if repeat not in history.get((layout, reference_sha.get(layout, "")), set())
        ]
        rows.append({"id": choice["id"], "label": choice["label"],
                     "total": len(layouts) * repeats, "missing": missing})
    return rows


def next_repeat(used: set[int], limit: int | None) -> int:
    repeat = 1
    while repeat in used:
        repeat += 1
    if limit is not None and repeat > limit:
        raise ValueError(f"All {limit} repeats are scored for this layout. Choose another layout.")
    return repeat


def _record_files(root: Path | None, history_sessions, suffix: str):
    sessions = list(root.glob("session-*")) if root is not None and root.is_dir() else []
    sessions.extend(history_sessions)
    seen = set()
    for session in sessions:
        for path in session.glob("rollout-*/" + suffix):
            canonical = path.resolve()
            if canonical not in seen:
                seen.add(canonical)
                yield canonical


def unscored_experiments(root: Path | None, task: str, *, history_sessions: tuple[Path, ...] = ()) -> dict[str, Path]:
    """Find completed, started study trials awaiting a human label in this method directory."""
    choices: dict[str, Path] = {}
    for path in sorted(_record_files(root, history_sessions, "meta.json"), reverse=True):
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
