from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

TaskResult = Literal["success", "failure", "invalid"]
ReviewMode = Literal["live", "video"]


def write_manual_evaluation(
    episode_dir: Path,
    *,
    task_result: TaskResult,
    smoothness_score: int,
    failure_tags: list[str],
    operator_note: str,
    reviewer_id: str,
    review_mode: ReviewMode = "live",
) -> Path:
    episode_dir = episode_dir.expanduser().resolve()
    if episode_dir.name.endswith(".partial"):
        raise ValueError("cannot evaluate an incomplete episode")
    if not episode_dir.is_dir():
        raise FileNotFoundError(f"episode directory does not exist: {episode_dir}")
    for required in ("meta.json", "result.json"):
        if not (episode_dir / required).is_file():
            raise ValueError(f"episode is missing {required}: {episode_dir}")
    if task_result not in {"success", "failure", "invalid"}:
        raise ValueError(f"unsupported task result: {task_result!r}")
    if not 1 <= smoothness_score <= 5:
        raise ValueError("smoothness_score must be between 1 and 5")

    evaluation_dir = episode_dir / "evaluation"
    evaluation_dir.mkdir(exist_ok=True)
    target = evaluation_dir / "human-label.json"
    temporary = evaluation_dir / f".human-label-{uuid.uuid4().hex}.tmp"
    payload = {
        "task_result": task_result,
        "smoothness_score": smoothness_score,
        "failure_tags": sorted(set(failure_tags)),
        "operator_note": operator_note.strip(),
        "reviewer_id": reviewer_id.strip() or "operator",
        "review_mode": review_mode,
        "label_schema": "human-label-v1",
        "created_at": datetime.now(UTC).isoformat(),
    }
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
