from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from manimux.evaluation.rubric import count_result, evaluation_parameters

TaskResult = Literal["success", "failure", "invalid"]
ReviewMode = Literal["live", "video"]


def write_manual_evaluation(
    episode_dir: Path,
    *,
    task_result: TaskResult | None = None,
    failure_tags: list[str],
    operator_note: str,
    reviewer_id: str,
    review_mode: ReviewMode = "live",
    completed_count: int | None = None,
) -> Path:
    """Save a v2 label, deriving optional count fields from recorded evaluation rules."""
    episode_dir = episode_dir.expanduser().resolve()
    if episode_dir.name.endswith(".partial"):
        raise ValueError("cannot evaluate an incomplete episode")
    if not episode_dir.is_dir():
        raise FileNotFoundError(f"episode directory does not exist: {episode_dir}")
    for required in ("meta.json", "result.json"):
        if not (episode_dir / required).is_file():
            raise ValueError(f"episode is missing {required}: {episode_dir}")
    if task_result not in (None, "success", "failure", "invalid"):
        raise ValueError(f"unsupported task result: {task_result!r}")

    metadata = json.loads((episode_dir / "meta.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping):
        raise ValueError("episode meta.json must contain an object")
    profile = evaluation_parameters(metadata.get("evaluation"))
    count_fields: dict[str, Any] = {}
    if profile["kind"] == "count":
        count_fields = {
            "completed_count": completed_count,
            "target_count": profile["target_count"],
        }
        if completed_count is None:
            if task_result != "invalid":
                raise ValueError("count evaluation requires completed_count")
        else:
            derived_result = count_result(profile, completed_count)
            if task_result != "invalid":
                if task_result is not None and task_result != derived_result:
                    raise ValueError("task_result does not match completed_count and target_count")
                task_result = derived_result
                count_fields["score"] = completed_count / profile["target_count"]
    else:
        if completed_count is not None:
            raise ValueError("completed_count requires a count evaluation")
        if task_result is None:
            raise ValueError("binary evaluation requires task_result")

    evaluation_dir = episode_dir / "evaluation"
    evaluation_dir.mkdir(exist_ok=True)
    target = evaluation_dir / "human-label.json"
    temporary = evaluation_dir / f".human-label-{uuid.uuid4().hex}.tmp"
    payload = {
        "task_result": task_result,
        "failure_tags": sorted(set(failure_tags)),
        "operator_note": operator_note.strip(),
        "reviewer_id": reviewer_id.strip() or "operator",
        "review_mode": review_mode,
        "label_schema": "human-label-v2",
        "evaluation": profile,
        **count_fields,
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
