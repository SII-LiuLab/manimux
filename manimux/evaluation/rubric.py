"""Explicit task scoring rules shared by experiment configuration and labels."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal


def evaluation_parameters(value: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Normalize a scoring rule without inferring a task from its prompt."""
    if value is None:
        return {"kind": "binary"}
    if not isinstance(value, Mapping):
        raise ValueError("evaluation must be a mapping or null")

    kind = value.get("kind", "binary")
    if kind not in ("binary", "count", "stages"):
        raise ValueError("evaluation.kind must be 'binary', 'count' or 'stages'")
    text_fields: tuple[str, ...] = ("task_id", "rubric_id", "label")
    allowed = {"kind", *text_fields}
    if kind == "count":
        text_fields = (*text_fields, "count_label")
        allowed.update(("count_label", "target_count"))
    if kind == "stages":
        allowed.add("options")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown evaluation fields: {', '.join(sorted(map(str, unknown)))}")
    if (
        kind == "binary"
        and any(key in value for key in text_fields)
        and not all(key in value for key in text_fields)
    ):
        raise ValueError("named binary evaluation requires task_id, rubric_id, and label")

    profile: dict[str, Any] = {"kind": kind}
    for key in text_fields:
        if kind == "binary" and key not in value:
            continue
        item = value.get(key)
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"evaluation.{key} must be a non-empty string")
        profile[key] = item.strip()
    if kind == "count":
        target_count = value.get("target_count")
        if type(target_count) is not int or target_count <= 0:
            raise ValueError("evaluation.target_count must be a positive integer")
        profile["target_count"] = target_count
    if kind == "stages":
        options = value.get("options")
        if not isinstance(options, list) or len(options) < 2:
            raise ValueError("stages evaluation requires at least two options")
        normalized, ids, labels = [], set(), set()
        for option in options:
            if (
                not isinstance(option, Mapping)
                or set(option) != {"id", "label", "score", "success"}
            ):
                raise ValueError("each stage requires id, label, score and success")
            item = dict(option)
            for key, seen in (("id", ids), ("label", labels)):
                text = item[key]
                if not isinstance(text, str) or not text.strip() or text.strip() in seen:
                    raise ValueError(f"stage {key} must be nonempty and unique")
                item[key] = text.strip()
                seen.add(item[key])
            score = item["score"]
            if (type(score) not in (int, float) or not math.isfinite(score)
                    or not 0 <= score <= 1):
                raise ValueError("stage score must be finite and between 0 and 1")
            if type(item["success"]) is not bool:
                raise ValueError("stage success must be boolean")
            if item["success"] != (score == 1):
                raise ValueError("stage success must be true exactly for score 1")
            item["score"] = float(score)
            normalized.append(item)
        if not any(o["success"] for o in normalized):
            raise ValueError("stages evaluation requires a complete-success option")
        profile["options"] = normalized
    return profile


def count_result(
    profile: Mapping[str, Any], completed_count: int
) -> Literal["success", "failure"]:
    """Validate a count and derive complete success from the configured target."""
    normalized = evaluation_parameters(profile)
    if normalized["kind"] != "count":
        raise ValueError("completed_count requires a count evaluation")
    target_count = normalized["target_count"]
    if type(completed_count) is not int or not 0 <= completed_count <= target_count:
        raise ValueError(f"completed_count must be an integer from 0 through {target_count}")
    return "success" if completed_count == target_count else "failure"


def stage_option(profile: Mapping[str, Any], stage_id: str) -> dict[str, Any]:
    """Resolve one total-score option from the recorded rubric, never sum stages."""
    normalized = evaluation_parameters(profile)
    if normalized["kind"] != "stages":
        raise ValueError("stage_id requires a stages evaluation")
    for option in normalized["options"]:
        if option["id"] == stage_id:
            return option
    raise ValueError(f"unknown stage_id: {stage_id!r}")
