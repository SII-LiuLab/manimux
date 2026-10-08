"""Explicit task scoring rules shared by experiment configuration and labels."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal


def evaluation_parameters(value: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Normalize a scoring rule without inferring a task from its prompt."""
    if value is None:
        return {"kind": "binary"}
    if not isinstance(value, Mapping):
        raise ValueError("evaluation must be a mapping or null")

    kind = value.get("kind", "binary")
    if kind not in ("binary", "count"):
        raise ValueError("evaluation.kind must be 'binary' or 'count'")
    text_fields: tuple[str, ...] = ("task_id", "rubric_id", "label")
    allowed = {"kind", *text_fields}
    if kind == "count":
        text_fields = (*text_fields, "count_label")
        allowed.update(("count_label", "target_count"))
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
