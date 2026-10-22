"""Compact presentation of recorded agent tool choices, without planner internals."""
from __future__ import annotations

import html
import json
import re
from collections.abc import Iterable, Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any


def _text(value: Any, limit: int | None = 360) -> str:
    text = " ".join(str(value).split())
    if limit is not None and len(text) > limit:
        text = text[:limit] + "…"
    return re.sub(r"([\\`*_{}\[\]()#+!|>-])", r"\\\1", html.escape(text, quote=False))


def _validate_decision(decision: Any) -> dict[str, Any]:
    if not isinstance(decision, dict):
        raise ValueError("agent decision must be an object")
    sequence = decision.get("sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
        raise ValueError("agent decision sequence must be a positive integer")
    if not isinstance(decision.get("tool"), str) or not decision["tool"].strip():
        raise ValueError("agent decision tool must be a nonempty string")
    phase = decision.get("phase")
    if not isinstance(phase, str) or phase not in {"selected", "completed", "failed"}:
        raise ValueError("agent decision phase must be selected, completed or failed")
    decision_id = decision.get("decision_id")
    if decision_id is not None and (not isinstance(decision_id, str) or not decision_id):
        raise ValueError("agent decision decision_id must be a nonempty string")
    if "planner" in decision and not isinstance(decision["planner"], str):
        raise ValueError("agent decision planner must be a string")
    for field in ("arguments", "outcome"):
        value = decision.get(field)
        if value is not None and not isinstance(value, dict):
            raise ValueError(f"agent decision {field} must be an object")
    arrival = (decision.get("outcome") or {}).get("arrival")
    if arrival is not None and not isinstance(arrival, dict):
        raise ValueError("agent decision outcome.arrival must be an object")
    return decision


def merge_decisions(
    existing: Iterable[dict[str, Any]], updates: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Merge heartbeat and recorded choices without reverting a terminal outcome."""
    merged: dict[tuple[str, str | int], dict[str, Any]] = {}
    for batch in (existing, updates):
        for decision in batch:
            _validate_decision(decision)
            key = (
                ("id", decision["decision_id"])
                if decision.get("decision_id") else ("sequence", decision["sequence"])
            )
            previous = merged.get(key)
            if previous is not None and previous["phase"] in {"completed", "failed"}:
                # Runtime outcomes are final; late heartbeat selections cannot erase them.
                continue
            merged[key] = deepcopy(decision)
    return sorted(merged.values(), key=lambda decision: decision["sequence"])


def read_decisions(episode_dir: Path) -> list[dict[str, Any]]:
    """Read recorded tool choices, tolerating only an unfinished trailing JSONL write."""
    episode_dir = Path(episode_dir)
    if not episode_dir.exists() and not episode_dir.name.endswith(".partial"):
        episode_dir = episode_dir.with_name(episode_dir.name + ".partial")
    events_path = episode_dir / "events.jsonl"

    def events() -> Iterator[dict[str, Any]]:
        with events_path.open("rb") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    event = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    if not line.endswith(b"\n"):
                        # The writer may currently be appending this last event.
                        break
                    raise ValueError(
                        f"Malformed recording event at {events_path}:{line_number}"
                    ) from exc
                if not isinstance(event, dict):
                    raise ValueError(
                        f"Recording event must be an object at {events_path}:{line_number}"
                    )
                if event.get("kind") != "agent_decision":
                    continue
                try:
                    yield _validate_decision(event)
                except ValueError as exc:
                    raise ValueError(f"{events_path}:{line_number}: {exc}") from exc

    return merge_decisions((), events())


def decision_status(decision: dict[str, Any]) -> str:
    outcome = decision.get("outcome") or {}
    if outcome.get("state") == "cancelled":
        return "Cancelled"
    if decision.get("phase") == "failed":
        return "Failed"
    if decision.get("phase") == "selected":
        return "Selected · awaiting tool result"
    if outcome.get("reason") == "measured_arrival" and (
        outcome.get("arrival") or {}
    ).get("reached"):
        return "Arrived"
    if decision.get("tool") == "execute_policy":
        return "Budget completed · inspect task result"
    if decision.get("tool") == "finish":
        return "Agent finished · " + str(outcome.get("status", "verdict recorded"))
    return "Completed"


def decisions_markdown(decisions: list[dict[str, Any]], *, show_all: bool = False) -> str:
    """Render recent choices or the complete chronological record with uncut text."""
    rows = []
    visible = sorted(decisions, key=lambda decision: decision["sequence"]) if show_all else (
        reversed(decisions[-8:])
    )
    limit = None if show_all else 360
    for decision in visible:
        tool = str(decision.get("tool", "unknown"))
        args = decision.get("arguments") or {}
        outcome = decision.get("outcome") or {}
        row = (f"**{_text(decision.get('sequence', '?'))}. {_text(tool)}** · "
               f"{_text(decision_status(decision))}")
        details = []
        if args.get("group"):
            details.append(f"Group: {_text(args['group'])} · base frame")
        for key, label, unit in (("xyz", "XYZ", "m"), ("delta_xyz", "Δ XYZ", "m"),
                                 ("delta_rpy", "Δ RPY", "rad"),
                                 ("quaternion_wxyz", "wxyz", "")):
            if args.get(key) is not None:
                details.append(f"{label}: {_text(json.dumps(args[key]))} {unit}")
        if "opening" in args:
            details.append(f"Opening: {_text(args['opening'])} (0 closed, 1 open)")
        if tool == "execute_policy":
            details.append(_text(args.get("prompt", ""), limit))
            chunks = args.get('chunks', 1)
            details.append(f"{_text(chunks)} {'chunk' if chunks == 1 else 'chunks'} × "
                           f"{_text(args.get('use_length', 5))} policy rows")
        if tool == "finish":
            details.append(_text(outcome.get("summary") or args.get("summary", ""), limit))
        if outcome.get("error"):
            details.append(_text(outcome["error"], limit))
        if details:
            row += "\n\n" + "  \n".join(filter(None, details))
        rows.append(row)
    planner = _text(decisions[-1].get("planner", "RPent")) if decisions else "RPent"
    return planner + "\n\n" + "\n\n---\n\n".join(rows) if rows else "Waiting for RPent decisions."
