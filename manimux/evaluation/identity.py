"""Freeze per-rollout research metadata independently of a particular study."""

from __future__ import annotations

import re
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any


def research_template(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Validate optional study constraints at the configuration boundary."""
    if value is None:
        return None
    template = dict(value)
    unknown = template.keys() - {"name", "layout_ids", "repeats", "require_reference"}
    if unknown:
        raise ValueError(f"Unknown experiment_template fields: {sorted(unknown)}")
    template.setdefault("name", "Custom study")
    template.setdefault("layout_ids", [])
    template.setdefault("repeats", None)
    template.setdefault("require_reference", False)
    if not isinstance(template["name"], str) or not template["name"].strip():
        raise ValueError("experiment_template.name must be a nonempty string")
    ids = template["layout_ids"]
    if not isinstance(ids, list) or any(not isinstance(x, str) or not x.strip() for x in ids):
        raise ValueError("experiment_template.layout_ids must be a list of nonempty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("experiment_template.layout_ids must be unique")
    repeats = template["repeats"]
    if repeats is not None and (type(repeats) is not int or repeats < 1):
        raise ValueError("experiment_template.repeats must be a positive integer")
    if not isinstance(template["require_reference"], bool):
        raise ValueError("experiment_template.require_reference must be boolean")
    return template


def rollout_identity(run: Mapping[str, Any]) -> dict[str, Any]:
    """Snapshot this Prepare's metadata; read no files and construct no devices."""
    mode = run.get("experiment_mode", False)
    if not isinstance(mode, bool):
        raise ValueError("run.experiment_mode must be a boolean")
    identity = {key: run.get(key, "") for key in ("experiment_name", "condition", "notes")}
    if any(not isinstance(value, str) for value in identity.values()):
        raise ValueError("experiment_name, condition and notes must be strings")
    identity.update(experiment_mode=mode, layout_id="", repeat_id=None, reference_layout=None)
    if not mode:
        return identity
    template = research_template(run.get("experiment_template"))
    layout_id = run.get("layout_id", "")
    repeat_id = run.get("repeat_id")
    if not isinstance(layout_id, str):
        raise ValueError("run.layout_id must be a string")
    if repeat_id is not None and (type(repeat_id) is not int or repeat_id < 1):
        raise ValueError("run.repeat_id must be a positive integer or null")
    if template:
        if template["layout_ids"] and layout_id not in template["layout_ids"]:
            raise ValueError("Choose a layout_id from experiment_template.layout_ids")
        if template["repeats"] is not None and (
            repeat_id is None or repeat_id > template["repeats"]
        ):
            raise ValueError("repeat_id is outside experiment_template.repeats")
    reference = run.get("reference_layout")
    if reference is None:
        if template and template["require_reference"]:
            raise ValueError("This experiment template requires a reference image")
    else:
        if not isinstance(reference, Mapping) or not layout_id:
            raise ValueError("reference_layout requires a layout_id and task/path/sha256")
        task, path, digest = (reference.get(key) for key in ("task", "path", "sha256"))
        if not isinstance(task, str) or not re.fullmatch(r"[\w-]{1,100}", task):
            raise ValueError("reference_layout.task must be a reference gallery name")
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError("reference_layout.path must be an absolute image path")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("reference_layout.sha256 must identify the selected image bytes")
        reference = {"task": task, "path": path, "sha256": digest}
    identity.update(layout_id=layout_id, repeat_id=repeat_id, reference_layout=reference)
    if template is not None:
        identity["experiment_template"] = deepcopy(template)
    return identity
