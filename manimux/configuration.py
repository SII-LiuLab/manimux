"""File-relative experiment composition, without constructing runtime components."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path

import yaml


def safe_id(value, label: str) -> str:
    """Path-safe stable ID shared by collections, tasks, poses and reports."""
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"{label} must be a path-safe nonempty ID")
    return value


def initial_pose_metadata(config: dict) -> dict:
    """Describe the configured start without pretending every start is Home."""
    options = config.get("robot", {}).get("options", {})
    if not options.get("move_to_start_on_connect", False):
        return {"id": "hold", "label": "Keep current pose", "joints": None}
    named = config.get("run", {}).get("initial_pose")
    if named:
        return deepcopy(named)
    targets = options.get("start_joints")
    duration = options.get("start_duration_s")
    digest = hashlib.sha256(json.dumps([targets, duration], sort_keys=True).encode()).hexdigest()[:12]
    return {"id": f"configured-{digest}", "label": "Configured start pose",
            "joints": deepcopy(targets), "duration_s": duration}


def initial_pose_targets(config: dict):
    """Only record a commanded start target when startup motion is enabled."""
    if not config.get("robot", {}).get("options", {}).get("execute", False):
        return None
    return initial_pose_metadata(config).get("joints")


def compose(path: str | Path, stack: tuple[Path, ...] = ()) -> tuple[dict, dict[str, str]]:
    from manimux.cli import _merge

    source = Path(path).expanduser().resolve()
    if source in stack:
        raise ValueError(f"Configuration inheritance cycle: {(*stack, source)}")
    if len(stack) >= 4:
        raise ValueError("Configuration inheritance is limited to four files per chain")
    content = source.read_bytes()
    raw = yaml.safe_load(content)
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration must be a mapping: {source}")
    bases = raw.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    if not isinstance(bases, list) or any(not isinstance(p, str) for p in bases):
        raise ValueError(f"_base_ must be a path or list of paths: {source}")
    result, sources = {}, {}
    for reference in bases:
        base, dependencies = compose(source.parent / reference, (*stack, source))
        result = _merge(result, base)
        sources.update(dependencies)

    def resolve(value):
        if isinstance(value, list):
            return [resolve(v) for v in value]
        if not isinstance(value, dict):
            return value
        return {
            key: str((source.parent / Path(v).expanduser()).resolve())
            if key in {"config", "control_profile", "local"} and isinstance(v, str)
            else resolve(v)
            for key, v in value.items()
        }

    sources[str(source)] = hashlib.sha256(content).hexdigest()
    return _merge(result, resolve(deepcopy(raw))), sources


def _validated_pose(pose_id: str, pose: dict, dimensions: dict, default_duration) -> dict:
    import math

    safe_id(pose_id, "Initial pose ID")
    joints = pose["joints"]
    if set(joints) != set(dimensions):
        raise ValueError("Initial pose must specify every robot group")
    for group, size in dimensions.items():
        q = joints[group]
        if (not isinstance(q, list) or len(q) != size or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in q)):
            raise ValueError(f"Invalid initial pose joints: {group}")
    duration = pose.get("duration_s", default_duration)
    if (isinstance(duration, bool) or not isinstance(duration, (float, int))
            or not math.isfinite(duration) or duration <= 0):
        raise ValueError("Initial pose duration must be finite and positive")
    label = pose.get("label", pose_id)
    if not isinstance(label, str) or not label.strip():
        raise ValueError("Initial pose label must be nonempty")
    return {"id": pose_id, "label": label, "joints": deepcopy(joints),
            "duration_s": duration, "source": deepcopy(pose.get("source"))}


def apply_initial_pose(raw: dict) -> None:
    """Resolve the selected pose and expose validated embodiment presets to the UI.

    Loading and selecting configurations never connects hardware. An explicit task
    preset file takes precedence over the assembly's preset library.
    """
    from manimux.cli import read_yaml

    selection = raw.pop("initial_pose", None)
    # Presets offered in RoboGUI; the active start is always offered. UI only, not a condition.
    choices = raw.get("run", {}).pop("initial_pose_choices", None)
    robot = raw["robot"]
    options = robot.setdefault("options", {})
    path = None
    if selection is not None:
        if not isinstance(selection, dict) or set(selection) != {"config", "id"}:
            raise ValueError("initial_pose requires config and id")
        if not options.get("move_to_start_on_connect"):
            raise ValueError("Initial pose requires move_to_start_on_connect")
        path = Path(selection["config"])
    elif robot.get("config"):
        assembly_path = Path(robot["config"])
        reference = read_yaml(assembly_path).get("initial_poses")
        if reference:
            path = (assembly_path.parent / reference).resolve()
    if path is None:
        return
    presets = read_yaml(path)["poses"]
    poses = [_validated_pose(key, value, robot.get("group_dims", {}),
                             options.get("start_duration_s"))
             for key, value in presets.items()]
    catalog = poses
    if choices is not None:
        unknown = set(choices) - {p["id"] for p in poses}
        if unknown:
            raise ValueError(f"Unknown initial_pose_choices: {sorted(unknown)}")
        poses = [p for p in poses if p["id"] in choices]
    if len({p["label"] for p in poses}) != len(poses):
        raise ValueError("Initial pose labels must be unique")
    run = raw.setdefault("run", {})
    if selection is not None:
        selected = next((p for p in poses if p["id"] == selection["id"]), None)
        if selected is None:
            raise ValueError(f"Unknown initial pose: {selection['id']}")
        options.update(start_joints=deepcopy(selected["joints"]),
                       start_duration_s=selected["duration_s"])
        run["initial_pose"] = deepcopy(selected)
    current = initial_pose_metadata(raw)
    if current["id"] not in {p["id"] for p in poses}:
        # Preserve legacy inline starts or intentional hold as the default choice.
        current.setdefault("duration_s", options.get("start_duration_s"))
        # A configured start identical to a preset takes its name and replaces it in the list.
        same = next((p for p in catalog if p["joints"] == current["joints"]
                     and p["duration_s"] == current["duration_s"]), None)
        if same is not None:
            current["label"] = same["label"]
            poses = [p for p in poses if p is not same]
        poses.insert(0, current)
    run["initial_pose_options"] = poses
    raw.setdefault("config_sources", {})[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()


def select_initial_pose(config: dict, pose_id: str) -> dict:
    """Return a fresh configuration for a validated preset, without hardware I/O."""
    result = deepcopy(config)
    poses = result["run"].get("initial_pose_options", [])
    selected = next((p for p in poses if p["id"] == pose_id), None)
    if selected is None:
        raise ValueError(f"Unknown initial pose: {pose_id}")
    options = result["robot"].setdefault("options", {})
    if pose_id == "hold":
        options["move_to_start_on_connect"] = False
        result["run"].pop("initial_pose", None)
    else:
        selected = _validated_pose(pose_id, selected, result["robot"]["group_dims"],
                                   options.get("start_duration_s"))
        options.update(move_to_start_on_connect=True, start_joints=deepcopy(selected["joints"]),
                       start_duration_s=selected["duration_s"])
        result["run"]["initial_pose"] = selected
    return result
