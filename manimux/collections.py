"""Collection-owned experiment identity, storage and service selection.

Loading is read-only. A collection never overrides a variant's policy parameters.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from copy import deepcopy
from datetime import UTC
from pathlib import Path

from manimux.configuration import initial_pose_metadata, safe_id
from manimux.experiment_list import ExperimentChoice, ExperimentList, shared_service_contract


def _has_rollout_records(session: Path) -> bool:
    # A session manifest is written before Prepare. Empty sessions from setup
    # failures contain no trials to protect; any rollout, including a partial,
    # retains the full condition-compatibility check.
    return any(path.is_dir() for path in session.glob("rollout-*"))


def _real_paths(value):
    """Identify installed files by their real path, not by how a path is spelled.

    A repository move (e.g. XPolicyLab -> third_party/XPolicyLab, kept reachable by a
    symlink) or a checkpoint symlink does not change scientific conditions.
    """
    if isinstance(value, dict):
        return {key: _real_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_real_paths(item) for item in value]
    if isinstance(value, str) and value.startswith("/") and os.path.exists(value):
        return os.path.realpath(value)
    return value


def fingerprint(config: dict) -> str:
    """Effective scientific conditions; exclude storage, provenance and UI cosmetics."""
    from manimux.evaluation.rubric import explicit_evaluation
    from manimux.runtime import inference_parameters

    fields = {
        key: deepcopy(config.get(key))
        for key in (
            "robot",
            "policy",
            "sensors",
            "executor",
            "execution",
            "camera_server",
            "policy_server",
            "evaluation",
            "recording",
        )
    }
    # Display-only cold-start wait does not change rollout inference deadlines,
    # actions or scoring. Preserve its value in manifests, not scientific identity.
    if isinstance(fields.get("policy"), dict):
        fields["policy"].get("options", {}).pop("warmup_request_timeout_s", None)
    # Xiaomi's mask is consumed only by RTC guidance. Enabling it on a shared
    # server must not hide completed ordinary/PAINT trials from the same study.
    if ((fields.get("policy_server") or {}).get("policy_name") == "Xiaomi_Robotics_1"
            and config.get("inference", {}).get("algorithm") in {"serial", "async", "paint"}):
        fields["policy_server"].pop("rtc_mask_padding", None)
        fields.get("policy", {}).get("expected_backend", {}).get("model", {}).pop(
            "rtc_mask_padding", None
        )
    fields["inference"] = inference_parameters(
        executor=config.get("executor", {}), **config.get("inference", {})
    )
    run = config.get("run", {})
    prompt = run.get("task_prompt", run.get("task"))
    fields["evaluation"] = explicit_evaluation(
        config.get("evaluation"), task_id=prompt or "task"
    )
    fields["run"] = {
        key: config.get("run", {}).get(key)
        for key in (
            "max_control_steps",
            "warmup_before_start",
            "sensor_reading",
            "experiment_name",
            "condition",
        )
    }
    # Keep the scientific identity stable across the prompt field rename.
    fields["run"]["task"] = prompt
    debug_age = (config.get("agent") or {}).get("debug_max_plan_age_s")
    if debug_age is not None:
        # Preserve every historical fingerprint when this opt-in is absent.
        fields["agent"] = {"debug_max_plan_age_s": float(debug_age)}
    if (config.get("agent") or {}).get("policy_enabled") is False:
        fields.setdefault("agent", {})["policy_enabled"] = False
    # Normalization of legacy, pre-defaulted manifests is done by their loader.
    fields = _real_paths(json.loads(json.dumps(fields, default=str)))
    return hashlib.sha256(json.dumps(fields, sort_keys=True, default=str).encode()).hexdigest()


def service_contract(config: dict) -> dict:
    contract = shared_service_contract(config)
    robot = deepcopy(contract["robot"])
    for key in ("start_joints", "start_duration_s", "move_to_start_on_connect"):
        robot.get("options", {}).pop(key, None)
    contract["robot"] = robot
    return contract


def pose_configuration(base: dict, pose_id: str) -> dict:
    """Keep the default destination; isolate recordings of an operator-selected pose."""
    from manimux.configuration import select_initial_pose
    from manimux.cli import prepare_experiment

    if pose_id == initial_pose_metadata(base)["id"]:
        return deepcopy(base)
    config = select_initial_pose(base, pose_id)
    root = Path(base["run"]["output_dir"])
    config["run"].update(output_dir=root.with_name(root.name + "__pose_" + safe_id(pose_id, "pose id")),
                         history_sessions=[])
    if "collection" in config:
        variant = base["collection"]["variant"] + "__pose_" + pose_id
        config["experiment"]["variant"] = variant
        config["collection"].update(variant=variant, conditions_sha256=fingerprint(config),
                                     history_rejected=[])
        roots = [Path(path).with_name(Path(path).name + "__pose_" + pose_id)
                 for path in base["collection"].get("history_roots", [])]
        sessions, rejected = checked_history(config, roots)
        config["run"]["history_sessions"] = sessions
        config["collection"]["history_rejected"] = rejected
    for manifest in config["run"]["output_dir"].glob("session-*/session-manifest.json"):
        if not _has_rollout_records(manifest.parent):
            continue
        saved = json.loads(manifest.read_text())
        if not compatible_history(prepare_experiment(**saved["config"]), config):
            raise ValueError(f"Conditions changed in selected-pose records: {manifest.parent}")
    return config


def compatible_history(saved: dict, current: dict) -> bool:
    """Retain progress for a stop-budget change or versioned rubric extension.

    Session fingerprints still include the stop budget. Continuing a study does
    not make different stopping rules or rubrics equivalent for analysis.
    """
    from manimux.evaluation.rubric import explicit_evaluation

    saved = deepcopy(saved)
    # Sessions before the shared IK policy stored it as policy.adapter.ik_failure_policy.
    legacy = saved.get("policy", {}).get("adapter", {}).pop("ik_failure_policy", None)
    if legacy is not None:
        saved.setdefault("execution", {}).setdefault("ik", {})["failure_policy"] = legacy
    if fingerprint(saved) == fingerprint(current):
        return True
    saved.setdefault("run", {})["max_control_steps"] = current.get("run", {}).get("max_control_steps")
    if fingerprint(saved) == fingerprint(current):
        return True
    def profile(config):
        run = config.get("run", {})
        return explicit_evaluation(
            config.get("evaluation"), task_id=run.get("task_prompt", run.get("task", "task"))
        )
    old, new = profile(saved), profile(current)
    if (old["task_id"] != new["task_id"] or old["label"] != new["label"]
            or old["rubric_id"] == new["rubric_id"]
            or len(new["options"]) <= len(old["options"])
            or any(option not in new["options"] for option in old["options"])):
        return False
    comparable = {**saved, "evaluation": current["evaluation"]}
    return fingerprint(comparable) == fingerprint(current)


def checked_history(config: dict, roots: list[Path]) -> tuple[list[str], list[dict]]:
    """Import individual sessions only when saved effective conditions match."""
    from manimux.cli import prepare_experiment

    accepted, rejected, seen = [], [], set()
    for root in roots:
        if not root.is_dir():
            rejected.append({"path": str(root), "reason": "history directory missing"})
            continue
        for path in sorted(root.glob("session-*/session-manifest.json")):
            session = path.parent.resolve()
            if session in seen:
                continue
            seen.add(session)
            try:
                saved = json.loads(path.read_text())["config"]
                saved = prepare_experiment(**saved)
                if not compatible_history(saved, config):
                    raise ValueError("saved experiment conditions differ")
            except (OSError, ValueError, KeyError, TypeError) as error:
                rejected.append({"path": str(session), "reason": str(error)})
            else:
                accepted.append(str(session))
    return accepted, rejected


def _read_collection(path: str | Path, local: Path | None):
    """Read collection metadata without importing runtime/recording dependencies."""
    from manimux.cli import load_local, read_yaml, resolve_local_path

    source = Path(path).expanduser().resolve()
    raw = read_yaml(source)
    if not isinstance(raw, dict) or set(raw) - {"id", "recording", "experiments", "local"}:
        raise ValueError("Collection accepts id, recording, experiments and optional local")
    safe_id(raw.get("id"), "collection id")
    recording = raw.get("recording", {})
    if (
        not isinstance(recording, dict)
        or set(recording) != {"root"}
        or not isinstance(recording["root"], str)
        or not recording["root"].strip()
    ):
        raise ValueError("Collection recording.root is required")
    root = (source.parent / Path(recording["root"]).expanduser()).resolve()
    station = resolve_local_path(source, local)
    if "output_dir" in load_local(station).get("paths", {}):
        raise ValueError("Remove station paths.output_dir: collection recording.root owns storage")
    entries = raw.get("experiments")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Collection experiments must be a nonempty list")
    ids = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) - {
            "id",
            "config",
            "label",
            "history",
            "trials",
        }:
            raise ValueError("Collection entries accept id, config, label, history and trials")
        choice_id = safe_id(entry.get("id"), "experiment id")
        if choice_id in ids:
            raise ValueError(f"Duplicate experiment id: {choice_id}")
        ids.add(choice_id)
        if not isinstance(entry.get("config"), str) or not entry["config"].strip():
            raise ValueError("Collection entry config must be a nonempty path")
    return source, raw, root, station


def load_collection(
    path: str | Path, *, local: Path | None = None, experiment: str | None = None
) -> ExperimentList:
    from manimux.cli import _load_config

    source, raw, root, station = _read_collection(path, local)
    collection_id, entries = raw["id"], raw["experiments"]
    choices, outputs = [], set()
    for entry in entries:
        choice_id = entry["id"]
        config_path = (source.parent / entry["config"]).resolve()
        config = _load_config(config_path, local=station)
        from manimux.configuration import compose

        if "output_dir" in compose(config_path)[0].get("run", {}):
            raise ValueError(f"{config_path}: remove run.output_dir; collection owns storage")
        identity = config.get("experiment", {})
        if not isinstance(identity, dict) or set(identity) != {"task", "variant"}:
            raise ValueError(f"{config_path}: experiment requires task and variant IDs")
        task = safe_id(identity["task"], "task id")
        variant = safe_id(identity["variant"], "variant id")
        output = root / collection_id / task / variant
        if output in outputs:
            raise ValueError(f"Duplicate experiment destination: {output}")
        outputs.add(output)
        if not config["robogui"]["enabled"]:
            raise ValueError("Collection experiments require robogui.enabled")
        trials = entry.get("trials", {})
        if not isinstance(trials, dict) or set(trials) - {"layout_ids", "repeats"}:
            raise ValueError("trials supports layout_ids and repeats only")
        if trials:
            from manimux.evaluation.identity import research_template

            template = deepcopy(config["run"].get("experiment_template") or {})
            available = template.get("layout_ids", [])
            if "layout_ids" in trials and not set(trials["layout_ids"]).issubset(available):
                raise ValueError("Trial layouts must be a subset of the task layouts")
            template.update(trials)
            config["run"]["experiment_template"] = research_template(template)
        condition_hash = fingerprint(config)
        for manifest in output.glob("session-*/session-manifest.json"):
            if not _has_rollout_records(manifest.parent):
                continue
            saved = json.loads(manifest.read_text())
            if saved.get("collection", {}).get("conditions_sha256") != condition_hash:
                # Older manifests hashed count/binary shorthand. Compare the
                # saved effective configuration, never relabel its recordings.
                from manimux.cli import prepare_experiment
                prior = saved.get("config")
                if not isinstance(prior, dict) or not compatible_history(prepare_experiment(**prior), config):
                    raise ValueError(
                        f"Conditions changed for {task}/{variant}; choose a new variant ID"
                    )
        history = entry.get("history", [])
        if not isinstance(history, list) or any(not isinstance(p, str) for p in history):
            raise ValueError("history must be a list of paths relative to recording.root")
        sessions, rejected = checked_history(config, [(root / p).resolve() for p in history])
        config["run"].update(output_dir=output, history_sessions=sessions)
        config["collection"] = {
            "id": collection_id,
            "path": str(source),
            "root": str(root),
            "experiment_id": choice_id,
            "task": task,
            "variant": variant,
            "conditions_sha256": condition_hash,
            "history_rejected": rejected,
            "history_roots": [str((root / p).resolve()) for p in history],
        }
        label = entry.get("label", config["robogui"]["policy_label"] or choice_id)
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Experiment label must be nonempty")
        pose_id = initial_pose_metadata(config)["id"]
        if any(
            c.config["run"]["task_prompt"] == config["run"]["task_prompt"]
            and c.label == label
            and initial_pose_metadata(c.config)["id"] == pose_id
            for c in choices
        ):
            raise ValueError(
                f"Ambiguous UI label for task/pose: {label}; give variants different labels"
            )
        choices.append(
            ExperimentChoice(
                choice_id,
                label,
                config_path,
                config,
                hashlib.sha256(config_path.read_bytes()).hexdigest(),
            )
        )
    catalog = ExperimentList(source, choices, hashlib.sha256(source.read_bytes()).hexdigest())
    if experiment is not None:
        if experiment not in catalog.choices:
            raise ValueError(
                f"Unknown experiment {experiment}; choose from {list(catalog.choices)}"
            )
        catalog.default_id = experiment
    return catalog


def select_service(
    path: Path, service: str, *, local: Path | None = None, experiment: str | None = None
) -> dict:
    # Model environments deliberately do not install robot/recording dependencies.
    # Resolve recipes here; full runtime/identity/history validation belongs to inspect/serve.
    from manimux.cli import read_experiment

    source, raw, _, station = _read_collection(path, local)
    entries = raw["experiments"]
    if experiment is not None:
        entries = [entry for entry in entries if entry["id"] == experiment]
        if not entries:
            raise ValueError(f"Unknown experiment {experiment}")
    chosen = None
    reference = None
    for entry in entries:
        config = read_experiment(source.parent / entry["config"], local=station)
        current = config.get(service)
        if not isinstance(current, dict) or not current:
            raise ValueError(f"{entry['id']} has no {service} recipe")
        if service == "camera_server":
            from manimux.servers.camera.server import camera_config

            current = (current, camera_config(config))
        if chosen is None:
            chosen, reference = config, current
        elif current != reference:
            raise ValueError(
                f"Collection has different {service} configurations; select --experiment-id"
            )
    return deepcopy(chosen)


def inventory(root: Path) -> list[dict]:
    """Visit real session directories once; never descend into videos or zarr arrays."""
    sessions, seen = [], set()
    if not root.exists():
        return sessions
    for directory, children, _ in os.walk(root, followlinks=False):
        descend = []
        for name in sorted(children):
            path = Path(directory) / name
            if path.is_symlink():
                continue
            if not name.startswith("session-"):
                if not name.startswith("rollout-") and name not in {"data.zarr", "_archive"}:
                    descend.append(name)
                continue
            canonical = path.resolve()
            if canonical in seen:
                continue
            seen.add(canonical)
            entry = {"session_id": name, "path": str(canonical), "rollouts": []}
            try:
                saved = json.loads((path / "session-manifest.json").read_text())
                entry["collection"] = saved.get("collection")
                run = saved["config"]["run"]
                entry["task"] = run.get("task_prompt", run.get("task"))
                entry["inference"] = saved["config"].get("inference")
            except (OSError, ValueError, KeyError, TypeError) as error:
                entry["manifest_error"] = str(error)
            for episode in sorted(path.glob("rollout-*")):
                if not episode.is_dir() or episode.is_symlink():
                    continue
                row = {
                    "id": episode.name,
                    "path": str(episode.resolve()),
                    "partial": episode.name.endswith(".partial"),
                    "finalized": (episode / "result.json").is_file(),
                }
                for field, filename in [
                    ("metadata", "meta.json"),
                    ("evaluation", "evaluation/human-label.json"),
                ]:
                    try:
                        data = json.loads((episode / filename).read_text())
                        row[field] = (
                            {
                                key: data.get(key)
                                for key in (
                                    "task",
                                    "experiment_mode",
                                    "identity_frozen",
                                    "layout_id",
                                    "repeat_id",
                                    "experiment_name",
                                    "condition",
                                    "reference_layout",
                                    "collection",
                                )
                            }
                            if field == "metadata"
                            else data
                        )
                    except FileNotFoundError:
                        row[field] = None
                    except (OSError, ValueError) as error:
                        row[field + "_error"] = str(error)
                entry["rollouts"].append(row)
            sessions.append(entry)
        children[:] = descend
    return sorted(sessions, key=lambda item: item["path"])


def write_inventory(root: Path) -> Path:
    """Atomic derived index; raw records and saved manifests are never rewritten."""
    from datetime import datetime

    payload = {
        "schema_version": 1,
        "root": str(root.resolve()),
        "updated_at": datetime.now(UTC).isoformat(),
        "sessions": inventory(root),
    }
    root.mkdir(parents=True, exist_ok=True)
    output = root / "recordings-index-v2.json"
    with tempfile.NamedTemporaryFile(mode="w", dir=root, delete=False, encoding="utf-8") as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    try:
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output
