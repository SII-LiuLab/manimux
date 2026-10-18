"""Validated experiments sharing a station, with independently served policies."""

from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from manimux.policy_service import PolicyServiceSpec, read_policy_services, service_address


@dataclass(frozen=True)
class ExperimentChoice:
    id: str
    label: str
    path: Path
    config: dict
    sha256: str
    policy: str = "default"

    def metadata(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "config_path": str(self.path),
            "task": self.config["run"]["task"],
            "method": self.config["inference"]["algorithm"],
            "policy": self.policy,
            "initial_pose": deepcopy(self.config["run"].get(
                "initial_pose", {"id": "default", "label": "Default pose"},
            )),
        }


def shared_service_contract(config: dict) -> dict:
    """Settings shared by all methods within one policy group.

    Model initialization arguments remain fixed, including sampler defaults and
    normalization. Task instructions belong to run.task, not model identity.
    Runtime strategies/executors are rebuilt by Prepare and may differ.
    """
    policy = config["policy"]
    return {
        "policy_server": config.get("policy_server"),
        "camera_server": config.get("camera_server"),
        "robot": config["robot"],
        "sensors": config["sensors"],
        "robogui": config["robogui"]["robot"],
        "policy": {
            key: policy.get(key)
            for key in (
                "worker",
                "service",
                "adapter",
                "options",
                "expected_backend",
                "action_dt_s",
                "horizon_policy_steps",
            )
        },
    }


class ExperimentList:
    def __init__(
        self,
        path: Path,
        choices: list[ExperimentChoice],
        digest: str,
        policy_services: dict[str, PolicyServiceSpec] | None = None,
    ):
        self.path = path
        self.choices = {choice.id: choice for choice in choices}
        self.sha256 = digest
        self.default_id = choices[0].id
        self.policy_services = policy_services or {}

    def metadata(self) -> list[dict]:
        return [choice.metadata() for choice in self.choices.values()]

    def selection(self, choice_id: str) -> dict:
        return {
            "list_path": str(self.path),
            "list_sha256": self.sha256,
            "config_sha256": self.choices[choice_id].sha256,
            **self.choices[choice_id].metadata(),
        }


def _pose_choices(choice: ExperimentChoice, path: Path) -> list[ExperimentChoice]:
    """Expand trusted pose presets without copying experiment YAMLs or moving hardware."""
    from manimux.cli import read_yaml

    raw = read_yaml(path)
    if set(raw) != {"poses"} or not isinstance(raw["poses"], dict):
        raise ValueError("Initial pose file requires a poses mapping")
    options = choice.config["robot"]["options"]
    if not options.get("move_to_start_on_connect") or not isinstance(options.get("start_joints"), dict):
        raise ValueError("Initial pose presets require configured move_to_start_on_connect/start_joints")
    dimensions = choice.config["robot"]["group_dims"]
    result = [choice]
    labels = {"Default pose"}
    for pose_id, pose in raw["poses"].items():
        if not isinstance(pose_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", pose_id) or pose_id == "default":
            raise ValueError("Initial pose ID must be a non-default path-safe name")
        if not isinstance(pose, dict) or set(pose) - {"label", "joints", "duration_s", "source"}:
            raise ValueError(f"Invalid initial pose {pose_id}")
        label = pose.get("label", pose_id)
        if not isinstance(label, str) or not label.strip() or label in labels:
            raise ValueError(f"Duplicate or empty initial pose label: {label}")
        labels.add(label)
        joints = pose.get("joints")
        if not isinstance(joints, dict) or set(joints) != set(dimensions):
            raise ValueError(f"Initial pose {pose_id} must specify every robot group")
        for group, size in dimensions.items():
            values = joints[group]
            if not isinstance(values, list) or len(values) != size or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values
            ):
                raise ValueError(f"Invalid initial pose coordinates for {group}")
        duration = pose.get("duration_s", options.get("start_duration_s"))
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            raise ValueError("Initial pose duration must be finite and positive")
        digest = hashlib.sha256(json.dumps({"joints": joints, "duration_s": duration}, sort_keys=True).encode()).hexdigest()
        config = deepcopy(choice.config)
        config["robot"]["options"].update(start_joints=deepcopy(joints), start_duration_s=duration)
        config["run"]["initial_pose"] = {"id": pose_id, "label": label, "joints": deepcopy(joints),
            "duration_s": duration, "sha256": digest, "source": deepcopy(pose.get("source"))}
        output = Path(config["run"]["output_dir"]) / "initial_pose" / pose_id
        for manifest in output.glob("session-*/session-manifest.json"):
            saved = json.loads(manifest.read_text())["config"]["run"].get("initial_pose", {})
            if saved.get("sha256") != digest:
                raise ValueError(f"Initial pose {pose_id} changed after collection; use a new pose ID")
        config["run"]["output_dir"] = output
        result.append(ExperimentChoice(f"{choice.id}::pose:{pose_id}", choice.label, choice.path, config, choice.sha256, choice.policy))
    return result


def load_experiment_list(
    path: str | Path,
    *,
    local: Path | None = None,
    executor: str | None = None,
    control_timing: bool = False,
) -> ExperimentList:
    from manimux.cli import _load_config, read_yaml

    source = Path(path).expanduser().resolve()
    raw = read_yaml(source)
    if (
        set(raw) - {"experiments", "policy_services", "initial_poses"}
        or not isinstance(raw.get("experiments"), list)
        or not raw["experiments"]
    ):
        raise ValueError("Experiment list requires a nonempty experiments list")
    services = read_policy_services(raw.get("policy_services", {}), source.parent)
    choices: list[ExperimentChoice] = []
    ids: set[str] = set()
    outputs: set[Path] = set()
    station_contract = None
    policy_contracts: dict[str, dict] = {}
    service_recipes: dict[str, dict] = {}
    for entry in raw["experiments"]:
        if not isinstance(entry, dict) or set(entry) - {"id", "config", "label", "policy", "initial_poses"}:
            raise ValueError("Each experiment entry accepts id, config and optional label/policy/initial_poses")
        policy_name = entry.get("policy", "default")
        if not isinstance(policy_name, str) or not policy_name.strip():
            raise ValueError("Experiment policy must be a nonempty string")
        choice_id = entry.get("id")
        if not isinstance(choice_id, str) or not choice_id.strip() or choice_id in ids:
            raise ValueError(f"Experiment id must be a unique nonempty string: {choice_id!r}")
        if not isinstance(entry.get("config"), str):
            raise ValueError(f"Experiment {choice_id} requires a config path")
        config_path = (source.parent / entry["config"]).resolve()
        config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
        config = _load_config(config_path, executor, local=local, control_timing=control_timing)
        if services:
            if policy_name not in services:
                raise ValueError(f"Missing policy_services entry for {policy_name!r}")
            service_address(config)
        if not config["robogui"]["enabled"]:
            raise ValueError(f"Experiment {choice_id} requires robogui.enabled=true")
        current = shared_service_contract(config)
        station = {key: current[key] for key in ("camera_server", "robot", "robogui")}
        if station_contract is not None and station != station_contract:
            changed = [key for key in station if station[key] != station_contract[key]]
            raise ValueError(
                f"Experiment {choice_id} cannot reuse the same station: {', '.join(changed)}"
            )
        station_contract = station
        contract = policy_contracts.get(policy_name)
        if contract is not None and current != contract:
            changed = [key for key in current if current[key] != contract[key]]
            raise ValueError(
                f"Experiment {choice_id} cannot reuse the same services in policy "
                f"{policy_name!r}: {', '.join(changed)}"
            )
        policy_contracts[policy_name] = current
        # Two different model recipes cannot be live at the same client address.
        endpoint = config["policy"]["options"].get("server")
        if isinstance(endpoint, str):
            identity = {
                "worker": config["policy"]["worker"],
                "policy_server": config.get("policy_server"),
                "expected_backend": config["policy"].get("expected_backend"),
            }
            if (
                not services
                and endpoint in service_recipes
                and service_recipes[endpoint] != identity
            ):
                raise ValueError(
                    f"Different policy services share endpoint {endpoint!r}; "
                    "assign distinct endpoints"
                )
            service_recipes[endpoint] = identity
        output = Path(config["run"]["output_dir"]).resolve()
        if output in outputs:
            raise ValueError(f"Experiment {choice_id} must have its own output_dir: {output}")
        label = entry.get("label", choice_id)
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Experiment {choice_id} requires a nonempty label")
        task = config["run"]["task"]
        if any(
            c.policy == policy_name and c.config["run"]["task"] == task and c.label == label
            for c in choices
        ):
            raise ValueError(
                f"Duplicate config label for policy {policy_name}, task {task}: {label}"
            )
        ids.add(choice_id)
        outputs.add(output)
        choice = ExperimentChoice(
            choice_id, label, config_path, deepcopy(config), config_sha256, policy_name
        )
        expanded = [choice]
        pose_file = entry.get("initial_poses", raw.get("initial_poses"))
        if pose_file is not None:
            if not isinstance(pose_file, str):
                raise ValueError("initial_poses must reference a pose file")
            expanded = _pose_choices(choice, (source.parent / pose_file).resolve())
        for variant in expanded[1:]:
            variant_output = Path(variant.config["run"]["output_dir"]).resolve()
            if variant.id in ids or variant_output in outputs:
                raise ValueError("Duplicate initial pose choice or output path")
            ids.add(variant.id)
            outputs.add(variant_output)
        choices.extend(expanded)
    unused = set(services) - set(policy_contracts)
    if unused:
        raise ValueError(f"Unused policy_services entries: {sorted(unused)}")
    return ExperimentList(
        source, choices, hashlib.sha256(source.read_bytes()).hexdigest(), services
    )
