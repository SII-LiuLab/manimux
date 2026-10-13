"""A finite set of validated experiments sharing one live station/model service."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path


@dataclass(frozen=True)
class ExperimentChoice:
    id: str
    label: str
    path: Path
    config: dict
    sha256: str

    def metadata(self) -> dict:
        return {
            "id": self.id, "label": self.label, "config_path": str(self.path),
            "task": self.config["run"]["task"],
            "method": self.config["inference"]["algorithm"],
        }


def shared_service_contract(config: dict) -> dict:
    """Fail before launch if switching would require different external services.

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
        "policy": {key: policy.get(key) for key in (
            "worker", "service", "adapter", "options", "expected_backend",
            "action_dt_s", "horizon_policy_steps",
        )},
    }


class ExperimentList:
    def __init__(self, path: Path, choices: list[ExperimentChoice], digest: str):
        self.path = path
        self.choices = {choice.id: choice for choice in choices}
        self.sha256 = digest
        self.default_id = choices[0].id

    def metadata(self) -> list[dict]:
        return [choice.metadata() for choice in self.choices.values()]

    def selection(self, choice_id: str) -> dict:
        return {"list_path": str(self.path), "list_sha256": self.sha256,
                "config_sha256": self.choices[choice_id].sha256,
                **self.choices[choice_id].metadata()}


def load_experiment_list(path: str | Path, *, local: Path | None = None,
                         executor: str | None = None, control_timing: bool = False) -> ExperimentList:
    from manimux.cli import _load_config, read_yaml

    source = Path(path).expanduser().resolve()
    raw = read_yaml(source)
    if set(raw) != {"experiments"} or not isinstance(raw["experiments"], list) or not raw["experiments"]:
        raise ValueError("Experiment list requires a nonempty experiments list")
    choices: list[ExperimentChoice] = []
    ids: set[str] = set()
    outputs: set[Path] = set()
    contract = None
    for entry in raw["experiments"]:
        if not isinstance(entry, dict) or set(entry) - {"id", "config", "label"}:
            raise ValueError("Each experiment entry accepts id, config and optional label")
        choice_id = entry.get("id")
        if not isinstance(choice_id, str) or not choice_id.strip() or choice_id in ids:
            raise ValueError(f"Experiment id must be a unique nonempty string: {choice_id!r}")
        if not isinstance(entry.get("config"), str):
            raise ValueError(f"Experiment {choice_id} requires a config path")
        config_path = (source.parent / entry["config"]).resolve()
        config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
        config = _load_config(config_path, executor, local=local, control_timing=control_timing)
        if not config["robogui"]["enabled"]:
            raise ValueError(f"Experiment {choice_id} requires robogui.enabled=true")
        current = shared_service_contract(config)
        if contract is not None and current != contract:
            changed = [key for key in current if current[key] != contract[key]]
            raise ValueError(f"Experiment {choice_id} cannot reuse the same services: {', '.join(changed)}")
        contract = current
        output = Path(config["run"]["output_dir"]).resolve()
        if output in outputs:
            raise ValueError(f"Experiment {choice_id} must have its own output_dir: {output}")
        label = entry.get("label", choice_id)
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Experiment {choice_id} requires a nonempty label")
        task = config["run"]["task"]
        if any(c.config["run"]["task"] == task and c.label == label for c in choices):
            raise ValueError(f"Duplicate config label for task {task}: {label}")
        ids.add(choice_id)
        outputs.add(output)
        choices.append(ExperimentChoice(choice_id, label, config_path, deepcopy(config), config_sha256))
    return ExperimentList(source, choices, hashlib.sha256(source.read_bytes()).hexdigest())
