"""Selection metadata for a finite collection of resolved experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from manimux.configuration import initial_pose_metadata


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
            "task": self.config["run"]["task_prompt"],
            "method": self.config["inference"]["algorithm"],
            "initial_pose": initial_pose_metadata(self.config),
        }


def shared_service_contract(config: dict) -> dict:
    """Fail before launch if switching would require different external services.

    Model initialization arguments remain fixed, including sampler defaults and
    normalization. Task instructions belong to run.task_prompt, not model identity.
    Runtime strategies/executors are rebuilt by Prepare and may differ.
    """
    policy = dict(config["policy"])
    # The chunk IK solver runs in the runtime's decoder, rebuilt for every rollout;
    # switching mink <-> dls needs no service restart.
    policy["adapter"] = {
        key: value for key, value in (policy.get("adapter") or {}).items()
        if key not in {"ik_backend", "dls_ik"}
    }
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
