from __future__ import annotations

import json
from pathlib import Path

import pytest

import manimux.servers.groot as gr00t_server
from manimux.cli import load_config
from manimux.policies.base import action_interval
from manimux.servers.groot import _runtime_readiness, _validate_checkpoint

STATE_KEYS = ("left_arm", "left_gripper", "right_arm", "right_gripper")
DIMS = {"left_arm": 6, "left_gripper": 1, "right_arm": 6, "right_gripper": 1}


def _write_checkpoint(root: Path) -> None:
    root.mkdir()
    model_config = {"architectures": ["Gr00tN1d7"], "num_inference_timesteps": 4}
    (root / "config.json").write_text(json.dumps(model_config), encoding="utf-8")
    modality = {
        "video": {"modality_keys": ["base_view", "left_wrist_view", "right_wrist_view"]},
        "state": {"modality_keys": list(STATE_KEYS)},
        "action": {
            "modality_keys": list(STATE_KEYS),
            "delta_indices": list(range(16)),
            "action_configs": [{"rep": "ABSOLUTE"} for _ in STATE_KEYS],
        },
    }
    processor = {"processor_kwargs": {"modality_configs": {"new_embodiment": modality}}}
    (root / "processor_config.json").write_text(json.dumps(processor), encoding="utf-8")
    field_names = ("min", "max", "mean", "std", "q01", "q99")
    statistics = {
        "new_embodiment": {
            scope: {key: {field: [0.0] * dim for field in field_names} for key, dim in DIMS.items()}
            for scope in ("state", "action")
        }
    }
    (root / "statistics.json").write_text(json.dumps(statistics), encoding="utf-8")
    (root / "model-00001-of-00001.safetensors").write_bytes(b"test")
    index = {"weight_map": {"model.test": "model-00001-of-00001.safetensors"}}
    (root / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")


def test_gr00t_checkpoint_contract(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint)

    resolved, horizon = _validate_checkpoint(
        {"model_dir": str(checkpoint), "native_frequency_hz": 30.0}
    )

    assert resolved == checkpoint.resolve()
    assert horizon == 16


def test_gr00t_checkpoint_rejects_frequency_drift(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint)

    with pytest.raises(ValueError, match="native_frequency_hz must be 30.0"):
        _validate_checkpoint({"model_dir": str(checkpoint), "native_frequency_hz": 20.0})


def test_gr00t_manimux_config_preserves_native_contract() -> None:
    config = load_config("manimux/configs/experiments/pick_box/yam_groot_manimux.yaml")

    assert config["policy"]["worker"] == "xpolicylab_ws"
    assert config["policy"]["adapter"]["type"] == "manimux.policy_adapter.joint:JointAdapter"
    assert config["policy"]["horizon_policy_steps"] == 16
    assert action_interval(config["policy"]) == pytest.approx(1.0 / 30.0)
    assert config["robot"]["group_dims"] == {"left_arm": 7, "right_arm": 7}
    assert config["inference"]["algorithm"] == "manimux"


def test_runtime_readiness_does_not_call_contract_ready_inference_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing_python = tmp_path / "missing/.venv/bin/python"
    monkeypatch.setattr(gr00t_server, "MODEL_PYTHON", missing_python)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))

    report = _runtime_readiness({"cosmos_model_path": "nvidia/Cosmos-Reason2-2B"})

    assert report["runtime_status"] == "blocked_incomplete_model_environment"
    assert report["inference_status"] == "not_verified"
    assert report["cosmos_cached"] is False


def test_runtime_readiness_rejects_empty_environment_and_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    python = tmp_path / "env/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("not python", encoding="utf-8")
    python.chmod(0o755)
    (tmp_path / "env/pyvenv.cfg").write_text("home = test\n", encoding="utf-8")
    snapshot = tmp_path / "hub/models--nvidia--Cosmos-Reason2-2B/snapshots/empty"
    snapshot.mkdir(parents=True)
    monkeypatch.setattr(gr00t_server, "MODEL_PYTHON", python)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))

    report = _runtime_readiness({"cosmos_model_path": "nvidia/Cosmos-Reason2-2B"})

    assert report["runtime_status"] == "blocked_incomplete_model_environment"
    assert report["cosmos_cached"] is False
    assert report["inference_status"] == "not_verified"
