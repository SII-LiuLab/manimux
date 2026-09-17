from __future__ import annotations

import pytest

import scripts.servers.xiaomi_xr1_yam_server as xr1_server
from scripts.servers.xiaomi_xr1_yam_server import _validate


def _minimal_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "policy_name": "Xiaomi_Robotics_1",
        "protocol": "ws",
        "action_type": "ee",
        "output_format": "packed_ee_delta",
        "action_length": 30,
    }
    config.update(overrides)
    return config


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"protocol": "http"}, "protocol must be ws"),
        ({"action_type": "joint"}, "action_type must be ee"),
        ({"action_length": 16}, "action_length must be 30"),
    ],
)
def test_xr1_server_rejects_contract_drift(
    override: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _validate(_minimal_config(**override))


def test_xr1_runtime_status_does_not_claim_inference_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    python = tmp_path / "env/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setattr(xr1_server, "MODEL_PYTHON", python)

    config = xr1_server._load_config(
        xr1_server.DEFAULT_CONFIG
    )
    report = _validate(config)

    assert report["contract_status"] == "ready"
    assert report["runtime_status"] == "environment_present_gpu_forward_not_verified"
    assert report["inference_status"] == "not_verified"
    assert report["policy_status"] == "base_checkpoint_capability_unvalidated"


def test_xr1_base_config_does_not_claim_task_capability() -> None:
    config = xr1_server._load_config(
        xr1_server.REPO_ROOT / "configs/xiaomi-xr1/yam/server/base.yaml"
    )
    report = _validate(config)
    assert report["contract_status"] == "ready"
    assert report["checkpoint_variant"] == "xiaomi_robotics_1_5b_base_with_yam_stats"
    assert report["policy_status"] == "base_checkpoint_capability_unvalidated"
    assert report["norm_stats_role"] == "yam_projection_only_not_checkpoint_matched"


@pytest.mark.parametrize(
    ("config_name", "checkpoint_variant"),
    [
        (
            "finetune-assemble-screwdriver-step12000.yaml",
            "xiaomi_xr1_yam_assemble_screwdriver_step_12000",
        ),
        (
            "put-bottles/step30000.yaml",
            "xiaomi_xr1_yam_put_bottles_into_the_bin_step_30000",
        ),
    ],
)
def test_xr1_finetune_has_checkpoint_matched_contract(
    config_name: str, checkpoint_variant: str
) -> None:
    config = xr1_server._load_config(
        xr1_server.REPO_ROOT
        / "configs/xiaomi-xr1/yam/server"
        / config_name
    )
    report = _validate(config)

    assert report["checkpoint_variant"] == checkpoint_variant
    assert report["checkpoint_role"] == "yam_finetuned_policy"
    assert report["policy_status"] == "yam_finetune_not_evaluated"
    assert report["norm_stats_role"] == "checkpoint_matched_yam_finetune"


def test_xr1_declared_finetune_rejects_base_projection_stats() -> None:
    config = xr1_server._load_config(xr1_server.DEFAULT_CONFIG)
    config["checkpoint_role"] = "yam_finetuned_policy"
    with pytest.raises(ValueError, match="checkpoint-matched normalization"):
        _validate(config)
