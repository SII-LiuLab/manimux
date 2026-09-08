from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.validation.check_lingbot_vla2_yam import _validate

REPO_ROOT = Path(__file__).resolve().parents[2]


def _minimal_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "policy_name": "LingBot_VLA2",
        "protocol": "ws",
        "action_type": "joint",
        "lingbot_vla2_root": "/missing/source",
        "qwen3vl_path": "/missing/qwen3vl",
        "model_root": "/missing/checkpoint",
        "training_config_path": "/missing/lingbotvla_cli.yaml",
        "robot_config_path": "/missing/robot_config.yaml",
        "norm_stats_path": "/missing/norm_stats.json",
        "action_horizon": 50,
        "native_hz": 30.0,
    }
    config.update(overrides)
    return config


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"policy_name": "LingBot_VLA"}, "policy_name must be LingBot_VLA2"),
        ({"protocol": "http"}, "protocol must be ws"),
        ({"action_type": "ee"}, "action_type must be joint"),
    ],
)
def test_server_rejects_contract_drift(override, message) -> None:
    with pytest.raises(ValueError, match=message):
        _validate(_minimal_config(**override))


def test_missing_posttraining_artifacts_are_blocked() -> None:
    report = _validate(_minimal_config())
    assert report["status"] == "blocked"
    assert report["rtc_capability"] == "pi_guided_v1_sampler"
    assert any("missing files" in error for error in report["errors"])


def test_infra_must_match_server_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    import XPolicyLab.policy.LingBot_VLA2.model as adapter

    monkeypatch.setattr(
        adapter,
        "validate_deployment",
        lambda _: {
            "status": "ready",
            "errors": [],
            "native_hz": 20.0,
            "action_horizon": 25,
            "action_semantics": "absolute_joint_position",
        },
    )
    infra = {
        "robot": {"control_hz": 30.0},
        "policy": {
            "adapter": "xpolicylab",
            "horizon_steps": 50,
            "action_dt_s": 1 / 30,
        },
        "execution": {"runtime": "manimux"},
    }
    report = _validate(_minimal_config(), infra)
    assert report["status"] == "blocked"
    assert len(report["errors"]) == 2


def test_structurally_feasible_rtc_config_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import XPolicyLab.policy.LingBot_VLA2.model as adapter

    monkeypatch.setattr(
        adapter,
        "validate_deployment",
        lambda _: {
            "status": "ready",
            "errors": [],
            "native_hz": 30.0,
            "action_horizon": 50,
            "action_semantics": "absolute_joint_position",
            "rtc_capability": "pi_guided_v1_sampler",
        },
    )
    infra = {
        "robot": {"control_hz": 30.0},
        "policy": {
            "adapter": "xpolicylab",
            "horizon_steps": 50,
            "action_dt_s": 1 / 30,
        },
        "execution": {
            "runtime": "rtc",
            "rtc": {
                "initial_delay_steps": 12,
                "min_execute_steps": 20,
                "delay_buffer_size": 10,
                "beta": 5.0,
            },
        },
    }
    report = _validate(_minimal_config(), infra)
    assert report["status"] == "ready"
    assert report["rtc_capability"] == "pi_guided_v1_sampler"


def test_base_variant_is_reported_without_claiming_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import XPolicyLab.policy.LingBot_VLA2.model as adapter

    monkeypatch.setattr(
        adapter,
        "validate_deployment",
        lambda _: {
            "status": "ready",
            "errors": [],
            "native_hz": 30.0,
            "action_horizon": 50,
            "action_semantics": "absolute_joint_position",
            "rtc_capability": "pi_guided_v1_sampler",
        },
    )
    config = _minimal_config(
        checkpoint_variant="lingbot_vla2_6b_base_with_yam_stats",
        norm_stats_role="yam_projection_only_not_checkpoint_matched",
    )
    infra = {
        "robot": {"control_hz": 30.0},
        "policy": {
            "adapter": "xpolicylab",
            "horizon_steps": 50,
            "action_dt_s": 1 / 30,
        },
        "execution": {"runtime": "manimux"},
    }
    report = _validate(config, infra)
    assert report["status"] == "ready"
    assert report["inference_status"] == "not_verified"
    assert report["policy_status"] == "base_checkpoint_capability_unvalidated"


def test_standard_xpolicy_launcher_owns_lingbot_server() -> None:
    launcher = REPO_ROOT / "XPolicyLab/policy/LingBot_VLA2/setup_eval_policy_server.sh"
    text = launcher.read_text(encoding="utf-8")
    assert "XPolicyLab/setup_policy_server.py" not in text
    assert '"${XPOLICYLAB_ROOT}/setup_policy_server.py"' in text
    assert "scripts/lingbot_vla2_yam_server.py" not in text


def test_xpolicy_and_manimux_configs_use_explicit_artifact_paths() -> None:
    xpolicy = yaml.safe_load(
        (REPO_ROOT / "XPolicyLab/policy/LingBot_VLA2/deploy.yml").read_text(
            encoding="utf-8"
        )
    )
    manimux = yaml.safe_load(
        (REPO_ROOT / "configs/lingbot-vla2/yam/server/base.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert xpolicy["policy_name"] == manimux["policy_name"] == "LingBot_VLA2"
    assert xpolicy["model_root"] is None
    assert manimux["model_root"].endswith(
        "checkpoints/pretrained/lingbot-vla-v2-6b-yam-projection/runs/yam/hf_ckpt"
    )
    assert manimux["norm_stats_path"].endswith("norm_stats.json")
    assert manimux["qwen3vl_path"].endswith("qwen3_vl_4b_processor")
    assert xpolicy["checkpoint_variant"] == "yam_finetuned"
    assert manimux["checkpoint_variant"] == "lingbot_vla2_6b_base_with_yam_stats"
