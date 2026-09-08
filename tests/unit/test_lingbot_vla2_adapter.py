from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

MODEL_PATH = Path(__file__).resolve().parents[2] / "XPolicyLab/policy/LingBot_VLA2/model.py"
REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOT_INFO = {"arm_dim": [6, 6], "ee_dim": [1, 1]}


def _load_model_module():
    spec = importlib.util.spec_from_file_location("lingbot_vla2_adapter_test", MODEL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _observation():
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    return {
        "vision": {
            "cam_head": {"color": image},
            "cam_left_wrist": {"color": image},
            "cam_right_wrist": {"color": image},
        },
        "state": {
            "left_arm_joint_state": np.arange(6, dtype=np.float32),
            "left_ee_joint_state": np.array([6], dtype=np.float32),
            "right_arm_joint_state": np.arange(10, 16, dtype=np.float32),
            "right_ee_joint_state": np.array([16], dtype=np.float32),
        },
        "instruction": "pick the block",
    }


def test_encode_yam_observation_uses_official_feature_names() -> None:
    module = _load_model_module()
    encoded = module.encode_observation(_observation(), "fallback", ROBOT_INFO)
    np.testing.assert_array_equal(
        encoded["observation.state.arm.position"],
        np.array([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        encoded["observation.state.effector.position"],
        np.array([6, 16], dtype=np.float32),
    )
    assert encoded["task"] == "pick the block"


def test_encode_relative_yam_observation_uses_packed_training_keys() -> None:
    module = _load_model_module()
    encoded = module.encode_observation(
        _observation(),
        "fallback",
        ROBOT_INFO,
        module.RELATIVE_ACTION_SEMANTICS,
    )
    np.testing.assert_array_equal(
        encoded["observation.state"],
        np.array(
            [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16],
            dtype=np.float32,
        ),
    )
    assert "observation.images.top_rgb" in encoded
    assert "observation.state.arm.position" not in encoded


def test_decode_absolute_joint_chunk() -> None:
    module = _load_model_module()
    arms = np.arange(36, dtype=np.float32).reshape(3, 12)
    effectors = np.arange(6, dtype=np.float32).reshape(3, 2)
    actions = module.decode_actions(
        {
            "action.arm.position": arms,
            "action.effector.position": effectors,
        },
        ROBOT_INFO,
    )
    assert len(actions) == 3
    np.testing.assert_array_equal(actions[1]["left_arm_joint_state"], arms[1, :6])
    np.testing.assert_array_equal(actions[1]["right_arm_joint_state"], arms[1, 6:])
    np.testing.assert_array_equal(actions[1]["left_ee_joint_state"], effectors[1, :1])
    np.testing.assert_array_equal(actions[1]["right_ee_joint_state"], effectors[1, 1:])


def test_decode_rejects_wrong_action_width() -> None:
    module = _load_model_module()
    with pytest.raises(ValueError, match="must be \\(H, 12\\)"):
        module.decode_actions(
            {
                "action.arm.position": np.zeros((2, 14), dtype=np.float32),
                "action.effector.position": np.zeros((2, 2), dtype=np.float32),
            },
            ROBOT_INFO,
        )


def test_official_feature_literals_are_normalized_for_runtime() -> None:
    module = _load_model_module()
    data_config = SimpleNamespace(
        joints=[{"arm.position": 14}, "{'effector.position': 2}"],
        norm_type=[{"arm.position": "meanstd"}],
    )

    normalized = module._normalize_official_feature_literals(data_config)

    assert normalized.joints == ["{'arm.position': 14}", "{'effector.position': 2}"]
    assert normalized.norm_type == ["{'arm.position': 'meanstd'}"]


def test_standard_xpolicy_run_selects_latest_complete_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_model_module()
    run_root = tmp_path / "yam_real-demo-yam_dual-joint-0"
    older = run_root / "checkpoints/global_step_100/hf_ckpt"
    latest = run_root / "checkpoints/global_step_200/hf_ckpt"
    older.mkdir(parents=True)
    latest.mkdir(parents=True)
    (older / "model.safetensors.index.json").write_text("{}")
    (latest / "model.safetensors.index.json").write_text("{}")
    monkeypatch.setattr(module, "CHECKPOINTS_DIR", tmp_path)

    resolved = module._resolve_model_root(
        {
            "bench_name": "yam_real",
            "ckpt_name": "demo",
            "env_cfg_type": "yam_dual",
            "action_type": "joint",
            "seed": 0,
        }
    )

    assert resolved == latest


def test_get_action_rtc_dispatches_sampler_level_bridge() -> None:
    module = _load_model_module()

    class FakeBridge:
        @staticmethod
        def infer(observation, condition, weights, beta):
            assert observation == {"task": "pick"}
            assert condition.shape == (2, 14)
            assert weights.shape == (2,)
            assert beta == 5.0
            return {
                "action.arm.position": np.zeros((2, 12), dtype=np.float32),
                "action.effector.position": np.zeros((2, 2), dtype=np.float32),
            }

    model = object.__new__(module.Model)
    model._observations = [{"task": "pick"}]
    model.action_horizon = 2
    model.action_dim = 14
    model.robot_info = ROBOT_INFO
    model.action_semantics = module.ABSOLUTE_ACTION_SEMANTICS
    model._rtc_bridge = FakeBridge()
    actions = model.get_action_rtc(
        {
            "action_condition": np.zeros((2, 14), dtype=np.float32),
            "condition_weights": np.array([1.0, 0.5], dtype=np.float32),
            "beta": 5.0,
        }
    )
    assert len(actions) == 2
    assert set(actions[0]) == {
        "left_arm_joint_state",
        "left_ee_joint_state",
        "right_arm_joint_state",
        "right_ee_joint_state",
    }


def test_get_action_rtc_preserves_relative_action_metadata() -> None:
    module = _load_model_module()

    class FakeBridge:
        @staticmethod
        def infer(*_):
            return {
                "action.arm.position": np.zeros((2, 12), dtype=np.float32),
                "action.effector.position": np.ones((2, 2), dtype=np.float32),
            }

    model = object.__new__(module.Model)
    model._observations = [{"task": "assemble"}]
    model.action_horizon = 2
    model.action_dim = 14
    model.robot_info = ROBOT_INFO
    model.action_semantics = module.RELATIVE_ACTION_SEMANTICS
    model._rtc_bridge = FakeBridge()

    result = model.get_action_rtc(
        {
            "action_condition": np.zeros((2, 14), dtype=np.float32),
            "condition_weights": np.array([1.0, 0.5], dtype=np.float32),
            "beta": 5.0,
        }
    )

    assert result["action_semantics"] == module.RELATIVE_ACTION_SEMANTICS
    assert len(result["actions"]) == 2
