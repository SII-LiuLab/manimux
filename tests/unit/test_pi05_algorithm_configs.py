from pathlib import Path

import pytest
import yaml

from manimux.cli import load_config
from manimux.policies.base import action_interval
from manimux.runtime.inference import build_inference_strategy

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "manimux/configs/experiments"
SUFFIX = "assemble-screwdriver-step15000"
METHODS = {
    "manimux": ("manimux", "default", "skip_elapsed_steps"),
    "rtc": ("rtc", "rtc", "skip_elapsed_steps"),
    "act-temporal-ensemble": ("act_temporal_ensemble", "default", "skip_elapsed_steps"),
    "aac": ("aac", "aac", "first_step_when_ready"),
    "paint": ("paint", "paint", "skip_elapsed_steps"),
    "autohorizon": ("autohorizon", "autohorizon", "first_step_when_ready"),
    "dvac": ("dvac", "dvac", "first_step_when_ready"),
}


@pytest.mark.parametrize("method", METHODS)
def test_screwdriver_algorithms_share_model_and_executor(method: str) -> None:
    baseline = load_config(EXPERIMENTS / "assemble_screwdriver/pi05/yam_pi05_manimux_step15000.yaml")
    config = load_config(
        EXPERIMENTS
        / f"assemble_screwdriver/pi05/yam_pi05_{method.replace(chr(45), chr(95))}_step15000.yaml"
    )
    runtime, sampling, action_start_mode = METHODS[method]

    assert config["inference"]["algorithm"] == runtime
    strategy = build_inference_strategy(config)
    assert strategy.name == runtime
    assert strategy.required_sampling_modes == frozenset({sampling})
    assert config["inference"]["action_start_mode"] == action_start_mode

    # Algorithm selection must not change hardware, observation or last-mile control.
    assert config["robot"] == baseline["robot"]
    assert config["robot"]["control_hz"] == 100.0
    assert config["sensors"] == baseline["sensors"]
    assert config["recording"] == baseline["recording"]
    assert config["run"]["task"] == baseline["run"]["task"]
    assert config["run"]["max_control_steps"] == baseline["run"]["max_control_steps"]
    assert config["run"]["output_dir"] == baseline["run"]["output_dir"].parent / method
    assert config["executor"]["type"] == "smooth"
    assert config["executor"]["smooth"] == baseline["executor"]["smooth"]
    assert config["executor"]["command_safety"] == baseline["executor"]["command_safety"]
    gripper = config["executor"]["smooth"]["gripper"]
    assert gripper is not None
    assert gripper["mode"] == "continuous"
    assert gripper["group_indices"] == {"left_arm": 6, "right_arm": 6}
    assert gripper["max_velocity"] == 1.0
    assert gripper["max_acceleration"] == 12.0

    # Cold-start timeouts may differ; checkpoint, stats and action semantics may not.
    for key in (
        "worker",
        "adapter",
        "action_dt_s",
        "horizon_policy_steps",
        "expected_backend",
    ):
        assert config["policy"][key] == baseline["policy"][key]
    options = {k: v for k, v in config["policy"]["options"].items() if k != "request_timeout_s"}
    base_options = {
        k: v for k, v in baseline["policy"]["options"].items() if k != "request_timeout_s"
    }
    assert options == base_options
    assert config["policy"]["horizon_policy_steps"] == 50
    assert action_interval(config["policy"]) == pytest.approx(1 / 30)
    if method not in {"manimux", "rtc"}:
        assert config["inference"]["blend_policy_steps"] == 0
        # Keep algorithm defaults except the screwdriver ACT query interval.
        previous = load_config(
            EXPERIMENTS
            / f"pick_red_object/pi05/yam_pi05_{method.replace(chr(45), chr(95))}_step1000.yaml"
        )
        for key in ("paint", "aac", "dvac", "temporal_ensemble"):
            expected = previous["inference"][key]
            if method == "act-temporal-ensemble" and key == "temporal_ensemble":
                expected = {**expected, "query_interval_policy_steps": 20}
            assert config["inference"][key] == expected
        assert config["policy"]["timeout_s"] == previous["policy"]["timeout_s"]
        assert (
            config["policy"]["options"]["request_timeout_s"]
            == previous["policy"]["options"]["request_timeout_s"]
        )


def test_screwdriver_backend_identity_matches_shared_server() -> None:
    config = load_config(EXPERIMENTS / "assemble_screwdriver/pi05/yam_pi05_manimux_step15000.yaml")
    server_path = ROOT / f"manimux/configs/policy/pi05/yam/finetune-{SUFFIX}.yaml"
    server = yaml.safe_load(server_path.read_text())
    assert config["policy"]["expected_backend"] is not None
    identity = config["policy"]["expected_backend"]["model"]
    for key in (
        "task_name",
        "checkpoint_variant",
        "checkpoint_source",
        "norm_stats_source",
        "train_config_name",
        "norm_stats_path",
        "action_horizon",
        "num_steps",
    ):
        assert identity[key] == server[key]
    assert identity["model_root"] == server["model_path"]
    assert config["policy"]["service"] == "policy"


def test_screwdriver_aac_scoring_stats_are_present_and_separate() -> None:
    config = load_config(EXPERIMENTS / "assemble_screwdriver/pi05/yam_pi05_aac_step15000.yaml")
    stats = config["inference"]["aac"]["ee_stats_path"]
    assert stats is not None
    assert (ROOT / stats).is_file()
    assert config["policy"]["expected_backend"] is not None
    assert stats != config["policy"]["expected_backend"]["model"]["norm_stats_path"]
