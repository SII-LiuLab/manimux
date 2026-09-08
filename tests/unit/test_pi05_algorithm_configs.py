from pathlib import Path

import pytest
import yaml

from manimux.config import load_config
from manimux.runtime.inference import build_inference_strategy

ROOT = Path(__file__).resolve().parents[2]
INFRA = ROOT / "configs/pi05/yam/infra"
SUFFIX = "assemble-screwdriver-step15000"
METHODS = {
    "manimux": ("manimux", "default"),
    "rtc": ("rtc", "rtc"),
    "act-temporal-ensemble": ("act_temporal_ensemble", "default"),
    "aac": ("aac", "aac"),
    "paint": ("paint", "paint"),
    "autohorizon": ("autohorizon", "autohorizon"),
    "dvac": ("dvac", "dvac"),
}


@pytest.mark.parametrize("method", METHODS)
def test_screwdriver_algorithms_share_model_and_executor(method: str) -> None:
    baseline = load_config(INFRA / f"manimux-{SUFFIX}.yaml")
    config = load_config(INFRA / f"{method}-{SUFFIX}.yaml")
    runtime, sampling = METHODS[method]

    assert config.execution.runtime == runtime
    strategy = build_inference_strategy(config)
    assert strategy.name == runtime
    assert strategy.required_sampling_modes == frozenset({sampling})

    # Algorithm selection must not change hardware, observation or last-mile control.
    assert config.robot == baseline.robot
    assert config.robot.control_hz == 100.0
    assert config.sensors == baseline.sensors
    assert config.recording == baseline.recording
    assert config.run.task == baseline.run.task
    assert config.run.max_steps == baseline.run.max_steps
    assert config.run.output_dir == baseline.run.output_dir.parent / method
    assert config.execution.executor == "smooth"
    assert config.execution.smooth == baseline.execution.smooth
    assert config.execution.command_safety == baseline.execution.command_safety
    gripper = config.execution.smooth.gripper
    assert gripper is not None
    assert gripper.mode == "continuous"
    assert gripper.group_indices == {"left_arm": 6, "right_arm": 6}
    assert gripper.max_velocity == 1.0
    assert gripper.max_acceleration == 12.0

    # Cold-start timeouts may differ; checkpoint, stats and action semantics may not.
    for key in ("worker", "adapter", "action_dt_s", "horizon_steps", "expected_backend"):
        assert getattr(config.policy, key) == getattr(baseline.policy, key)
    options = {k: v for k, v in config.policy.options.items() if k != "request_timeout_s"}
    base_options = {
        k: v for k, v in baseline.policy.options.items() if k != "request_timeout_s"
    }
    assert options == base_options
    assert config.policy.horizon_steps == 50
    assert config.policy.effective_action_dt_s == pytest.approx(1 / 30)
    if method not in {"manimux", "rtc"}:
        assert config.execution.blend_steps == 0
        # Keep the existing algorithm's defaults, independently of the executor.
        previous = load_config(INFRA / f"{method}-pick-red-ball-box-step1000.yaml")
        for key in ("paint", "aac", "dvac", "temporal_ensemble"):
            assert getattr(config.execution, key) == getattr(previous.execution, key)
        assert config.policy.timeout_s == previous.policy.timeout_s
        assert config.policy.options["request_timeout_s"] == previous.policy.options[
            "request_timeout_s"
        ]


def test_screwdriver_backend_identity_matches_shared_server() -> None:
    config = load_config(INFRA / f"manimux-{SUFFIX}.yaml")
    server_path = ROOT / f"configs/pi05/yam/server/finetune-{SUFFIX}.yaml"
    server = yaml.safe_load(server_path.read_text())
    assert config.policy.expected_backend is not None
    identity = config.policy.expected_backend.model
    for key in (
        "task_name", "checkpoint_variant", "checkpoint_source", "norm_stats_source",
        "train_config_name", "norm_stats_path", "action_horizon", "num_steps",
    ):
        assert identity[key] == server[key]
    assert identity["model_root"] == server["model_path"]
    assert config.policy.options["server"] == f"ws://{server['host']}:{server['port']}"


def test_screwdriver_aac_scoring_stats_are_present_and_separate() -> None:
    config = load_config(INFRA / f"aac-{SUFFIX}.yaml")
    stats = config.execution.aac.ee_stats_path
    assert stats is not None
    assert (ROOT / stats).is_file()
    assert config.policy.expected_backend is not None
    assert stats != config.policy.expected_backend.model["norm_stats_path"]
