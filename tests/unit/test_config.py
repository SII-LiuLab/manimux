from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from manimux.config import ManiMuxConfig, load_config
from manimux.plugins import PluginError
from manimux.runtime import build_runtime


@pytest.mark.parametrize("action_variant", ["joint", "joint-ee"])
def test_yam_control_profile_preserves_executor_choices(action_variant):
    collection = load_config("configs/collection/yam/control.yaml")
    inference = load_config(
        f"configs/pi05/yam/infra/put-bottles/rtc-{action_variant}-step30000.yaml"
    )
    assert collection.control_profile == inference.control_profile
    assert collection.robot.group_dims == inference.robot.group_dims
    assert collection.policy.action_dt_s == inference.policy.action_dt_s
    assert collection.execution.command_safety == inference.execution.command_safety
    for side in ("left", "right"):
        assert (
            collection.robot.options[f"{side}_hardware_options"]
            == inference.robot.options[f"{side}_hardware_options"]
        )
    assert collection.robot.control_hz == 30
    assert inference.robot.control_hz == 100
    assert collection.execution.executor == "direct"
    assert inference.execution.smooth.max_velocity is None
    assert inference.execution.smooth.max_acceleration is None
    assert inference.execution.smooth.gripper.max_velocity is None
    assert inference.execution.smooth.gripper.max_acceleration is None
    assert inference.execution.smooth.gripper.max_closing_velocity == 1.0
    assert inference.execution.smooth.cutoff_hz == 8.0
    server = yaml.safe_load(Path(
        f"configs/pi05/yam/server/put-bottles/{action_variant}-step30000.yaml"
    ).read_text())
    expected = inference.policy.expected_backend.model
    for field in (
        "task_name", "checkpoint_variant", "checkpoint_source", "train_config_name",
        "norm_stats_path", "norm_stats_source", "action_horizon", "num_steps",
    ):
        assert server[field] == expected[field]
    assert server["model_path"] == expected["model_root"]
    assert inference.policy.options["server"] == f"ws://{server['host']}:{server['port']}"


@pytest.mark.parametrize("field,value", [
    ("robot.driver", "mock_dual_arm"),
    ("robot.group_dims", {"left_arm": 6}),
    ("robot.options.left_channel", "other_bus"),
    ("policy.action_dt_s", 0.1),
    ("policy.trajectory_duration_s", 0.1),
    ("execution.smooth.max_velocity", 0.25),
    ("execution.smooth.gripper.max_velocity", 1.0),
    ("execution.smooth.gripper.max_closing_velocity", None),
])
def test_control_profile_rejects_local_conflicts(tmp_path, field, value):
    raw = load_config("configs/collection/yam/control.yaml").model_dump(mode="json")
    target = raw
    parts = field.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    path = tmp_path / "local.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="conflicts with control_profile"):
        load_config(path)


def test_control_profile_relative_path_and_safety_contract(tmp_path):
    profile = yaml.safe_load(Path("configs/robots/yam/common.yaml").read_text())
    envelope = {
        "position_lower": {group: [-3.2] * 7 for group in profile["robot"]["group_dims"]},
        "position_upper": {group: [3.2] * 7 for group in profile["robot"]["group_dims"]},
        "max_velocity": {group: [0.5] * 7 for group in profile["robot"]["group_dims"]},
        "max_acceleration": {group: [1.0] * 7 for group in profile["robot"]["group_dims"]},
    }
    profile["command_safety"] = envelope
    (tmp_path / "shared.yaml").write_text(yaml.safe_dump(profile))
    raw = yaml.safe_load(Path("configs/collection/yam/control.yaml").read_text())
    raw["control_profile"] = "shared.yaml"
    path = tmp_path / "local.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = load_config(path)
    assert config.execution.command_safety.model_dump() == envelope
    assert config.control_profile == tmp_path / "shared.yaml"
    raw["execution"]["command_safety"] = {}
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="command_safety conflicts"):
        load_config(path)


def test_control_profile_rejects_recursive_inheritance(tmp_path):
    profile = yaml.safe_load(Path("configs/robots/yam/common.yaml").read_text())
    profile["control_profile"] = "shared.yaml"
    (tmp_path / "shared.yaml").write_text(yaml.safe_dump(profile))
    raw = yaml.safe_load(Path("configs/collection/yam/control.yaml").read_text())
    raw["control_profile"] = "shared.yaml"
    path = tmp_path / "local.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValidationError, match="Extra inputs"):
        load_config(path)


def test_shared_finite_motion_limits_are_resolved_for_smooth_and_direct(tmp_path):
    profile = yaml.safe_load(Path("configs/robots/yam/common.yaml").read_text())
    profile["motion_limits"]["arm"] = {"max_velocity": 0.5, "max_acceleration": 2.0}
    profile["motion_limits"]["gripper"].update(max_velocity=2.0, max_acceleration=10.0)
    (tmp_path / "shared.yaml").write_text(yaml.safe_dump(profile))
    raw = yaml.safe_load(Path("configs/collection/yam/control.yaml").read_text())
    raw["control_profile"] = "shared.yaml"
    path = tmp_path / "local.yaml"
    for executor in ("direct", "smooth"):
        raw["execution"]["executor"] = executor
        path.write_text(yaml.safe_dump(raw))
        config = load_config(path)
        assert config.execution.motion_limits.arm.max_velocity == 0.5
        assert config.execution.smooth.max_velocity == 0.5
        assert config.execution.smooth.max_acceleration == 2.0
        assert config.execution.smooth.gripper.max_velocity == 2.0
    raw["execution"]["executor"] = "mpc"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="not mpc"):
        load_config(path)


@pytest.mark.parametrize("value", [-1, 0, float("nan"), float("inf")])
def test_motion_limits_reject_invalid_numeric_values(value):
    from manimux.config import MotionRateConfig

    with pytest.raises(ValidationError):
        MotionRateConfig(max_velocity=value)
    with pytest.raises(ValidationError):
        MotionRateConfig(max_acceleration=value)


def test_mock_config_loads() -> None:
    config = load_config(Path("configs/mock.yaml"))
    assert config.robot.driver == "mock_dual_arm"
    assert config.execution.executor == "smooth"
    assert config.execution.inference_schedule == "deadline"
    assert config.robot.group_dims["left_arm"] == 6


def test_total_trajectory_duration_overrides_point_spacing() -> None:
    config = load_config(Path("configs/molmoact2/yam/infra/manimux.yaml"))

    assert config.policy.trajectory_duration_s is None
    assert config.policy.effective_action_dt_s == pytest.approx(0.05)
    assert config.execution.smooth.max_velocity == 0.25
    assert config.execution.smooth.max_acceleration == 0.5


def test_unknown_config_field_fails() -> None:
    with pytest.raises(ValidationError):
        ManiMuxConfig.model_validate(
            {
                "run": {"task": "x", "unknown": True},
                "robot": {"driver": "mock", "group_dims": {"arm": 1}},
                "policy": {"worker": "fake", "adapter": "identity"},
            }
        )


def test_expected_backend_requires_a_stable_identity_field() -> None:
    config = load_config(Path("configs/mock.yaml"))
    payload = config.model_dump(mode="python")
    payload["policy"]["expected_backend"] = {}

    with pytest.raises(ValidationError, match="must declare server or model identity"):
        ManiMuxConfig.model_validate(payload)


def test_all_infra_configs_load() -> None:
    for path in sorted(Path("configs").glob("*/yam/infra/*.yaml")):
        load_config(path)


def test_rtc_rejects_default_scheduler_fields_that_it_does_not_use() -> None:
    config = load_config(Path("configs/mock.yaml"))
    payload = config.model_dump(mode="python")
    payload["execution"]["runtime"] = "rtc"
    payload["execution"]["refill_threshold_s"] = 0.2

    with pytest.raises(ValidationError, match="not used by RTC"):
        ManiMuxConfig.model_validate(payload)


def test_recording_cannot_be_silently_disabled() -> None:
    config = load_config(Path("configs/mock.yaml"))
    payload = config.model_dump(mode="python")
    payload["recording"]["enabled"] = False

    with pytest.raises(ValidationError):
        ManiMuxConfig.model_validate(payload)


def test_experiment_and_video_recording_defaults_are_opt_in() -> None:
    config = load_config(Path("configs/mock.yaml"))

    assert not config.run.experiment_mode
    assert config.run.layout_id == ""
    assert config.recording.video_fps == 0


def test_command_safety_must_match_every_robot_group_dimension() -> None:
    config = load_config(Path("configs/mock.yaml"))
    payload = config.model_dump(mode="python")
    groups = payload["robot"]["group_dims"]
    payload["execution"]["command_safety"] = {
        "position_lower": {name: [-1.0] * dim for name, dim in groups.items()},
        "position_upper": {name: [1.0] * dim for name, dim in groups.items()},
        "max_velocity": {name: [1.0] * dim for name, dim in groups.items()},
        "max_acceleration": {name: [2.0] * dim for name, dim in groups.items()},
    }
    first_group = next(iter(groups))
    payload["execution"]["command_safety"]["max_velocity"][first_group].pop()

    with pytest.raises(ValidationError, match="vectors must share"):
        ManiMuxConfig.model_validate(payload)


def test_unknown_inference_strategy_fails_before_runtime_construction(tmp_path: Path) -> None:
    config = load_config(Path("configs/mock.yaml"))
    config.execution.runtime = "missing_strategy"

    with pytest.raises(PluginError, match="manimux.inference_strategies"):
        build_runtime(config, tmp_path)


def test_execution_prefix_rejects_incompatible_runtime_and_oversized_horizon():
    cfg = load_config('configs/mock.yaml').model_dump()
    cfg['execution']['max_chunk_steps'] = 25
    with pytest.raises(ValidationError, match='must not exceed'):
        ManiMuxConfig.model_validate(cfg)
    cfg['policy']['horizon_steps'] = 50
    assert ManiMuxConfig.model_validate(cfg).execution.max_chunk_steps == 25
    cfg['execution']['runtime'] = 'rtc'
    with pytest.raises(ValidationError, match='ordinary ManiMux'):
        ManiMuxConfig.model_validate(cfg)
