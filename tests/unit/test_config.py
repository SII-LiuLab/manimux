from __future__ import annotations

from copy import deepcopy
from json import dumps, loads
from pathlib import Path

import pytest
import yaml

from manimux.cli import load_config, prepare_experiment, read_experiment
from manimux.plugins import PluginError
from manimux.policies.base import action_interval
from manimux.runtime import build_runtime
from manimux.runtime.executors.limits import arm_motion_parameters
from manimux.runtime.executors.smooth import gripper_hysteresis_parameters
from manimux.runtime.safety import command_safety_parameters


def _collection_recipe() -> dict:
    raw = yaml.safe_load(Path("manimux/configs/collection/yam/control.yaml").read_text())
    raw["robot"]["config"] = str(
        Path("manimux/configs/embodiment/robot/yam_dual.yaml").resolve()
    )
    return raw


@pytest.mark.parametrize("action_variant", ["joint", "joint-ee"])
def test_yam_control_profile_preserves_executor_choices(action_variant, tmp_path):
    station = tmp_path / "station.yaml"
    station.write_text(Path("manimux/configs/local/yam.example.yaml").read_text())
    collection = load_config("manimux/configs/collection/yam/control.yaml", local=station)
    inference = load_config(
        "manimux/configs/experiments/put_bottles/"
        f"yam_pi05_rtc_{action_variant.replace(chr(45), chr(95))}_step30000.yaml",
        local=station,
    )
    assert collection["control_profile"] == inference["control_profile"]
    assert collection["robot"]["group_dims"] == inference["robot"]["group_dims"]
    assert collection["policy"]["action_dt_s"] == inference["policy"]["action_dt_s"]
    assert collection["executor"]["command_safety"] == inference["executor"]["command_safety"]
    for side in ("left", "right"):
        assert (
            collection["robot"]["options"]["component_hardware"][f"{side}_yam"]
            == inference["robot"]["options"]["component_hardware"][f"{side}_yam"]
        )
    assert collection["robot"]["control_hz"] == 30
    recipe = yaml.safe_load(
        Path(
            "manimux/configs/experiments/put_bottles/"
            f"yam_pi05_rtc_{action_variant.replace(chr(45), chr(95))}_step30000.yaml"
        ).read_text()
    )
    assert inference["robot"]["control_hz"] == recipe["robot"]["control_hz"]
    assert collection["executor"]["type"] == "direct"
    assert inference["executor"]["smooth"]["max_velocity"] is None
    assert inference["executor"]["smooth"]["max_acceleration"] is None
    assert inference["executor"]["smooth"]["gripper"]["max_velocity"] is None
    assert inference["executor"]["smooth"]["gripper"]["max_acceleration"] is None
    assert inference["executor"]["smooth"]["gripper"]["max_closing_velocity"] == 1.0
    assert inference["executor"]["smooth"]["cutoff_hz"] == 8.0
    server = inference["policy_server"]
    expected = inference["policy"]["expected_backend"]["model"]
    for field in (
        "task_name",
        "checkpoint_variant",
        "checkpoint_source",
        "train_config_name",
        "norm_stats_path",
        "norm_stats_source",
        "action_horizon",
        "num_steps",
    ):
        assert server[field] == expected[field]
    assert server["model_path"] == expected["model_root"]
    assert inference["policy"]["options"]["server"] == f"ws://{server['host']}:{server['port']}"


@pytest.mark.parametrize(
    "field,value",
    [
        ("robot.type", "tests.support.robot:build_robot"),
        ("robot.group_dims", {"left_arm": 6}),
        ("executor.smooth.max_velocity", 0.25),
        ("executor.smooth.gripper.max_velocity", 1.0),
        ("executor.smooth.gripper.max_closing_velocity", None),
    ],
)
def test_control_profile_rejects_local_conflicts(tmp_path, field, value):
    raw = loads(
        dumps(deepcopy(load_config("manimux/configs/collection/yam/control.yaml")), default=str)
    )
    target = raw
    parts = field.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    path = tmp_path / "local.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(
        ValueError,
        match="conflicts with (control_profile|embodiment action_contract)",
    ):
        load_config(path)


def test_control_profile_relative_path_and_safety_contract(tmp_path):
    profile = yaml.safe_load(Path("manimux/configs/embodiment/robot/yam_control.yaml").read_text())
    envelope = {
        "position_lower": {group: [-3.2] * 7 for group in profile["robot"]["group_dims"]},
        "position_upper": {group: [3.2] * 7 for group in profile["robot"]["group_dims"]},
        "max_velocity": {group: [0.5] * 7 for group in profile["robot"]["group_dims"]},
        "max_acceleration": {group: [1.0] * 7 for group in profile["robot"]["group_dims"]},
    }
    profile["command_safety"] = envelope
    (tmp_path / "shared.yaml").write_text(yaml.safe_dump(profile))
    raw = _collection_recipe()
    raw["control_profile"] = "shared.yaml"
    path = tmp_path / "local.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = load_config(path)
    assert deepcopy(config["executor"]["command_safety"]) == envelope
    assert config["control_profile"] == tmp_path / "shared.yaml"
    raw["executor"]["command_safety"] = {}
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="command_safety conflicts"):
        load_config(path)


def test_policy_timing_is_independent_of_shared_robot_limits(tmp_path):
    """换模型动作时间轴不修改本体限制，也不改变主循环下发频率。"""
    raw = loads(
        dumps(deepcopy(load_config("manimux/configs/collection/yam/control.yaml")), default=str)
    )
    raw["policy"]["action_dt_s"] = 0.05
    raw["robot"]["control_hz"] = 100.0
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = load_config(path)
    assert action_interval(config["policy"]) == 0.05
    assert config["robot"]["control_hz"] == 100.0
    assert config["executor"]["motion_limits"] == raw["executor"]["motion_limits"]


def test_control_profile_rejects_recursive_inheritance(tmp_path):
    profile = yaml.safe_load(Path("manimux/configs/embodiment/robot/yam_control.yaml").read_text())
    profile["control_profile"] = "shared.yaml"
    (tmp_path / "shared.yaml").write_text(yaml.safe_dump(profile))
    raw = _collection_recipe()
    raw["control_profile"] = "shared.yaml"
    path = tmp_path / "local.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="recursive control_profile"):
        load_config(path)


def test_shared_finite_motion_limits_are_resolved_for_smooth_and_direct(tmp_path):
    profile = yaml.safe_load(Path("manimux/configs/embodiment/robot/yam_control.yaml").read_text())
    profile["motion_limits"]["arm"] = {"max_velocity": 0.5, "max_acceleration": 2.0}
    profile["motion_limits"]["gripper"].update(max_velocity=2.0, max_acceleration=10.0)
    (tmp_path / "shared.yaml").write_text(yaml.safe_dump(profile))
    raw = _collection_recipe()
    raw["control_profile"] = "shared.yaml"
    path = tmp_path / "local.yaml"
    for executor in ("direct", "smooth"):
        raw["executor"]["type"] = executor
        path.write_text(yaml.safe_dump(raw))
        config = load_config(path)
        assert config["executor"]["motion_limits"]["arm"]["max_velocity"] == 0.5
        assert config["executor"]["smooth"]["max_velocity"] == 0.5
        assert config["executor"]["smooth"]["max_acceleration"] == 2.0
        assert config["executor"]["smooth"]["gripper"]["max_velocity"] == 2.0
    raw["executor"]["type"] = "mpc"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="not mpc"):
        load_config(path)


@pytest.mark.parametrize("value", [-1, 0, float("nan"), float("inf")])
def test_motion_limits_reject_invalid_numeric_values(value):
    with pytest.raises(ValueError):
        arm_motion_parameters(max_velocity=value)
    with pytest.raises(ValueError):
        arm_motion_parameters(max_acceleration=value)
    with pytest.raises(ValueError):
        arm_motion_parameters(max_step_dt_s=value)


def test_motion_mode_defaults_and_explicit_selection(tmp_path):
    assert arm_motion_parameters()["mode"] == "per_joint"
    with pytest.raises(ValueError):
        arm_motion_parameters(mode="unknown")
    raw = _collection_recipe()
    profile = yaml.safe_load(Path("manimux/configs/embodiment/robot/yam_control.yaml").read_text())
    profile["motion_limits"]["arm"].update(mode="isotropic", max_step_dt_s=0.016)
    (tmp_path / "shared.yaml").write_text(yaml.safe_dump(profile))
    raw["control_profile"] = "shared.yaml"
    path = tmp_path / "local.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = load_config(path)
    assert config["executor"]["motion_limits"]["arm"]["mode"] == "isotropic"
    assert config["executor"]["smooth"]["mode"] == "isotropic"
    assert config["executor"]["smooth"]["max_step_dt_s"] == 0.016
    # Arm limit mode and gripper aperture mode are different configuration fields.
    assert config["executor"]["smooth"]["gripper"]["mode"] == "continuous"
    # Repeated shared settings remain equivalent when older YAML omits new defaults.
    raw["executor"]["motion_limits"] = profile["motion_limits"]
    path.write_text(yaml.safe_dump(raw))
    assert load_config(path)["executor"]["smooth"]["mode"] == "isotropic"
    raw["executor"]["smooth"] = {"mode": "per_joint"}
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="conflicts with control_profile"):
        load_config(path)


@pytest.mark.parametrize("variant", ["default", "rtc"])
def test_umi_pass_ball_enables_close_latch_with_shared_motion_limits(variant):
    config = load_config(f"manimux/configs/experiments/pass_ball/tianji_umi_dp_{variant}.yaml")
    gripper = config["executor"]["smooth"]["gripper"]
    assert gripper["mode"] == "close_latch"
    assert (gripper["close_threshold"], gripper["open_threshold"], gripper["closed_value"]) == (
        0.6,
        0.75,
        0.2,
    )
    assert (gripper["min_closed_s"], gripper["open_confirm_s"]) == (0.0, 0.0)
    assert gripper["group_indices"] == {"left_arm": 7, "right_arm": 7}
    assert (
        gripper["max_velocity"],
        gripper["max_acceleration"],
        gripper["max_closing_velocity"],
    ) == (
        3,
        12,
        1,
    )
    assert config["robot"]["options"]["execute"] is False
    assert config["robot"]["options"]["end_effector_control"] is False


def test_tianji_rate_contract_derives_one_controller_and_runtime_speed():
    assembly = yaml.safe_load(
        Path("manimux/configs/embodiment/robot/tianji_taccap.yaml").read_text()
    )
    profile = yaml.safe_load(
        Path("manimux/configs/embodiment/robot/tianji_control.yaml").read_text()
    )
    assert "motion_limits" not in profile
    assert "max_velocity" not in profile["command_safety"]

    config = load_config(
        "manimux/configs/experiments/pass_ball/tianji_taccap_umi_dp.yaml"
    )
    hardware = assembly["hardware"]
    controller_speed = (
        hardware["rated_joint_velocity_rad_s"] * hardware["velocity_ratio"] / 100
    )
    executor_speed = controller_speed * profile["rate_contract"]["command_margin"]
    assert config["executor"]["motion_limits"]["arm"]["max_velocity"] == pytest.approx(
        executor_speed
    )
    assert config["executor"]["smooth"]["max_velocity"] == pytest.approx(executor_speed)
    for rates in config["executor"]["command_safety"]["max_velocity"].values():
        assert rates[:7] == pytest.approx([controller_speed] * 7)


def test_tianji_taccap_experiments_use_shared_control_and_thin_executor_profiles():
    for path in Path("manimux/configs/experiments").glob("**/tianji_taccap_*.yaml"):
        raw = yaml.safe_load(path.read_text())
        expected = (
            "tianji_control_live.yaml"
            if path.stem.endswith("_live")
            else "tianji_control.yaml"
        )
        assert raw["control_profile"].endswith(f"/{expected}")
        assert raw["executor"]["config"].endswith(
            "/tianji_smooth_control_profile.yaml"
        )


@pytest.mark.parametrize(
    "override",
    [
        {"closed_value": 0.6},
        {"closed_value": 0.7},
        {"open_value": 0.7},
        {"close_threshold": 0.75},
        {"open_threshold": float("nan")},
    ],
)
def test_close_latch_rejects_inconsistent_thresholds(override):
    fields = dict(
        mode="close_latch",
        group_indices={"arm": 1},
        close_threshold=0.6,
        open_threshold=0.75,
        closed_value=0.2,
    )
    with pytest.raises(ValueError):
        gripper_hysteresis_parameters(**(fields | override))


def test_safety_optional_acceleration_validates_every_supplied_group():
    envelope = {
        "position_lower": {"arm": [-1.0]},
        "position_upper": {"arm": [1.0]},
        "max_velocity": {"arm": [2.0]},
    }
    assert command_safety_parameters(**envelope)["max_acceleration"] == {}
    assert command_safety_parameters(**envelope, max_acceleration=None)["max_acceleration"] == {}
    for invalid in ({"wrong": [1.0]}, {"arm": []}, {"arm": [-1.0]}, {"arm": [float("nan")]}):
        with pytest.raises(ValueError):
            command_safety_parameters(**envelope, max_acceleration=invalid)
    with pytest.raises(ValueError, match="together"):
        command_safety_parameters(max_acceleration={"arm": [1.0]})


def test_mock_config_loads() -> None:
    config = load_config(Path("tests/fixtures/runtime.yaml"))
    assert config["robot"]["type"] == "tests.support.robot:build_robot"
    assert config["executor"]["type"] == "smooth"
    assert config["inference"]["inference_schedule"] == "deadline"
    assert config["robot"]["group_dims"]["left_arm"] == 6


def test_total_trajectory_duration_overrides_point_spacing() -> None:
    config = load_config(
        Path("manimux/configs/experiments/pick_red_object/yam_molmoact2_manimux.yaml")
    )

    assert config["policy"]["trajectory_duration_s"] is None
    assert action_interval(config["policy"]) == pytest.approx(0.05)
    assert config["executor"]["smooth"]["max_velocity"] == 0.25
    assert config["executor"]["smooth"]["max_acceleration"] == 0.5


def test_yaml_reader_preserves_application_fields(tmp_path):
    from manimux.cli import read_yaml

    path = tmp_path / "custom.yaml"
    path.write_text("custom:\n  option: 12\n")
    assert read_yaml(path) == {"custom": {"option": 12}}


def test_expected_backend_requires_a_stable_identity_field() -> None:
    config = load_config(Path("tests/fixtures/runtime.yaml"))
    payload = deepcopy(config)
    payload["policy"]["expected_backend"] = {}

    with pytest.raises(ValueError, match="must declare server or model identity"):
        prepare_experiment(**payload)


def test_policy_recipe_generates_backend_identity() -> None:
    path = Path("manimux/configs/experiments/assemble_screwdriver/yam_pi05_manimux_step15000.yaml")
    raw = read_experiment(path, bind_local=False)
    server = raw["policy_server"]
    identity = raw["policy"]["expected_backend"]["model"]

    assert "backend_identity" not in server
    assert identity["checkpoint_variant"] == server["checkpoint_variant"]
    assert identity["model_root"] == server["model_path"]


def test_explicit_step_units_are_the_runtime_contract(tmp_path) -> None:
    raw = yaml.safe_load(Path("tests/fixtures/runtime.yaml").read_text())
    path = tmp_path / "runtime.yaml"
    path.write_text(yaml.safe_dump(raw))

    config = load_config(path)
    assert config["run"]["max_control_steps"] == 120
    assert config["policy"]["horizon_policy_steps"] == 20
    assert config["inference"]["blend_policy_steps"] == 2


@pytest.mark.parametrize(
    "path,field",
    [
        (("run",), "max_steps"),
        (("policy",), "horizon_steps"),
        (("executor", "mpc"), "horizon_steps"),
        (("inference",), "blend_steps"),
        (("inference",), "chunk_steps"),
        (("inference",), "max_chunk_steps"),
        (("inference", "rtc"), "initial_delay_steps"),
        (("inference", "rtc"), "min_execute_steps"),
        (("inference", "temporal_ensemble"), "query_interval_steps"),
        (("inference", "paint"), "execution_steps"),
        (("inference", "paint"), "initial_delay_steps"),
        (("inference", "dvac"), "tail_steps"),
        (("inference", "dvac"), "min_execution_steps"),
        (("inference", "dvac"), "max_execution_steps"),
    ],
)
def test_removed_step_fields_are_rejected(path: tuple[str, ...], field: str) -> None:
    config = load_config("tests/fixtures/runtime.yaml")
    target = config
    for section in path:
        target = target[section]
    target[field] = 1

    with pytest.raises(ValueError, match="unsupported"):
        prepare_experiment(**config)


def test_all_infra_configs_load() -> None:
    for path in sorted(Path("configs").glob("*/yam/infra/*.yaml")):
        load_config(path)


def test_rtc_rejects_default_scheduler_fields_that_it_does_not_use() -> None:
    config = load_config(Path("tests/fixtures/runtime.yaml"))
    payload = deepcopy(config)
    payload["inference"]["algorithm"] = "rtc"
    payload["inference"]["refill_threshold_s"] = 0.2

    with pytest.raises(ValueError, match="not used by RTC"):
        prepare_experiment(**payload)


def test_recording_cannot_be_silently_disabled() -> None:
    config = load_config(Path("tests/fixtures/runtime.yaml"))
    payload = deepcopy(config)
    payload["recording"]["enabled"] = False

    with pytest.raises(ValueError):
        prepare_experiment(**payload)


def test_experiment_and_video_recording_defaults_are_opt_in() -> None:
    config = load_config(Path("tests/fixtures/runtime.yaml"))

    assert not config["run"]["experiment_mode"]
    assert config["run"]["layout_id"] == ""
    assert config["recording"]["video_fps"] == 0


def test_command_safety_must_match_every_robot_group_dimension() -> None:
    config = load_config(Path("tests/fixtures/runtime.yaml"))
    payload = deepcopy(config)
    groups = payload["robot"]["group_dims"]
    payload["executor"]["command_safety"] = {
        "position_lower": {name: [-1.0] * dim for name, dim in groups.items()},
        "position_upper": {name: [1.0] * dim for name, dim in groups.items()},
        "max_velocity": {name: [1.0] * dim for name, dim in groups.items()},
        "max_acceleration": {name: [2.0] * dim for name, dim in groups.items()},
    }
    first_group = next(iter(groups))
    payload["executor"]["command_safety"]["max_velocity"][first_group].pop()

    with pytest.raises(ValueError, match="vectors must share"):
        prepare_experiment(**payload)


def test_unknown_inference_strategy_fails_before_runtime_construction(tmp_path: Path) -> None:
    config = load_config(Path("tests/fixtures/runtime.yaml"))
    config["inference"]["algorithm"] = "missing_strategy"

    with pytest.raises(PluginError, match="manimux.inference_strategies"):
        build_runtime(config, tmp_path)


def test_execution_prefix_rejects_incompatible_runtime_and_oversized_horizon():
    cfg = deepcopy(load_config("tests/fixtures/runtime.yaml"))
    cfg["inference"]["max_chunk_policy_steps"] = 25
    with pytest.raises(ValueError, match="must not exceed"):
        prepare_experiment(**cfg)
    cfg["policy"]["horizon_policy_steps"] = 50
    assert prepare_experiment(**cfg)["inference"]["max_chunk_policy_steps"] == 25
    cfg["inference"]["algorithm"] = "rtc"
    with pytest.raises(ValueError, match="ordinary ManiMux"):
        prepare_experiment(**cfg)
