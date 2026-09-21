"""Tianji station wiring and UMI exports, without loading the SDK or a model."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from manimux import cli
from manimux.servers import umi_dp
from manimux.servers.camera.server import camera_config

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "manimux/configs/experiments/pass_ball/tianji_taccap_umi_dp.yaml"
TEMPLATE = ROOT / "manimux/configs/local/tianji_taccap.example.yaml"


def write_station(tmp_path):
    station = yaml.safe_load(TEMPLATE.read_text())
    station["robot"]["hardware"]["ip"] = "192.0.2.10"
    station["services"]["policy"] = {
        "endpoint": "ws://192.0.2.30:8561",
        "bind_host": "0.0.0.0",
    }
    station["services"]["camera"] = {
        "endpoint": "tcp://192.0.2.20:5558",
        "request_endpoint": "tcp://192.0.2.20:5557",
        "bind_endpoint": "tcp://*:5558",
        "bind_request_endpoint": "tcp://*:5557",
    }
    station["paths"] = {"checkpoint": "weights/pass_ball"}
    path = tmp_path / "station.yaml"
    path.write_text(yaml.safe_dump(station))
    return path


@pytest.mark.parametrize(
    "filename",
    [
        "tianji_taccap_umi_dp.yaml",
        "tianji_taccap_umi_dp_diff.yaml",
        "tianji_umi_dp_default.yaml",
        "tianji_umi_dp_rtc.yaml",
    ],
)
def test_tianji_station_binds_controller_grippers_and_camera_streams(tmp_path, filename):
    station = write_station(tmp_path)
    experiment = EXPERIMENT.with_name(filename)
    bound = cli.load_config(experiment, local=station)
    options = bound["robot"]["options"]
    assert options["hardware"]["ip"] == "192.0.2.10"
    assert options["execute"] is False
    assert options["end_effector_control"] is False
    components = options["component_hardware"]
    cameras = camera_config(bound)["sensors"]["cameras"]
    for side in ("left", "right"):
        assert components[f"{side}_end_effector"]["serial"] == f"REPLACE_{side.upper()}_GRIPPER"
        assert cameras[f"{side}_wrist"]["camera_serial"] == (
            components[f"{side}_wrist_camera"]["camera_serial"]
        )
    assert bound["sensors"][0]["options"]["endpoint"] == "tcp://192.0.2.20:5558"
    assert bound["camera_server"]["pub_endpoint"] == "tcp://*:5558"
    assert bound["camera_server"]["rep_endpoint"] == "tcp://*:5557"
    assert bound["policy_server"]["checkpoint_path"] == str(tmp_path / "weights/pass_ball")
    assert bound["policy"]["options"]["server"] == "ws://192.0.2.30:8561"
    assert bound["policy_server"]["host"] == "0.0.0.0"
    assert bound["policy_server"]["port"] == 8561
    if filename.startswith("tianji_umi_dp_"):
        # Existing model input names stay unchanged when device bindings move to local.
        assert bound["policy"]["adapter"]["camera_map"] == {
            "cam_left_wrist": "left_wrist",
            "cam_right_wrist": "right_wrist",
            "cam_left_wrist_prev": "left_wrist_prev",
            "cam_right_wrist_prev": "right_wrist_prev",
        }
        assert bound["robot"]["control_hz"] == 100.0
        assert bound["policy"]["action_dt_s"] == 1 / 30


@pytest.mark.parametrize("selection", ["default", "experiment_reference"])
def test_umi_export_reuses_automatically_selected_station(tmp_path, monkeypatch, selection):
    station = write_station(tmp_path)
    monkeypatch.setattr(cli, "DEFAULT_LOCAL_PATH", station)
    experiment_path = EXPERIMENT
    if selection == "experiment_reference":
        raw = cli.read_experiment(EXPERIMENT, bind_local=False)
        raw["local"] = "station.yaml"
        experiment_path = tmp_path / "experiment.yaml"
        experiment_path.write_text(yaml.safe_dump(raw))
        monkeypatch.setattr(cli, "DEFAULT_LOCAL_PATH", tmp_path / "unused.yaml")

    # Artifact identity is supplied by a fake provider: no model or hardware is loaded.
    report = {
        "policy_family": "umi_dp",
        "observation_profile": "tianji_umi_relative2",
        "action_semantics": "absolute_per_arm_base_xyz_wxyz",
        "checkpoint_sha256": "offline-test",
        "training_config_sha256": "offline-test",
        "checkpoint_path": str(tmp_path / "weights/pass_ball"),
        "weight_key": "ema",
        "rgb_normalize": True,
        "action_horizon": 16,
        "action_dt_s": 1 / 30,
        "first_action_offset_s": 1 / 30,
        "observation_period_s": 0.1,
    }

    def validate(config):
        assert config["checkpoint_path"] == report["checkpoint_path"]
        assert config["host"] == "0.0.0.0"
        assert config["port"] == 8561
        return report

    monkeypatch.setitem(
        sys.modules,
        "XPolicyLab.policy.UMI_DP.artifact_identity",
        SimpleNamespace(validate_deployment=validate),
    )
    monkeypatch.setattr(sys, "path", list(sys.path))
    output = tmp_path / "exported/run.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "umi_dp",
            "--experiment",
            str(experiment_path),
            "--bind-runtime-config",
            str(output),
        ],
    )
    umi_dp.main()

    written = yaml.safe_load(output.read_text())
    assert written["local"] == str(station)
    assert "component_hardware" not in written["robot"]["options"]
    assert "ip" not in written["robot"]["options"].get("hardware", {})
    assert "endpoint" not in written["sensors"][0]["options"]
    assert "server" not in written["policy"]["options"]
    snapshot_path = output.with_name("run-server.yaml")
    snapshot = yaml.safe_load(snapshot_path.read_text())
    assert snapshot["host"] == "0.0.0.0"
    assert snapshot["port"] == 8561
    assert snapshot["expected_artifacts"] == report

    # The runtime and --experiment server reread bindings; --config uses the saved snapshot.
    updated = yaml.safe_load(station.read_text())
    updated["robot"]["hardware"]["ip"] = "192.0.2.11"
    updated["robot"]["components"]["right_wrist_camera"]["camera_serial"] = "NEW_RIGHT_CAMERA"
    updated["services"]["policy"]["endpoint"] = "ws://192.0.2.31:8562"
    station.write_text(yaml.safe_dump(updated))
    bound = cli.read_experiment(output)
    assert bound["robot"]["options"]["hardware"]["ip"] == "192.0.2.11"
    assert camera_config(bound)["sensors"]["cameras"]["right_wrist"]["camera_serial"] == (
        "NEW_RIGHT_CAMERA"
    )
    assert bound["policy"]["options"]["server"] == "ws://192.0.2.31:8562"
    assert bound["policy_server"]["port"] == 8562
    assert yaml.safe_load(snapshot_path.read_text())["port"] == 8561
