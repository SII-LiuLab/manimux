"""Deployment entry points share one station without opening devices or model weights."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from manimux import cli
from manimux.servers import pi05
from manimux.servers.camera import server as camera

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "manimux/configs/experiments/put_bottles/pi05/yam_pi05_rtc_joint_step30000.yaml"
SERVER = ROOT / "manimux/configs/policy/pi05/yam/put-bottles/joint-step30000.yaml"


@pytest.fixture
def station(tmp_path, monkeypatch):
    data = yaml.safe_load((ROOT / "manimux/configs/local/yam.example.yaml").read_text())
    data["robot"]["components"]["left_yam"]["channel"] = "can_test_left"
    data["robot"]["components"]["right_yam"]["channel"] = "can_test_right"
    data["robot"]["components"]["front_camera"]["camera_serial"] = "TEST_FRONT_CAMERA"
    data["services"]["policy"] = {
        "endpoint": "ws://192.0.2.20:18500", "bind_host": "0.0.0.0",
    }
    data["services"]["camera"] = {
        "endpoint": "tcp://127.0.0.1:15555",
        "request_endpoint": "tcp://127.0.0.1:15555",
        "bind_endpoint": "tcp://127.0.0.1:15556",
        "bind_request_endpoint": "tcp://127.0.0.1:15555",
    }
    data["paths"] = {"checkpoints": "weights"}
    path = tmp_path / "station.yaml"
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setattr(cli, "DEFAULT_LOCAL_PATH", path)
    return path


def test_station_selection_is_explicit_then_saved_then_default(station, tmp_path, monkeypatch):
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text("local: saved.yaml\n")
    monkeypatch.chdir(tmp_path.parent)
    assert cli.resolve_local_path() == station
    assert cli.resolve_local_path(experiment) == tmp_path / "saved.yaml"
    assert cli.resolve_local_path(experiment, station) == station
    assert cli.resolve_local_path(EXPERIMENT) == station


def test_deployment_requires_station_but_offline_reading_does_not(tmp_path, monkeypatch):
    missing = tmp_path / "missing.yaml"
    monkeypatch.setattr(cli, "DEFAULT_LOCAL_PATH", missing)
    assert cli.load_config(EXPERIMENT)["robot"]["control_hz"] == 30.0
    with pytest.raises(FileNotFoundError) as exc:
        cli._load_config(EXPERIMENT)
    assert exc.value.filename == str(missing)


def test_runtime_and_camera_entrypoints_read_same_default_station(station, monkeypatch):
    seen = {}

    def run(path, executor, *, local):
        seen["runtime"] = cli._load_config(path, executor, local=local)
        return 0

    class CameraServer:
        def __init__(self, **options):
            seen["listener"] = options

        def bind(self):
            pass

        def run(self):
            seen["cameras"] = self.cameras

        def shutdown(self):
            pass

    monkeypatch.setattr(cli, "_run", run)
    monkeypatch.setattr(cli.signal, "signal", lambda *args: None)
    monkeypatch.setattr(camera, "CameraServer", CameraServer)
    monkeypatch.setattr(camera, "_build_cameras", lambda config: config["sensors"]["cameras"])
    assert cli.main(["run", "--config", str(EXPERIMENT)]) == 0
    assert camera.main(["--experiment", str(EXPERIMENT)]) == 0

    config = seen["runtime"]
    devices = config["robot"]["options"]["component_hardware"]
    assert devices["left_yam"]["channel"] == "can_test_left"
    assert devices["right_yam"]["channel"] == "can_test_right"
    assert seen["cameras"]["front_camera"]["camera_serial"] == "TEST_FRONT_CAMERA"
    assert config["sensors"][0]["options"]["endpoint"] == seen["listener"]["rep_endpoint"]
    assert seen["listener"]["pub_endpoint"] == "tcp://127.0.0.1:15556"
    assert config["robot"]["control_hz"] == 30.0
    assert config["policy"]["action_dt_s"] == pytest.approx(1 / 30)
    assert config["robot"]["options"]["execute"] is True


@pytest.mark.parametrize("source,path", [("--experiment", EXPERIMENT), ("--config", SERVER)])
def test_pi05_server_uses_same_station_without_loading_model(station, monkeypatch, source, path):
    captured = {}
    monkeypatch.setattr(
        pi05, "_validate_paths",
        lambda config: (Path(config["model_path"]), Path(config["norm_stats_path"])),
    )
    monkeypatch.setattr(pi05, "_resolved_contract", lambda *args: {})
    monkeypatch.setitem(
        sys.modules, "setup_policy_server",
        SimpleNamespace(main=lambda config: captured.update(config)),
    )
    assert pi05.main([source, str(path)]) == 0
    runtime = cli._load_config(EXPERIMENT)
    assert captured == runtime["policy_server"]
    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 18500
    assert runtime["policy"]["options"]["server"] == "ws://192.0.2.20:18500"
    model = runtime["policy"]["expected_backend"]["model"]
    assert model["model_root"] == captured["model_path"]
    assert model["norm_stats_path"] == captured["norm_stats_path"]
    assert Path(captured["model_path"]).is_relative_to(station.parent / "weights")
    assert captured["checkpoint_variant"] == "pi05_yam_put_bottles_joint_step_30000"


def test_station_root_preserves_checkpoint_selection_when_experiment_changes(station):
    joint = cli._load_config(EXPERIMENT)
    ee = cli._load_config(EXPERIMENT.with_name("yam_pi05_rtc_joint_ee_step30000.yaml"))
    assert joint["policy_server"]["model_path"] != ee["policy_server"]["model_path"]
    assert joint["policy_server"]["train_config_name"] == "pi05_yam"
    assert ee["policy_server"]["train_config_name"] == "pi05_yam_joint_ee"
    assert joint["policy_server"]["checkpoint_variant"] != ee["policy_server"]["checkpoint_variant"]
    assert Path(ee["policy_server"]["model_path"]).is_relative_to(station.parent / "weights")
