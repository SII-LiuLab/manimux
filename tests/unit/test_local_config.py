"""Local bindings must change devices without changing robot/policy semantics."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from manimux.cli import load_config, load_local, read_experiment
from manimux.clock import SystemClock
from manimux.embodiments.robot import build_robot
from manimux.policies.base import action_interval
from manimux.servers.camera.server import camera_config

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "manimux/configs/experiments/pass_ball/tianji_taccap_umi_dp.yaml"
TEMPLATE = ROOT / "manimux/configs/local/tianji_taccap.example.yaml"


def write_local(tmp_path, **updates):
    data = yaml.safe_load(TEMPLATE.read_text())
    data.update(updates)
    path = tmp_path / "station.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_local_binds_same_camera_for_robot_and_server_without_opening_it(tmp_path):
    local = write_local(tmp_path)
    cfg = load_config(EXPERIMENT, local=local)
    robot = build_robot(cfg["robot"], SystemClock())
    cameras = camera_config(read_experiment(EXPERIMENT, local=local))["sensors"]["cameras"]
    try:
        assert robot.controller._robot is None
        assert all(tool._device is None for tool in robot.end_effectors.values())
        for side in ("left", "right"):
            name = f"{side}_wrist_camera"
            assert robot.sensors[name]._camera is None
            assert cameras[f"{side}_wrist"]["camera_serial"] == robot.sensors[name]._serial
        assert cfg["sensors"][0]["options"]["output_names"] == {
            "left_wrist": "left_wrist_camera",
            "right_wrist": "right_wrist_camera",
        }
    finally:
        robot.close()


def test_local_does_not_change_fk_or_execution_settings(tmp_path):
    base = load_config(EXPERIMENT)
    bound = load_config(EXPERIMENT, local=write_local(tmp_path))
    assert bound["inference"] == base["inference"]
    assert bound["executor"] == base["executor"]
    assert action_interval(bound["policy"]) == action_interval(base["policy"])
    assert bound["robot"]["control_hz"] == base["robot"]["control_hz"]
    assert bound["robot"]["options"]["execute"] is False
    assert bound["robot"]["options"]["end_effector_control"] is False
    robots = [build_robot(config["robot"], SystemClock()) for config in (base, bound)]
    q = {
        side: np.r_[np.radians([21.8, -41, -4.74, -63.67, 10.15, 14.72, 7.68]), 0.8]
        for side in base["robot"]["group_dims"]
    }
    try:
        a, b = [robot.fk(q) for robot in robots]
        for name in a:
            np.testing.assert_array_equal(a[name], b[name])
    finally:
        for robot in robots:
            robot.close()


def test_relative_local_paths_and_cli_selection_are_independent_of_cwd(tmp_path, monkeypatch):
    local = write_local(
        tmp_path,
        paths={
            "checkpoint": "weights/model",
            "norm_stats": "weights/normalize.json",
            "vlm_processor": "weights/processor",
            "output_dir": "runs",
        },
    )
    raw = read_experiment(EXPERIMENT)
    raw["local"] = "station.yaml"
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(yaml.safe_dump(raw))
    monkeypatch.chdir(ROOT / "manimux")
    cfg = load_config(experiment)
    assert cfg["local"] == local
    assert cfg["policy_server"]["checkpoint_path"] == str(tmp_path / "weights/model")
    assert cfg["policy_server"]["norm_stats_path"] == str(
        tmp_path / "weights/normalize.json"
    )
    assert cfg["policy_server"]["vlm_processor_path"] == str(tmp_path / "weights/processor")
    assert cfg["run"]["output_dir"] == tmp_path / "runs"
    other = tmp_path / "other.yaml"
    other.write_text(yaml.safe_dump({"robot": {"hardware": {"ip": "192.0.2.99"}}}))
    assert (
        load_config(experiment, local=other)["robot"]["options"]["hardware"]["ip"] == "192.0.2.99"
    )


@pytest.mark.parametrize(
    "parameters", [{"channel": "can_left"}, {"port": "/dev/ttyUSB0"}, {"ip": "192.0.2.5"}]
)
def test_component_bindings_do_not_require_ip_serial_or_separate_gripper(tmp_path, parameters):
    local = write_local(tmp_path, robot={"components": {"left_arm": parameters}})
    loaded = load_local(local)
    assert loaded["robot"].get("hardware", {}) == {}
    assert loaded["robot"]["components"] == {"left_arm": parameters}


def test_typo_in_component_name_is_not_silently_ignored(tmp_path):
    local = write_local(
        tmp_path, robot={"components": {"wrong_wrist_camera": {"camera_serial": "x"}}}
    )
    with pytest.raises(KeyError, match="wrong_wrist_camera"):
        build_robot(load_config(EXPERIMENT, local=local)["robot"], SystemClock())


def test_local_cannot_enable_execution(tmp_path):
    local = write_local(tmp_path, robot={"execute": True})
    with pytest.raises(ValueError, match="execute"):
        load_config(EXPERIMENT, local=local)


def test_remote_service_bind_addresses_are_distinct_from_client_addresses(tmp_path):
    services = {
        "policy": {"endpoint": "ws://192.0.2.30:8561", "bind_host": "0.0.0.0"},
        "camera": {
            "endpoint": "tcp://192.0.2.20:5558",
            "request_endpoint": "tcp://192.0.2.20:5557",
            "bind_endpoint": "tcp://*:5558",
            "bind_request_endpoint": "tcp://*:5557",
        },
    }
    cfg = load_config(EXPERIMENT, local=write_local(tmp_path, services=services))
    assert cfg["policy"]["options"]["server"] == services["policy"]["endpoint"]
    assert cfg["policy_server"]["host"] == "0.0.0.0" and cfg["policy_server"]["port"] == 8561
    assert cfg["sensors"][0]["options"]["endpoint"] == services["camera"]["endpoint"]
    assert cfg["camera_server"]["pub_endpoint"] == "tcp://*:5558"


def test_camera_cli_uses_shared_local_without_real_devices(tmp_path, monkeypatch):
    from manimux.servers.camera import server

    opened, events = {}, []
    local = write_local(tmp_path)

    def open_camera(name, spec, by_id_root):
        opened[name] = spec
        return SimpleNamespace(close=lambda: None)

    class FakeServer:
        def __init__(self, **kwargs):
            self.cameras = kwargs["cameras"]
            assert kwargs["pub_endpoint"] == "tcp://127.0.0.1:5556"

        def bind(self):
            events.append("bind")

        def run(self):
            events.append("run")

        def shutdown(self):
            events.append("close")

    monkeypatch.setattr(server, "CameraServer", FakeServer)
    monkeypatch.setattr(server, "_open_taccap", open_camera)
    monkeypatch.setattr(server.signal, "signal", lambda *args: None)
    assert server.main(["--experiment", str(EXPERIMENT), "--local", str(local)]) == 0
    assert events == ["bind", "run", "close"]
    assert opened["left_wrist"]["camera_serial"] == "REPLACE_LEFT_CAMERA"


def test_runtime_cli_passes_local_without_starting_runtime(tmp_path, monkeypatch):
    from manimux import cli

    local = write_local(tmp_path)
    seen = {}

    def run(path, executor, *, local):
        seen["config"] = cli._load_config(path, executor, local=local)
        return 0

    monkeypatch.setattr(cli, "_run", run)
    monkeypatch.setattr(cli.signal, "signal", lambda *args: None)
    assert cli.main(["run", "--config", str(EXPERIMENT), "--local", str(local)]) == 0
    assert seen["config"]["local"] == local


def test_checkpoint_binding_keeps_local_camera_and_remote_policy_addresses(tmp_path, monkeypatch):
    import runpy
    import sys

    from test_tianji_policy_assembly import configured

    report = dict(configured()["policy"]["expected_backend"]["model"])
    local = write_local(
        tmp_path,
        paths={"checkpoint": "weights/pass_ball"},
        services={
            "policy": {"endpoint": "ws://192.0.2.30:8561", "bind_host": "0.0.0.0"},
            "camera": {
                "endpoint": "tcp://127.0.0.1:5556",
                "request_endpoint": "tcp://127.0.0.1:5555",
            },
        },
    )
    checkpoint = str(tmp_path / "weights/pass_ball")
    report["checkpoint_path"] = checkpoint

    def validate(config):
        assert config["checkpoint_path"] == checkpoint
        assert config["host"] == "0.0.0.0" and config["port"] == 8561
        return report

    monkeypatch.setitem(
        sys.modules,
        "XPolicyLab.policy.UMI_DP.artifact_identity",
        SimpleNamespace(validate_deployment=validate),
    )
    monkeypatch.setattr(sys, "path", list(sys.path))
    output = tmp_path / "nested/deployment/run.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "umi_dp_tianji_server.py",
            "--experiment",
            str(EXPERIMENT),
            "--local",
            str(local),
            "--bind-runtime-config",
            str(output),
        ],
    )
    runpy.run_path(str(ROOT / "manimux/servers/umi_dp.py"), run_name="__main__")
    cfg = load_config(output)
    assert cfg["local"] == local
    assert cfg["robot"]["config"] == ROOT / "manimux/configs/embodiment/robot/tianji_taccap.yaml"
    assert cfg["policy"]["options"]["server"] == "ws://192.0.2.30:8561"
    assert cfg["policy"]["expected_backend"]["model"] == report
    written = yaml.safe_load(output.read_text())
    assert "component_hardware" not in written["robot"]["options"]
    assert "ip" not in written["robot"]["options"].get("hardware", {})
    assert "endpoint" not in written["sensors"][0]["options"]
    assert "server" not in written["policy"]["options"]
    cameras = camera_config(read_experiment(output))
    assert cameras["sensors"]["cameras"]["right_wrist"]["camera_serial"] == "REPLACE_RIGHT_CAMERA"
    server = yaml.safe_load(output.with_name("run-server.yaml").read_text())
    assert server["expected_artifacts"] == report
    assert server["checkpoint_path"] == checkpoint
    assert server["host"] == "0.0.0.0" and server["port"] == 8561
    # 绑定文件生成后更换工位，相机和 runtime 都读取最新 serial。
    bindings = yaml.safe_load(local.read_text())
    bindings["robot"]["components"]["right_wrist_camera"]["camera_serial"] = "NEW_RIGHT_CAMERA"
    local.write_text(yaml.safe_dump(bindings))
    updated = load_config(output)
    assert updated["robot"]["options"]["component_hardware"]["right_wrist_camera"] == {
        "camera_serial": "NEW_RIGHT_CAMERA"
    }
    assert (
        camera_config(read_experiment(output))["sensors"]["cameras"]["right_wrist"]["camera_serial"]
        == "NEW_RIGHT_CAMERA"
    )
