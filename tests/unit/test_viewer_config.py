import json
import sys

import pytest
import yaml

from manimux.viewer import dashboard
from manimux.viewer.dashboard import load_robot_view, load_viewer_config


def _write_config(tmp_path, **options):
    # 通用 Viewer 配置测试使用随包发布的 YAM 模型，不依赖私有天机 SDK。
    config = load_viewer_config(robot="yam")
    # 加载后的模型和场景资源都是 Path；写回自定义 YAML 时一起转成字符串。
    config = json.loads(json.dumps(config, default=str))
    config.update(options)
    path = tmp_path / "viewer.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_tianji_defaults_have_only_the_two_wrist_cameras():
    config = load_viewer_config()
    assert config["camera_mode"] == "policy"
    assert [c["source"] for c in config["cameras"]] == ["left_wrist", "right_wrist"]
    assert [c["slot"] for c in config["cameras"]] == ["left", "right"]
    assert config["model"].name == "tianji_taccap.yaml"


def test_model_path_is_relative_to_selected_yaml(tmp_path):
    model = load_viewer_config()["model"]
    path = tmp_path / "viewer.yaml"
    import os

    path.write_text(yaml.safe_dump({"model": os.path.relpath(model, tmp_path), "cameras": []}))
    config = load_viewer_config(path)
    assert config["model"] == model
    assert config["cameras"] == []


def test_camera_list_can_add_agent_view_without_changing_robot(tmp_path):
    cameras = [
        {"source": "agent_view", "label": "External", "slot": "external"},
        *load_viewer_config(robot="yam")["cameras"],
    ]
    config = load_viewer_config(_write_config(tmp_path, cameras=cameras, camera_mode="manual"))
    robot = load_robot_view(config)
    assert [c["source"] for c in config["cameras"]] == [
        "agent_view",
        "front_camera",
        "left_camera",
        "right_camera",
    ]
    assert set(robot.model.groups) == {"left_arm", "right_arm"}


@pytest.mark.parametrize(
    "options",
    [
        {"camera_mode": "guess"},
        {"cameras": {"top": "camera"}},
        {"cameras": [{}]},
        {"cameras": [{"source": ""}]},
        {"cameras": [{"source": "a", "slot": "top"}, {"source": "b", "slot": "top"}]},
    ],
)
def test_invalid_config_fails_before_viewer_startup(tmp_path, options):
    with pytest.raises(ValueError):
        load_viewer_config(_write_config(tmp_path, **options))


def test_startup_uses_selected_model_scene_and_cameras(monkeypatch, tmp_path):
    config = _write_config(
        tmp_path,
        camera_mode="manual",
        cameras=[{"source": "agent_view", "label": "External", "slot": "top"}],
    )
    monkeypatch.setattr(sys, "argv", ["manimux-viewer", "--config", str(config)])
    selected = {}

    class StopBeforeServing(Exception):
        pass

    def capture(*args, **kwargs):
        selected["robot"] = args[4]
        selected["config"] = kwargs["viewer_config"]
        raise StopBeforeServing

    monkeypatch.setattr(dashboard, "PolicyViewer", capture)
    with pytest.raises(StopBeforeServing):
        dashboard.main()
    assert selected["robot"].name == "yam"
    assert selected["config"]["camera_mode"] == "manual"
    assert selected["config"]["cameras"][0]["source"] == "agent_view"
    assert selected["robot"].scene_boxes[0].position == (0.38, 0, 0.02)
