"""Configured assembly, official FK/IK and exported URDF agree without hardware."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation
from yourdfpy import URDF

from manimux.embodiments.robot import RobotModel
from manimux.embodiments.robot.tianji_taccap import TianjiTaccapRobot

CONFIG = Path(__file__).resolve().parents[2] / "manimux/configs/embodiment/robot/tianji_taccap.yaml"


def configuration(model):
    return {
        name: np.r_[np.radians([21.8, -41, -4.74, -63.67, 10.15, 14.72, 7.68]), 0.5]
        for name in model.groups
    }


def edited_config(tmp_path, edit):
    spec = yaml.safe_load(CONFIG.read_text())
    for component in spec["components"].values():
        component["config"] = str((CONFIG.parent / component["config"]).resolve())
    edit(spec)
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return path


@pytest.mark.parametrize("modified", [False, True])
def test_mounts_and_tcp_are_shared_by_official_fk_ik_and_urdf(tmp_path, modified):
    def edit(spec):
        if modified:
            for name in ("left_end_effector", "right_end_effector"):
                spec["components"][name]["mount"] = {
                    "xyz": [0.03, -0.01, 0.07],
                    "rpy": [0.1, 0.4, -0.2],
                }
                tcp = np.eye(4)
                tcp[:3, :3] = Rotation.from_euler("xyz", [0.3, 0.1, -0.2]).as_matrix()
                tcp[:3, 3] = [0.1, -0.01, 0.03]
                spec["components"][name]["options"] = {"tcp_transform": tcp.tolist()}

    model = RobotModel.from_config(edited_config(tmp_path, edit))
    q = configuration(model)
    targets = model.kinematics.fk(q)
    results = model.kinematics.ik(
        targets, q, fixed_coordinates={name: {"gripper": 0.5} for name in q}
    )
    for name, mounted in model.groups.items():
        assert results[name].converged, results[name].reason
        assert mounted.kinematics.base_frame == mounted.arm.base_frame
        expected = (
            mounted.arm.kinematics.fk_flange(q[name][:7])
            @ mounted.mount.matrix()
            @ mounted.end_effector.geometry.tcp_transform(q[name][7:])
        )
        np.testing.assert_allclose(targets[name], expected, atol=1e-10)
        urdf = URDF.load(mounted.visual_urdf(), load_meshes=False)
        urdf.update_cfg(mounted.visual_configuration(q[name]))
        visual_tcp = urdf.get_transform("ee_tcp")
        np.testing.assert_allclose(visual_tcp, expected, atol=1e-4)


def test_missing_hardware_is_explicit_but_does_not_block_offline_model():
    model = RobotModel.from_config(CONFIG)
    assert model.components["left_wrist_camera"]["mount"] is None
    robot = TianjiTaccapRobot.from_config(CONFIG)
    assert robot.controller._robot is None
    q = configuration(model)
    for name, pose in robot.fk(q).items():
        np.testing.assert_allclose(pose, model.kinematics.fk(q)[name])
    with pytest.raises(ValueError):
        robot.connect()
    robot.close()


def test_hardware_construction_reuses_models_and_is_inert():
    robot = TianjiTaccapRobot.from_config(
        CONFIG,
        hardware={"ip": "192.0.2.1"},
        component_hardware={
            "left_end_effector": {"serial": "TEST_LEFT", "kp": 8.0, "kd": 0.3},
            "right_end_effector": {"serial": "TEST_RIGHT", "kp": 8.0, "kd": 0.3},
            "left_wrist_camera": {"camera_serial": "TEST_LEFT_CAMERA"},
            "right_wrist_camera": {"camera_serial": "TEST_RIGHT_CAMERA"},
        },
    )
    try:
        assert robot.kinematics is robot.model.kinematics
        assert robot.controller._robot is None and not robot.controller._owns_session
        assert list(robot.arm_components) == ["left_arm", "right_arm"]
        for name, arm in robot.arm_components.items():
            assert arm.kinematics is robot.model.groups[name].arm.kinematics
            assert arm.controller is robot.controller
        assert all(sensor._camera is None for sensor in robot.sensors.values())
        assert all(tool._device is None for tool in robot.end_effectors.values())
        with pytest.raises(RuntimeError, match="not connected"):
            robot.get_state()
    finally:
        robot.close()


@pytest.mark.parametrize("case", ["parent", "duplicate", "unused", "nonfinite"])
def test_invalid_assembly_is_rejected(tmp_path, case):
    def edit(spec):
        if case == "parent":
            spec["components"]["left_end_effector"]["parent"] = "right_arm.flange"
        elif case == "duplicate":
            spec["groups"]["right_arm"]["arm"] = "left_arm"
        elif case == "unused":
            del spec["groups"]["right_arm"]
        elif case == "nonfinite":
            spec["components"]["left_end_effector"]["mount"]["xyz"][0] = float("nan")

    with pytest.raises(ValueError):
        RobotModel.from_config(edited_config(tmp_path, edit))


def test_robot_model_uses_configured_group_layout(tmp_path):
    def edit(spec):
        spec["groups"] = {"manipulator": spec["groups"]["right_arm"]}
        for name in ("left_arm", "left_end_effector", "left_wrist_camera"):
            del spec["components"][name]

    model = RobotModel.from_config(edited_config(tmp_path, edit))
    assert list(model.groups) == ["manipulator"]
    q = configuration(model)
    assert model.kinematics.fk(q)["manipulator"].shape == (4, 4)
    assert model.groups["manipulator"].visual_configuration(q["manipulator"]).shape == (8,)
    with pytest.raises(ValueError):
        model.groups["manipulator"].visual_configuration(np.zeros(16))


def test_scene_placement_does_not_change_control_fk_or_ik():
    from manimux.viewer.dashboard import load_robot_view, load_viewer_config

    config = load_viewer_config(robot="tianji")
    original_view = load_robot_view(config)
    relocated = deepcopy(config)
    for style in relocated["groups"].values():
        style["viewer_display_frame"] = {
            "xyz": [3, -4, 5],
            "rpy": [0.3, -0.4, 0.7],
        }
    moved_view = load_robot_view(relocated)
    original, moved = original_view.model, moved_view.model
    q = configuration(original)
    targets = original.kinematics.fk(q)
    before = original.kinematics.ik(
        targets,
        q,
        fixed_coordinates={n: {"gripper": 0.5} for n in q},
    )
    after = moved.kinematics.ik(
        targets,
        q,
        fixed_coordinates={n: {"gripper": 0.5} for n in q},
    )
    for name in q:
        np.testing.assert_allclose(moved.kinematics.fk(q)[name], targets[name], atol=1e-12)
        assert before[name].converged and after[name].converged
        np.testing.assert_allclose(before[name].joints, after[name].joints, atol=1e-12)
        assert not np.allclose(
            original_view.group(name).base_position, moved_view.group(name).base_position
        )
