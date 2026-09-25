"""一体 YAM 的装配、控制和显示契约；所有硬件调用使用假 SDK。"""

import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from yourdfpy import URDF

from manimux.cli import load_config
from manimux.clock import SystemClock
from manimux.embodiments.arm.base import ArmModel
from manimux.embodiments.arm.yam.kinematics import YamKinematics
from manimux.embodiments.robot import RobotModel, build_robot
from manimux.embodiments.robot.yam import YamRobot
from manimux.kinematics import (
    FixedToolGeometry,
    FlangeKinematicsBase,
    IKResult,
    KinematicCoordinate,
)
from manimux.policy_adapter import build_policy_adapter
from manimux.types import ActionContext, RobotCommand
from manimux.viewer.dashboard import load_robot_view, load_viewer_config
from manimux.viewer.robots.yam import YamAdapter

ROOT = Path(__file__).resolve().parents[2]
ASSEMBLY = ROOT / "manimux/configs/embodiment/robot/yam_dual.yaml"
EXPERIMENT = ROOT / "manimux/configs/experiments/put_bottles/pi05/yam_pi05_joint.yaml"
START = np.array([-0.6094, 0.5835, 0.8425, -1.0168, -0.1108, -0.4580, 0.65])


@pytest.fixture
def fake_sdk(monkeypatch):
    devices = {}
    events = []

    class Device:
        def __init__(self, channel, **options):
            self.q = START.copy()
            self.commands = []
            self.channel = channel
            self.options = options
            self.closed = 0
            self._stop_event = SimpleNamespace(set=lambda: None)
            self._server_thread = SimpleNamespace(join=lambda **kw: None, is_alive=lambda: False)
            self.motor_chain = SimpleNamespace(running=True, close=self.close)
            devices[channel] = self
            events.append(("connect", channel))

        def get_joint_pos(self):
            events.append(("read", self.channel))
            return self.q.copy()

        def command_joint_pos(self, q):
            self.commands.append(q.copy())
            self.q = q.copy()

        def move_joints(self, q, time_interval_s):
            events.append(("move", self.channel, q.copy(), time_interval_s))
            self.q = q.copy()

        def close(self):
            self.closed += 1

    monkeypatch.setitem(sys.modules, "i2rt.robots.get_robot", SimpleNamespace(get_yam_robot=Device))
    enum = SimpleNamespace(from_string_name=lambda name: name)
    monkeypatch.setitem(
        sys.modules, "i2rt.robots.utils", SimpleNamespace(ArmType=enum, GripperType=enum)
    )
    return devices, events


def make_robot(**options):
    return YamRobot.from_config(
        ASSEMBLY,
        component_hardware={"left_yam": {"channel": "left"}, "right_yam": {"channel": "right"}},
        **options,
    )


def test_null_keeps_integrated_tcp_layout_and_model_identity(fake_sdk):
    robot = make_robot()
    assert fake_sdk[0] == {}
    assert not robot.end_effectors
    reference = YamKinematics()
    for name, group in robot.model.groups.items():
        assert group.end_effector is None
        assert group.kinematics is group.arm.kinematics is robot.arm_components[name].kinematics
        assert group.kinematics.num_coordinates == 7
        assert group.kinematics.coordinates[-1] == KinematicCoordinate("gripper", "normalized")
        assert group.kinematics.tcp_frame == "grasp_site"
        # 左右 Viewer 安装位置不同，控制位姿仍与原来的各臂基座 FK 一致。
        np.testing.assert_allclose(group.kinematics.fk(START), reference.pose(START), atol=1e-12)
    robot.close()
    assert fake_sdk[0] == {}


def test_integrated_read_and_write_use_seven_values_once_per_group(fake_sdk):
    devices, events = fake_sdk
    robot = make_robot(execute=True)
    robot.connect()
    robot.connect()
    assert len([event for event in events if event[0] == "connect"]) == 2
    assert all(not device.commands for device in devices.values())
    before = len(events)
    state = robot.get_state()
    assert [event[0] for event in events[before:]] == ["read", "read"]
    targets = {"left_arm": START + 0.02, "right_arm": START - 0.03}
    robot.send_command(RobotCommand(targets, monotonic_ns=state.monotonic_ns, plan_id="test"))
    for name, channel in [("left_arm", "left"), ("right_arm", "right")]:
        assert len(devices[channel].commands) == 1
        np.testing.assert_array_equal(devices[channel].commands[0], targets[name])
    robot.stop()
    for device in devices.values():
        np.testing.assert_array_equal(device.commands[-1], device.q)
    robot.close()
    robot.close()
    assert all(device.closed == 1 for device in devices.values())


def test_home_keeps_gripper_until_both_arm_moves_finish(fake_sdk):
    robot = make_robot(execute=True, home_duration_s=2.0, home_gripper_release_duration_s=0.4)
    robot.connect()
    robot.home()
    moves = [event for event in fake_sdk[1] if event[0] == "move"]
    assert len(moves) == 4
    for _, _, q, duration in moves[:2]:
        np.testing.assert_array_equal(q, np.r_[np.zeros(6), START[6]])
        assert duration == 2.0
    for _, _, q, duration in moves[2:]:
        np.testing.assert_array_equal(q, np.r_[np.zeros(6), 1.0])
        assert duration == 0.4
    robot.close()


def test_execute_false_skips_start_home_and_target_dispatch(fake_sdk):
    robot = make_robot(execute=False, move_to_start_on_connect=True)
    robot.connect()
    robot.home()
    robot.send_command(RobotCommand({name: START for name in robot.model.groups}, 1, "test"))
    assert not any(event[0] == "move" for event in fake_sdk[1])
    assert all(not device.commands for device in fake_sdk[0].values())
    robot.close()


def test_start_pose_is_explicit_and_not_repeated_on_connect(fake_sdk):
    targets = {"left_arm": START + 0.01, "right_arm": START - 0.01}
    robot = make_robot(
        execute=True, move_to_start_on_connect=True, start_joints=targets, start_duration_s=1.5
    )
    robot.connect()
    robot.connect()
    assert len([event for event in fake_sdk[1] if event[0] == "move"]) == 2
    for name, q in robot.get_state().groups.items():
        np.testing.assert_array_equal(q, targets[name])
    robot.close()


@pytest.mark.parametrize("opening", [0.0, 0.4, 1.0])
def test_viewer_uses_same_pose_and_finger_mapping_as_legacy(opening):
    view = load_robot_view(load_viewer_config(robot="yam"))
    legacy = YamAdapter()
    q = START.copy()
    q[6] = opening
    for group, old_name in [("left_arm", "left"), ("right_arm", "right")]:
        np.testing.assert_allclose(view.pose(group, q), legacy.pose(old_name, q), atol=1e-12)
        visual = view.visual_configuration(group, q)
        np.testing.assert_array_equal(visual, legacy.visual_configuration(old_name, q))
        urdf = URDF.load(view.group(group).urdf_path, load_meshes=False)
        urdf.update_cfg(visual)
        assert len(urdf.actuated_joint_names) == 8


def test_ik_passes_target_gripper_before_solving_and_keeps_seed(monkeypatch):
    model = RobotModel.from_config(ASSEMBLY).groups["left_arm"].kinematics
    target = model.fk(START)
    seed = START.copy()
    calls = []

    def solve(pose, arm_seed, gripper):
        calls.append((pose, arm_seed.copy(), gripper))
        return True, arm_seed + 0.01

    monkeypatch.setattr(model.solver, "ik", solve)
    result = model.ik(target, seed, fixed_coordinates={"gripper": 0.2})
    assert result.converged
    np.testing.assert_array_equal(result.joints, np.r_[START[:6] + 0.01, 0.2])
    np.testing.assert_array_equal(seed, START)
    assert calls[0][2] == 0.2
    monkeypatch.setattr(model.solver, "ik", lambda *args: (False, START[:6]))
    result = model.ik(target, seed, fixed_coordinates={"gripper": 0.2})
    assert not result.converged and result.joints is None


def test_real_ik_round_trip_for_integrated_model():
    pytest.importorskip("mink")
    pytest.importorskip("i2rt")
    model = RobotModel.from_config(ASSEMBLY).groups["left_arm"].kinematics
    target = model.fk(START)
    seed = START + np.r_[np.full(6, 0.03), 0]
    result = model.ik(target, seed, fixed_coordinates={"gripper": START[6]})
    assert result.converged
    assert result.joints[6] == START[6]
    np.testing.assert_allclose(model.fk(result.joints), target, atol=1e-3)


def test_viewer_display_frame_does_not_change_control_fk_or_ik():
    pytest.importorskip("mink")
    pytest.importorskip("i2rt")
    config = load_viewer_config(robot="yam")
    original = load_robot_view(config)
    relocated = deepcopy(config)
    for style in relocated["groups"].values():
        style["viewer_display_frame"] = {"xyz": [3, -4, 5], "rpy": [0.3, -0.4, 0.7]}
    moved = load_robot_view(relocated)
    seed = START + np.r_[np.full(6, 0.03), 0]
    for name in original.model.groups:
        before = original.model.groups[name].kinematics
        after = moved.model.groups[name].kinematics
        target = before.fk(START)
        np.testing.assert_array_equal(after.fk(START), target)
        np.testing.assert_array_equal(moved.pose(name, START), original.pose(name, START))
        result_before = before.ik(target, seed, fixed_coordinates={"gripper": START[6]})
        result_after = after.ik(target, seed, fixed_coordinates={"gripper": START[6]})
        assert result_before.converged and result_after.converged
        np.testing.assert_array_equal(result_after.joints, result_before.joints)
        np.testing.assert_array_equal(moved.group(name).base_position, [3, -4, 5])
        assert moved.group(name).base_orientation != original.group(name).base_orientation


def test_viewer_scene_mesh_paths_and_optional_display_frame(tmp_path):
    # Scene resources resolve beside the Viewer YAML or inside their named package.
    path = tmp_path / "viewer.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model": str(ASSEMBLY),
                "scene": {
                    "meshes": {
                        "relative": {"urdf": "stand.urdf"},
                        "packaged": {
                            "urdf": "package://manimux.embodiments.arm.yam/assets/i2rt/robot_models/arm/yam/yam.urdf",
                            "viewer_display_frame": {"xyz": [0, 0, 1]},
                        },
                    }
                },
            }
        )
    )
    view = load_robot_view(load_viewer_config(path))
    meshes = {mesh.name: mesh for mesh in view.static_meshes}
    relative, packaged = meshes["relative"], meshes["packaged"]
    assert relative.urdf_path == tmp_path / "stand.urdf"
    assert relative.position == (0, 0, 0)
    assert packaged.urdf_path.is_file()
    assert packaged.position == (0, 0, 1)
    # A model can be viewed at the origin without adding display fields to its assembly.
    assert all(group.base_position == (0, 0, 0) for group in view.groups)


def test_new_recipe_preserves_joint_policy_and_execution_contract(fake_sdk):
    new = load_config(EXPERIMENT, local=ROOT / "manimux/configs/local/yam.example.yaml")
    old = load_config(
        ROOT / "manimux/configs/experiments/put_bottles/pi05/yam_pi05_rtc_joint_step30000.yaml"
    )
    assert new["inference"] == old["inference"]
    assert new["executor"] == old["executor"]
    assert new["robot"]["group_dims"] == old["robot"]["group_dims"]
    # 模型动作契约共用；各实验可独立选择下发频率，不要求示例与实跑相同。
    from manimux.cli import read_yaml

    assert new["robot"]["control_hz"] == read_yaml(EXPERIMENT)["robot"]["control_hz"]
    assert new["policy"]["action_dt_s"] == old["policy"]["action_dt_s"]
    assert new["policy"]["horizon_policy_steps"] == old["policy"]["horizon_policy_steps"]
    robot = build_robot(new["robot"], SystemClock())
    assert not fake_sdk[0]
    adapter = build_policy_adapter(new["robot"], new["policy"], kinematics=robot.kinematics)
    adapter.validate(new["robot"], new["policy"])
    steps = [
        {
            "left_arm_joint_state": START[:6],
            "left_ee_joint_state": [0.4],
            "right_arm_joint_state": START[:6] + 0.01,
            "right_ee_joint_state": [0.7],
        }
        for _ in range(new["policy"]["horizon_policy_steps"])
    ]
    chunk = adapter.decode_action(
        steps,
        ActionContext(
            request_seq=1,
            observation_time_ns=10,
            created_time_ns=20,
        ),
    )
    assert chunk.dt_ns == int(1e9 / 30)
    assert chunk.groups["left_arm"].shape == (50, 7)
    np.testing.assert_allclose(chunk.groups["left_arm"][0], np.r_[START[:6], 0.4])
    np.testing.assert_allclose(chunk.groups["right_arm"][0], np.r_[START[:6] + 0.01, 0.7])
    robot.close()


@pytest.mark.parametrize("with_tool", [False, True])
def test_bare_flange_and_separate_tool_still_compose(tmp_path, monkeypatch, with_tool):
    # 不依赖缺失的 Tianji 私有 SDK，直接覆盖公共装配层的两条已有组合路径。
    from manimux.embodiments.robot import base

    class Solver(FlangeKinematicsBase):
        num_arm_joints = 1

        def fk_flange(self, joints):
            pose = np.eye(4)
            pose[0, 3] = joints[0]
            return pose

        def ik_flange(self, target, seed):
            return IKResult(True, np.array([target[0, 3]]))

    arm = ArmModel(Solver(), (KinematicCoordinate("slide", "m"),), tmp_path / "arm.urdf", "tip")
    offset = np.eye(4)
    offset[0, 3] = 0.1
    factories = {
        "arm": SimpleNamespace(load_model=lambda **options: arm),
        "tool": SimpleNamespace(
            load_model=lambda **options: SimpleNamespace(
                geometry=FixedToolGeometry(offset),
            )
        ),
    }
    monkeypatch.setattr(base, "load_plugin", lambda name, **options: factories[name])
    (tmp_path / "arm.yaml").write_text("type: arm\nimplementation: arm\n")
    (tmp_path / "tool.yaml").write_text("type: end_effector\nimplementation: tool\n")
    components = {
        "a": {
            "config": "arm.yaml",
        }
    }
    if with_tool:
        components["t"] = {
            "config": "tool.yaml",
            "parent": "a.flange",
            "mount": {"xyz": [0.02, 0, 0]},
        }
    path = tmp_path / "robot.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "test",
                "components": components,
                "groups": {
                    "arbitrary_name": {"arm": "a", "end_effector": "t" if with_tool else None}
                },
            }
        )
    )
    model = RobotModel.from_config(path).groups["arbitrary_name"].kinematics
    target = model.fk(np.array([0.3]))
    assert target[0, 3] == pytest.approx(0.42 if with_tool else 0.3)
    result = model.ik(target, np.array([0.0]), fixed_coordinates={})
    np.testing.assert_allclose(result.joints, [0.3])
