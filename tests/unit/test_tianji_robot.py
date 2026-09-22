"""Fake hardware assembly tests; no control library loaded or device connected."""

from types import SimpleNamespace

import numpy as np
import pytest

from manimux.embodiments.end_effector import GripperBase, GripperState
from manimux.embodiments.robot.tianji_taccap import TianjiArmConfig, TianjiTaccapRobot
from manimux.kinematics import IKResult, KinematicCoordinate, ManipulatorKinematicsBase
from manimux.types import RobotCommand, SensorFrame


class Clock:
    now = 1_000_000_000

    def now_ns(self):
        return self.now


class Gripper(GripperBase):
    def __init__(self, clock):
        self.clock = clock
        self.calls = []
        self.fail = False

    def connect(self):
        self.calls.append("connect")

    def get_state(self):
        return GripperState(0.4, self.clock.now_ns() / 1e9)

    def send_command(self, command):
        self.calls.append(command.opening)
        if self.fail:
            raise RuntimeError("gripper failed")

    def stop(self):
        self.calls.append("stop")

    def close(self):
        self.calls.append("close")


class Model(ManipulatorKinematicsBase):
    def __init__(self, name, gripper=True):
        self._base = f"tianji_{name}_base"
        self._coords = tuple(KinematicCoordinate(f"joint_{i}", "rad") for i in range(1, 8))
        if gripper:
            self._coords += (KinematicCoordinate("gripper", "normalized"),)

    @property
    def coordinates(self):
        return self._coords

    @property
    def base_frame(self):
        return self._base

    @property
    def tcp_frame(self):
        return "tcp"

    def fk(self, configuration):
        self.fk_input = configuration.copy()
        pose = np.eye(4)
        pose[0, 3] = configuration[0] + 0.16945
        return pose

    def ik(self, target_tcp, seed_configuration, *, fixed_coordinates):
        q = seed_configuration.copy()
        q[0] = target_tcp[0, 3] - 0.16945
        q[-1] = fixed_coordinates["gripper"]
        return IKResult(True, q)


class SDK:
    def __init__(self):
        self.frames = [1, 1]
        self.advance = True
        self.modes = [0, 0]
        self.faults = [0, 0]
        self.ratios = [{"joint_vel_ratio": 0, "joint_acc_ratio": 0} for _ in range(2)]
        self.pending = []
        self.batches = []
        self.connects = self.releases = 0
        self.fail_disable = set()
        self.fail_release = False

    def connect(self, ip):
        self.connects += 1
        return True

    def subscribe(self, buf):
        if self.advance:
            self.frames = [n + 1 for n in self.frames]
        return {
            "outputs": [
                {"frame_serial": self.frames[i], "fb_joint_pos": [10.0 * (i + 1)] * 7}
                for i in range(2)
            ],
            "states": [{"cur_state": self.modes[i], "err_code": self.faults[i]} for i in range(2)],
            "inputs": self.ratios,
        }

    def clear_set(self):
        self.pending = []
        return True

    def set_joint_cmd_pose(self, arm, joints):
        self.pending.append(("target", arm, joints))
        return True

    def set_vel_acc(self, arm, vel, acc):
        self.pending.append(("ratio", arm, vel, acc))
        return True

    def set_state(self, arm, mode):
        if mode == 0 and arm in self.fail_disable:
            return False
        self.pending.append(("mode", arm, mode))
        return True

    def send_cmd(self):
        self.batches.append(list(self.pending))
        for entry in self.pending:
            index = 0 if entry[1] == "A" else 1
            if entry[0] == "mode":
                self.modes[index] = entry[2]
            if entry[0] == "ratio":
                self.ratios[index] = {"joint_vel_ratio": entry[2], "joint_acc_ratio": entry[3]}
        return True

    def release_robot(self):
        self.releases += 1
        return not self.fail_release


@pytest.fixture
def setup(monkeypatch):
    clock, sdk = Clock(), SDK()
    monkeypatch.setattr(
        "manimux.embodiments.arm.tianji.arm.importlib.import_module",
        lambda name: SimpleNamespace(Marvin_Robot=lambda: sdk, DCSS=object),
    )
    configs = {
        name: TianjiArmConfig(
            Model(name), (np.full(7, -2.0), np.full(7, 2.0)), 10, 20, Gripper(clock)
        )
        for name in ("left", "right")
    }
    robot = TianjiTaccapRobot(ip="192.168.1.10", arms=configs, clock=clock, ready_timeout_s=0.02)
    yield robot, sdk, clock, configs
    sdk.fail_disable.clear()
    sdk.fail_release = False
    sdk.advance = True
    sdk.faults = [0, 0]
    robot.close()


def command(**overrides):
    groups = {n: np.r_[np.full(7, 0.3), 0.7] for n in ("left", "right")}
    groups.update(overrides)
    return RobotCommand(groups, 0, None)


def test_dual_lifecycle_and_batched_units(setup):
    robot, sdk, _, cfg = setup
    assert sdk.connects == 0
    robot.connect()
    robot.connect()
    assert sdk.connects == 1 and sdk.batches == []
    state = robot.get_state()
    np.testing.assert_allclose(state.groups["left"][:7], np.radians(10))
    np.testing.assert_allclose(state.groups["right"][:7], np.radians(20))
    assert state.groups["right"][-1] == 0.4
    robot.send_command(command())
    assert len(sdk.batches) == 2  # measured hold/enable, then both arm targets
    targets = sdk.batches[-1]
    assert [t[1] for t in targets] == ["A", "B"]
    np.testing.assert_allclose(targets[0][2], np.degrees(np.full(7, 0.3)))
    assert all(c.gripper.calls[-1] == 0.7 for c in cfg.values())
    assert robot.get_state().groups["left"][-1] == 0.4
    robot.stop()
    assert sdk.modes == [0, 0]
    robot.send_command(command())
    assert sdk.modes == [1, 1]
    robot.close()
    robot.close()
    assert sdk.releases == 1 and sdk.modes == [0, 0]
    assert all(c.gripper.calls.count("close") == 1 for c in cfg.values())


@pytest.mark.parametrize("bad", [np.zeros(7), np.r_[np.full(7, 3.0), 0.5], np.r_[np.zeros(7), 1.1]])
def test_validate_all_groups_before_any_motion(setup, bad):
    robot, sdk, _, _ = setup
    robot.connect()
    with pytest.raises(ValueError):
        robot.send_command(command(right=bad))
    assert sdk.batches == []


def test_missing_group_rejected(setup):
    robot, sdk, _, _ = setup
    robot.connect()
    with pytest.raises(ValueError):
        robot.send_command(RobotCommand({"left": np.zeros(8)}, 0, None))
    assert sdk.batches == []


def test_partial_gripper_failure_stops_everything(setup):
    robot, sdk, _, cfg = setup
    robot.connect()
    cfg["right"].gripper.fail = True
    with pytest.raises(RuntimeError, match="gripper failed"):
        robot.send_command(command())
    assert sdk.modes == [0, 0]
    assert all("stop" in c.gripper.calls for c in cfg.values())


def test_stop_attempts_other_components_and_retry(setup):
    robot, sdk, _, cfg = setup
    robot.connect()
    robot.send_command(command())
    sdk.fail_disable.add("A")
    with pytest.raises(ExceptionGroup):
        robot.close()
    assert sdk.modes[1] == 0 and sdk.releases == 0
    assert all("close" in c.gripper.calls for c in cfg.values())
    sdk.fail_disable.clear()
    robot.close()
    assert sdk.releases == 1


def test_estop_fault_state_releases_controller_session(setup):
    robot, sdk, _, _ = setup
    robot.connect()
    robot.send_command(command())
    before = len(sdk.batches)

    sdk.modes = [100, 100]
    sdk.faults = [13, 13]
    robot.close()

    assert len(sdk.batches) == before
    assert sdk.releases == 1


def test_stale_arm_feedback(setup):
    robot, sdk, clock, _ = setup
    robot.connect()
    sdk.advance = False
    clock.now += 300_000_000
    with pytest.raises(RuntimeError, match="stale"):
        robot.get_state()


def test_fault_connect_cleanup_without_clear(setup):
    robot, sdk, _, _ = setup
    sdk.faults[1] = 13
    with pytest.raises(RuntimeError, match="fault"):
        robot.connect()
    assert sdk.releases == 1 and sdk.batches == []


def test_grouped_tcp_models_used_without_hardware(setup):
    robot, sdk, _, _ = setup
    q = command().groups
    targets = robot.kinematics.fk(q)
    assert targets["left"][0, 3] == pytest.approx(0.46945)
    result = robot.kinematics.ik(targets, q, fixed_coordinates={n: {"gripper": 0.8} for n in q})
    assert all(r.converged and r.joints[-1] == 0.8 for r in result.values())
    assert sdk.connects == 0
    with pytest.raises(ValueError):
        robot.kinematics.fk({"unknown": np.zeros(8)})
    with pytest.raises(ValueError):
        robot.kinematics.ik(targets, q, fixed_coordinates={})
    with pytest.raises(TypeError):
        robot.kinematics.models["left"] = Model("right")


def test_single_arm_uses_shared_class(setup):
    robot, sdk, clock, cfg = setup
    single = TianjiTaccapRobot(ip="192.168.1.10", arms={"right": cfg["right"]}, clock=clock)
    try:
        single.connect()
        single.send_command(RobotCommand({"right": np.r_[np.zeros(7), 0.5]}, 0, None))
        assert all(entry[1] == "B" for batch in sdk.batches for entry in batch)
        with pytest.raises(RuntimeError, match="session"):
            robot.connect()
    finally:
        single.close()


def test_layout_and_side_mismatch_rejected(setup):
    _, _, clock, cfg = setup
    with pytest.raises(ValueError, match="base"):
        TianjiTaccapRobot(ip="192.168.1.10", arms={"left": cfg["right"]}, clock=clock)
    with pytest.raises(ValueError, match="layout"):
        TianjiArmConfig(Model("left"), (np.full(7, -2.0), np.full(7, 2.0)), 10, 20)


def test_home_explicitly_unsupported(setup):
    with pytest.raises(NotImplementedError):
        setup[0].home()


def test_speed_change_stops_owned_arms(setup):
    robot, sdk, _, _ = setup
    robot.connect()
    robot.send_command(command())
    sdk.ratios[1]["joint_vel_ratio"] = 99
    with pytest.raises(RuntimeError, match="speed settings"):
        robot.send_command(command())
    assert sdk.modes == [0, 0]


def test_bare_arm_configuration(setup):
    _, sdk, clock, _ = setup
    cfg = TianjiArmConfig(Model("left", gripper=False), (np.full(7, -2.0), np.full(7, 2.0)), 10, 20)
    bare = TianjiTaccapRobot(ip="192.168.1.10", arms={"left": cfg}, clock=clock)
    try:
        bare.connect()
        assert bare.get_state().groups["left"].shape == (7,)
        bare.send_command(RobotCommand({"left": np.zeros(7)}, 0, None))
        assert sdk.modes == [1, 0]
    finally:
        bare.close()


def test_arm_components_share_controller_and_can_address_one_channel(setup):
    from manimux.embodiments.arm import ArmBase

    robot, sdk, _, _ = setup
    left, right = robot.arm_components["left"], robot.arm_components["right"]
    assert isinstance(left, ArmBase) and isinstance(right, ArmBase)
    assert left.controller is right.controller is robot.controller
    robot.connect()
    left.send_command(np.full(7, 0.2))
    assert sdk.modes == [1, 0]
    assert all(entry[1] == "A" for batch in sdk.batches for entry in batch)
    np.testing.assert_allclose(right.get_state().joints, np.radians(20))


def test_arm_target_validation_precedes_enabling(setup):
    robot, sdk, _, _ = setup
    robot.connect()
    left = robot.arm_components["left"]
    for target in (np.zeros(8), np.full(7, 3.0), np.full(7, np.nan)):
        with pytest.raises(ValueError):
            left.send_command(target)
    assert sdk.batches == []


def test_fk_entry_uses_assembled_model_without_arm_sdk(setup):
    robot, sdk, _, _ = setup
    q = command().groups
    assert all(arm._kinematics is None for arm in robot.arm_components.values())
    targets = robot.fk(q)
    results = robot.ik(targets, q, fixed_coordinates={n: {"gripper": 0.4} for n in q})
    assert all(result.converged for result in results.values())
    assert sdk.connects == 0


@pytest.mark.parametrize("offset", [-300_000_000, 1_000_000])
def test_robot_rejects_arm_clock_mismatch(setup, monkeypatch, offset):
    from manimux.embodiments.arm.base import ArmState

    robot, _, clock, _ = setup
    robot.connect()
    states = {name: ArmState(np.zeros(7), clock.now_ns() + offset, 1) for name in ("left", "right")}
    monkeypatch.setattr(robot.controller, "get_states", lambda: states)
    with pytest.raises(RuntimeError, match="stale arm feedback or clock mismatch"):
        robot.get_state()


def test_robot_sensor_failed_cleanup_requires_retry(setup):
    from manimux.embodiments.robot.base import RobotBase
    from manimux.embodiments.sensor.base import SensorBase

    original, _, clock, cfg = setup

    frame = SensorFrame("wrist", np.zeros((2, 2, 3), dtype=np.uint8), clock.now_ns(), 1)

    class Sensor(SensorBase):
        fail = True
        starts = 0

        def start(self):
            self.starts += 1
            if self.fail:
                raise RuntimeError("start failed")

        def read(self):
            return frame

        def close(self):
            if self.fail:
                raise RuntimeError("close failed")

    sensor = Sensor()

    class Robot(RobotBase):
        def __init__(self):
            super().__init__(
                arm_components={"arm": original.arm_components["left"]},
                end_effectors={"arm": cfg["left"].gripper},
                models={"arm": cfg["left"].kinematics},
                sensors={"wrist": sensor},
                clock=clock,
            )

    robot = Robot()
    with pytest.raises(ExceptionGroup, match="sensor startup and cleanup failed"):
        robot.start_sensors()
    with pytest.raises(RuntimeError, match="cleanup incomplete"):
        robot.start_sensors()
    assert sensor.starts == 1
    sensor.fail = False
    robot.close_sensors()
    robot.start_sensors()
    robot.start_sensors()
    assert sensor.starts == 2
    assert robot.read_sensors()["wrist"] is frame
    robot.close()


def test_robot_coordination_is_independent_of_tianji_group_names(setup):
    from manimux.embodiments.robot.base import RobotBase

    robot, sdk, clock, cfg = setup

    class AssembledRobot(RobotBase):
        def __init__(self):
            super().__init__(
                arm_components={"manipulator": robot.arm_components["right"]},
                end_effectors={"manipulator": cfg["right"].gripper},
                models={"manipulator": cfg["right"].kinematics},
                clock=clock,
            )

    assembled = AssembledRobot()
    try:
        assembled.connect()
        assembled.send_command(RobotCommand({"manipulator": np.r_[np.zeros(7), 0.6]}, 0, None))
        assert list(assembled.get_state().groups) == ["manipulator"]
        assert sdk.modes == [0, 1]
    finally:
        assembled.close()
