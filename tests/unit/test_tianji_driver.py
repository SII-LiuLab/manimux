"""Tianji driver against fake Marvin and TacCap SDKs: no hardware, no vendor libraries."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from manimux.config import ControlProfileConfig, RobotConfig
from manimux.robots import build_robot
from manimux.robots.tianji import driver as driver_module
from manimux.robots.tianji import marvin as marvin_module
from manimux.robots.tianji import sdk
from manimux.types import RobotCommand

REPO = Path(__file__).resolve().parents[2]
HOME = {
    0: np.array([90.0, -90.0, -90.0, -90.0, 0.0, 0.0, 0.0]),
    1: np.array([-90.0, -90.0, 90.0, -90.0, 0.0, 0.0, 0.0]),
}
EXECUTE = {"execute": True, "vel_ratio": 32, "max_joint_rate_deg_s": 51.84}
ARM_INDEX = {"A": 0, "B": 1}


class FakeController:
    def __init__(self) -> None:
        self.joints = {index: value.copy() for index, value in HOME.items()}
        self.cur_state = [0, 0]
        self.err_code = [0, 0]
        self.vel_ratio = [0, 0]
        self.acc_ratio = [0, 0]
        self.serial = 0
        self.freeze_serial = False
        self.follow = True
        self.pending: dict[str, np.ndarray] = {}
        self.sent: list[dict[str, np.ndarray]] = []
        self.calls: list[tuple[object, ...]] = []
        self.released = False

    def module(self) -> SimpleNamespace:
        controller = self

        class Marvin_Robot:  # noqa: N801 - vendor name
            def connect(self, ip: str) -> bool:
                controller.calls.append(("connect", ip))
                return True

            def log_switch(self, flag: str) -> None: ...

            def local_log_switch(self, flag: str) -> None: ...

            def get_param(self, kind: str, name: str) -> tuple[bool, int]:
                return True, 1017

            def subscribe(self, dcss: object) -> dict[str, object]:
                if not controller.freeze_serial:
                    controller.serial += 1
                return {
                    "states": [
                        {
                            "cur_state": controller.cur_state[i],
                            "cmd_state": 0,
                            "err_code": controller.err_code[i],
                        }
                        for i in (0, 1)
                    ],
                    "outputs": [
                        {
                            "frame_serial": controller.serial,
                            "fb_joint_pos": controller.joints[i].tolist(),
                        }
                        for i in (0, 1)
                    ],
                    "inputs": [
                        {
                            "joint_vel_ratio": controller.vel_ratio[i],
                            "joint_acc_ratio": controller.acc_ratio[i],
                        }
                        for i in (0, 1)
                    ],
                }

            def clear_set(self) -> None:
                controller.pending = {}

            def set_vel_acc(self, arm: str, velRatio: int, AccRatio: int) -> None:  # noqa: N803
                controller.vel_ratio[ARM_INDEX[arm]] = velRatio
                controller.acc_ratio[ARM_INDEX[arm]] = AccRatio
                controller.calls.append(("set_vel_acc", arm, velRatio))

            def set_state(self, arm: str, state: int) -> None:
                controller.cur_state[ARM_INDEX[arm]] = state
                controller.calls.append(("set_state", arm, state))

            def set_joint_cmd_pose(self, arm: str, joints: list[float]) -> None:
                controller.pending[arm] = np.asarray(joints, dtype=np.float64)

            def send_cmd(self) -> None:
                if controller.pending:
                    controller.sent.append(dict(controller.pending))
                    if controller.follow:
                        for arm, joints in controller.pending.items():
                            controller.joints[ARM_INDEX[arm]] = joints.copy()
                controller.pending = {}

            def stop_running(self, arm: str) -> None:
                controller.calls.append(("stop_running", arm))

            def release_robot(self) -> None:
                controller.released = True

        return SimpleNamespace(Marvin_Robot=Marvin_Robot, DCSS=lambda: object())


class FakeGrippers:
    def __init__(self) -> None:
        self.positions = {"/dev/left": 0.8, "/dev/right": 0.6}
        self.torque = 0.0
        self.enabled: set[str] = set()
        self.targets: list[tuple[str, float]] = []
        self.stopped_transports: list[str] = []
        self.reads = 0

    def module(self) -> SimpleNamespace:
        world = self
        endpoints = [
            SimpleNamespace(
                firmware_sn="SN-L", side="Left", role="Follower", mcu_device="/dev/left"
            ),
            SimpleNamespace(
                firmware_sn="SN-R", side="Right", role="Follower", mcu_device="/dev/right"
            ),
        ]

        class Motor:
            def __init__(self, device: str) -> None:
                self.device = device

            def clear_fault(self) -> None: ...

            def enable(self) -> None:
                world.enabled.add(self.device)

            def disable(self) -> None:
                world.enabled.discard(self.device)

        class FollowerGripper:
            def __init__(self, mcu_device: str) -> None:
                self.device = mcu_device
                self.motor = Motor(mcu_device)
                self.transport = SimpleNamespace(
                    stop=lambda: world.stopped_transports.append(mcu_device)
                )

            def get_gripper_config(self) -> SimpleNamespace:
                return SimpleNamespace(flags=1, min_open_rad=0.0, max_open_rad=1.3)

            def position(self) -> float:
                world.reads += 1
                return world.positions[self.device]

        class ControlLoop:
            def __init__(self, gripper: FollowerGripper, hz: int, kp: float, kd: float) -> None:
                self.gripper = gripper

            def start(self) -> None: ...

            def stop(self) -> None: ...

            def set_target(self, value: float) -> None:
                world.targets.append((self.gripper.device, value))

            def observation(self) -> SimpleNamespace:
                return SimpleNamespace(
                    valid=True,
                    position=world.positions[self.gripper.device],
                    velocity=0.0,
                    torque=world.torque,
                    age_ms=5.0,
                )

        return SimpleNamespace(
            Side=SimpleNamespace(Left="Left", Right="Right"),
            Role=SimpleNamespace(Follower="Follower", Leader="Leader"),
            scan_grippers=lambda: endpoints,
            FollowerGripper=FollowerGripper,
            ControlLoop=ControlLoop,
        )


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000_000_000

    def now_ns(self) -> int:
        return self.now

    def sleep_until_ns(self, target_ns: int) -> None:
        self.now = max(self.now, target_ns)


@pytest.fixture
def rig(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    controller, grippers, loaded = FakeController(), FakeGrippers(), []
    marvin, taccap = controller.module(), grippers.module()

    def load_taccap() -> SimpleNamespace:
        loaded.append("taccap")
        return taccap

    def load_marvin_robot(sdk_root: object = None) -> SimpleNamespace:
        loaded.append("marvin")
        return marvin

    monkeypatch.setattr(sdk, "load_taccap", load_taccap)
    monkeypatch.setattr(sdk, "load_marvin_robot", load_marvin_robot)
    monkeypatch.setattr(marvin_module, "time", SimpleNamespace(sleep=lambda _: None))
    return SimpleNamespace(
        controller=controller, grippers=grippers, loaded=loaded, clock=FakeClock()
    )


def robot_config(width: int = 8, **options: object) -> RobotConfig:
    base: dict[str, object] = {"left_gripper_sn": "SN-L", "right_gripper_sn": "SN-R"}
    base.update(options)
    return RobotConfig(
        driver="tianji_dual", group_dims={"left_arm": width, "right_arm": width}, options=base
    )


def command(
    state_groups: dict[str, np.ndarray], t_ns: int, **overrides: np.ndarray
) -> RobotCommand:
    groups = {
        name: np.asarray(value, dtype=np.float64).copy() for name, value in state_groups.items()
    }
    groups.update(overrides)
    return RobotCommand(groups=groups, monotonic_ns=t_ns, plan_id=None)


def test_options_are_validated() -> None:
    clock = FakeClock()
    with pytest.raises(ValueError, match="unknown robot.options"):
        build_robot(robot_config(speed=3), clock)
    with pytest.raises(ValueError, match="dimension 8"):
        build_robot(robot_config(width=7), clock)
    with pytest.raises(ValueError, match="vel_ratio and max_joint_rate_deg_s"):
        build_robot(robot_config(execute=True), clock)
    with pytest.raises(ValueError, match="percentage"):
        build_robot(robot_config(vel_ratio=0), clock)
    bare = build_robot(robot_config(width=7, end_effector="none"), clock)
    # Overrides may only narrow the controller's own limits.
    widened = driver_module.TianjiDualArmDriver(
        robot_config(limit_margin_deg=0.0, joint_limits_deg={"left_arm": {1: [-400, 400]}}), clock
    )
    assert widened._lower["left_arm"][0] == -170.0
    assert bare is not None


def test_importing_the_driver_package_loads_no_vendor_sdk() -> None:
    probe = (
        "import sys, manimux.robots.tianji, manimux.robots; "
        "print(any(name.startswith(('_manimux_marvin_', 'xense')) for name in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, cwd=REPO
    )
    assert result.stdout.strip() == "False"


def test_read_only_session_reads_state_and_never_moves_anything(rig: SimpleNamespace) -> None:
    robot = build_robot(robot_config(), rig.clock)
    robot.connect()
    assert rig.loaded == ["taccap", "marvin"]
    state = robot.get_state()
    np.testing.assert_allclose(state.groups["left_arm"][:7], np.radians(HOME[0]))
    assert state.groups["left_arm"][7] == pytest.approx(0.8)
    assert state.groups["right_arm"][7] == pytest.approx(0.6)

    robot.send_command(command(state.groups, rig.clock.now + 10_000_000))
    with pytest.raises(ValueError, match="J1 command"):
        outside = state.groups["left_arm"].copy()
        outside[0] = np.radians(179.0)
        robot.send_command(command(state.groups, rig.clock.now + 20_000_000, left_arm=outside))
    robot.home()
    robot.stop()
    assert rig.controller.sent == []
    assert not any(call[0] in {"set_state", "set_vel_acc"} for call in rig.controller.calls)
    assert rig.grippers.enabled == set()

    robot.close()
    assert rig.controller.released
    assert sorted(rig.grippers.stopped_transports) == ["/dev/left", "/dev/right"]


def test_read_only_gripper_polling_is_throttled(rig: SimpleNamespace) -> None:
    robot = build_robot(robot_config(), rig.clock)
    robot.connect()
    for _ in range(20):
        robot.get_state()
    assert rig.grippers.reads <= 4
    robot.close()


def test_controller_faults_and_stalled_feedback_raise(rig: SimpleNamespace) -> None:
    robot = build_robot(robot_config(), rig.clock)
    robot.connect()
    robot.get_state()
    rig.controller.err_code[1] = 13
    with pytest.raises(RuntimeError, match="emergency stop"):
        robot.get_state()
    rig.controller.err_code[1] = 0
    rig.controller.freeze_serial = True
    robot.get_state()
    rig.clock.now += 200_000_000
    with pytest.raises(RuntimeError, match="stalled"):
        robot.get_state()
    robot.close()


def test_execute_enters_position_mode_and_streams_checked_commands(rig: SimpleNamespace) -> None:
    robot = build_robot(robot_config(**EXECUTE), rig.clock)
    robot.connect()
    assert ("set_vel_acc", "A", 32) in rig.controller.calls
    assert ("set_state", "B", 1) in rig.controller.calls
    assert rig.grippers.enabled == {"/dev/left", "/dev/right"}

    state = robot.get_state()
    moved = state.groups["left_arm"].copy()
    moved[1] += np.radians(0.3)
    moved[7] = 0.0
    t = rig.clock.now + 10_000_000
    robot.send_command(command(state.groups, t, left_arm=moved))
    sent = rig.controller.sent[-1]
    np.testing.assert_allclose(sent["A"][1], HOME[0][1] + 0.3)
    # The gripper target is clamped to within grip_margin of the measured aperture.
    assert ("/dev/left", pytest.approx(0.8 - 0.036)) in rig.grippers.targets

    jump = moved.copy()
    jump[1] += np.radians(5.0)
    with pytest.raises(RuntimeError, match="joint step"):
        robot.send_command(command(state.groups, t + 10_000_000, left_arm=jump))
    assert ("stop_running", "A") in rig.controller.calls

    robot.close()
    assert ("set_state", "A", 0) in rig.controller.calls
    assert ("set_state", "B", 0) in rig.controller.calls
    assert rig.controller.released and rig.grippers.enabled == set()


def test_sustained_tracking_error_halts(rig: SimpleNamespace) -> None:
    robot = build_robot(robot_config(**EXECUTE), rig.clock)
    robot.connect()
    rig.controller.follow = False
    state = robot.get_state()
    target = state.groups["left_arm"].copy()
    with pytest.raises(RuntimeError, match="tracking error"):
        for _ in range(200):
            rig.clock.now += 10_000_000
            robot.get_state()
            target[1] += np.radians(0.5)
            robot.send_command(command(state.groups, rig.clock.now, left_arm=target.copy()))
    robot.close()


def test_gripper_protection_fault_halts(rig: SimpleNamespace) -> None:
    robot = build_robot(robot_config(**EXECUTE), rig.clock)
    robot.connect()
    state = robot.get_state()
    rig.grippers.torque = 2.0
    with pytest.raises(RuntimeError, match="gripper protection"):
        robot.send_command(command(state.groups, rig.clock.now + 10_000_000))
    robot.close()


def test_bare_arms_skip_the_gripper_sdk(rig: SimpleNamespace) -> None:
    robot = build_robot(robot_config(width=7, end_effector="none"), rig.clock)
    robot.connect()
    assert rig.loaded == ["marvin"]
    assert robot.get_state().groups["right_arm"].shape == (7,)
    robot.close()


def test_home_passes_through_waypoints(rig: SimpleNamespace) -> None:
    waypoint = [-75.0, -10.0, 20.0, -80.0, 0.0, 0.0, 0.0]
    rig.controller.joints[1] = np.array([-60.0, -30.0, 30.0, -70.0, 0.0, 0.0, 0.0])
    robot = build_robot(
        robot_config(
            **EXECUTE,
            active_arms=["right_arm"],
            home_speed_deg_s=2000.0,
            home_waypoints_deg={"right_arm": [waypoint]},
        ),
        rig.clock,
    )
    robot.connect()
    robot.home()
    right = [np.asarray(batch["B"]) for batch in rig.controller.sent if "B" in batch]
    assert all("A" not in batch for batch in rig.controller.sent)
    assert any(np.allclose(joints, waypoint) for joints in right)
    np.testing.assert_allclose(right[-1], HOME[1])
    robot.close()


def test_control_profile_builds_a_read_only_driver() -> None:
    profile = ControlProfileConfig.model_validate(
        yaml.safe_load((REPO / "configs/robots/tianji/common.yaml").read_text())
    )
    assert profile.robot.options["execute"] is False
    robot = build_robot(
        RobotConfig(
            driver=profile.robot.driver,
            group_dims=profile.robot.group_dims,
            options=profile.robot.options,
        ),
        FakeClock(),
    )
    assert robot._upper["right_arm"][5] == pytest.approx(53.0)
