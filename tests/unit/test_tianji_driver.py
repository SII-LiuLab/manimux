"""Tianji driver against fake Marvin and TacCap SDKs: no hardware, no vendor libraries."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from manimux.config import ControlProfileConfig, RobotConfig
from manimux.kinematics.tianji import DH_TABLE_M6_40, JOINT_LIMITS_DEG, TianjiKinematics
from manimux.robots import build_robot
from manimux.robots.tianji import marvin as marvin_module
from manimux.robots.tianji import sdk
from manimux.types import RobotCommand

REPO = Path(__file__).resolve().parents[2]
HOME = {
    0: np.array([90.0, -90.0, -90.0, -90.0, 0.0, 0.0, 0.0]),
    1: np.array([-90.0, -90.0, 90.0, -90.0, 0.0, 0.0, 0.0]),
}
ARM_INDEX = {"A": 0, "B": 1}
PNVA = [[high, low, 180.0, 450.0] for low, high in JOINT_LIMITS_DEG]


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
        self.fk_offset_mm = 0.0
        self.pending: dict[str, np.ndarray] = {}
        self.sent: list[dict[str, np.ndarray]] = []
        self.calls: list[tuple[object, ...]] = []
        self.released = False

    def robot_module(self) -> SimpleNamespace:
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

    def kine_module(self) -> SimpleNamespace:
        controller = self
        dh = TianjiKinematics()

        class Marvin_Kine:  # noqa: N801 - vendor name
            def log_switch(self, flag: int) -> None: ...

            def load_config(self, arm_type: int, config_path: str) -> dict[str, object]:
                table = [list(row) for row in DH_TABLE_M6_40]
                return {"TYPE": [1017, 1017], "DH": [table, table], "PNVA": [PNVA, PNVA]}

            def initial_kine(self, robot_type: int, dh: object, pnva: object, j67: object) -> bool:
                return True

            def fk(self, joints: list[float]) -> list[list[float]]:
                pose = dh.flange(np.radians(joints))
                pose[:3, 3] = pose[:3, 3] * 1e3 + controller.fk_offset_mm
                return pose.tolist()

        return SimpleNamespace(Marvin_Kine=Marvin_Kine)


class FakeGrippers:
    def __init__(self) -> None:
        self.positions = {"/dev/left": 0.8, "/dev/right": 0.6}
        self.torque = 0.0
        self.enabled: set[str] = set()
        self.targets: list[tuple[str, float]] = []
        self.stopped_transports: list[str] = []
        self.serials = {"/dev/left": "SN-L", "/dev/right": "SN-R"}
        self.reads = 0

    def module(self) -> SimpleNamespace:
        world = self

        def scan_grippers() -> list[SimpleNamespace]:
            return [
                SimpleNamespace(
                    firmware_sn=world.serials[device],
                    side=side,
                    role="Follower",
                    mcu_device=device,
                )
                for device, side in (("/dev/left", "Left"), ("/dev/right", "Right"))
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
            scan_grippers=scan_grippers,
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
    modules = {
        "taccap": grippers.module(),
        "marvin_kine": controller.kine_module(),
        "marvin_robot": controller.robot_module(),
    }

    def loader(name: str):  # type: ignore[no-untyped-def]
        def load(*_: object) -> SimpleNamespace:
            loaded.append(name)
            return modules[name]

        return load

    monkeypatch.setattr(sdk, "load_taccap", loader("taccap"))
    monkeypatch.setattr(sdk, "load_marvin_kine", loader("marvin_kine"))
    monkeypatch.setattr(sdk, "load_marvin_robot", loader("marvin_robot"))
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


def command(groups: dict[str, np.ndarray], **overrides: np.ndarray) -> RobotCommand:
    values = {name: np.asarray(value, dtype=np.float64).copy() for name, value in groups.items()}
    values.update(overrides)
    return RobotCommand(groups=values, monotonic_ns=0, plan_id=None)


def test_options_are_validated() -> None:
    clock = FakeClock()
    with pytest.raises(ValueError, match="unknown robot.options"):
        build_robot(robot_config(speed=3), clock)
    with pytest.raises(ValueError, match="dimension 8"):
        build_robot(robot_config(width=7), clock)
    with pytest.raises(ValueError, match="requires vel_ratio"):
        build_robot(robot_config(execute=True), clock)
    with pytest.raises(ValueError, match="gripper_control requires execute"):
        build_robot(robot_config(execute=True, vel_ratio=32, gripper_control=True,
                                 active_arms=["left_arm"]), clock)  # fmt: skip
    with pytest.raises(ValueError, match="home_on_close"):
        build_robot(robot_config(home_on_close="yes"), clock)
    build_robot(robot_config(width=7, end_effector="none"), clock)


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
    assert rig.loaded == ["taccap", "marvin_kine", "marvin_robot"]
    state = robot.get_state()
    np.testing.assert_allclose(state.groups["left_arm"][:7], np.radians(HOME[0]))
    assert state.groups["left_arm"][7] == pytest.approx(0.8)
    assert state.groups["right_arm"][7] == pytest.approx(0.6)

    far = state.groups["left_arm"].copy()
    far[1] += np.radians(30.0)
    robot.send_command(command(state.groups, left_arm=far))
    bad = state.groups["right_arm"].copy()
    bad[7] = 1.2
    with pytest.raises(ValueError, match="outside"):
        robot.send_command(command(state.groups, right_arm=bad))
    robot.home()
    robot.stop()

    assert rig.controller.sent == []
    assert not any(call[0] in {"set_state", "set_vel_acc"} for call in rig.controller.calls)
    assert rig.grippers.enabled == set()
    robot.close()
    assert rig.controller.released
    assert sorted(rig.grippers.stopped_transports) == ["/dev/left", "/dev/right"]


def test_state_reads_do_not_depend_on_the_call_rate(rig: SimpleNamespace) -> None:
    robot = build_robot(robot_config(), rig.clock)
    robot.connect()
    reads_after_start = rig.grippers.reads
    rig.controller.freeze_serial = True
    started = time.monotonic()
    for _ in range(500):  # far faster than the controller refreshes
        rig.clock.now += 50_000
        robot.get_state()
    elapsed = time.monotonic() - started
    # The aperture comes from the reader thread at ~30 Hz, not from each call.
    assert rig.grippers.reads - reads_after_start <= 2 * (elapsed * 30 + 2)
    rig.clock.now += 200_000_000
    with pytest.raises(RuntimeError, match="did not advance"):
        robot.get_state()
    robot.close()


def test_controller_faults_raise(rig: SimpleNamespace) -> None:
    robot = build_robot(robot_config(), rig.clock)
    robot.connect()
    rig.controller.err_code[1] = 13
    with pytest.raises(RuntimeError, match="emergency stop"):
        robot.get_state()
    robot.close()


def test_fk_mismatch_is_refused_at_connect(rig: SimpleNamespace) -> None:
    rig.controller.fk_offset_mm = 0.05
    robot = build_robot(robot_config(), rig.clock)
    with pytest.raises(ValueError, match="SDK FK versus DH"):
        robot.connect()
    assert rig.controller.released


def test_connect_requires_refreshing_frames(rig: SimpleNamespace) -> None:
    rig.controller.freeze_serial = True
    robot = build_robot(robot_config(), rig.clock)
    with pytest.raises(RuntimeError, match="not refreshing"):
        robot.connect()
    assert rig.controller.released


def test_gripper_serial_mismatch_is_refused(rig: SimpleNamespace) -> None:
    rig.grippers.serials["/dev/right"] = "SN-OTHER"
    robot = build_robot(robot_config(), rig.clock)
    with pytest.raises(ValueError, match="mismatched Right follower"):
        robot.connect()
    assert rig.controller.released


def test_execute_sends_commands_as_given_and_trips_on_tracking(rig: SimpleNamespace) -> None:
    robot = build_robot(
        robot_config(execute=True, vel_ratio=32, active_arms=["left_arm"]), rig.clock
    )
    robot.connect()
    assert ("set_vel_acc", "A", 32) in rig.controller.calls
    assert ("set_state", "A", 1) in rig.controller.calls
    assert not any(call[1:2] == ("B",) for call in rig.controller.calls)

    state = robot.get_state()
    target = state.groups["left_arm"].copy()
    target[1] += np.radians(3.0)
    robot.send_command(command(state.groups, left_arm=target))
    np.testing.assert_allclose(rig.controller.sent[-1]["A"][1], HOME[0][1] + 3.0)
    assert set(rig.controller.sent[-1]) == {"A"}

    state = robot.get_state()
    target[1] += np.radians(6.0)
    with pytest.raises(RuntimeError, match="tracking error"):
        robot.send_command(command(state.groups, left_arm=target))
    assert ("stop_running", "A") in rig.controller.calls

    rig.controller.cur_state[0] = 0
    with pytest.raises(RuntimeError, match="left position mode"):
        robot.get_state()
    robot.close()
    assert ("set_state", "A", 0) in rig.controller.calls
    assert rig.controller.released


def test_gripper_control_clamps_targets_and_trips_on_torque(rig: SimpleNamespace) -> None:
    robot = build_robot(robot_config(execute=True, vel_ratio=32, gripper_control=True), rig.clock)
    robot.connect()
    assert rig.grippers.enabled == {"/dev/left", "/dev/right"}
    state = robot.get_state()
    closed = state.groups["left_arm"].copy()
    closed[7] = 0.0
    robot.send_command(command(state.groups, left_arm=closed))
    assert ("/dev/left", pytest.approx(0.8 - 0.036)) in rig.grippers.targets

    rig.grippers.torque = 2.0
    with pytest.raises(RuntimeError, match="gripper protection"):
        robot.send_command(command(robot.get_state().groups))
    assert rig.grippers.enabled == set()
    assert ("stop_running", "B") in rig.controller.calls
    robot.close()


def test_home_passes_through_waypoints(rig: SimpleNamespace) -> None:
    waypoint = [-75.0, -10.0, 20.0, -80.0, 0.0, 0.0, 0.0]
    rig.controller.joints[1] = np.array([-60.0, -30.0, 30.0, -70.0, 0.0, 0.0, 0.0])
    robot = build_robot(
        robot_config(
            execute=True,
            vel_ratio=32,
            active_arms=["right_arm"],
            home_speed_deg_s=2000.0,
            home_waypoints_deg={"right_arm": [waypoint]},
        ),
        rig.clock,
    )
    robot.connect()
    robot.home()
    right = [np.asarray(batch["B"]) for batch in rig.controller.sent]
    assert all(set(batch) == {"B"} for batch in rig.controller.sent)
    assert any(np.allclose(joints, waypoint) for joints in right)
    np.testing.assert_allclose(right[-1], HOME[1])
    robot.close()


def test_bare_arms_skip_the_gripper_sdk(rig: SimpleNamespace) -> None:
    robot = build_robot(robot_config(width=7, end_effector="none"), rig.clock)
    robot.connect()
    assert "taccap" not in rig.loaded
    assert robot.get_state().groups["right_arm"].shape == (7,)
    robot.close()


def test_control_profile_envelopes_are_consistent() -> None:
    profile = ControlProfileConfig.model_validate(
        yaml.safe_load((REPO / "configs/robots/tianji/common.yaml").read_text())
    )
    assert profile.robot.options["execute"] is False
    build_robot(
        RobotConfig(
            driver=profile.robot.driver,
            group_dims=profile.robot.group_dims,
            options=profile.robot.options,
        ),
        FakeClock(),
    )
    safety, motion = profile.command_safety, profile.motion_limits
    assert safety is not None and motion is not None
    for group in ("left_arm", "right_arm"):
        lower = np.asarray(safety.position_lower[group])
        upper = np.asarray(safety.position_upper[group])
        home = np.radians(HOME[0 if group == "left_arm" else 1])
        assert lower.shape == upper.shape == (8,)
        assert np.all(lower[:7] < home) and np.all(home < upper[:7])
        # The executor shapes commands strictly inside the runtime guard.
        assert motion.arm.max_velocity < min(safety.max_velocity[group][:7])
        assert motion.arm.max_acceleration < min(safety.max_acceleration[group][:7])
        index = motion.gripper.group_indices[group]
        assert motion.gripper.max_velocity < safety.max_velocity[group][index]
        assert motion.gripper.max_acceleration < safety.max_acceleration[group][index]
    assert safety.position_upper["right_arm"][5] < safety.position_upper["left_arm"][5]
