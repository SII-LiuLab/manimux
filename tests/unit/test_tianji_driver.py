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
from manimux.robots.tianji import driver as driver_module
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
        self.clear_attempts = [0, 0]
        self.clear_after = [1, 1]
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

            def clear_error(self, arm: str) -> bool:
                index = ARM_INDEX[arm]
                controller.calls.append(("clear_error", arm))
                controller.clear_attempts[index] += 1
                if controller.clear_attempts[index] >= controller.clear_after[index]:
                    controller.err_code[index] = 0
                    controller.cur_state[index] = 0
                # SDK success is not proof that a physical E-stop was released.
                return True

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
        self.follow = False

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
                if world.follow:
                    world.positions[self.gripper.device] = value

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


@pytest.mark.parametrize("control_hz", [100.0, 250.0])
@pytest.mark.parametrize("finish_home", [None, False, True])
def test_viewer_controls_reach_tianji_and_smooth_ticks_send_commands(
    rig: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    control_hz: float,
    finish_home: bool | None,
) -> None:
    from manimux.config import load_config
    from manimux.runtime import edge
    from manimux.types import ActionChunk, InferenceResponse
    from manimux.viewer import transport

    controls = iter([
        {"paused": True}, {"paused": False}, {"paused": False},
        {"paused": True}, {"paused": False},
        {"paused": True, "home_requested": True},
        {"paused": True}, {"paused": False},
        {"paused": True, "finish_requested": True, "finish_home": finish_home},
    ])
    messages = []
    monkeypatch.setattr(transport, "ControlClient", lambda: SimpleNamespace(
        poll=lambda: next(controls), close=lambda: None,
    ))
    monkeypatch.setattr(transport, "ViewerPublisher", lambda: SimpleNamespace(
        publish=lambda message: messages.append(message.to_wire()), close=lambda: None,
    ))

    class InstantPolicy:
        is_alive = True
        pending = None

        def start(self):
            pass

        def submit_latest(self, request):
            # Constant FK-independent targets isolate GUI/control scheduling.
            chunk = ActionChunk(
                plan_id=f"test-{request.request_seq}", request_seq=request.request_seq,
                observation_time_ns=request.observation_time_ns,
                created_time_ns=rig.clock.now_ns(), action_space="joint_position",
                dt_ns=33_333_333,
                groups={name: np.tile(values, (20, 1))
                        for name, values in request.observation.state.groups.items()},
            )
            self.pending = InferenceResponse(
                request.session_id, request.request_seq, rig.clock.now_ns(), 0.0, chunk,
                observation_time_ns=request.observation_time_ns,
            )

        def poll(self):
            result, self.pending = self.pending, None
            return result

        def close(self):
            self.is_alive = False

    monkeypatch.setattr(edge, "PolicyWorkerClient", lambda *_: InstantPolicy())
    config = load_config(REPO / "configs/mock.yaml")
    config.robot = robot_config(execute=True, vel_ratio=32)
    # Explicit Finish without homing must override even home_on_close=true.
    config.robot.options["home_on_close"] = finish_home is False
    config.robot.control_hz = control_hz
    config.sensors = []
    config.viewer.enabled = True
    config.viewer.robot_adapter = "tianji"
    config.execution.inference_schedule = "single_inflight"
    config.execution.commit_lead_s = 0
    config.run.max_steps = 20
    runtime = edge.EdgeRuntime(config, tmp_path, clock=rig.clock)
    sends, smoothing, homes = [], [], []
    original_send = runtime._robot.send_command
    original_step = runtime._executor.step
    original_home = runtime._robot.home

    def send(command):
        sends.append(rig.clock.now_ns())
        original_send(command)

    def step(*args):
        smoothing.append(rig.clock.now_ns())
        return original_step(*args)

    def home():
        homes.append(rig.clock.now_ns())
        original_home()

    monkeypatch.setattr(runtime._robot, "send_command", send)
    monkeypatch.setattr(runtime._robot, "home", home)
    monkeypatch.setattr(runtime._executor, "step", step)
    result = runtime.run()

    assert result.terminal_reason == "viewer_finish_requested"
    assert result.steps == 4
    assert len(homes) == (2 if finish_home is True else 1)
    assert len(smoothing) == 4
    assert len(sends) == len(rig.controller.sent) == 7  # includes paused holds
    np.testing.assert_array_equal(np.diff(sends), np.full(6, round(1e9 / control_hz)))
    assert set(smoothing).issubset(sends)
    assert rig.controller.released
    assert any(call[0] == "stop_running" for call in rig.controller.calls)
    assert any(message.get("event") == "episode_finished" for message in messages)


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


def test_normal_connect_does_not_clear_estop(rig: SimpleNamespace) -> None:
    rig.controller.cur_state = [100, 100]
    rig.controller.err_code = [13, 13]
    robot = driver_module.build_robot(robot_config(execute=True, vel_ratio=32), rig.clock)
    with pytest.raises(RuntimeError, match="clear it first"):
        robot.connect()
    assert rig.controller.clear_attempts == [0, 0]
    assert not any(call[:1] == ("set_vel_acc",) for call in rig.controller.calls)
    assert rig.controller.sent == []
    assert rig.controller.released


@pytest.mark.parametrize("state, error", [(100, 13), (0, 13), (100, 0), (100, 2)])
def test_recovery_clears_both_arms_before_enabling(
    rig: SimpleNamespace,
    state: int,
    error: int,
) -> None:
    rig.controller.cur_state = [state, state]
    rig.controller.err_code = [error, error]
    rig.controller.clear_after = [1, 3]
    robot = driver_module.build_robot(robot_config(execute=True, vel_ratio=32), rig.clock)
    try:
        robot.connect(recover_errors=True)
        assert rig.controller.clear_attempts == [1, 3]
        assert rig.controller.err_code == [0, 0]
        assert rig.controller.cur_state == [1, 1]
        calls = rig.controller.calls
        last_clear = max(i for i, call in enumerate(calls) if call[0] == "clear_error")
        first_enable = next(i for i, call in enumerate(calls) if call[0] == "set_vel_acc")
        assert last_clear < first_enable
        robot.get_state()
    finally:
        robot.close()


def test_recovery_with_persistent_estop_never_enables_arms_or_grippers(
    rig: SimpleNamespace,
) -> None:
    rig.controller.cur_state = [100, 100]
    rig.controller.err_code = [13, 13]
    rig.controller.clear_after = [1, 100]
    robot = driver_module.build_robot(
        robot_config(execute=True, vel_ratio=32, gripper_control=True), rig.clock
    )
    with pytest.raises(RuntimeError, match="release the physical E-stop.*arm B.*err_code 13"):
        robot.connect(recover_errors=True)
    assert rig.controller.clear_attempts == [1, marvin_module.ERROR_CLEAR_ATTEMPTS]
    assert not any(call[0] in {"set_vel_acc", "set_state"} for call in rig.controller.calls)
    assert rig.grippers.enabled == set()
    assert rig.controller.sent == []
    assert rig.controller.released


def test_recovery_requires_fresh_feedback_after_clearing(
    rig: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig.controller.err_code = [13, 13]
    original_read = marvin_module.MarvinSession.read

    def read(session):
        if any(rig.controller.clear_attempts):
            rig.controller.freeze_serial = True
        return original_read(session)

    monkeypatch.setattr(marvin_module.MarvinSession, "read", read)
    robot = driver_module.build_robot(robot_config(execute=True, vel_ratio=32), rig.clock)
    with pytest.raises(RuntimeError, match="Feedback frames did not advance"):
        robot.connect(recover_errors=True)
    assert not any(call[0] == "set_state" for call in rig.controller.calls)
    assert rig.controller.sent == []
    assert rig.controller.released


def test_recovery_does_not_clear_faults_outside_active_arms(rig: SimpleNamespace) -> None:
    rig.controller.err_code = [13, 13]
    robot = driver_module.build_robot(
        robot_config(execute=True, vel_ratio=32, active_arms=["left_arm"]), rig.clock
    )
    with pytest.raises(RuntimeError, match="fault outside active_arms.*arm B"):
        robot.connect(recover_errors=True)
    assert rig.controller.clear_attempts == [0, 0]
    assert not any(call[0] == "set_state" for call in rig.controller.calls)
    assert rig.controller.released


def test_read_only_connect_cannot_request_recovery(rig: SimpleNamespace) -> None:
    robot = driver_module.build_robot(robot_config(), rig.clock)
    with pytest.raises(ValueError, match="requires execute=true"):
        robot.connect(recover_errors=True)
    assert rig.loaded == []
    assert rig.controller.calls == []


def test_recovery_does_not_clear_healthy_arms(rig: SimpleNamespace) -> None:
    rig.controller.err_code[0] = 13
    robot = driver_module.build_robot(
        robot_config(execute=True, vel_ratio=32, active_arms=["left_arm"]), rig.clock
    )
    try:
        robot.connect(recover_errors=True)
        assert rig.controller.clear_attempts == [1, 0]
        assert not any(call[1:2] == ("B",) for call in rig.controller.calls)
    finally:
        robot.close()


def test_recovery_mode_switch_fault_disables_partially_enabled_arm(
    rig: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_read = marvin_module.MarvinSession.read

    def read(session):
        if rig.controller.cur_state[0] == 1:
            rig.controller.err_code[0] = 13
        return original_read(session)

    monkeypatch.setattr(marvin_module.MarvinSession, "read", read)
    robot = driver_module.build_robot(robot_config(execute=True, vel_ratio=32), rig.clock)
    with pytest.raises(RuntimeError, match="mode switch failed.*err_code 13"):
        robot.connect(recover_errors=True)
    assert ("set_state", "A", 0) in rig.controller.calls
    assert ("set_state", "B", 1) not in rig.controller.calls
    assert rig.controller.clear_attempts == [0, 0]
    assert rig.controller.sent == []
    assert rig.controller.released


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
    assert rig.grippers.targets == []
    robot.close()


@pytest.fixture
def home_rig(rig: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    def sleep(seconds: float) -> None:
        rig.clock.now += round(seconds * 1e9)

    monkeypatch.setattr(
        driver_module, "time",
        SimpleNamespace(monotonic=lambda: rig.clock.now_ns() / 1e9, sleep=sleep),
    )
    rig.grippers.positions = {"/dev/left": 0.0, "/dev/right": 0.1}
    rig.grippers.follow = True
    return rig


def test_return_home_recovers_estop_and_starts_at_measured_pose(
    home_rig: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from manimux.config import load_config
    from manimux.robots.tianji import recovery

    rig = home_rig
    rig.controller.joints[0][0] -= 4.0
    rig.controller.joints[1][0] += 3.0
    start = {arm: rig.controller.joints[index].copy() for arm, index in ARM_INDEX.items()}
    rig.controller.err_code = [13, 13]
    rig.controller.cur_state = [100, 100]
    config = load_config("configs/mock.yaml")
    config.robot = robot_config(execute=True, vel_ratio=32, gripper_control=True)
    config.viewer.robot_adapter = "tianji"
    monkeypatch.setattr(recovery, "SystemClock", lambda: rig.clock)
    controller = recovery.TianjiRecovery(config)
    controller.update({"recovery_request_id": "1", "recovery_request": "home"})
    controller._home_thread.join(2)
    controller.update({})
    assert not controller.busy
    assert controller.metadata()["error"] == ""
    assert rig.controller.clear_attempts == [1, 1]
    for arm, index in ARM_INDEX.items():
        np.testing.assert_allclose(rig.controller.sent[0][arm], start[arm])
        np.testing.assert_allclose(rig.controller.joints[index], HOME[index])
    assert all(value == pytest.approx(1.0) for value in rig.grippers.positions.values())
    assert rig.controller.cur_state == [0, 0]
    assert rig.controller.released
    assert rig.grippers.enabled == set()


@pytest.mark.parametrize("fault", ["estop", "disabled", "stale", "state_error"])
def test_home_aborts_on_new_fault_without_clearing_or_sending_more_targets(
    home_rig: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    rig = home_rig
    rig.controller.joints[0][0] -= 4.0
    robot = driver_module.build_robot(robot_config(execute=True, vel_ratio=32), rig.clock)
    robot.connect(recover_errors=True)
    session = robot._require_session()
    send = session.send_joints

    def send_then_fault(targets):
        send(targets)
        if fault == "estop":
            rig.controller.err_code[0] = 13
        elif fault == "disabled":
            rig.controller.cur_state[0] = 0
        elif fault == "state_error":
            rig.controller.cur_state[0] = 100
        else:
            rig.controller.freeze_serial = True
            rig.clock.now += 200_000_000

    monkeypatch.setattr(session, "send_joints", send_then_fault)
    try:
        with pytest.raises(RuntimeError, match="emergency stop|did not advance|robot error"):
            robot.home()
        assert len(rig.controller.sent) == 1
        assert rig.controller.clear_attempts == [0, 0]
        assert ("stop_running", "A") in rig.controller.calls
        assert ("stop_running", "B") in rig.controller.calls
    finally:
        robot.close()


@pytest.mark.parametrize("stop_first", [False, True])
@pytest.mark.parametrize("already_home", [False, True])
def test_home_opens_grippers_only_after_both_arms_arrive(
    home_rig: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
    stop_first: bool, already_home: bool,
) -> None:
    rig = home_rig
    if not already_home:
        rig.controller.joints[0][0] -= 4.0
        rig.controller.joints[1][0] += 4.0
    robot = build_robot(
        robot_config(execute=True, vel_ratio=32, gripper_control=True), rig.clock
    )
    robot.connect()
    try:
        if stop_first:
            robot.stop()
            assert rig.grippers.enabled == set()
        grippers = robot._grippers
        assert grippers is not None
        original_target = grippers.set_target
        original_send = robot._session.send_joints
        final_sends = 0

        def send(joints):
            nonlocal final_sends
            original_send(joints)
            if not already_home and all(
                np.array_equal(joints[arm], HOME[index]) for arm, index in ARM_INDEX.items()
            ):
                final_sends += 1
                # The controller accepts the final target, but arm B takes
                # several feedback cycles to arrive. Never release during lag.
                if final_sends < 4:
                    rig.controller.joints[1][0] += 1.0

        def target(arm, aperture):
            for index in (0, 1):
                np.testing.assert_allclose(rig.controller.joints[index], HOME[index])
            assert aperture == 1.0
            device = "/dev/left" if arm == "A" else "/dev/right"
            before = rig.grippers.positions[device]
            value = original_target(arm, aperture)
            assert 0.0 <= value - before <= 0.036 + 1e-12
            return value

        monkeypatch.setattr(robot._session, "send_joints", send)
        monkeypatch.setattr(grippers, "set_target", target)
        robot.home()
        assert rig.grippers.enabled == {"/dev/left", "/dev/right"}
        assert all(value >= 0.98 for value in rig.grippers.positions.values())
        assert len(rig.grippers.targets) > 2  # opening requires repeated limited targets
        if not already_home:
            assert final_sends >= 4
    finally:
        robot.close()
    assert rig.grippers.enabled == set()


def test_home_does_not_open_grippers_if_joints_fail_to_arrive(home_rig: SimpleNamespace) -> None:
    rig = home_rig
    rig.controller.joints[1][0] += 1.0  # below tracking limit, outside home tolerance
    rig.controller.follow = False
    robot = build_robot(
        robot_config(execute=True, vel_ratio=32, gripper_control=True), rig.clock
    )
    robot.connect()
    try:
        with pytest.raises(RuntimeError, match="joints did not reach home"):
            robot.home()
        assert rig.grippers.targets == []
        assert rig.grippers.enabled == set()
        assert ("stop_running", "B") in rig.controller.calls
    finally:
        robot.close()


def test_home_waits_for_both_grippers_and_times_out_if_one_is_stuck(
    home_rig: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = home_rig
    robot = build_robot(
        robot_config(execute=True, vel_ratio=32, gripper_control=True), rig.clock
    )
    robot.connect()
    original_target = robot._grippers.set_target

    def target(arm, aperture):
        value = original_target(arm, aperture)
        if arm == "B":
            rig.grippers.positions["/dev/right"] = 0.1
        return value

    monkeypatch.setattr(robot._grippers, "set_target", target)
    try:
        with pytest.raises(RuntimeError, match="grippers did not open"):
            robot.home()
        assert rig.grippers.positions["/dev/left"] >= 0.98
        assert rig.grippers.positions["/dev/right"] == 0.1
        assert rig.grippers.enabled == set()
        assert ("stop_running", "A") in rig.controller.calls
    finally:
        robot.close()


@pytest.mark.parametrize("fault", ["torque", "stale", "controller"])
def test_home_opening_stops_on_fault(
    home_rig: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    rig = home_rig
    robot = build_robot(
        robot_config(execute=True, vel_ratio=32, gripper_control=True), rig.clock
    )
    robot.connect()
    original_target = robot._grippers.set_target

    def target(arm, aperture):
        value = original_target(arm, aperture)
        if arm == "B":
            if fault == "torque":
                rig.grippers.torque = 2.0
            elif fault == "controller":
                rig.controller.err_code[0] = 13
            else:
                loop = robot._grippers._loops["A"]
                observation = loop.observation()
                observation.age_ms = 1000.0
                monkeypatch.setattr(loop, "observation", lambda: observation)
        return value

    monkeypatch.setattr(robot._grippers, "set_target", target)
    try:
        with pytest.raises(RuntimeError, match="gripper protection|stale|emergency stop"):
            robot.home()
        assert len(rig.grippers.targets) == 2
        assert rig.grippers.enabled == set()
        assert ("stop_running", "B") in rig.controller.calls
    finally:
        robot.close()


def test_home_does_not_restart_grippers_after_a_latched_fault(home_rig: SimpleNamespace) -> None:
    rig = home_rig
    robot = build_robot(
        robot_config(execute=True, vel_ratio=32, gripper_control=True), rig.clock
    )
    robot.connect()
    try:
        rig.grippers.torque = 2.0
        with pytest.raises(RuntimeError, match="gripper protection"):
            robot.send_command(command(robot.get_state().groups))
        rig.grippers.torque = 0.0
        with pytest.raises(RuntimeError, match="gripper protection"):
            robot.home()
        assert rig.grippers.enabled == set()
        assert rig.grippers.targets == []
    finally:
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
    assert motion.arm.mode == "isotropic"
    assert motion.arm.max_step_dt_s == 0.016
    assert motion.arm.max_acceleration is None and safety.max_acceleration == {}
    for group in ("left_arm", "right_arm"):
        lower = np.asarray(safety.position_lower[group])
        upper = np.asarray(safety.position_upper[group])
        home = np.radians(HOME[0 if group == "left_arm" else 1])
        assert lower.shape == upper.shape == (8,)
        assert np.all(lower[:7] < home) and np.all(home < upper[:7])
        # The executor shapes commands strictly inside the runtime guard.
        assert motion.arm.max_velocity < min(safety.max_velocity[group][:7])
        index = motion.gripper.group_indices[group]
        assert motion.gripper.max_velocity < safety.max_velocity[group][index]
    assert safety.position_upper["right_arm"][5] < safety.position_upper["left_arm"][5]
