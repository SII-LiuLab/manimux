from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from manimux.cli import load_config
from manimux.clock import SystemClock
from manimux.integrations.molmoact_yam.policy_plugin import (
    MolmoActHttpPolicyModel,
    MolmoActYamAdapter,
)
from manimux.policies import build_policy_adapter, build_policy_model
from manimux.policies.base import action_interval
from manimux.robots import build_robot
from manimux.robots.yam import YamDualArmDriver
from manimux.embodiments.sensor import build_sensor
from manimux.server.sensor.taccap import CameraServerSensorDriver
from manimux.types import ActionContext, RobotCommand


def test_molmoact_yam_run_config_selects_real_plugins_without_touching_hardware() -> None:
    config = load_config(Path("configs/molmoact2/yam/infra/manimux.yaml"))

    assert isinstance(build_robot(config["robot"], SystemClock()), YamDualArmDriver)
    assert isinstance(build_sensor(config["sensors"][0], SystemClock()), CameraServerSensorDriver)
    assert isinstance(build_policy_model(config["policy"]), MolmoActHttpPolicyModel)
    assert isinstance(build_policy_adapter(config["robot"], config["policy"]), MolmoActYamAdapter)


def test_molmoact_adapter_splits_raw_actions_into_canonical_yam_groups() -> None:
    config = load_config("configs/molmoact2/yam/infra/manimux.yaml")
    adapter = build_policy_adapter(config["robot"], config["policy"])
    raw = np.arange(30 * 14, dtype=np.float64).reshape(30, 14)

    chunk = adapter.decode_action(
        raw,
        ActionContext(request_seq=4, observation_time_ns=10, created_time_ns=20),
    )

    assert chunk.action_space == "joint_position"
    assert chunk.request_seq == 4
    assert chunk.dt_ns == int(config["policy"]["action_dt_s"] * 1_000_000_000)
    np.testing.assert_array_equal(chunk.groups["left_arm"], raw[:, :7])
    np.testing.assert_array_equal(chunk.groups["right_arm"], raw[:, 7:])


def test_molmoact_adapter_rejects_wrong_action_width() -> None:
    config = load_config("configs/molmoact2/yam/infra/manimux.yaml")
    adapter = build_policy_adapter(config["robot"], config["policy"])

    with pytest.raises(ValueError, match="shape"):
        adapter.decode_action(
            np.zeros((30, 13)),
            ActionContext(request_seq=1, observation_time_ns=10, created_time_ns=20),
        )


class _FakeBimanualYam:
    def __init__(self) -> None:
        self.state = np.linspace(0.0, 1.3, 14)
        self.command: np.ndarray | None = None

    def get_joint_state(self) -> np.ndarray:
        return self.state.copy()

    def command_joint_state(self, command: np.ndarray) -> None:
        self.command = np.asarray(command, dtype=np.float64).copy()


class _FakeNativeYam:
    def __init__(self, position: np.ndarray) -> None:
        self.position = np.asarray(position, dtype=np.float64).copy()
        self.moves: list[tuple[np.ndarray, float]] = []
        self.closed = False

    def move_joints(self, target: np.ndarray, time_interval_s: float) -> None:
        self.position = np.asarray(target, dtype=np.float64).copy()
        self.moves.append((self.position.copy(), time_interval_s))

    def close(self) -> None:
        self.closed = True


class _FakeYamArm:
    def __init__(self, position: np.ndarray) -> None:
        self.robot = _FakeNativeYam(position)

    def get_joint_state(self) -> np.ndarray:
        return self.robot.position.copy()

    def close(self) -> None:
        self.robot.close()


class _FakeBimanualHardware:
    def __init__(self) -> None:
        self._robot_l = _FakeYamArm(np.linspace(-0.5, 0.5, 7))
        self._robot_r = _FakeYamArm(np.linspace(0.5, -0.5, 7))

    def get_joint_state(self) -> np.ndarray:
        return np.concatenate([self._robot_l.get_joint_state(), self._robot_r.get_joint_state()])

    def command_joint_state(self, command: np.ndarray) -> None:
        self._robot_l.robot.position = np.asarray(command[:7], dtype=np.float64).copy()
        self._robot_r.robot.position = np.asarray(command[7:], dtype=np.float64).copy()


def test_yam_driver_maps_grouped_move_to_existing_joint_command() -> None:
    config = load_config("configs/molmoact2/yam/infra/manimux.yaml")
    driver = YamDualArmDriver(config["robot"], SystemClock())
    backend = _FakeBimanualYam()
    driver._robot = backend

    state = driver.get_state()
    np.testing.assert_array_equal(state.groups["left_arm"], backend.state[:7])
    np.testing.assert_array_equal(state.groups["right_arm"], backend.state[7:])

    command = RobotCommand(
        groups={"left_arm": np.ones(7), "right_arm": np.full(7, 2.0)},
        monotonic_ns=100,
        plan_id="plan",
    )
    driver.send_command(command)
    assert backend.command is not None
    expected = np.r_[np.ones(7), np.full(7, 2.0)]
    np.testing.assert_array_equal(backend.command, expected)


def test_an_unknown_robot_option_is_refused_before_the_arms_move() -> None:
    """``robot.options`` is free-form, so a typo would be ignored in silence.

    The arms would still move -- just not the way the config says. Reject the
    key at construction instead, before anything opens CAN.
    """
    config = load_config("configs/molmoact2/yam/infra/manimux.yaml")
    config["robot"]["options"]["start_duration"] = 1.0  # missing the _s suffix

    with pytest.raises(ValueError, match="start_duration"):
        YamDualArmDriver(config["robot"], SystemClock())


def test_live_config_enables_explicit_start_and_verified_home() -> None:
    config = load_config("configs/molmoact2/yam/infra/manimux.yaml")

    assert config["robot"]["options"]["move_to_start_on_connect"] is True
    assert config["robot"]["options"]["home_on_close"] is True
    assert config["robot"]["control_hz"] == 100.0
    assert config["robot"]["options"]["start_duration_s"] == 3.0
    assert config["robot"]["options"]["home_duration_s"] == 3.0
    assert action_interval(config["policy"]) == pytest.approx(0.05)


def test_yam_live_close_releases_without_implicit_motion() -> None:
    config = load_config("configs/molmoact2/yam/infra/manimux.yaml")
    driver = YamDualArmDriver(config["robot"], SystemClock())
    backend = _FakeBimanualHardware()
    driver._robot = backend

    driver.close()

    for arm in (backend._robot_l, backend._robot_r):
        assert arm.robot.moves == []
        assert arm.robot.closed
    assert driver._robot is None


def test_yam_wrapper_joins_control_thread_before_closing_can() -> None:
    try:
        from manimux.robots.yam import YAMRobot
    except ImportError:
        pytest.skip("i2rt hardware extra is not installed")

    events: list[str] = []

    class _MotorChain:
        def __init__(self) -> None:
            self.running = True

        def control_loop(self) -> None:
            while self.running:
                time.sleep(0.001)
            time.sleep(0.02)
            events.append("control_stopped")

        def close(self) -> None:
            assert "control_stopped" in events
            events.append("can_closed")

    class _Native:
        def __init__(self) -> None:
            self._stop_event = threading.Event()
            self.motor_chain = _MotorChain()
            self._server_thread = threading.Thread(target=self.server_loop)
            self._server_thread.start()

        def server_loop(self) -> None:
            self._stop_event.wait()
            events.append("server_stopped")

    native = _Native()
    control_thread = threading.Thread(target=native.motor_chain.control_loop)
    control_thread.start()
    arm = object.__new__(YAMRobot)
    arm.robot = native

    arm.close()

    assert events == ["server_stopped", "control_stopped", "can_closed"]
    assert not control_thread.is_alive()
