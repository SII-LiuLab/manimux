from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from manimux.embodiments.sensor.camera_server import CameraServerSensorDriver
from manimux.cli import load_config
from manimux.clock import SystemClock
from manimux.embodiments.robot import build_robot
from manimux.embodiments.robot.yam import YamRobot
from manimux.embodiments.sensor import build_sensor
from manimux.policies.molmoact import MolmoActHttpPolicyModel
from manimux.policies import build_policy_model
from manimux.policies.base import action_interval
from manimux.policy_adapter import build_policy_adapter
from manimux.policy_adapter.molmoact_yam import MolmoActYamAdapter
from manimux.types import ActionContext


def test_molmoact_yam_run_config_selects_real_plugins_without_touching_hardware() -> None:
    config = load_config(Path("manimux/configs/experiments/pick_red_object/molmoact2/yam_molmoact2_manimux.yaml"))

    assert isinstance(build_robot(config["robot"], SystemClock()), YamRobot)
    assert isinstance(build_sensor(config["sensors"][0], SystemClock()), CameraServerSensorDriver)
    assert isinstance(build_policy_model(config["policy"]), MolmoActHttpPolicyModel)
    assert isinstance(build_policy_adapter(config["robot"], config["policy"]), MolmoActYamAdapter)


def test_molmoact_adapter_splits_raw_actions_into_canonical_yam_groups() -> None:
    config = load_config("manimux/configs/experiments/pick_red_object/molmoact2/yam_molmoact2_manimux.yaml")
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
    config = load_config("manimux/configs/experiments/pick_red_object/molmoact2/yam_molmoact2_manimux.yaml")
    adapter = build_policy_adapter(config["robot"], config["policy"])

    with pytest.raises(ValueError, match="shape"):
        adapter.decode_action(
            np.zeros((30, 13)),
            ActionContext(request_seq=1, observation_time_ns=10, created_time_ns=20),
        )


def test_an_unknown_robot_option_is_refused_before_the_arms_move() -> None:
    """``robot.options`` is free-form, so a typo would be ignored in silence.

    The arms would still move -- just not the way the config says. Reject the
    key at construction instead, before anything opens CAN.
    """
    config = load_config("manimux/configs/experiments/pick_red_object/molmoact2/yam_molmoact2_manimux.yaml")
    config["robot"]["options"]["start_duration"] = 1.0  # missing the _s suffix

    with pytest.raises(TypeError, match="start_duration"):
        build_robot(config["robot"], SystemClock())


def test_live_config_enables_explicit_start_and_verified_home() -> None:
    config = load_config("manimux/configs/experiments/pick_red_object/molmoact2/yam_molmoact2_manimux.yaml")

    assert config["robot"]["options"]["move_to_start_on_connect"] is True
    assert config["robot"]["options"]["home_on_close"] is True
    assert config["robot"]["control_hz"] == 100.0
    assert config["robot"]["options"]["start_duration_s"] == 3.0
    assert config["robot"]["options"]["home_duration_s"] == 3.0
    assert action_interval(config["policy"]) == pytest.approx(0.05)


def test_yam_wrapper_joins_control_thread_before_closing_can() -> None:
    from manimux.embodiments.arm.yam import YamController

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
    arm = YamController(channel="test")
    arm.robot = native

    arm.close()

    assert events == ["server_stopped", "control_stopped", "can_closed"]
    assert not control_thread.is_alive()
