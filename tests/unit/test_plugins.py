from __future__ import annotations

import numpy as np
import pytest

from manimux.cli import load_config
from manimux.clock import SystemClock
from manimux.embodiments.robot import robot_parameters
from manimux.plugins import PluginError, load_plugin
from manimux.policies import build_policy_adapter, build_policy_model
from manimux.policies.base import decode_policy_action, policy_parameters
from manimux.policies.fake import FakePolicyAdapter, FakePolicyModel
from manimux.robots import build_robot
from manimux.robots.mock import MockDualArmDriver
from manimux.embodiments.sensor import build_sensor, sensor_parameters
from manimux.embodiments.sensor.mock import MockCameraDriver
from manimux.types import ActionChunk, ActionContext


def test_existing_mock_config_uses_compatible_plugin_defaults() -> None:
    config = load_config("configs/mock.yaml")

    assert config["robot"]["options"] == {}
    assert config["sensors"][0]["options"] == {}
    assert config["policy"]["options"] == {}
    assert isinstance(build_robot(config["robot"], SystemClock()), MockDualArmDriver)
    assert isinstance(build_sensor(config["sensors"][0], SystemClock()), MockCameraDriver)
    assert isinstance(build_policy_model(config["policy"]), FakePolicyModel)
    assert isinstance(build_policy_adapter(config["robot"], config["policy"]), FakePolicyAdapter)


def test_plugin_options_allow_config_driven_extensions() -> None:
    robot = robot_parameters(
        driver="mock_dual_arm",
        group_dims={"arm": 2},
        options={"vendor_mode": "position"},
    )
    sensor = sensor_parameters(
        name="camera",
        driver="mock_camera",
        options={"endpoint": "tcp://127.0.0.1:5555"},
    )
    policy = policy_parameters(
        worker="fake",
        adapter="identity",
        options={"server": "http://127.0.0.1:8202"},
    )

    assert robot["options"]["vendor_mode"] == "position"
    assert sensor["options"]["endpoint"] == "tcp://127.0.0.1:5555"
    assert policy["options"]["server"] == "http://127.0.0.1:8202"


def test_explicit_module_plugin_is_loadable_without_installing_entry_point() -> None:
    plugin = load_plugin(
        "manimux.policies.fake:FakePolicyAdapter",
        group="manimux.policies.adapters",
        builtins={},
    )
    assert plugin is FakePolicyAdapter


def test_unknown_plugin_fails_before_runtime_touches_hardware() -> None:
    with pytest.raises(PluginError, match="unknown manimux.robots plugin"):
        build_robot(
            robot_parameters(driver="not_registered", group_dims={"arm": 1}),
            SystemClock(),
        )


def test_original_one_argument_policy_adapter_remains_compatible() -> None:
    chunk = ActionChunk(
        plan_id="legacy",
        request_seq=1,
        observation_time_ns=1,
        created_time_ns=2,
        action_space="joint_position",
        dt_ns=10,
        groups={"arm": np.zeros((2, 1))},
    )

    class LegacyAdapter:
        def decode_action(self, raw: object) -> ActionChunk:
            assert isinstance(raw, ActionChunk)
            return raw

    decoded = decode_policy_action(
        LegacyAdapter(),  # type: ignore[arg-type]
        chunk,
        ActionContext(request_seq=1, observation_time_ns=1, created_time_ns=2),
    )
    assert decoded is chunk
