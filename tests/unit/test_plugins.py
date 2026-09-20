from __future__ import annotations

import numpy as np
import pytest

from manimux.cli import load_config
from manimux.clock import SystemClock
from manimux.embodiments.robot import build_robot, robot_parameters
from manimux.embodiments.sensor import build_sensor, sensor_parameters
from manimux.plugins import PluginError, load_plugin
from manimux.policies import build_policy_model
from manimux.policies.base import policy_parameters
from manimux.policies.fake import FakePolicyAdapter, FakePolicyModel
from manimux.policy_adapter import build_policy_adapter
from manimux.types import ActionChunk, ActionContext
from tests.support.robot import RobotDouble
from tests.support.sensor import SensorDouble


def test_test_fixture_uses_explicit_robot_plugin() -> None:
    config = load_config("tests/fixtures/runtime.yaml")

    assert config["robot"]["options"] == {}
    assert config["sensors"][0]["options"] == {}
    assert config["policy"]["options"] == {}
    assert isinstance(build_robot(config["robot"], SystemClock()), RobotDouble)
    assert isinstance(build_sensor(config["sensors"][0], SystemClock()), SensorDouble)
    assert isinstance(build_policy_model(config["policy"]), FakePolicyModel)
    assert isinstance(build_policy_adapter(config["robot"], config["policy"]), FakePolicyAdapter)


def test_plugin_options_allow_config_driven_extensions() -> None:
    robot = robot_parameters(
        type="tests.support.robot:build_robot",
        group_dims={"arm": 2},
        options={"vendor_mode": "position"},
    )
    sensor = sensor_parameters(
        name="camera",
        driver="tests.support.sensor:build_sensor",
        options={"endpoint": "tcp://127.0.0.1:5555"},
    )
    policy = policy_parameters(
        worker="fake",
        adapter={"type": "manimux.policies.fake:FakePolicyAdapter"},
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


@pytest.mark.parametrize(
    "name", ["not_registered", "mock_dual_arm", "maniunicon_meshcat_dual_arm", "tianji_dual"]
)
def test_unknown_plugin_fails_before_runtime_touches_hardware(name) -> None:
    with pytest.raises(PluginError, match="unknown manimux.embodiments.robot plugin"):
        build_robot(
            robot_parameters(type=name, group_dims={"arm": 1}),
            SystemClock(),
        )


def test_policy_adapter_receives_the_decode_context() -> None:
    chunk = ActionChunk(
        plan_id="context",
        request_seq=1,
        observation_time_ns=1,
        created_time_ns=2,
        action_space="joint_position",
        dt_ns=10,
        groups={"arm": np.zeros((2, 1))},
    )

    class ContextAdapter:
        def decode_action(self, raw: object, context: ActionContext) -> ActionChunk:
            assert context.request_seq == raw.request_seq
            assert isinstance(raw, ActionChunk)
            return raw

    decoded = ContextAdapter().decode_action(  # type: ignore[arg-type]
        chunk,
        ActionContext(request_seq=1, observation_time_ns=1, created_time_ns=2),
    )
    assert decoded is chunk
