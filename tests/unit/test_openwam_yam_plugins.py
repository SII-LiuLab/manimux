from __future__ import annotations

import json

import numpy as np
import pytest

import manimux.integrations.openwam_yam.policy_plugin as openwam_plugin
from manimux.config import PolicyConfig, RobotConfig, load_config
from manimux.integrations.openwam_yam.policy_plugin import (
    ACTION_DIM,
    DEFAULT_PROMPT_TEMPLATE,
    OpenWAMInferenceRequest,
    OpenWAMWsPolicyModel,
    OpenWAMYamAdapter,
    matrix_to_rot6d,
    rot6d_to_matrix,
)
from manimux.integrations.openwam_yam.ws_client import (
    OpenWAMProtocolError,
    OpenWAMWsClient,
)
from manimux.policies import build_policy_adapter, build_policy_model
from manimux.types import (
    ActionContext,
    InferenceRequest,
    ObservationSnapshot,
    RobotState,
    SensorFrame,
)


class _FakeKinematics:
    num_arm_joints = 6

    def fk(self, joints, gripper):
        del gripper
        pose = np.eye(4)
        pose[:3, 3] = np.asarray(joints)[:3]
        return pose

    def ik(self, target, seed, gripper):
        del gripper
        solved = np.asarray(seed, dtype=np.float64).copy()
        solved[:3] = target[:3, 3]
        return True, solved

    def clip_arm_joints(self, joints):
        return np.asarray(joints, dtype=np.float64)


def _config() -> tuple[RobotConfig, PolicyConfig]:
    robot = RobotConfig(
        driver="fake",
        group_dims={"left_arm": 7, "right_arm": 7},
    )
    policy = PolicyConfig(
        worker="openwam_ws",
        adapter="openwam_yam",
        action_dt_s=1 / 30,
        timeout_s=300,
        horizon_steps=32,
        options={},
    )
    return robot, policy


def _snapshot() -> ObservationSnapshot:
    state = RobotState(
        groups={
            "left_arm": np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.25]),
            "right_arm": np.array([-0.1, -0.2, 0.4, 0.0, 0.0, 0.0, 0.75]),
        },
        monotonic_ns=1,
        sequence=1,
    )
    frames = {
        name: SensorFrame(name, np.zeros((4, 5, 3), dtype=np.uint8), 1, 1)
        for name in ("front_camera", "left_camera", "right_camera")
    }
    return ObservationSnapshot(state=state, frames=frames)


def test_rot6d_uses_first_two_rotation_columns() -> None:
    angle = 0.4
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    encoded = matrix_to_rot6d(rotation)
    np.testing.assert_allclose(encoded, np.concatenate([rotation[:, 0], rotation[:, 1]]))
    np.testing.assert_allclose(rot6d_to_matrix(encoded), rotation, atol=1e-12)


def test_prepare_request_packs_absolute_eef20(monkeypatch) -> None:
    monkeypatch.setattr(
        "manimux.kinematics.build_kinematics",
        lambda *args, **kwargs: _FakeKinematics(),
    )
    robot, policy = _config()
    adapter = OpenWAMYamAdapter(robot, policy)
    request = InferenceRequest("session", 4, 1, 2, _snapshot(), "put bottle in bin")
    prepared = adapter.prepare_request(request)

    assert isinstance(prepared, OpenWAMInferenceRequest)
    assert prepared.openwam_state.shape == (ACTION_DIM,)
    identity = [1, 0, 0, 0, 1, 0]
    np.testing.assert_allclose(prepared.openwam_state[:10], [0.1, 0.2, 0.3, *identity, 0.25])
    np.testing.assert_allclose(prepared.openwam_state[10:], [-0.1, -0.2, 0.4, *identity, 0.75])
    assert prepared.openwam_prompt == DEFAULT_PROMPT_TEMPLATE.format(
        instruction="put bottle in bin"
    )


def test_absolute_eef_chunk_is_solved_to_joint_chunk(monkeypatch) -> None:
    monkeypatch.setattr(
        "manimux.kinematics.build_kinematics",
        lambda *args, **kwargs: _FakeKinematics(),
    )
    robot, policy = _config()
    adapter = OpenWAMYamAdapter(robot, policy)
    request = InferenceRequest("session", 5, 1, 2, _snapshot(), "task")
    adapter.prepare_request(request)

    actions = np.zeros((32, ACTION_DIM), dtype=np.float64)
    identity = [1, 0, 0, 0, 1, 0]
    actions[:, 3:9] = identity
    actions[:, 13:19] = identity
    actions[:, :3] = np.linspace([0.1, 0.2, 0.3], [0.2, 0.3, 0.4], 32)
    actions[:, 10:13] = np.linspace([-0.1, -0.2, 0.4], [0.0, -0.1, 0.5], 32)
    actions[:, 9] = 0.25
    actions[:, 19] = 0.75

    chunk = adapter.decode_action(actions, ActionContext(5, 1, 2))
    assert chunk.horizon_steps == 32
    np.testing.assert_allclose(chunk.groups["left_arm"][:, :3], actions[:, :3])
    np.testing.assert_allclose(chunk.groups["right_arm"][:, :3], actions[:, 10:13])
    np.testing.assert_allclose(chunk.groups["left_arm"][:, 6], 0.25)
    np.testing.assert_allclose(chunk.groups["right_arm"][:, 6], 0.75)
    assert chunk.metadata["native_action_space"] == "absolute_eef20"


class _FakeConnection:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.sent = []

    def send(self, body):
        self.sent.append(json.loads(body))

    def recv(self, timeout=None):
        del timeout
        return json.dumps(next(self.replies))

    def close(self):
        return None


class _FakeOpenWAMClient:
    pong = {
        "type": "pong",
        "representation": "eef",
        "gripper_convention": "zero_closed_one_open",
        "transport_mode": "manimux_chunk",
        "action_horizon": 32,
    }

    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.closed = False

    def connect(self):
        return None

    def ping(self):
        return dict(self.pong)

    def reset(self):
        return None

    def close(self):
        self.closed = True


def test_model_handshake_pins_eef_gripper_and_chunk_contract(monkeypatch) -> None:
    monkeypatch.setattr(openwam_plugin, "OpenWAMWsClient", _FakeOpenWAMClient)
    _, policy = _config()
    model = OpenWAMWsPolicyModel(policy)
    model.reset("session")
    assert model.capabilities().backend_metadata["model"] == {
        key: value for key, value in _FakeOpenWAMClient.pong.items() if key != "type"
    }
    model.close()


def test_model_rejects_wrong_gripper_handshake(monkeypatch) -> None:
    class WrongGripperClient(_FakeOpenWAMClient):
        pong = {**_FakeOpenWAMClient.pong, "gripper_convention": "one_closed_zero_open"}

    monkeypatch.setattr(openwam_plugin, "OpenWAMWsClient", WrongGripperClient)
    _, policy = _config()
    model = OpenWAMWsPolicyModel(policy)
    with pytest.raises(RuntimeError, match="zero_closed_one_open"):
        model.reset("session")


def test_ws_client_receives_one_server_chunk() -> None:
    client = OpenWAMWsClient("ws://unused")
    connection = _FakeConnection(
        [{"type": "action", "action": [[1.0, 0.5], [2.0, 0.5], [3.0, 0.5]], "step": 1}]
    )
    client._conn = connection
    actions = client.infer_chunk(
        {"images": {}, "prompt": "task", "state": []}, horizon_steps=3
    )
    assert actions == [[1.0, 0.5], [2.0, 0.5], [3.0, 0.5]]
    assert len(connection.sent) == 1
    assert connection.sent[0]["type"] == "obs"


def test_ws_client_can_drain_official_stream_compatibility_mode() -> None:
    replies = [
        {"type": "action", "action": [float(step), 0.5], "step": step}
        for step in range(1, 4)
    ]
    client = OpenWAMWsClient("ws://unused")
    connection = _FakeConnection(replies)
    client._conn = connection
    actions = client.infer_chunk(
        {"images": {}, "prompt": "task", "state": []},
        horizon_steps=3,
        response_mode="stream",
    )
    assert actions == [[1.0, 0.5], [2.0, 0.5], [3.0, 0.5]]
    assert all(message["type"] == "obs" for message in connection.sent)


def test_ws_client_rejects_interleaved_server_steps() -> None:
    client = OpenWAMWsClient("ws://unused")
    client._conn = _FakeConnection(
        [
            {"type": "action", "action": [0.0], "step": 1},
            {"type": "action", "action": [0.0], "step": 3},
        ]
    )
    with pytest.raises(OpenWAMProtocolError, match="not exclusive"):
        client.infer_chunk(
            {"images": {}, "prompt": "task", "state": []},
            horizon_steps=2,
            response_mode="stream",
        )


def test_openwam_config_registers_model_and_adapter(monkeypatch) -> None:
    monkeypatch.setattr(
        "manimux.kinematics.build_kinematics",
        lambda *args, **kwargs: _FakeKinematics(),
    )
    config = load_config("configs/openwam/yam/infra/manimux.yaml")
    assert isinstance(build_policy_model(config.policy), OpenWAMWsPolicyModel)
    assert isinstance(build_policy_adapter(config.robot, config.policy), OpenWAMYamAdapter)
