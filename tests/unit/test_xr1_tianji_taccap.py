"""Offline contract tests for Xiaomi Robotics 1 on Tianji–TacCap."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from manimux.cli import load_config
from manimux.kinematics.robot import RobotKinematics
from manimux.kinematics.tianji_diff import rotation_matrix, rotation_vector
from manimux.policy_adapter.xr1.tianji import (
    ACTION_DIM,
    XR1TianjiTacCapAdapter,
)
from manimux.kinematics.base import IKResult
from manimux.types import (
    ActionContext,
    InferenceRequest,
    ObservationSnapshot,
    RobotState,
    SensorFrame,
)

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = (
    ROOT / "manimux/configs/experiments/pass_ball/tianji_taccap_xiaomi_xr1_step50000.yaml"
)


class FakeManipulator:
    def __init__(self, *, fail=False):
        self.arm = SimpleNamespace()
        self.fail = fail
        self.targets = []

    def reset(self):
        return None

    def fk(self, configuration):
        q = np.asarray(configuration, dtype=np.float64)
        pose = np.eye(4)
        pose[:3, 3] = q[:3]
        pose[:3, :3] = rotation_matrix(q[3:6])
        return pose

    def ik(self, target, seed, *, fixed_coordinates):
        self.targets.append(np.asarray(target).copy())
        if self.fail:
            return IKResult(False, reason="offline_test_failure")
        q = np.asarray(seed, dtype=np.float64).copy()
        q[:3] = target[:3, 3]
        q[3:6] = rotation_vector(target[:3, :3])
        q[-1] = fixed_coordinates["gripper"]
        return IKResult(True, joints=q)


def _adapter(*, fail=False):
    config = load_config(EXPERIMENT)
    kinematics = RobotKinematics(
        {name: FakeManipulator(fail=fail) for name in ("left_arm", "right_arm")}
    )
    adapter = XR1TianjiTacCapAdapter(
        config["robot"], config["policy"], kinematics=kinematics
    )
    adapter.validate(config["robot"], config["policy"])
    return config, adapter


def _request(adapter, request_seq=4):
    now = 10_000_000_000
    groups = {
        "left_arm": np.r_[np.zeros(7), 0.4],
        "right_arm": np.r_[np.zeros(7), 0.6],
    }
    frames = {
        name: SensorFrame(name, np.full((8, 12, 3), 7, dtype=np.uint8), now, 1)
        for name in adapter._required_cameras
    }
    snapshot = ObservationSnapshot(RobotState(groups, now, 1), frames)
    request = InferenceRequest("offline", request_seq, now, now + 10**9, snapshot)
    context = ActionContext(request_seq, now, now + 1, measured_state=snapshot.state)
    return request, context


def test_config_requires_only_two_physical_cameras_and_black_ego_identity():
    config, adapter = _adapter()
    assert set(adapter._camera_map) == {"cam_left_wrist", "cam_right_wrist"}
    assert "cam_head" not in adapter._camera_map
    identity = config["policy"]["expected_backend"]["model"]
    assert identity["ego_view_mode"] == "black"
    assert identity["observation_profile"] == "tianji_taccap_two_wrist_black_ego"
    assert config["executor"]["smooth"]["gripper"]["mode"] == "continuous"
    assert config["robot"]["options"]["execute"] is False


def test_anchor_relative_delta_decodes_to_tianji_joint_chunk():
    _, adapter = _adapter()
    request, context = _request(adapter)
    adapter.prepare_request(request)
    actions = np.zeros((30, ACTION_DIM), dtype=np.float32)
    actions[:, 0] = 0.01
    actions[:, 5] = 0.02
    actions[:, 6] = 0.1
    actions[:, 9] = -0.02
    actions[:, 14] = -0.1
    actions[:, 16:20] = 9.0  # Waist/base slots are unsupported and ignored explicitly.

    chunk = adapter.decode_action(actions, context)

    assert chunk.action_space == "joint_position"
    assert chunk.dt_ns == round(1e9 / 30)
    np.testing.assert_allclose(chunk.groups["left_arm"][:, 0], 0.01)
    np.testing.assert_allclose(chunk.groups["left_arm"][:, 5], 0.02)
    np.testing.assert_allclose(chunk.groups["left_arm"][:, -1], 0.5)
    np.testing.assert_allclose(chunk.groups["right_arm"][:, 1], -0.02)
    np.testing.assert_allclose(chunk.groups["right_arm"][:, -1], 0.5)
    assert chunk.metadata["ignored_waist_base_max_abs"] == 9.0


def test_bad_action_shape_and_gripper_range_are_rejected():
    _, adapter = _adapter()
    request, context = _request(adapter)
    adapter.prepare_request(request)
    with pytest.raises(ValueError, match="shape"):
        adapter.decode_action(np.zeros((29, ACTION_DIM)), context)

    request, context = _request(adapter, request_seq=5)
    adapter.prepare_request(request)
    actions = np.zeros((30, ACTION_DIM))
    actions[:, 6] = 0.7
    with pytest.raises(ValueError, match="outside"):
        adapter.decode_action(actions, context)


def test_any_ik_failure_rejects_the_whole_chunk():
    _, adapter = _adapter(fail=True)
    request, context = _request(adapter)
    adapter.prepare_request(request)
    with pytest.raises(ValueError, match="offline_test_failure"):
        adapter.decode_action(np.zeros((30, ACTION_DIM)), context)
