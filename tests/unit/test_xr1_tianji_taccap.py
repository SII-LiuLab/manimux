"""Offline contract tests for Xiaomi Robotics 1 on Tianji–TacCap."""

import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from manimux.cli import load_config
from manimux.kinematics.base import IKResult
from manimux.kinematics.robot import RobotKinematics
from manimux.kinematics.tianji_diff import rotation_matrix, rotation_vector
from manimux.policies.decoder import ActionDecoderClient
from manimux.policy_adapter.xr1.tianji import (
    ACTION_DIM,
    ARM_SLICES,
    XR1TianjiTacCapAdapter,
)
from manimux.types import (
    ActionContext,
    InferenceRequest,
    InferenceResponse,
    ObservationSnapshot,
    RobotState,
    SensorFrame,
)

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "manimux/configs/experiments/pass_ball/tianji_taccap_xiaomi_xr1_step50000.yaml"
START = np.radians([50.0, -40.0, -30.0, -100.0, -65.0, 0.0, 40.0])


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
    policy = deepcopy(config["policy"])
    # Unit fakes exercise the contract without Tianji's flange Jacobian.
    policy["action_decoding"] = "inline"
    policy["adapter"]["ik_backend"] = "analytic"
    policy["adapter"].pop("diff_ik", None)
    kinematics = RobotKinematics(
        {name: FakeManipulator(fail=fail) for name in ("left_arm", "right_arm")}
    )
    adapter = XR1TianjiTacCapAdapter(config["robot"], policy, kinematics=kinematics)
    adapter.validate(config["robot"], policy)
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
    assert config["policy"]["action_decoding"] == "process"
    assert config["policy"]["adapter"]["ik_backend"] == "diff"
    assert adapter.supports_context_only_decode
    assert adapter.decode_seed_source == "observation_state"
    diff_ik = config["policy"]["adapter"]["diff_ik"]
    motion = config["executor"]["motion_limits"]["arm"]
    assert diff_ik["max_velocity_rad_s"] == motion["max_velocity"]
    assert diff_ik["dt_max_s"] == motion["max_step_dt_s"]
    assert config["inference"]["strategy"] is None
    assert config["inference"]["commit_lead_s"] == 0.08
    assert config["inference"]["expected_decode_s"] == 0.08
    assert config["inference"]["decode_forecast_size"] == 10
    assert config["inference"]["independent_group_decoding"] is False
    assert adapter._gripper_clip_tolerance == 0.1
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
    assert chunk.metadata["gripper_clip_max_abs"] == 0.0
    assert chunk.metadata["ik_seed_source"] == "observation_state"
    assert chunk.metadata["ik_seed_time_ns"] == context.observation_time_ns


def test_small_gripper_boundary_overshoot_is_clipped():
    _, adapter = _adapter()
    request, context = _request(adapter)
    adapter.prepare_request(request)
    actions = np.zeros((30, ACTION_DIM), dtype=np.float32)
    actions[:, 6] = 0.65  # Left anchor 0.4 -> raw 1.05 -> clipped 1.0.
    actions[:, 14] = -0.65  # Right anchor 0.6 -> raw -0.05 -> clipped 0.0.

    chunk = adapter.decode_action(actions, context)

    np.testing.assert_allclose(chunk.groups["left_arm"][:, -1], 1.0)
    np.testing.assert_allclose(chunk.groups["right_arm"][:, -1], 0.0)
    assert chunk.metadata["gripper_clip_tolerance"] == 0.1
    assert chunk.metadata["gripper_clip_max_abs"] == pytest.approx(0.05)


def test_left_and_right_partition_decode_matches_full_chunk():
    actions = np.zeros((30, ACTION_DIM), dtype=np.float32)
    actions[:, 0] = 0.01
    actions[:, 6] = 0.65  # Global clip metadata must match in both partitions.
    actions[:, 9] = -0.02
    actions[:, 14] = -0.61
    actions[:, 16:20] = 3.0

    chunks = {}
    for partition in (None, "left_arm", "right_arm"):
        _, adapter = _adapter()
        _, context = _request(adapter, request_seq=8)
        chunks[partition] = (
            adapter.decode_action(actions, context)
            if partition is None
            else adapter.decode_action_partition(actions, context, partition)
        )

    full = chunks[None]
    assert XR1TianjiTacCapAdapter.decode_partitions == ("left_arm", "right_arm")
    for partition in XR1TianjiTacCapAdapter.decode_partitions:
        chunk = chunks[partition]
        assert set(chunk.groups) == {partition}
        np.testing.assert_allclose(chunk.groups[partition], full.groups[partition])
        assert chunk.metadata == full.metadata

    _, adapter = _adapter()
    with pytest.raises(ValueError, match="unknown XR-1 Tianji decode partition"):
        adapter.decode_action_partition(actions, ActionContext(9, 0, 0), "waist")


def test_decode_requires_the_request_observation_state():
    _, adapter = _adapter()
    actions = np.zeros((30, ACTION_DIM))
    with pytest.raises(ValueError, match="no observation state"):
        adapter.decode_action(actions, ActionContext(9, 10, 11))

    state = RobotState(
        {
            "left_arm": np.r_[np.zeros(7), 0.4],
            "right_arm": np.r_[np.zeros(7), 0.6],
        },
        9,
        1,
    )
    with pytest.raises(ValueError, match="observation state time"):
        adapter.decode_action(
            actions,
            ActionContext(9, 10, 11, measured_state=state),
        )


def test_bad_action_shape_and_gripper_range_are_rejected():
    _, adapter = _adapter()
    request, context = _request(adapter)
    adapter.prepare_request(request)
    with pytest.raises(ValueError, match="shape"):
        adapter.decode_action(np.zeros((29, ACTION_DIM)), context)

    request, context = _request(adapter, request_seq=5)
    adapter.prepare_request(request)
    actions = np.zeros((30, ACTION_DIM))
    actions[:, 6] = 0.71  # Left anchor 0.4 -> 1.11, beyond tolerance.
    with pytest.raises(ValueError, match="more than tolerance"):
        adapter.decode_action(actions, context)


def test_any_ik_failure_rejects_the_whole_chunk():
    _, adapter = _adapter(fail=True)
    request, context = _request(adapter)
    adapter.prepare_request(request)
    with pytest.raises(ValueError, match="offline_test_failure"):
        adapter.decode_action(np.zeros((30, ACTION_DIM)), context)


def _wait_for_decode(decoder: ActionDecoderClient):
    until = time.monotonic() + 30
    while time.monotonic() < until:
        result = decoder.poll()
        if result is not None:
            return result
        time.sleep(0.001)
    raise AssertionError("Xiaomi action decoder did not return a result")


def _diff_actions(adapter: XR1TianjiTacCapAdapter, observation: RobotState) -> np.ndarray:
    actions = np.zeros((30, ACTION_DIM), dtype=np.float64)
    for group, joint in (("left_arm", 0), ("right_arm", 1)):
        model = adapter.kinematics.models[group]
        anchor = model.fk(observation.groups[group])
        columns = ARM_SLICES[group]
        for index in range(30):
            target_joints = START.copy()
            target_joints[joint] += np.radians(0.02 * (index + 1))
            target = model.fk(np.r_[target_joints, 0.8])
            actions[index, columns["position"]] = anchor[:3, :3].T @ (target[:3, 3] - anchor[:3, 3])
            actions[index, columns["axis_angle"]] = rotation_vector(
                anchor[:3, :3].T @ target[:3, :3]
            )
    return actions


def test_diff_ik_uses_two_processes_and_matches_direct_decode():
    pytest.importorskip("osqp")
    config = load_config(EXPERIMENT)
    adapter = XR1TianjiTacCapAdapter(config["robot"], config["policy"])
    adapter.validate(config["robot"], config["policy"])
    now = time.monotonic_ns()
    observation = RobotState(
        {group: np.r_[START, 0.8] for group in ("left_arm", "right_arm")},
        now,
        1,
    )
    actions = _diff_actions(adapter, observation)

    def submit(seq: int, seed: RobotState):
        context = ActionContext(
            seq,
            now,
            time.monotonic_ns(),
            execution_time_ns=time.monotonic_ns() + 80_000_000,
            measured_state=seed,
        )
        response = InferenceResponse(
            "test", seq, time.monotonic_ns(), 1.0, actions, observation_time_ns=now
        )
        decoder.submit(response, context, time.monotonic_ns() + 30_000_000_000)
        return context

    decoder = ActionDecoderClient(config["robot"], config["policy"], adapter)
    assert [process.name for process in decoder._processes] == [
        "manimux-decode-left_arm",
        "manimux-decode-right_arm",
    ]
    try:
        decoder.start()
        context = submit(1, observation)
        direct = adapter.decode_action(actions, context)
        result = _wait_for_decode(decoder)

        assert result.error is None
        chunk = result.chunk
        assert chunk is not None
        assert chunk.metadata["decode_mode"] == "process"
        assert chunk.metadata["ik_backend"] == "diff"
        assert set(chunk.metadata["decode_partition_ms"]) == {
            "left_arm",
            "right_arm",
        }
        for group in ("left_arm", "right_arm"):
            np.testing.assert_allclose(chunk.groups[group], direct.groups[group], atol=1e-9, rtol=0)
            assert chunk.metadata["diff_ik_lag"][group] == pytest.approx(
                direct.metadata["diff_ik_lag"][group], abs=1e-9
            )

        # A bad right-arm observation must not let the completed left partition
        # escape as a partial chunk.
        bad_groups = {name: values.copy() for name, values in observation.groups.items()}
        bad_groups["right_arm"][0] = 3.0
        failed_seed = RobotState(bad_groups, now, 2)
        submit(2, failed_seed)
        failed = _wait_for_decode(decoder)
        assert failed.chunk is None
        assert failed.error is not None
        assert "rejecting the entire chunk" in failed.error
    finally:
        decoder.close()
