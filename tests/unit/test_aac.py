from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from manimux.cli import load_config, prepare_experiment
from manimux.policies.fake import FakePolicyAdapter
from manimux.policies.xpolicylab.aac import (
    AacPreviousAction,
    EeActionStats,
    build_ee_candidates,
    ee_motion_magnitude,
    ee_pose_increment,
    entropy_elbow,
    motion_floor,
    select_candidate,
    select_ee_chunk,
)
from manimux.policies.xpolicylab.codec import GroupLayout
from manimux.runtime.aac import AacInferenceRequest
from manimux.runtime.inference import RequestState, build_inference_strategy
from manimux.runtime.safety import RuntimeState
from manimux.runtime.timeline import ActionTimeline, CommitResult
from manimux.types import ActionChunk, InferenceResponse, ObservationSnapshot, RobotState

LAYOUTS = (
    GroupLayout(group="left_arm", prefix="left", arm_dofs=6, gripper_dofs=1),
    GroupLayout(group="right_arm", prefix="right", arm_dofs=6, gripper_dofs=1),
)

EE_STATS = EeActionStats(
    groups=("left_arm", "right_arm"),
    minimum=np.full((2, 6), -1.0),
    maximum=np.full((2, 6), 1.0),
)


class _PoseKinematics:
    num_arm_joints = 6

    def fk(self, joints: np.ndarray, gripper: float) -> np.ndarray:
        del gripper
        joints = np.asarray(joints, dtype=np.float64)
        angle = joints[3]
        cosine, sine = np.cos(angle), np.sin(angle)
        pose = np.eye(4)
        pose[:3, :3] = np.asarray([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
        pose[:3, 3] = joints[:3]
        return pose


def _candidate_chunks(samples: int = 3, horizon: int = 4) -> list[list[dict[str, np.ndarray]]]:
    candidates = []
    offsets = np.linspace(-0.1, 0.1, samples)
    for offset in offsets:
        steps = []
        for step in range(horizon):
            left = np.zeros(6, dtype=np.float32)
            right = np.zeros(6, dtype=np.float32)
            left[0] = 0.2 * step + offset
            right[0] = 0.2 * step - offset
            left[3] = 0.05 * step + offset
            right[3] = 0.05 * step - offset
            steps.append(
                {
                    "left_arm_joint_state": left,
                    "left_ee_joint_state": np.array([0.0], dtype=np.float32),
                    "right_arm_joint_state": right,
                    "right_ee_joint_state": np.array([0.0], dtype=np.float32),
                }
            )
        candidates.append(steps)
    return candidates


def _aac_config() -> dict:
    return load_config("manimux/configs/experiments/pick_box/yam_groot_aac.yaml")


def test_aac_elbow_and_motion_floor_match_official_indexing() -> None:
    assert entropy_elbow(np.asarray([0.0, 0.1, 1.1, 1.2])) == 2
    assert motion_floor(np.asarray([0.2, 0.8, 1.2]), threshold=0.5, horizon=4) == 3
    assert motion_floor(np.asarray([0.2, 0.3, 0.4]), threshold=0.5, horizon=4) == 4


def test_fk_ee_adaptation_selects_a_variable_horizon() -> None:
    selected, selection, previous = select_ee_chunk(
        _candidate_chunks(),
        layouts=LAYOUTS,
        current_groups={"left_arm": np.zeros(7), "right_arm": np.zeros(7)},
        kinematics=_PoseKinematics(),
        ee_stats=EE_STATS,
        motion_threshold=0.4,
    )

    assert selection.motion_floor == 3
    assert selection.chunk_size == 3
    assert selection.chunk_id == 0
    assert len(selected) == selection.chunk_size
    assert previous.ee_features.shape == (3, 4, 2, 7)


def test_fk_conversion_produces_per_step_base_frame_increments() -> None:
    features, _ = build_ee_candidates(
        _candidate_chunks(),
        layouts=LAYOUTS,
        current_groups={"left_arm": np.zeros(7), "right_arm": np.zeros(7)},
        kinematics=_PoseKinematics(),
    )

    np.testing.assert_allclose(features[0, :, 0, 0], [-0.1, 0.2, 0.2, 0.2])
    np.testing.assert_allclose(features[0, :, 0, 5], [-0.1, 0.05, 0.05, 0.05])
    np.testing.assert_allclose(features[0, :, 1, 0], [0.1, 0.2, 0.2, 0.2])


def test_rotation_increment_uses_official_left_multiplication_convention() -> None:
    previous = np.eye(4)
    previous[:3, :3] = Rotation.from_euler("x", 30, degrees=True).as_matrix()
    delta = Rotation.from_euler("y", 20, degrees=True)
    target = np.eye(4)
    target[:3, :3] = (delta * Rotation.from_matrix(previous[:3, :3])).as_matrix()

    increment = ee_pose_increment(previous, target)

    np.testing.assert_allclose(increment[3:], delta.as_rotvec(), atol=1e-12)


def test_ee_stats_apply_official_min_max_normalization() -> None:
    features = np.zeros((1, 1, 2, 7), dtype=np.float64)
    features[..., :6] = np.asarray(
        [[[-1.0, 0.0, 1.0, -0.5, 0.0, 0.5], [-1.0, 0.0, 1.0, -0.5, 0.0, 0.5]]]
    )
    features[..., 6] = 0.75

    normalized = EE_STATS.normalize(features)

    np.testing.assert_allclose(normalized[..., :6], features[..., :6])
    np.testing.assert_allclose(normalized[..., 6], 0.75)


def test_dual_arm_motion_is_the_mean_of_both_ee_magnitudes() -> None:
    features = np.zeros((1, 3, 2, 7), dtype=np.float64)
    features[0, 0, 0, 0] = 1.0
    features[0, 1, 0, 0] = 1.0
    features[0, 2, 0, 0] = 2.0
    np.testing.assert_allclose(ee_motion_magnitude(features), [1.0, 2.0])


def test_official_candidate_selectors_are_preserved() -> None:
    candidates = np.stack([np.zeros((4, 2, 7)), np.full((4, 2, 7), 10.0), np.ones((4, 2, 7))])
    assert (
        select_candidate(
            candidates,
            method="mean",
            chunk_size=3,
            previous=None,
            beta=0.99,
        )
        == 2
    )

    previous = AacPreviousAction(
        ee_features=np.stack([np.full((4, 2, 7), 10.0), np.zeros((4, 2, 7)), np.ones((4, 2, 7))]),
        chunk_size=2,
    )
    assert (
        select_candidate(
            candidates,
            method="backward",
            chunk_size=2,
            previous=previous,
            beta=0.99,
        )
        == 2
    )


def test_aac_strategy_uses_the_server_capability_and_waits_for_chunk_end() -> None:
    config = _aac_config()
    strategy = build_inference_strategy(config)
    snapshot = ObservationSnapshot(
        state=RobotState(
            groups={name: np.zeros(dim) for name, dim in config["robot"]["group_dims"].items()},
            monotonic_ns=1_000,
            sequence=1,
        )
    )
    request_state = RequestState(False, -1, 0)
    timeline = ActionTimeline(config["robot"]["group_dims"])
    kwargs = {
        "session_id": "session",
        "snapshot": snapshot,
        "adapter": FakePolicyAdapter({}, {}),
        "timeline": timeline,
        "request_state": request_state,
        "runtime_state": RuntimeState.RUNNING,
    }

    submission = strategy.build_submission(request_seq=1, now_ns=1_000, **kwargs)
    assert submission is not None
    assert isinstance(submission.request, AacInferenceRequest)
    assert submission.request.aac_num_samples == 20
    assert submission.request.aac_ee_stats_path.endswith("yam_60ep_ee_increment.json")
    assert strategy.required_sampling_modes == frozenset({"aac"})

    chunk = ActionChunk(
        plan_id="active",
        request_seq=1,
        observation_time_ns=1_000,
        created_time_ns=1_000,
        action_space="joint_position",
        dt_ns=100,
        groups={name: np.zeros((4, dim)) for name, dim in config["robot"]["group_dims"].items()},
    )
    result = timeline.commit(
        chunk,
        now_ns=1_000,
        commit_lead_ns=0,
        max_plan_age_ns=1_000,
        current_command=snapshot.state.groups,
        blend_steps=0,
    )
    assert result.accepted
    assert strategy.build_submission(request_seq=2, now_ns=1_200, **kwargs) is None
    assert strategy.build_submission(request_seq=2, now_ns=1_300, **kwargs) is not None


def test_aac_rebases_selected_chunk_after_synchronous_inference() -> None:
    strategy = build_inference_strategy(_aac_config())
    chunk = ActionChunk(
        plan_id="selected",
        request_seq=1,
        observation_time_ns=10,
        created_time_ns=20,
        action_space="joint_position",
        dt_ns=100,
        groups={"left_arm": np.zeros((3, 7)), "right_arm": np.zeros((3, 7))},
    )
    response = InferenceResponse("s", 1, 20, 1.0, raw_action=[])
    prepared = strategy.prepare_chunk(chunk=chunk, response=response, now_ns=1_000)
    assert prepared.observation_time_ns == 1_000
    assert prepared.horizon_steps == 3

    fields = strategy.on_plan_accepted(
        chunk=prepared,
        result=CommitResult(accepted=True, reason="accepted", trimmed_steps=0),
        response=InferenceResponse(
            "s",
            1,
            20,
            1.0,
            raw_action={
                "actions": [],
                "aac": {"chunk_id": 4, "entropy_elbow": 2, "motion_floor": 3},
            },
        ),
        now_ns=1_000,
    )
    assert fields == {
        "selected_horizon_steps": 3,
        "trimmed_steps": 0,
        "chunk_id": 4,
        "entropy_elbow": 2,
        "motion_floor": 3,
    }


def test_aac_config_rejects_runtime_blending() -> None:
    payload = deepcopy(_aac_config())
    payload["inference"].pop("inference_schedule")
    payload["inference"].pop("refill_threshold_s")
    payload["inference"]["blend_steps"] = 1
    with pytest.raises(ValueError, match="blend_steps=0"):
        prepare_experiment(**payload)


def test_aac_config_requires_fixed_ee_stats() -> None:
    payload = deepcopy(_aac_config())
    payload["inference"]["aac"]["ee_stats_path"] = None
    with pytest.raises(ValueError, match="ee_stats_path"):
        prepare_experiment(**payload)
