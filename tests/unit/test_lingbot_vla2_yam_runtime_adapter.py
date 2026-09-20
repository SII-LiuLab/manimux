from __future__ import annotations

import numpy as np
import pytest

from manimux.embodiments.robot import robot_parameters
from manimux.policies.base import policy_parameters
from manimux.policy_adapter.lingbot_vla2.yam import ACTION_SEMANTICS, LingBotVLA2YamAdapter
from manimux.types import (
    ActionContext,
    InferenceRequest,
    ObservationSnapshot,
    RobotState,
)


def _adapter() -> LingBotVLA2YamAdapter:
    robot = robot_parameters(
        type="fake",
        control_hz=100.0,
        group_dims={"left_arm": 7, "right_arm": 7},
    )
    policy = policy_parameters(
        worker="fake",
        adapter={"type": "manimux.policy_adapter.lingbot_vla2.yam:LingBotVLA2YamAdapter"},
        action_dt_s=1 / 30,
        horizon_steps=2,
        options={
            "group_order": ["left_arm", "right_arm"],
            "group_prefixes": {"left_arm": "left", "right_arm": "right"},
            "gripper_dofs": 1,
        },
    )
    return LingBotVLA2YamAdapter(robot, policy)


def _request() -> InferenceRequest:
    state = RobotState(
        groups={
            "left_arm": np.array([1, 2, 3, 4, 5, 6, 0.8], dtype=np.float64),
            "right_arm": np.array([10, 20, 30, 40, 50, 60, 0.9], dtype=np.float64),
        },
        monotonic_ns=100,
        sequence=1,
    )
    return InferenceRequest(
        session_id="test",
        request_seq=7,
        observation_time_ns=100,
        deadline_ns=200,
        observation=ObservationSnapshot(state=state),
        instruction="Assemble the screwdriver.",
    )


def _raw_action() -> dict[str, object]:
    return {
        "action_semantics": ACTION_SEMANTICS,
        "actions": [
            {
                "left_arm_joint_state": np.full(6, 0.1),
                "left_ee_joint_state": np.array([0.25]),
                "right_arm_joint_state": np.full(6, -0.2),
                "right_ee_joint_state": np.array([0.75]),
            },
            {
                "left_arm_joint_state": np.full(6, 0.3),
                "left_ee_joint_state": np.array([0.5]),
                "right_arm_joint_state": np.full(6, -0.4),
                "right_ee_joint_state": np.array([1.0]),
            },
        ],
    }


def test_relative_arms_use_request_anchor_and_grippers_stay_absolute() -> None:
    adapter = _adapter()
    request = _request()
    adapter.prepare_request(request)
    unrelated_latest_state = RobotState(
        groups={
            "left_arm": np.full(7, 99.0),
            "right_arm": np.full(7, 99.0),
        },
        monotonic_ns=190,
        sequence=2,
    )
    chunk = adapter.decode_action(
        _raw_action(),
        ActionContext(
            request_seq=7,
            observation_time_ns=100,
            created_time_ns=200,
            measured_state=unrelated_latest_state,
        ),
    )

    np.testing.assert_allclose(
        chunk.groups["left_arm"][0, :6],
        [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
    )
    np.testing.assert_allclose(
        chunk.groups["right_arm"][0, :6],
        [9.8, 19.8, 29.8, 39.8, 49.8, 59.8],
    )
    np.testing.assert_allclose(chunk.groups["left_arm"][:, 6], [0.25, 0.5])
    np.testing.assert_allclose(chunk.groups["right_arm"][:, 6], [0.75, 1.0])
    assert chunk.action_space == "joint_position"
    assert chunk.metadata["anchor_request_seq"] == 7


def test_relative_adapter_requires_matching_request_anchor() -> None:
    adapter = _adapter()
    with pytest.raises(ValueError, match="no observation anchor"):
        adapter.decode_action(
            _raw_action(),
            ActionContext(
                request_seq=7,
                observation_time_ns=100,
                created_time_ns=200,
            ),
        )
