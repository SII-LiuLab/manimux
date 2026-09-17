"""XR-1 is the first policy whose actions are Cartesian, so the adapter does real
geometry. These tests pin that geometry: a zero delta must be an exact no-op, a
known delta must reproduce exactly through FK, and the dimensions YAM does not
have (waist, mobile base) must never reach the arms.
"""

from __future__ import annotations

import numpy as np
import pytest

from manimux.cli import load_config
from manimux.clock import SystemClock
from manimux.integrations.xpolicylab.policy_plugin import XPolicyLabWsPolicyModel
from manimux.integrations.xr1_yam.policy_plugin import (
    ACTION_DIM,
    XR1YamAdapter,
    _axis_angle_to_rotation,
)
from manimux.policies import build_policy_adapter, build_policy_model
from manimux.robots import build_robot
from manimux.sensors import build_sensor
from manimux.sensors.camera_server import CameraServerSensorDriver
from manimux.types import ActionContext, InferenceRequest, ObservationSnapshot, RobotState

pytest.importorskip("mujoco")
pytest.importorskip("mink")
pytest.importorskip("i2rt")

ANCHOR = np.array(
    [
        -0.6094,
        0.5835,
        0.8425,
        -1.0168,
        -0.1108,
        -0.4580,
        0.55,
        -0.5000,
        0.4000,
        0.7000,
        -0.9000,
        0.2000,
        -0.3000,
        0.35,
    ]
)


@pytest.fixture(scope="module")
def adapter() -> XR1YamAdapter:
    config = load_config("configs/xiaomi-xr1/yam/infra/manimux.yaml")
    return build_policy_adapter(config["robot"], config["policy"])


def test_xr1_run_config_swaps_only_the_policy_layer() -> None:
    from manimux.robots.yam import YamDualArmDriver

    xr1 = load_config("configs/xiaomi-xr1/yam/infra/manimux.yaml")
    molmoact = load_config("configs/molmoact2/yam/infra/manimux.yaml")

    assert isinstance(build_robot(xr1["robot"], SystemClock()), YamDualArmDriver)
    assert isinstance(build_sensor(xr1["sensors"][0], SystemClock()), CameraServerSensorDriver)
    assert isinstance(build_policy_model(xr1["policy"]), XPolicyLabWsPolicyModel)

    assert xr1["robot"]["type"] == molmoact["robot"]["type"]
    assert xr1["sensors"][0]["driver"] == molmoact["sensors"][0]["driver"]
    assert xr1["viewer"]["robot"] == molmoact["viewer"]["robot"]


def test_zero_delta_holds_the_measured_pose(adapter: XR1YamAdapter) -> None:
    """Every step is a delta against the same anchor, so all-zero must not move."""
    chunk = adapter.decode_action(
        {"actions": np.zeros((30, ACTION_DIM)), "state": ANCHOR},
        ActionContext(request_seq=1, observation_time_ns=0, created_time_ns=0),
    )

    for group, offset in (("left_arm", 0), ("right_arm", 7)):
        joints = chunk.groups[group]
        assert joints.shape == (30, 7)
        assert np.abs(joints[:, :6] - ANCHOR[offset : offset + 6]).max() < 2e-3
        assert np.abs(joints[:, 6] - ANCHOR[offset + 6]).max() < 1e-12


def test_known_delta_round_trips_through_forward_kinematics(adapter: XR1YamAdapter) -> None:
    from manimux.kinematics import build_kinematics

    rng = np.random.default_rng(3)
    actions = np.zeros((30, ACTION_DIM))
    d_pos = rng.uniform(-0.05, 0.05, (30, 3))
    d_aa = rng.uniform(-0.15, 0.15, (30, 3))
    actions[:, 0:3] = d_pos
    actions[:, 3:6] = d_aa
    actions[:, 6] = np.linspace(0.0, 0.3, 30)
    # Waist and mobile-base columns: YAM has neither, they must be dropped.
    actions[:, 16] = 9.9
    actions[:, 17:20] = 9.9

    chunk = adapter.decode_action(
        {"actions": actions, "state": ANCHOR},
        ActionContext(request_seq=2, observation_time_ns=0, created_time_ns=0),
    )

    kinematics = build_kinematics("yam")
    anchor_pose = kinematics.fk(ANCHOR[:6], ANCHOR[6])
    rotation, position = anchor_pose[:3, :3], anchor_pose[:3, 3]

    for step in range(30):
        expected_p = position + rotation @ d_pos[step]
        expected_r = rotation @ _axis_angle_to_rotation(d_aa[step])
        achieved = kinematics.fk(chunk.groups["left_arm"][step, :6], ANCHOR[6])
        assert np.linalg.norm(achieved[:3, 3] - expected_p) < 2e-3
        residual = expected_r.T @ achieved[:3, :3]
        angle = abs(np.arccos(np.clip((np.trace(residual) - 1) / 2, -1.0, 1.0)))
        assert angle < np.radians(0.2)

    gripper = chunk.groups["left_arm"][:, 6]
    assert gripper[0] == pytest.approx(ANCHOR[6])
    assert gripper[-1] == pytest.approx(ANCHOR[6] + 0.3)
    assert chunk.action_space == "joint_position"
    assert chunk.plan_id.startswith("xr1-")


def test_adapter_rejects_a_bare_action_matrix(adapter: XR1YamAdapter) -> None:
    with pytest.raises(ValueError, match="no observation anchor"):
        adapter.decode_action(
            np.zeros((30, ACTION_DIM)),
            ActionContext(request_seq=1, observation_time_ns=0, created_time_ns=0),
        )


def test_prepared_request_supplies_anchor_to_bare_action_matrix(
    adapter: XR1YamAdapter,
) -> None:
    request = InferenceRequest(
        session_id="session",
        request_seq=10,
        observation_time_ns=1,
        deadline_ns=2,
        observation=ObservationSnapshot(
            state=RobotState(
                groups={"left_arm": ANCHOR[:7], "right_arm": ANCHOR[7:]},
                monotonic_ns=1,
                sequence=1,
            )
        ),
    )
    adapter.prepare_request(request)
    chunk = adapter.decode_action(
        np.zeros((30, ACTION_DIM)),
        ActionContext(request_seq=10, observation_time_ns=1, created_time_ns=2),
    )

    np.testing.assert_allclose(chunk.groups["left_arm"][:, 6], ANCHOR[6])
    np.testing.assert_allclose(chunk.groups["right_arm"][:, 6], ANCHOR[13])


def test_adapter_rejects_wrong_action_width(adapter: XR1YamAdapter) -> None:
    with pytest.raises(ValueError, match="shape"):
        adapter.decode_action(
            {"actions": np.zeros((30, 14)), "state": ANCHOR},
            ActionContext(request_seq=1, observation_time_ns=0, created_time_ns=0),
        )
