from __future__ import annotations

import numpy as np

from XPolicyLab.policy.SAPolicy.model import (
    Model,
    relative_actions_to_wire,
)


def test_sapolicy_dry_run_model_returns_wire_shape() -> None:
    model = Model(
        {
            "action_type": "ee",
            "env_cfg_type": "yam_dual",
            "output_format": "packed_ee_wire",
            "action_horizon": 16,
            "dry_run": True,
            "camera_names": ["agentview"],
        }
    )
    model.update_obs(
        {
            "vision": {
                "agentview": {"color": np.zeros((8, 8, 3), dtype=np.uint8), "shape": [8, 8]}
            },
            "instruction": "put bottles in bin",
            "state": {},
            "additional_info": {
                "frequency": 30.0,
                "sapolicy": {
                    "left_endpose": np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float64),
                    "right_endpose": np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float64),
                    "left_gripper": 0.5,
                    "right_gripper": 0.5,
                    "intrinsics": {"agentview": np.eye(3)},
                },
            },
        }
    )
    actions = model.get_action()
    assert actions.shape == (16, 16)
    np.testing.assert_allclose(actions[:, :7], np.broadcast_to([0, 0, 0, 0, 0, 0, 1], (16, 7)))
    np.testing.assert_allclose(actions[:, 7], 0.5)
    np.testing.assert_allclose(actions[:, 8:15], np.broadcast_to([0, 0, 0, 0, 0, 0, 1], (16, 7)))
    np.testing.assert_allclose(actions[:, 15], 0.5)
    assert model.runtime_metadata()["policy_family"] == "sapolicy"


def test_to_spatial_obs_forwards_image_native_hw() -> None:
    model = Model(
        {
            "dry_run": True,
            "camera_names": ["top", "left", "right"],
        }
    )
    native = {"top": [480, 640], "left": [480, 640], "right": [480, 640]}
    spatial = model._to_spatial_obs(
        {
            "vision": {
                name: {"color": np.zeros((168, 224, 3), dtype=np.uint8), "shape": [168, 224]}
                for name in ("top", "left", "right")
            },
            "additional_info": {
                "sapolicy": {
                    "left_endpose": np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float64),
                    "right_endpose": np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float64),
                    "left_gripper": 0.5,
                    "right_gripper": 0.5,
                    "camera_names": ["top", "left", "right"],
                    "intrinsics": {name: np.eye(3) for name in ("top", "left", "right")},
                    "image_native_hw": native,
                }
            },
        }
    )
    assert spatial["image_native_hw"] == native
    assert spatial["images"]["top"].shape == (168, 224, 3)


def test_relative_actions_to_wire_identity_prefix() -> None:
    # DiT layout is [pose18 | grip2]. Identity rot6d + zero translation
    # should keep the measured xyzw endpose.
    rel = np.zeros((4, 20), dtype=np.float64)
    identity_6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    rel[:, 3:9] = identity_6d
    rel[:, 12:18] = identity_6d
    rel[:, 18] = 0.25
    rel[:, 19] = 0.75
    left = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    right = np.array([-0.1, -0.2, 0.4, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    wire = relative_actions_to_wire(rel, left, right, body_frame=True)
    assert wire.shape == (4, 16)
    np.testing.assert_allclose(wire[:, :3], np.broadcast_to(left[:3], (4, 3)), atol=1e-5)
    np.testing.assert_allclose(wire[:, 3:7], np.broadcast_to(left[3:7], (4, 4)), atol=1e-5)
    np.testing.assert_allclose(wire[:, 8:11], np.broadcast_to(right[:3], (4, 3)), atol=1e-5)
    np.testing.assert_allclose(wire[:, 11:15], np.broadcast_to(right[3:7], (4, 4)), atol=1e-5)
    np.testing.assert_allclose(wire[:, 7], 0.25)
    np.testing.assert_allclose(wire[:, 15], 0.75)


def test_pack_state_uses_grouped_pose18_grip2() -> None:
    model = Model({"dry_run": True, "camera_names": ["agentview"]})
    left = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    right = np.array([-0.1, -0.2, 0.4, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    state = model._pack_state(left, right, 0.25, 0.75)
    assert state.shape == (20,)
    np.testing.assert_allclose(state[0:3], left[:3])
    np.testing.assert_allclose(state[3:9], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(state[9:12], right[:3])
    np.testing.assert_allclose(state[12:18], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(state[18:20], [0.25, 0.75])
