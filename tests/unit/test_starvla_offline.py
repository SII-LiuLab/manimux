"""Checked-in StarVLA recipes traverse the same loaders as deployed policies."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from manimux.cli import load_config
from manimux.clock import SystemClock
from manimux.embodiments.robot import build_robot
from manimux.policies.xpolicylab.codec import GroupLayout, decode_policy_actions, encode_observation
from manimux.policy_adapter import build_policy_adapter
from manimux.runtime import build_runtime
from manimux.types import ActionContext, ObservationSnapshot, RobotState

EXPERIMENTS = Path("manimux/configs/experiments/offline/starvla")


@pytest.mark.parametrize(
    "path",
    sorted(p for p in EXPERIMENTS.glob("*.yaml") if not p.name.startswith("station")),
    ids=lambda p: p.stem,
)
def test_paired_recipe_uses_only_test_devices_and_checks_backend(path, tmp_path):
    station = tmp_path / "station.yaml"
    checkpoint = tmp_path / "model.pt"
    station.write_text(yaml.safe_dump({"paths": {"checkpoint": str(checkpoint)}}))
    config = load_config(path, local=station)
    assert config["robot"]["type"] == "tests.support.robot:build_robot"
    assert all(s["driver"] == "tests.support.sensor:build_sensor" for s in config["sensors"])
    identity = config["policy"]["expected_backend"]
    assert identity["model"]["checkpoint_path"] == str(checkpoint)
    assert identity["model"]["policy_name"] == "starVLA"
    robot = build_robot(config["robot"], SystemClock())
    adapter = build_policy_adapter(config["robot"], config["policy"], kinematics=robot.kinematics)
    if config["policy"]["options"]["action_format"] == "pose":
        assert adapter.kinematics is robot.kinematics
    # Construction proves the real strategy/timeline/executor config is coherent;
    # run() and sockets are deliberately exercised by the checkpoint probe.
    runtime = build_runtime(config, tmp_path / "run")
    algorithm = config["inference"]["algorithm"]
    expected_mode = (
        algorithm if algorithm in {"rtc", "paint", "aac", "autohorizon", "dvac"} else "default"
    )
    assert runtime._strategy.required_sampling_modes == frozenset({expected_mode})


def test_single_joint_schema_matches_xpolicylab_shared_helpers():
    from XPolicyLab.utils.process_data import pack_robot_state

    layouts = (GroupLayout("arm", "", 7, 1),)
    snapshot = ObservationSnapshot(RobotState({"arm": np.arange(8.0)}, 1, 0))
    observation = encode_observation(
        snapshot, layouts=layouts, camera_map={}, instruction="test", frequency=20
    )
    packed = pack_robot_state(observation, "joint", {"arm_dim": [7], "ee_dim": [1]})
    np.testing.assert_array_equal(packed, np.arange(8.0))
    for key in ("joint_state", "arm_joint_state"):
        payload = decode_policy_actions(
            [{key: np.arange(7.0), "ee_joint_state": [0.5]}], layouts=layouts, format="joint"
        )
        np.testing.assert_array_equal(payload["actions"]["arm"], [np.r_[np.arange(7.0), 0.5]])


def test_full_joint_chunk_decodes_standard_provider_actions():
    from manimux.policy_adapter.joint import JointAdapter
    from XPolicyLab.utils.process_data import unpack_robot_state

    dimensions = {"arm_dim": [6, 6], "ee_dim": [1, 1]}
    actions = unpack_robot_state(
        np.arange(700, dtype=np.float32).reshape(50, 14), "joint", dimensions
    )
    layouts = (GroupLayout("left_arm", "left", 6, 1), GroupLayout("right_arm", "right", 6, 1))
    raw = decode_policy_actions(actions, layouts=layouts, format="joint")
    robot = {"group_dims": {"left_arm": 7, "right_arm": 7}}
    policy = {
        "adapter": {
            "group_layouts": {
                "left_arm": {"arm_dofs": 6, "gripper_dofs": 1},
                "right_arm": {"arm_dofs": 6, "gripper_dofs": 1},
            }
        },
        "horizon_policy_steps": 50,
        "action_dt_s": 0.04,
        "trajectory_duration_s": None,
    }
    adapter = JointAdapter(robot, policy)
    chunk = adapter.decode_action(raw, ActionContext(1, 10, 20))
    assert chunk.horizon_steps == 50 and chunk.dt_ns == 40_000_000
    np.testing.assert_array_equal(chunk.groups["left_arm"][0], np.arange(7))
    np.testing.assert_array_equal(chunk.groups["right_arm"][-1], np.arange(693, 700))
