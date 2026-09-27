"""Representation, anchoring and IK failures without model weights or hardware."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from manimux.kinematics import RobotKinematics
from manimux.kinematics.poses import matrix_pose
from manimux.policy_adapter import build_policy_adapter
from manimux.types import ActionContext, InferenceRequest, ObservationSnapshot, RobotState
from tests.support.cartesian_geometry import CartesianGeometry


def make_adapter(semantics="absolute_per_arm_base_xyz_wxyz", *, dual=False, steps=2):
    sides = {"left_arm": "left", "right_arm": "right"} if dual else {"arm": ""}
    robot = {"group_dims": {g: 7 for g in sides}}
    policy = {
        "adapter": {
            "type": "manimux.policy_adapter.pose:PoseAdapter",
            "group_prefixes": sides,
            "action_semantics": semantics,
            "decode_policy_steps": steps,
        },
        "expected_backend": {"model": {"action_semantics": semantics}},
        "action_dt_s": 0.05,
        "horizon_policy_steps": 2,
    }
    kin = RobotKinematics({g: CartesianGeometry() for g in sides})
    adapter = build_policy_adapter(robot, policy, kinematics=kin)
    state = RobotState({g: np.array([0.2, -0.1, 0.3, 0, 0, np.pi / 2, 0.5]) for g in sides}, 123, 0)
    return adapter, state, kin, robot, policy


@pytest.mark.parametrize("dual", [False, True])
@pytest.mark.parametrize(
    "semantics",
    [
        "absolute_per_arm_base_xyz_wxyz",
        "delta_observation_base_xyz_wxyz",
        "delta_observation_tool_xyz_wxyz",
        "delta_step_base_xyz_wxyz",
    ],
)
def test_factory_fk_pose_codec_and_ik(dual, semantics):
    from manimux.policies.xpolicylab.codec import GroupLayout, decode_policy_actions

    count = 1 if semantics == "delta_step_base_xyz_wxyz" else 2
    adapter, state, kin, _, _ = make_adapter(semantics, dual=dual, steps=count)
    assert adapter.supports_context_only_decode == (semantics != "delta_step_base_xyz_wxyz")
    assert adapter.decode_seed_source == (
        "observation_state" if semantics.startswith("delta_observation_") else "execution_reference"
    )
    request = adapter.prepare_request(
        InferenceRequest("episode", 9, 123, 1000, ObservationSnapshot(state))
    )
    delta = np.eye(4)
    delta[:3, 3] = [0.02, 0, 0]
    delta[:3, :3] = Rotation.from_rotvec([0.1, 0, 0]).as_matrix()
    row = {}
    layouts = []
    for group, side in adapter.prefixes.items():
        prefix = side + "_" if side else ""
        np.testing.assert_allclose(
            request.model_state[prefix + "ee_pose"],
            matrix_pose(kin.models[group].fk(state.groups[group])),
        )
        row[prefix + "ee_pose"] = matrix_pose(delta)
        row[prefix + "ee_joint_state"] = np.array([0.7])
        layouts.append(GroupLayout(group, side, 6, 1))
    payload = decode_policy_actions([row, row], layouts=tuple(layouts), format="pose")
    chunk = adapter.decode_action(payload, ActionContext(9, 123, 456, measured_state=state))
    assert chunk.horizon_steps == max(2, count) and chunk.dt_ns == 50_000_000
    assert chunk.metadata["decoded_policy_steps"] == count
    for group in adapter.prefixes:
        anchor = kin.models[group].fk(state.groups[group])
        expected = delta.copy()
        if semantics == "delta_observation_tool_xyz_wxyz":
            expected = anchor @ delta
        elif semantics != "absolute_per_arm_base_xyz_wxyz":
            expected[:3, 3] += anchor[:3, 3]
            expected[:3, :3] = delta[:3, :3] @ anchor[:3, :3]
        for solved in chunk.groups[group]:
            np.testing.assert_allclose(kin.models[group].fk(solved), expected, atol=1e-9)
            assert solved[-1] == 0.7


def test_feedback_delta_cannot_be_silently_accumulated():
    with pytest.raises(ValueError, match="decode_policy_steps=1"):
        make_adapter("delta_step_base_xyz_wxyz")


def test_missing_state_invalid_quaternion_gripper_and_failed_ik():
    adapter, state, kin, _, _ = make_adapter()
    rows = np.tile([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5], (2, 1))
    raw = {"format": "pose", "actions": {"arm": rows}}
    with pytest.raises(ValueError, match="measured joints"):
        adapter.decode_action(raw, ActionContext(1, 1, 2))
    context = ActionContext(1, 1, 2, measured_state=state)
    rows[1, 3] = 0
    with pytest.raises(ValueError, match="unit length"):
        adapter.decode_action(raw, context)
    rows[1, 3] = 1
    rows[1, 7] = 2
    with pytest.raises(ValueError, match="normalized opening"):
        adapter.decode_action(raw, context)
    rows[1, 7] = 0.5
    kin.models["arm"].fail = True
    with pytest.raises(ValueError, match="outside_test_workspace"):
        adapter.decode_action(raw, context)


def test_backend_identity_and_sampler_conditions_are_not_bypassed():
    _, _, kin, robot, policy = make_adapter()
    policy["expected_backend"]["model"]["action_semantics"] = "joint"
    with pytest.raises(ValueError, match="expected_backend"):
        build_policy_adapter(robot, policy, kinematics=kin)


def test_specialized_request_cannot_silently_become_default_sampling():
    from manimux.runtime.rtc.request import RtcInferenceRequest

    adapter, state, _, _, _ = make_adapter()
    with pytest.raises(ValueError, match="default inference requests only"):
        adapter.prepare_request(
            RtcInferenceRequest("test", 0, 123, 1000, ObservationSnapshot(state))
        )


def test_feedback_delta_disallows_forecast_decoder_process_seed():
    _, _, kin, robot, policy = make_adapter("delta_step_base_xyz_wxyz", steps=1)
    policy["action_decoding"] = "process"
    with pytest.raises(ValueError, match="inline decoding"):
        build_policy_adapter(robot, policy, kinematics=kin)


def test_parallel_ik_requires_an_explicit_process_configuration():
    _, _, kin, robot, policy = make_adapter(dual=True)
    policy["adapter"]["parallel_ik"] = True
    with pytest.raises(ValueError, match="requires process"):
        build_policy_adapter(robot, policy, kinematics=kin)
    policy["action_decoding"] = "process"
    policy["adapter"]["parallel_ik"] = "true"
    with pytest.raises(ValueError, match="must be a boolean"):
        build_policy_adapter(robot, policy, kinematics=kin)


def test_warmup_requires_valid_seeds_and_surfaces_solver_failure():
    _, state, kin, robot, policy = make_adapter(dual=True)
    policy["adapter"]["ik_warmup_joints"] = {"left_arm": state.groups["left_arm"]}
    with pytest.raises(ValueError, match="every robot group"):
        build_policy_adapter(robot, policy, kinematics=kin)
    policy["adapter"]["ik_warmup_joints"] = state.groups
    kin.models["right_arm"].fail = True
    with pytest.raises(ValueError, match="warmup failed: group=right_arm"):
        build_policy_adapter(robot, policy, kinematics=kin)


@pytest.mark.parametrize("injected", [True, False])
@pytest.mark.parametrize("delta", [True, False])
def test_real_yam_geometry_and_independent_decoder_reconstruction(injected, delta):
    from pathlib import Path

    from manimux.embodiments.robot.base import RobotModel

    _, _, _, robot, policy = make_adapter(dual=True)
    robot["config"] = str(Path("manimux/configs/embodiment/robot/yam_dual.yaml").resolve())
    geometry = RobotModel.from_config(robot["config"]).kinematics
    if delta:
        semantics = "delta_observation_tool_xyz_wxyz"
        policy["adapter"]["action_semantics"] = semantics
        policy["expected_backend"]["model"]["action_semantics"] = semantics
    adapter = build_policy_adapter(robot, policy, kinematics=geometry if injected else None)
    q = np.array([0.15, 0.47, 0.8, -1.0, -0.33, 0.27, 0.5])
    state = RobotState({g: q.copy() for g in robot["group_dims"]}, 1, 0)
    targets, rows = {}, {}
    for group in state.groups:
        target = geometry.models[group].fk(q)
        targets[group] = target
        pose = matrix_pose(np.eye(4) if delta else target)
        rows[group] = np.tile(np.r_[pose, 0.5], (2, 1))
    chunk = adapter.decode_action(
        {"format": "pose", "actions": rows}, ActionContext(1, 1, 2, measured_state=state)
    )
    for group, values in chunk.groups.items():
        for solved in values:
            np.testing.assert_allclose(geometry.models[group].fk(solved), targets[group], atol=2e-4)
