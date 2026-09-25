"""Component layout and backend boundaries, without devices or learned models."""

import multiprocessing as mp

import numpy as np
import pytest
import yaml

from manimux.cli import load_config
from manimux.embodiments.layout import assembly_action_contract
from manimux.embodiments.robot import apply_action_contract
from manimux.embodiments.robot.base import RobotModel
from manimux.kinematics import IKResult, KinematicCoordinate, RobotKinematics
from manimux.policies import build_policy_model
from manimux.policies.capabilities import PolicyCapabilities
from manimux.policies.xpolicylab.codec import _layouts_from_options, decode_policy_actions
from manimux.policy_adapter import build_policy_adapter
from manimux.runtime.executors.direct import DirectExecutor
from manimux.runtime.executors.limits import motion_limits_parameters
from manimux.types import (
    ActionContext,
    ActionHorizon,
    InferenceRequest,
    ObservationSnapshot,
    RobotState,
)

JOINT = "manimux/configs/experiments/put_bottles/pi05/yam_pi05_joint.yaml"
POSE_CONFIGS = [
    "manimux/configs/experiments/put_bottles/dp/yam_dp_serial_eef_step100000.yaml",
    "manimux/configs/experiments/put_bottles/openwam/yam_openwam_manimux_step30000.yaml",
    "manimux/configs/experiments/put_bottles/xiaomi-xr1/yam_xiaomi_xr1_manimux_step30000.yaml",
]


def test_component_assembly_resolves_mixed_layout_and_executor(tmp_path):
    for filename, layout in [("arm", {"arm_dofs": 2}), ("tool", {"gripper_dofs": 1})]:
        (tmp_path / (filename + ".yaml")).write_text(yaml.safe_dump({"action_layout": layout}))
    assembly = {
        "components": {
            "left": {"config": "arm.yaml"},
            "right": {"config": "arm.yaml"},
            "tool": {"config": "tool.yaml"},
        },
        "groups": {
            "left": {"arm": "left", "end_effector": "tool"},
            "right": {"arm": "right", "end_effector": None},
        },
    }
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(assembly))
    config = {}
    indices = apply_action_contract(config, assembly_action_contract(path))
    assert indices == {"left": 2}
    assert config["robot"]["group_dims"] == {"left": 3, "right": 2}
    limits = motion_limits_parameters(
        arm={"max_velocity": 1}, gripper={"max_velocity": 0.1, "group_indices": indices}
    )
    executor = DirectExecutor(limits, control_dt_s=0.1)
    state = RobotState({"left": np.zeros(3), "right": np.zeros(2)}, 0, 0)
    reference = ActionHorizon(
        0, 100_000_000, "test", {"left": np.ones((2, 3)), "right": np.ones((2, 2))}
    )
    command = executor.step(100_000_000, state, reference)
    np.testing.assert_allclose(command.groups["left"], [0.1, 0.1, 0.01])
    np.testing.assert_allclose(command.groups["right"], [0.1, 0.1])
    # Removing the tool must remove its index, not mark the last arm joint as a gripper.
    del assembly["groups"]["left"]["end_effector"]
    path.write_text(yaml.safe_dump(assembly))
    assert apply_action_contract({}, assembly_action_contract(path)) == {}


def test_multi_coordinate_tool_is_explicitly_unsupported_by_scalar_executor():
    with pytest.raises(ValueError, match="multi-coordinate"):
        apply_action_contract({}, {"group_layouts": {"arm": {"arm_dofs": 6, "gripper_dofs": 2}}})


class Geometry:
    """Offset geometry exists only in this test; no runtime calibration hook."""

    coordinates = tuple(KinematicCoordinate(f"joint{i}", "rad") for i in range(6)) + (
        KinematicCoordinate("gripper", "normalized"),
    )

    def __init__(self, offset):
        self.offset = offset
        self.fail = False
        self.calls = []

    def fk(self, q):
        result = np.eye(4)
        result[0, 3] = q[0] + self.offset
        return result

    def ik(self, target, seed, *, fixed_coordinates):
        self.calls.append((target.copy(), seed.copy(), dict(fixed_coordinates)))
        if self.fail:
            return IKResult(False, reason="test_no_solution")
        q = seed.copy()
        q[0] = target[0, 3] - self.offset
        q[-1] = fixed_coordinates["gripper"]
        return IKResult(True, q)


@pytest.mark.parametrize("path", POSE_CONFIGS)
def test_adapters_use_supplied_per_group_geometry_and_failure(path):
    config = load_config(path)
    models = {"left_arm": Geometry(0.02), "right_arm": Geometry(-0.03)}
    kin = RobotKinematics(models)
    adapter = build_policy_adapter(config["robot"], config["policy"], kinematics=kin)
    assert adapter.kinematics is kin
    q = np.r_[0.1, np.zeros(5), 0.5]
    for group, model in models.items():
        target = adapter._fk(group, q[:6], q[-1])
        assert target[0, 3] == pytest.approx(q[0] + model.offset)
        ok, result = adapter._ik(group, target, q[:6], 0.5)
        assert ok
        np.testing.assert_allclose(result, q[:6])
        model.fail = True
        ok, result = adapter._ik(group, target, q[:6], 0.5)
        assert not ok
        np.testing.assert_allclose(result, q[:6])


def offline_model_probe(path, output):
    config = load_config(path)
    adapter = build_policy_adapter(config["robot"], config["policy"])
    q = np.r_[0.15, 0.47, 0.8, -1.0, -0.33, 0.27, 0.5]
    output.put({name: model.fk(q) for name, model in adapter.kinematics.models.items()})


@pytest.mark.parametrize("path", POSE_CONFIGS)
def test_decoder_process_loads_same_body_definition(path):
    config = load_config(path)
    model = RobotModel.from_config(config["robot"]["config"])
    q = np.r_[0.15, 0.47, 0.8, -1.0, -0.33, 0.27, 0.5]
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=offline_model_probe, args=(path, queue))
    process.start()
    try:
        actual = queue.get(timeout=10)
        process.join(10)
        assert process.exitcode == 0
        for name, expected in model.kinematics.fk({name: q for name in model.groups}).items():
            np.testing.assert_allclose(actual[name], expected, atol=1e-12)
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        queue.close()
        queue.join_thread()


class AlternateBackend:
    """Test-only backend, with a different external packet format."""

    def reset(self, session_id):
        self.session_id = session_id

    def infer(self, request):
        packet = {
            "targets": {
                name: [values.tolist()] * 3
                for name, values in request.observation.state.groups.items()
            }
        }
        return {
            "format": "joint",
            "actions": {name: np.array(rows) for name, rows in packet["targets"].items()},
        }

    def capabilities(self):
        return PolicyCapabilities()

    def close(self):
        pass


def build_alternative(config):
    return AlternateBackend()


def test_two_backend_clients_feed_same_joint_adapter(monkeypatch):
    from manimux.policies.xpolicylab import client

    config = load_config(JOINT)
    policy = config["policy"]
    policy["horizon_policy_steps"] = 3
    q = np.r_[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    state = RobotState({"left_arm": q, "right_arm": q + 0.01}, 1, 1)
    frames = {}
    request = InferenceRequest("test", 1, 1, 100, ObservationSnapshot(state, frames))
    policy["adapter"]["camera_map"] = {}

    class WireClient:
        def __init__(self, **options):
            pass

        def connect(self):
            pass

        def reset(self):
            pass

        def close(self):
            pass

        def infer(self, observation, **options):
            return [
                {
                    "left_arm_joint_state": q[:6],
                    "left_ee_joint_state": q[6:],
                    "right_arm_joint_state": (q + 0.01)[:6],
                    "right_ee_joint_state": (q + 0.01)[6:],
                }
            ] * 3

    monkeypatch.setattr(client, "XPolicyLabWsClient", WireClient)
    adapter = build_policy_adapter(config["robot"], policy)
    results = []
    for backend in ("xpolicylab_ws", __name__ + ":build_alternative"):
        model = build_policy_model({**policy, "worker": backend})
        model.reset("test")
        try:
            raw = model.infer(request)
            results.append(adapter.decode_action(raw, ActionContext(1, 1, 2)))
        finally:
            model.close()
    for name in state.groups:
        np.testing.assert_array_equal(results[0].groups[name], results[1].groups[name])


@pytest.mark.parametrize("metadata", ["aac", "paint", "dvac", "autohorizon"])
def test_wire_conversion_preserves_sampler_metadata(metadata):
    config = load_config(JOINT)
    layouts = _layouts_from_options(config["policy"]["adapter"], config["robot"]["group_dims"])
    step = {
        key: np.zeros(width)
        for side in ("left", "right")
        for key, width in [(side + "_arm_joint_state", 6), (side + "_ee_joint_state", 1)]
    }
    raw = {
        "actions": [step] * 3,
        metadata: {"execution_steps": 2},
        "action_semantics": "absolute_joint_position",
    }
    converted = decode_policy_actions(raw, layouts=layouts, format="joint")
    assert converted[metadata] == raw[metadata]
    assert converted["action_semantics"] == raw["action_semantics"]


@pytest.mark.parametrize("path", POSE_CONFIGS)
def test_pose_decode_preserves_targets_and_adapter_failure_policy(path):
    from manimux.kinematics.poses import matrix_pose

    config = load_config(path)
    config["policy"]["horizon_policy_steps"] = 2
    if config["policy"].get("expected_backend"):
        config["policy"]["expected_backend"]["model"]["action_horizon"] = 2
    models = {"left_arm": Geometry(0.02), "right_arm": Geometry(-0.03)}
    adapter = build_policy_adapter(
        config["robot"], config["policy"], kinematics=RobotKinematics(models)
    )
    q = np.r_[0.1, np.zeros(5), 0.5]
    state = RobotState({name: q.copy() for name in models}, 1, 0)
    request = InferenceRequest("test", 1, 1, 100, ObservationSnapshot(state))
    is_dp, is_xr1 = "/dp/" in path, "/xiaomi-xr1/" in path
    if not is_dp:
        adapter.prepare_request(request)
    raw = (
        np.zeros((2, 60))
        if is_xr1
        else {
            "format": "pose",
            "action_semantics": "absolute_per_arm_base_xyz_wxyz",
            "actions": {
                name: np.tile(np.r_[matrix_pose(model.fk(q)), 0.5], (2, 1))
                for name, model in models.items()
            },
        }
    )
    context = ActionContext(1, 1, 2, measured_state=state)
    chunk = adapter.decode_action(raw, context)
    for group in models:
        np.testing.assert_allclose(chunk.groups[group], np.tile(q, (2, 1)))
    for model in models.values():
        model.fail = True
    if not is_dp:
        adapter.prepare_request(request)
    if is_xr1:
        held = adapter.decode_action(raw, context)
        for group in models:
            np.testing.assert_allclose(held.groups[group], np.tile(q, (2, 1)))
    else:
        with pytest.raises(ValueError, match="IK failed"):
            adapter.decode_action(raw, context)


def test_xr1_rtc_conversion_uses_each_groups_geometry():
    from manimux.policy_adapter.xr1.yam import joint_condition_to_xr1_actions

    kin = RobotKinematics({"left_arm": Geometry(0.02), "right_arm": Geometry(-0.03)})
    q = np.r_[0.1, np.zeros(5), 0.5]
    anchor = {name: q.copy() for name in kin.models}
    condition = np.tile(np.r_[q, q], (2, 1))
    condition[:, 0] += 0.01
    condition[:, 7] -= 0.02
    actions = joint_condition_to_xr1_actions(
        condition, anchor, group_order=("left_arm", "right_arm"), kinematics=kin
    )
    np.testing.assert_allclose(actions[:, 0], 0.01, atol=1e-8)
    np.testing.assert_allclose(actions[:, 8], -0.02, atol=1e-8)
    np.testing.assert_allclose(actions[:, [6, 14]], 0)


def test_actual_yam_fk_ik_matches_previous_default_solver():
    from manimux.kinematics import build_kinematics

    config = load_config(POSE_CONFIGS[0])
    adapter = build_policy_adapter(config["robot"], config["policy"])
    previous = build_kinematics("yam")
    q = np.r_[0.15, 0.47, 0.8, -1.0, -0.33, 0.27, 0.5]
    target = previous.fk(q[:6], q[-1])
    np.testing.assert_allclose(adapter._fk("left_arm", q[:6], q[-1]), target, atol=1e-12)
    old_ok, old_q = previous.ik(target, q[:6], q[-1])
    new_ok, new_q = adapter._ik("left_arm", target, q[:6], q[-1])
    assert old_ok and new_ok
    np.testing.assert_allclose(new_q, old_q, atol=1e-10)


def test_pose_wire_allows_arm_without_gripper():
    from manimux.policies.xpolicylab.codec import GroupLayout

    layout = GroupLayout("arm", "", 6, 0)
    converted = decode_policy_actions(
        [{"ee_pose": np.r_[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]}], layouts=(layout,), format="pose"
    )
    assert converted["actions"]["arm"].shape == (1, 7)


def test_absolute_adapter_rejects_relative_semantics():
    config = load_config(JOINT)
    adapter = build_policy_adapter(config["robot"], config["policy"])
    with pytest.raises(ValueError, match="absolute joint-position"):
        adapter.decode_action(
            {
                "format": "joint",
                "action_semantics": "anchor_relative_arm_absolute_gripper",
                "actions": {
                    name: np.zeros((50, width))
                    for name, width in config["robot"]["group_dims"].items()
                },
            },
            ActionContext(1, 1, 2),
        )


def test_relative_joint_adapter_keeps_anchor_and_absolute_gripper():
    config = load_config(
        "manimux/configs/experiments/assemble_screwdriver/lingbot-vla2/yam_lingbot_vla2_manimux_step15000.yaml"
    )
    adapter = build_policy_adapter(config["robot"], config["policy"])
    width = config["policy"]["horizon_policy_steps"]
    q = np.r_[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    state = RobotState({"left_arm": q, "right_arm": q + 0.01}, 1, 0)
    adapter.prepare_request(InferenceRequest("test", 1, 1, 100, ObservationSnapshot(state)))
    raw = {
        "format": "joint",
        "action_semantics": "anchor_relative_arm_absolute_gripper",
        "actions": {
            name: np.tile(np.r_[np.full(6, 0.1), 0.5], (width, 1)) for name in state.groups
        },
    }
    context = ActionContext(
        1, 1, 2, measured_state=RobotState({name: np.zeros(7) for name in state.groups}, 2, 1)
    )
    chunk = adapter.decode_action(raw, context)
    for name in state.groups:
        np.testing.assert_allclose(chunk.groups[name][0, :6], state.groups[name][:6] + 0.1)
        np.testing.assert_allclose(chunk.groups[name][:, 6], 0.5)


def test_codec_respects_mixed_group_layout_both_directions():
    from manimux.policies.xpolicylab.codec import encode_observation

    dimensions = {"left": 3, "right": 2}
    options = {
        "group_layouts": {
            "left": {"arm_dofs": 2, "gripper_dofs": 1},
            "right": {"arm_dofs": 2, "gripper_dofs": 0},
        },
        "group_prefixes": {"left": "l", "right": "r"},
    }
    layouts = _layouts_from_options(options, dimensions)
    groups = {"left": np.array([0.1, 0.2, 0.5]), "right": np.array([0.3, 0.4])}
    observation = encode_observation(
        ObservationSnapshot(RobotState(groups, 1, 1)),
        layouts=layouts,
        camera_map={},
        instruction="test",
        frequency=30,
    )
    assert "r_ee_joint_state" not in observation["state"]
    decoded = decode_policy_actions([observation["state"]] * 2, layouts=layouts, format="joint")
    for name in groups:
        # The existing observation wire intentionally encodes float32 values.
        np.testing.assert_allclose(
            decoded["actions"][name], np.tile(groups[name], (2, 1)), atol=1e-7, rtol=0
        )
