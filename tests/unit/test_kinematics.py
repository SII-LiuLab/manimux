"""YAM FK/IK: accurate enough to drive the arm, and identical to the recorder.

Policies that emit end-effector poses (XR-1 and friends) go through this, so a
silent convention drift here would show up as a wrong-looking arm rather than an
error. The round-trip test pins accuracy; the equivalence test pins the
convention against the FK that produced the recorded ``ee_pos`` / ``ee_rotm``.
"""

from __future__ import annotations

import platform
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from manimux.runtime.executors.smooth import (
    gripper_hysteresis_parameters,
    gripper_release_guard_parameters,
    smooth_parameters,
)

kinematics = pytest.importorskip("manimux.kinematics")

# First frame of a real red-ball episode (configs/molmoact_yam_left.yaml).
START_JOINTS = np.array([-0.6094, 0.5835, 0.8425, -1.0168, -0.1108, -0.4580])


@pytest.fixture(scope="module")
def yam():
    pytest.importorskip("mujoco")
    pytest.importorskip("mink")
    pytest.importorskip("i2rt")
    return kinematics.build_kinematics("yam")


def test_fk_ik_round_trip_is_sub_millimetre(yam) -> None:
    rng = np.random.default_rng(0)
    for _ in range(8):
        joints = START_JOINTS + rng.uniform(-0.25, 0.25, size=6)
        gripper = 0.5
        target = yam.fk(joints, gripper)

        # Seed from a perturbed pose, the way a chunk step seeds off the last one.
        seed = joints + rng.uniform(-0.08, 0.08, size=6)
        converged, solved = yam.ik(target, seed, gripper)
        assert converged

        achieved = yam.fk(solved, gripper)
        assert np.linalg.norm(achieved[:3, 3] - target[:3, 3]) < 1e-3  # < 1 mm
        rotation = target[:3, :3].T @ achieved[:3, :3]
        angle = abs(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1.0, 1.0)))
        assert angle < np.radians(0.05)


def test_fk_matches_the_recorded_episode_convention(yam) -> None:
    """Same convention as yam_abc_reproduce's ForwardKinematics.

    That implementation evaluates the site through mink; this one through
    ``mj_forward`` on the same spliced model, so the two agree to float noise
    (~1e-15) rather than bit-for-bit. ``qpos`` is still exactly equal, which is
    where a convention drift would actually show up.
    """
    import ast
    from pathlib import Path

    source = Path("/home/ubuntu/yam-abc-reproduce/yam_abc_reproduce/data/eepose.py")
    if not source.exists():
        pytest.skip("yam-abc-reproduce checkout is not available")

    import mujoco
    from i2rt.robots.kinematics import Kinematics
    from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

    tree = ast.parse(source.read_text())
    kept = [
        node
        for node in tree.body
        if (isinstance(node, ast.ClassDef) and node.name == "ForwardKinematics")
        or (isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "EE_SITE")
    ]
    namespace: dict[str, object] = {
        "mujoco": mujoco,
        "np": np,
        "Path": Path,
        "Kinematics": Kinematics,
        "ArmType": ArmType,
        "GripperType": GripperType,
        "combine_arm_and_gripper_xml": combine_arm_and_gripper_xml,
    }
    module = ast.Module(body=kept, type_ignores=[])
    exec(compile(module, str(source), "exec"), namespace, namespace)  # noqa: S102
    reference = namespace["ForwardKinematics"]("yam", "linear_4310", 6)  # type: ignore[operator]

    rng = np.random.default_rng(7)
    for _ in range(6):
        joints = START_JOINTS + rng.uniform(-0.4, 0.4, size=6)
        gripper = float(rng.uniform(0.0, 1.0))
        expected = reference.batch(joints[None, :], np.array([[gripper]]))["ee_transform"][0]
        np.testing.assert_allclose(yam.fk(joints, gripper), expected, atol=1e-12)
        np.testing.assert_array_equal(
            yam.robot_state_to_qpos(joints, gripper),
            reference.robot_state_to_qpos(joints, gripper),
        )


def test_ik_rejects_a_malformed_target(yam) -> None:
    with pytest.raises(ValueError, match="4x4"):
        yam.ik(np.eye(3), START_JOINTS, 0.5)


def test_clip_arm_joints_projects_mink_overshoot_onto_model_stops(yam) -> None:
    lower, upper = yam.joint_position_limits()
    overshot = lower.copy()
    overshot[3] = lower[3] - 8.53e-4
    clipped = yam.clip_arm_joints(overshot)
    assert clipped[3] == lower[3]
    np.testing.assert_allclose(yam.clip_arm_joints(upper + 1e-3), upper)


def test_joint4_lower_limit_matches_urdf_and_reaches_below_minus_90_degrees(yam):
    import xml.etree.ElementTree as ET

    from manimux.kinematics.yam import DEFAULT_ASSETS_ROOT

    urdf = ET.parse(DEFAULT_ASSETS_ROOT / "i2rt/robot_models/arm/yam/yam.urdf")
    joint = urdf.find("joint[@name='joint4']/limit")
    assert joint is not None
    lower, _ = yam.joint_position_limits()
    assert lower[3] == pytest.approx(float(joint.attrib["lower"]))
    assert lower[3] == pytest.approx(-1.69297)
    q = np.array([-0.01, 2.0, 2.72, -1.66, 0.4, -0.18])
    target = yam.fk(q, 0.5)
    seed = q.copy()
    seed[3] = -1.55
    converged, solved = yam.ik(target, seed, 0.5)
    assert converged
    assert lower[3] <= solved[3] < -np.pi / 2
    np.testing.assert_allclose(yam.fk(solved, 0.5), target, atol=1e-3)


def test_bounded_ik_respects_deadline_and_does_not_accept_bad_residual(yam):
    import time

    target = yam.fk(START_JOINTS, 0.5)
    ok, solved, info = yam.ik_bounded(
        target, START_JOINTS + 0.02, 0.5, deadline_ns=time.monotonic_ns() + 1_000_000_000
    )
    assert ok and info["reason"] == "converged"
    pos, rot = yam.pose_error(target, solved, 0.5)
    assert pos <= yam._pos_threshold and rot <= yam._ori_threshold
    target[0, 3] += 2
    ok, _, info = yam.ik_bounded(target, START_JOINTS, 0.5, deadline_ns=0)
    assert not ok and info["reason"] == "budget_exceeded" and info["iterations"] == 0
    ok, _, info = yam.ik_bounded(
        target, START_JOINTS, 0.5, deadline_ns=time.monotonic_ns() + 20_000_000
    )
    assert not ok and info["reason"] in ("budget_exceeded", "stagnation", "max_iters")


def test_bounded_ik_tracks_small_targets_inside_convergence_tolerance():
    import time

    yam = kinematics.build_kinematics("yam", pos_threshold=0.004, ori_threshold=0.002)
    seed = START_JOINTS.copy()
    initial_pose = yam.fk(seed, 0.5)
    desired_joints = seed.copy()
    desired_joints[0] += 0.001
    target = yam.fk(desired_joints, 0.5)
    initial_error = np.linalg.norm(target[:3, 3] - initial_pose[:3, 3])
    assert 0 < initial_error < 0.004
    assert yam.pose_error(target, seed, 0.5)[1] < 0.002
    ok, solved, info = yam.ik_bounded(
        target, seed, 0.5, deadline_ns=time.monotonic_ns() + 1_000_000_000
    )
    assert ok and info["iterations"] >= 1
    assert np.linalg.norm(solved - seed) > 1e-6
    assert yam.pose_error(target, solved, 0.5)[0] < initial_error * 0.5


def test_orientation_recovery_resolves_recorded_bin_target_without_relaxing_position():
    import time

    seed = np.array(
        [
            0.3709849698634322,
            2.023537041275654,
            2.5217441062027923,
            -1.459334706645306,
            -0.7856488899061542,
            0.2203021286335538,
        ]
    )
    # Recorded right-arm bin transfer target, expressed in the arm base frame.
    pose = np.array(
        [
            [-0.802328870068013, -0.5902139701852773, 0.0889710830918319, 0.33762224274627933],
            [-0.24389943266057884, 0.46023417534000416, 0.8536378450814466, 0.31983885377351484],
            [-0.5447765146902201, 0.6631982910164398, -0.5132120183639426, 0.4042985377307299],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    gripper = 0.0
    strict = kinematics.build_kinematics("yam", pos_threshold=0.004, ori_threshold=0.002)
    recovery = kinematics.build_kinematics(
        "yam",
        pos_threshold=0.004,
        ori_threshold=0.002,
        recovery_ori_threshold=np.deg2rad(5),
        recovery_max_joint_delta=0.35,
    )
    ok, _, _ = strict.ik_bounded(
        pose, seed, gripper, deadline_ns=time.monotonic_ns() + 1_000_000_000
    )
    assert not ok
    ok, q, info = recovery.ik_bounded(
        pose, seed, gripper, deadline_ns=time.monotonic_ns() + 1_000_000_000
    )
    assert ok and info["reason"] == "recovered_orientation"
    pos, rot = recovery.pose_error(pose, q, gripper)
    assert pos <= 0.004 and rot <= np.deg2rad(5)
    assert np.max(abs(q - seed)) <= 0.35
    assert np.min(recovery.joint_limit_margins(q)) >= 0
    recovery._recovery_max_joint_delta = 1e-6
    ok, _, info = recovery.ik_bounded(
        pose, seed, gripper, deadline_ns=time.monotonic_ns() + 1_000_000_000
    )
    assert not ok
    # Exercise recovery -> smooth commands -> measured release guard together.
    from manimux.runtime.executors.smooth import SmoothExecutor
    from manimux.types import ActionHorizon, RobotState

    cfg = smooth_parameters(
        tracking_mode="braking",
        max_velocity=0.6,
        max_acceleration=1.5,
        gripper=gripper_hysteresis_parameters(
            mode="continuous", group_indices={"right_arm": 6}, max_velocity=1, max_acceleration=12
        ),
        release_guard=gripper_release_guard_parameters(position_tolerance_m=0.02),
    )
    executor = SmoothExecutor(cfg, 0.01)
    measured = np.r_[seed, 0.0]
    goal = np.r_[q, 1.0]
    commands = []
    initial_pos = recovery.fk(seed, 0.0)[:3, 3]
    for tick in range(250):
        state = RobotState({"right_arm": measured.copy()}, tick * 10_000_000, tick)
        ref = ActionHorizon(
            tick * 10_000_000,
            10_000_000,
            "recovery",
            {"right_arm": np.tile(goal, (2, 1))},
            tracking_groups={"right_arm": goal},
        )
        command = executor.step(tick * 10_000_000, state, ref).groups["right_arm"]
        if tick < 50:
            assert command[6] == 0  # Fixed, lagging feedback must not release.
        if commands and command[6] > commands[-1][6] + 1e-9:
            actual = recovery.fk(measured[:6], measured[6])[:3, 3]
            desired = recovery.fk(q, 1.0)[:3, 3]
            assert np.linalg.norm(actual - desired) <= 0.02
        commands.append(command.copy())
        if tick >= 50:
            measured += 0.2 * (command - measured)  # Idealized delayed follower, no hardware.
    commands = np.array(commands)
    velocity = np.diff(np.vstack([np.r_[seed, 0.0], commands]), axis=0)[:, :6] / 0.01
    assert np.max(abs(velocity)) <= 0.6 + 1e-9
    assert np.max(abs(np.diff(np.vstack([np.zeros(6), velocity]), axis=0)) / 0.01) <= 1.5 + 1e-8
    assert commands[-1, 6] > 0.95
    assert np.linalg.norm(recovery.fk(measured[:6], measured[6])[:3, 3] - initial_pos) > 0.04


def test_recorded_last_bottle_target_recovers_inside_seed_bounds():
    import time

    seed = np.array(
        [
            -0.17299916075379507,
            1.9098573281452662,
            2.282940413519494,
            -1.6027695124742518,
            0.395017929350729,
            -0.37785152971694735,
        ]
    )
    pose = np.array(
        [
            [-0.9057774206997469, 0.13404043314241826, 0.40199555520742025, 0.3731892539081932],
            [-0.14393797326968363, 0.7949291097519043, -0.589380666698868, -0.15452294291036503],
            [-0.3985588086611185, -0.5917101255940973, -0.7007353303968549, 0.39082386654400114],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    yam = kinematics.build_kinematics(
        "yam",
        pos_threshold=0.004,
        ori_threshold=0.002,
        recovery_ori_threshold=float(np.deg2rad(5)),
        recovery_max_joint_delta=0.35,
    )
    physical_lower, physical_upper = yam.joint_position_limits()
    ok, q, info = yam.ik_bounded(pose, seed, 0.0, deadline_ns=time.monotonic_ns() + 1_000_000_000)
    assert ok and info["recovery_method"] == "seed_box"
    assert info["reason"] == "recovered_orientation"
    pos, rot = yam.pose_error(pose, q, 0.0)
    assert pos <= 0.004 and rot <= np.deg2rad(5)
    assert np.max(np.abs(q - seed)) <= 0.35
    assert np.all(q >= physical_lower) and np.all(q <= physical_upper)
    # Per-target bounds must never narrow the shared model or later solves.
    lower, upper = yam.joint_position_limits()
    np.testing.assert_array_equal(lower, physical_lower)
    np.testing.assert_array_equal(upper, physical_upper)
    np.testing.assert_array_equal(yam._joint_limits()[0].lower[:6], physical_lower)
    np.testing.assert_array_equal(yam._joint_limits()[0].upper[:6], physical_upper)
    yam._recovery_max_joint_delta = 1e-6
    ok, _, _ = yam.ik_bounded(pose, seed, 0.0, deadline_ns=time.monotonic_ns() + 1_000_000_000)
    assert not ok


# Flange poses from the vendor SDK's Marvin_Kine.fk() with ccs_m6_40.MvKDCfg
# (identical for arms A and B): joints in degrees -> [x, y, z] mm, rotation.
TIANJI_SDK_FLANGE = (
    ((0, 0, 0, 0, 0, 0, 0), (0.0, 0.0, 870.5), ((1, 0, 0), (0, 1, 0), (0, 0, 1))),
    ((90, -90, -90, -90, 0, 0, 0), (427.0, 305.0, 174.5), ((0, 0, 1), (-1, 0, 0), (0, -1, 0))),
    (
        (50, -40, -30, -100, -65, 0, 40),
        (464.6419245293656, 198.7742749840216, 177.3856794645234),
        (
            (0.1467561177185103, -0.04887015050140553, 0.9879647515247294),
            (-0.9890831803520693, -0.020687130011745283, 0.14589895464761),
            (0.013308051403891297, -0.9985908826890189, -0.05137260677358957),
        ),
    ),
    (
        (12, -34, 56, -78, 9, -21, 43),
        (348.71298220702715, 372.06072003625263, 313.7151512729679),
        (
            (-0.6938968578988153, -0.4128061801538545, 0.5899984815303397),
            (-0.1986138205678619, 0.8972957953782231, 0.3942243090415303),
            (-0.6921413878929669, 0.15636915682074604, -0.7046197456260208),
        ),
    ),
)


@pytest.mark.parametrize(("joints_deg", "position_mm", "rotation"), TIANJI_SDK_FLANGE)
def test_tianji_fk_reproduces_the_vendor_sdk(joints_deg, position_mm, rotation) -> None:
    tianji = kinematics.build_kinematics("tianji")
    pose = tianji.fk(np.radians(joints_deg), 1.0)
    np.testing.assert_allclose(pose[:3, 3], np.asarray(position_mm) * 1e-3, atol=1e-9)
    np.testing.assert_allclose(pose[:3, :3], rotation, atol=1e-9)


# teleop configs/tool/umi.yaml kine_offset for both arms (mm, degrees).
UMI_KINE_OFFSET = (-36.745, 0.0, 169.45, 0.0, -90.0, 180.0)


def test_tianji_tool_offset_uses_the_sdk_xyzabc_convention() -> None:
    from manimux.kinematics.tianji import xyzabc_to_matrix

    # Marvin_Kine.xyzabc_to_mat4x4([1, 2, 3, 10, 20, 30]).
    sample = xyzabc_to_matrix([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    np.testing.assert_allclose(
        sample[:3, :3],
        [
            [0.813797681349374, -0.4409696105298824, 0.37852230636979245],
            [0.4698463103929544, 0.8825641192593855, 0.018028311236297334],
            [-0.34202014332566877, 0.16317591116653488, 0.9254165783983237],
        ],
        atol=1e-12,
    )
    np.testing.assert_allclose(sample[:3, 3], [0.001, 0.002, 0.003])

    joints = np.radians([50, -40, -30, -100, -65, 0, 40])
    flange = kinematics.build_kinematics("tianji").fk(joints, 0.0)
    tcp = kinematics.build_kinematics("tianji", tool_xyzabc=UMI_KINE_OFFSET).fk(joints, 0.0)
    np.testing.assert_allclose(tcp, flange @ xyzabc_to_matrix(UMI_KINE_OFFSET), atol=1e-12)
    assert np.linalg.norm(tcp[:3, 3] - flange[:3, 3]) == pytest.approx(0.173388, abs=1e-6)
    # The bundled UMI end effector reproduces the controller's kine_offset.
    mounted = kinematics.build_kinematics("tianji", end_effector="umi_follower").fk(joints, 0.0)
    np.testing.assert_allclose(mounted, tcp, atol=1e-9)
    with pytest.raises(ValueError, match="either"):
        kinematics.build_kinematics(
            "tianji", tool_xyzabc=UMI_KINE_OFFSET, end_effector="umi_follower"
        )


def test_end_effector_spec_drives_urdf_joints() -> None:
    from manimux.kinematics.end_effector import available_end_effectors, load_end_effector

    assert "umi_follower" in available_end_effectors()
    umi = load_end_effector("umi_follower")
    assert umi.inputs == 1 and umi.actuated_joints == ("joint1",)
    assert umi.joint_positions([0.0]) == pytest.approx([-0.44])
    assert umi.joint_positions([1.0]) == pytest.approx([0.0])
    assert umi.joint_positions([1.7]) == pytest.approx([0.0])
    assert umi.joint_positions() == pytest.approx([0.0])
    with pytest.raises(ValueError, match="expects 1 inputs"):
        umi.joint_positions([0.5, 0.5])
    with pytest.raises(FileNotFoundError, match="available: .*umi_follower"):
        load_end_effector("no_such_hand")


def test_end_effector_is_attached_to_the_arm_flange() -> None:
    yourdfpy = pytest.importorskip("yourdfpy")
    from manimux.kinematics.end_effector import TCP_LINK, attach_end_effector, load_end_effector
    from importlib.resources import files

    umi = load_end_effector("umi_follower")
    arm = Path(str(files("manimux.embodiments.arm.tianji"))) / "assets/left/arm.urdf"
    assert attach_end_effector(arm, "Flange_L", None) == arm.resolve()
    combined = attach_end_effector(arm, "Flange_L", umi)
    assert attach_end_effector(arm, "Flange_L", umi) == combined
    robot = yourdfpy.URDF.load(combined)
    assert robot.actuated_joint_names == [*(f"Joint{i}_L" for i in range(1, 8)), "ee_joint1"]
    robot.update_cfg(np.zeros(8))
    np.testing.assert_allclose(
        robot.get_transform(TCP_LINK, "Flange_L"), umi.tool_transform(), atol=1e-12
    )
    for mesh in ET.parse(combined).getroot().iter("mesh"):
        assert Path(mesh.get("filename")).is_file()
    with pytest.raises(ValueError, match="flange link"):
        attach_end_effector(arm, "Flange_X", umi)


def test_end_effector_spec_rejects_inconsistent_descriptions(tmp_path) -> None:
    import shutil

    from manimux.kinematics.end_effector import DEFAULT_END_EFFECTOR_ROOT, load_end_effector

    source = DEFAULT_END_EFFECTOR_ROOT / "umi_follower"
    for name, replace, message in (
        ("wrong_joint", ("joint: joint1", "joint: joint2"), "actuated joints"),
        ("rest_mismatch", ("rest_inputs: [1.0]", "rest_inputs: []"), "rest_inputs"),
        ("unknown_key", ("inputs: 1\n", "inputs: 1\ncolour: red\n"), "colour"),
    ):
        target = tmp_path / name
        shutil.copytree(source, target)
        spec = target / "end_effector.yaml"
        spec.write_text(spec.read_text().replace(*replace))
        with pytest.raises(ValueError, match=message):
            load_end_effector(target)


# The analytic IK runs the vendored Linux x86-64 Marvin kinematics library offline.
requires_marvin_kine = pytest.mark.skipif(
    not (sys.platform.startswith("linux") and platform.machine() == "x86_64"),
    reason="vendored Marvin kinematics library is Linux x86-64 only",
)
UMI_START_A_DEG = [50.0, -40.0, -30.0, -100.0, -65.0, 0.0, 40.0]


@requires_marvin_kine
def test_tianji_analytic_ik_round_trips_the_umi_tool_frame() -> None:
    tianji = kinematics.build_kinematics("tianji", end_effector="umi_follower")
    joints = np.radians(UMI_START_A_DEG)
    target = tianji.fk(joints, 1.0)
    seed = joints + np.radians([0.3, -0.3, 0.3, -0.3, 0.3, -0.3, 0.3])
    converged, solved = tianji.ik(target, seed, 1.0)
    assert converged
    # A redundant arm: the SDK returns the solution nearest the seed, not the pose's
    # original joints, so check the pose and the continuity bound instead.
    assert np.max(np.abs(np.degrees(solved - seed))) <= 1.8
    position_error, rotation_error = tianji.pose_error(target, solved, 1.0)
    assert position_error < 1e-5 and rotation_error < np.radians(0.2)

    ok, _, info = tianji.ik_bounded(target, seed, 1.0, deadline_ns=time.monotonic_ns() + 10**9)
    assert ok and info["reason"] == "converged" and info["iterations"] == 1


@requires_marvin_kine
def test_tianji_analytic_ik_rejects_like_arm_ik() -> None:
    tianji = kinematics.build_kinematics("tianji", end_effector="umi_follower")
    joints = np.radians(UMI_START_A_DEG)
    target = tianji.fk(joints, 1.0)
    far_seed = joints + np.radians([4.0, -4.0, 4.0, -4.0, 4.0, -4.0, 4.0])
    ok, returned, info = tianji.ik_bounded(
        target, far_seed, 1.0, deadline_ns=time.monotonic_ns() + 10**9
    )
    assert not ok and info["reason"] == "branch_jump"
    np.testing.assert_allclose(returned, tianji.clip_arm_joints(far_seed))
    relaxed = kinematics.build_kinematics("tianji", end_effector="umi_follower", max_step_deg=20.0)
    assert relaxed.ik(target, far_seed, 1.0)[0]

    unreachable = target.copy()
    unreachable[:3, 3] += [2.0, 0.0, 0.0]
    ok, _, info = tianji.ik_bounded(
        unreachable, joints, 1.0, deadline_ns=time.monotonic_ns() + 10**9
    )
    assert not ok and info["reason"] in {"ik_failed", "ik_nsp_failed", "fk_mismatch"}
    ok, _, info = tianji.ik_bounded(target, joints, 1.0, deadline_ns=0)
    assert not ok and info["reason"] == "budget_exceeded"


def test_tianji_joint_limit_override_narrows_the_right_arm() -> None:
    right = kinematics.build_kinematics(
        "tianji", arm="right", joint_limits_deg={6: [-58.0, 58.0], 1: [-400.0, 400.0]}
    )
    lower, upper = right.joint_position_limits()
    assert np.degrees(upper[5]) == pytest.approx(58.0)
    assert np.degrees(upper[0]) == pytest.approx(170.0)


def test_tianji_limits_are_radians_from_the_controller_table() -> None:
    tianji = kinematics.build_kinematics("tianji")
    lower, upper = tianji.joint_position_limits()
    np.testing.assert_allclose(np.degrees(lower), [-170, -120, -170, -145, -170, -60, -90])
    np.testing.assert_allclose(np.degrees(upper), [170, 120, 170, 60, 170, 60, 90])
    clipped = tianji.clip_arm_joints(np.radians([180, 0, 0, 90, 0, -70, 0]))
    np.testing.assert_allclose(np.degrees(clipped), [170, 0, 0, 60, 0, -60, 0])
    assert tianji.num_arm_joints == 7
