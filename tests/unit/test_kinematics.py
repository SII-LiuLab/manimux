"""YAM FK/IK: accurate enough to drive the arm, and identical to the recorder.

Policies that emit end-effector poses (XR-1 and friends) go through this, so a
silent convention drift here would show up as a wrong-looking arm rather than an
error. The round-trip test pins accuracy; the equivalence test pins the
convention against the FK that produced the recorded ``ee_pos`` / ``ee_rotm``.
"""

from __future__ import annotations

import numpy as np
import pytest

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

    urdf = ET.parse(DEFAULT_ASSETS_ROOT / 'i2rt/robot_models/arm/yam/yam.urdf')
    joint = urdf.find("joint[@name='joint4']/limit")
    assert joint is not None
    lower, _ = yam.joint_position_limits()
    assert lower[3] == pytest.approx(float(joint.attrib['lower']))
    assert lower[3] == pytest.approx(-1.69297)
    q = np.array([-.01, 2., 2.72, -1.66, .4, -.18])
    target = yam.fk(q, .5)
    seed = q.copy()
    seed[3] = -1.55
    converged, solved = yam.ik(target, seed, .5)
    assert converged
    assert lower[3] <= solved[3] < -np.pi / 2
    np.testing.assert_allclose(yam.fk(solved, .5), target, atol=1e-3)


def test_bounded_ik_respects_deadline_and_does_not_accept_bad_residual(yam):
    import time
    target = yam.fk(START_JOINTS, .5)
    ok, solved, info = yam.ik_bounded(target, START_JOINTS + .02, .5,
                                      deadline_ns=time.monotonic_ns() + 1_000_000_000)
    assert ok and info['reason'] == 'converged'
    pos, rot = yam.pose_error(target, solved, .5)
    assert pos <= yam._pos_threshold and rot <= yam._ori_threshold
    target[0, 3] += 2
    ok, _, info = yam.ik_bounded(target, START_JOINTS, .5, deadline_ns=0)
    assert not ok and info['reason'] == 'budget_exceeded' and info['iterations'] == 0
    ok, _, info = yam.ik_bounded(target, START_JOINTS, .5,
                                 deadline_ns=time.monotonic_ns() + 20_000_000)
    assert not ok and info['reason'] in ('budget_exceeded', 'stagnation', 'max_iters')


def test_bounded_ik_tracks_small_targets_inside_convergence_tolerance():
    import time
    yam = kinematics.build_kinematics('yam', pos_threshold=.004, ori_threshold=.002)
    seed = START_JOINTS.copy()
    initial_pose = yam.fk(seed, .5)
    desired_joints = seed.copy()
    desired_joints[0] += .001
    target = yam.fk(desired_joints, .5)
    initial_error = np.linalg.norm(target[:3, 3] - initial_pose[:3, 3])
    assert 0 < initial_error < .004
    assert yam.pose_error(target, seed, .5)[1] < .002
    ok, solved, info = yam.ik_bounded(target, seed, .5,
                                     deadline_ns=time.monotonic_ns() + 1_000_000_000)
    assert ok and info['iterations'] >= 1
    assert np.linalg.norm(solved - seed) > 1e-6
    assert yam.pose_error(target, solved, .5)[0] < initial_error * .5


def test_orientation_recovery_resolves_recorded_bin_target_without_relaxing_position():
    import time
    seed = np.array([.3709849698634322, 2.023537041275654, 2.5217441062027923,
                     -1.459334706645306, -.7856488899061542, .2203021286335538])
    # Recorded right-arm bin transfer target, expressed in the arm base frame.
    pose = np.array([
        [-.802328870068013, -.5902139701852773, .0889710830918319, .33762224274627933],
        [-.24389943266057884, .46023417534000416, .8536378450814466, .31983885377351484],
        [-.5447765146902201, .6631982910164398, -.5132120183639426, .4042985377307299],
        [0., 0., 0., 1.],
    ])
    gripper = 0.0
    strict = kinematics.build_kinematics('yam', pos_threshold=.004, ori_threshold=.002)
    recovery = kinematics.build_kinematics('yam', pos_threshold=.004, ori_threshold=.002,
                                         recovery_ori_threshold=np.deg2rad(5), recovery_max_joint_delta=.35)
    ok, _, _ = strict.ik_bounded(pose, seed, gripper, deadline_ns=time.monotonic_ns()+1_000_000_000)
    assert not ok
    ok, q, info = recovery.ik_bounded(pose, seed, gripper, deadline_ns=time.monotonic_ns()+1_000_000_000)
    assert ok and info['reason'] == 'recovered_orientation'
    pos, rot = recovery.pose_error(pose, q, gripper)
    assert pos <= .004 and rot <= np.deg2rad(5)
    assert np.max(abs(q-seed)) <= .35
    assert np.min(recovery.joint_limit_margins(q)) >= 0
    recovery._recovery_max_joint_delta = 1e-6
    ok, _, info = recovery.ik_bounded(pose, seed, gripper, deadline_ns=time.monotonic_ns()+1_000_000_000)
    assert not ok
    # Exercise recovery -> smooth commands -> measured release guard together.
    from manimux.config import SmoothConfig, GripperHysteresisConfig, GripperReleaseGuardConfig
    from manimux.runtime.executors.smooth import SmoothExecutor
    from manimux.types import ActionHorizon, RobotState
    cfg = SmoothConfig(tracking_mode='braking', max_velocity=.6, max_acceleration=1.5,
        gripper=GripperHysteresisConfig(mode='continuous', group_indices={'right_arm':6},
                                      max_velocity=1, max_acceleration=12),
        release_guard=GripperReleaseGuardConfig(position_tolerance_m=.02))
    executor = SmoothExecutor(cfg, .01)
    measured = np.r_[seed, 0.]
    goal = np.r_[q, 1.]
    commands = []
    initial_pos = recovery.fk(seed, 0.)[:3,3]
    for tick in range(250):
        state = RobotState({'right_arm':measured.copy()},tick*10_000_000,tick)
        ref = ActionHorizon(tick*10_000_000,10_000_000,'recovery',
            {'right_arm':np.tile(goal,(2,1))}, tracking_groups={'right_arm':goal})
        command = executor.step(tick*10_000_000,state,ref).groups['right_arm']
        if tick < 50:
            assert command[6] == 0  # Fixed, lagging feedback must not release.
        if commands and command[6] > commands[-1][6]+1e-9:
            actual = recovery.fk(measured[:6],measured[6])[:3,3]
            desired = recovery.fk(q,1.)[:3,3]
            assert np.linalg.norm(actual-desired) <= .02
        commands.append(command.copy())
        if tick >= 50:
            measured += .2*(command-measured)  # Idealized delayed follower, no hardware.
    commands = np.array(commands)
    velocity = np.diff(np.vstack([np.r_[seed,0.],commands]),axis=0)[:,:6]/.01
    assert np.max(abs(velocity)) <= .6+1e-9
    assert np.max(abs(np.diff(np.vstack([np.zeros(6),velocity]),axis=0))/.01) <= 1.5+1e-8
    assert commands[-1,6] > .95
    assert np.linalg.norm(recovery.fk(measured[:6],measured[6])[:3,3]-initial_pos) > .04


def test_recorded_last_bottle_target_recovers_inside_seed_bounds():
    import time
    seed = np.array([-0.17299916075379507, 1.9098573281452662, 2.282940413519494, -1.6027695124742518, 0.395017929350729, -0.37785152971694735])
    pose = np.array([[-0.9057774206997469, 0.13404043314241826, 0.40199555520742025, 0.3731892539081932], [-0.14393797326968363, 0.7949291097519043, -0.589380666698868, -0.15452294291036503], [-0.3985588086611185, -0.5917101255940973, -0.7007353303968549, 0.39082386654400114], [0.0, 0.0, 0.0, 1.0]])
    yam = kinematics.build_kinematics('yam', pos_threshold=.004, ori_threshold=.002,
                                     recovery_ori_threshold=float(np.deg2rad(5)),
                                     recovery_max_joint_delta=.35)
    physical_lower, physical_upper = yam.joint_position_limits()
    ok, q, info = yam.ik_bounded(pose, seed, 0.,
                                deadline_ns=time.monotonic_ns() + 1_000_000_000)
    assert ok and info['recovery_method'] == 'seed_box'
    assert info['reason'] == 'recovered_orientation'
    pos, rot = yam.pose_error(pose, q, 0.)
    assert pos <= .004 and rot <= np.deg2rad(5)
    assert np.max(np.abs(q - seed)) <= .35
    assert np.all(q >= physical_lower) and np.all(q <= physical_upper)
    # Per-target bounds must never narrow the shared model or later solves.
    lower, upper = yam.joint_position_limits()
    np.testing.assert_array_equal(lower, physical_lower)
    np.testing.assert_array_equal(upper, physical_upper)
    np.testing.assert_array_equal(yam._joint_limits()[0].lower[:6], physical_lower)
    np.testing.assert_array_equal(yam._joint_limits()[0].upper[:6], physical_upper)
    yam._recovery_max_joint_delta = 1e-6
    ok, _, _ = yam.ik_bounded(pose, seed, 0.,
                             deadline_ns=time.monotonic_ns() + 1_000_000_000)
    assert not ok
