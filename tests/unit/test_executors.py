from __future__ import annotations

import numpy as np
import pytest

from manimux.config import GripperHysteresisConfig, MPCConfig, SmoothConfig
from manimux.runtime.executors import DirectExecutor, MPCExecutor, SmoothExecutor
from manimux.runtime.safety import SafetyGuard
from manimux.types import ActionHorizon, RobotCommand, RobotState


def _state() -> RobotState:
    return RobotState(
        groups={"left_arm": np.zeros(2), "right_arm": np.zeros(2)},
        monotonic_ns=0,
        sequence=0,
    )


def _reference(target: float, horizon_steps: int = 20) -> ActionHorizon:
    values = np.full((horizon_steps, 2), target, dtype=np.float64)
    return ActionHorizon(
        start_time_ns=0,
        dt_ns=10_000_000,
        plan_id="plan",
        groups={"left_arm": values, "right_arm": -values},
    )


def test_smooth_executor_obeys_acceleration_and_velocity_limits() -> None:
    executor = SmoothExecutor(
        SmoothConfig(
            cutoff_hz=100.0,
            max_velocity=1.0,
            max_acceleration=2.0,
            position_limit_abs=3.0,
        ),
        control_dt_s=0.01,
    )
    state = _state()
    executor.reset(state)
    first = executor.step(0, state, _reference(2.0))
    second = executor.step(10_000_000, state, _reference(2.0))
    np.testing.assert_allclose(first.groups["left_arm"], [0.0002, 0.0002])
    np.testing.assert_allclose(second.groups["left_arm"], [0.0006, 0.0006])


def test_smooth_executor_latches_gripper_closed_across_noisy_open_requests() -> None:
    executor = SmoothExecutor(
        SmoothConfig(
            cutoff_hz=100.0,
            max_velocity=1.0,
            max_acceleration=2.0,
            position_limit_abs=3.0,
            gripper=GripperHysteresisConfig(
                group_indices={"left_arm": 1, "right_arm": 1},
                close_threshold=0.55,
                open_threshold=0.85,
                min_closed_s=10.0,
                open_confirm_s=0.5,
                max_velocity=3.0,
                max_acceleration=12.0,
            ),
        ),
        control_dt_s=0.01,
    )
    state = RobotState(
        groups={
            "left_arm": np.array([0.0, 1.0]),
            "right_arm": np.array([0.0, 1.0]),
        },
        monotonic_ns=0,
        sequence=0,
    )
    executor.reset(state)

    closing = _reference(0.0)
    closing.groups["left_arm"][:, 1] = 0.2
    closing.groups["right_arm"][:, 1] = 0.2
    first = executor.step(0, state, closing)

    noisy_open = _reference(0.0)
    noisy_open.groups["left_arm"][:, 1] = 0.95
    noisy_open.groups["right_arm"][:, 1] = 0.95
    held = executor.step(5_000_000_000, state, noisy_open)
    candidate = executor.step(10_100_000_000, state, noisy_open)
    opened = executor.step(10_700_000_000, state, noisy_open)
    for tick in range(1, 8):
        opened = executor.step(10_700_000_000 + tick * 10_000_000, state, noisy_open)

    assert first.groups["left_arm"][1] < 1.0
    assert held.groups["left_arm"][1] < first.groups["left_arm"][1]
    assert candidate.groups["left_arm"][1] < held.groups["left_arm"][1]
    assert opened.groups["left_arm"][1] > candidate.groups["left_arm"][1]


def test_smooth_executor_continuous_gripper_has_independent_limits() -> None:
    executor = SmoothExecutor(
        SmoothConfig(
            cutoff_hz=100.0,
            max_velocity=0.25,
            max_acceleration=0.5,
            position_limit_abs=3.0,
            gripper=GripperHysteresisConfig(
                mode="continuous",
                group_indices={"left_arm": 1, "right_arm": 1},
                max_velocity=1.0,
                max_acceleration=12.0,
            ),
        ),
        control_dt_s=0.01,
    )
    state = RobotState(
        groups={
            "left_arm": np.array([0.0, 1.0]),
            "right_arm": np.array([0.0, 1.0]),
        },
        monotonic_ns=0,
        sequence=0,
    )
    executor.reset(state)
    reference = _reference(1.0)
    reference.groups["left_arm"][:, 1] = 0.4
    reference.groups["right_arm"][:, 1] = 0.4

    command = executor.step(0, state, reference)

    assert command.groups["left_arm"][0] == pytest.approx(0.00005)
    assert command.groups["left_arm"][1] == pytest.approx(0.9988)
    assert command.groups["left_arm"][1] < 1.0
    assert command.groups["left_arm"][1] > 0.4


def test_direct_executor_forwards_reference_without_shaping() -> None:
    executor = DirectExecutor()
    state = _state()
    reference = _reference(7.5)

    executor.reset(state)
    command = executor.step(10_000_000, state, reference)

    np.testing.assert_array_equal(command.groups["left_arm"], [7.5, 7.5])
    np.testing.assert_array_equal(command.groups["right_arm"], [-7.5, -7.5])
    assert command.plan_id == "plan"


def test_direct_execution_has_no_generic_position_limit() -> None:
    guard = SafetyGuard({"left_arm": 2, "right_arm": 2}, position_limit_abs=None)
    guard.validate_command(
        RobotCommand(
            groups={"left_arm": np.array([7.5, -7.5]), "right_arm": np.array([8.0, -8.0])},
            monotonic_ns=0,
            plan_id="direct",
        )
    )


def _per_joint_safety_guard(*, max_acceleration: float = 10.0) -> SafetyGuard:
    groups = {"left_arm": 2, "right_arm": 2}
    return SafetyGuard(
        groups,
        position_limit_abs=None,
        position_lower={name: [-1.0, 0.0] for name in groups},
        position_upper={name: [1.0, 1.0] for name in groups},
        max_velocity={name: [1.0, 2.0] for name in groups},
        max_acceleration={name: [max_acceleration, 20.0] for name in groups},
        control_dt_s=0.1,
    )


def test_command_safety_uses_true_per_joint_position_bounds() -> None:
    guard = _per_joint_safety_guard()
    state = RobotState(
        groups={
            "left_arm": np.array([0.0, 0.5]),
            "right_arm": np.array([0.0, 0.5]),
        },
        monotonic_ns=0,
        sequence=0,
    )
    guard.reset(state)

    with pytest.raises(ValueError, match="joint 1 position -0.100000"):
        guard.validate_command(
            RobotCommand(
                groups={
                    "left_arm": np.array([0.0, -0.1]),
                    "right_arm": np.array([0.0, 0.5]),
                },
                monotonic_ns=100_000_000,
                plan_id="unsafe-position",
            )
        )


def test_command_safety_allows_milliradian_joint_slop() -> None:
    guard = _per_joint_safety_guard()
    state = RobotState(
        groups={
            "left_arm": np.array([-1.0, 0.5]),
            "right_arm": np.array([0.0, 0.5]),
        },
        monotonic_ns=0,
        sequence=0,
    )
    guard.reset(state)
    guard.validate_command(
        RobotCommand(
            groups={
                "left_arm": np.array([-1.0 - 8.5e-4, 0.5]),
                "right_arm": np.array([0.0, 0.5]),
            },
            monotonic_ns=100_000_000,
            plan_id="encoder-slop",
        )
    )
    with pytest.raises(ValueError, match="joint 0 position -1.010000"):
        guard.validate_command(
            RobotCommand(
                groups={
                    "left_arm": np.array([-1.01, 0.5]),
                    "right_arm": np.array([0.0, 0.5]),
                },
                monotonic_ns=200_000_000,
                plan_id="real-excursion",
            )
        )


def test_command_safety_checks_velocity_and_acceleration_after_reset() -> None:
    guard = _per_joint_safety_guard()
    state = RobotState(
        groups={
            "left_arm": np.array([0.0, 0.5]),
            "right_arm": np.array([0.0, 0.5]),
        },
        monotonic_ns=0,
        sequence=0,
    )
    guard.reset(state)
    guard.validate_command(
        RobotCommand(
            groups={
                "left_arm": np.array([0.1, 0.5]),
                "right_arm": np.array([0.1, 0.5]),
            },
            monotonic_ns=100_000_000,
            plan_id="bounded",
        )
    )

    with pytest.raises(ValueError, match="velocity 1.500000"):
        guard.validate_command(
            RobotCommand(
                groups={
                    "left_arm": np.array([0.25, 0.5]),
                    "right_arm": np.array([0.2, 0.5]),
                },
                monotonic_ns=200_000_000,
                plan_id="unsafe-velocity",
            )
        )

    acceleration_guard = _per_joint_safety_guard(max_acceleration=5.0)
    acceleration_guard.reset(state)
    with pytest.raises(ValueError, match="acceleration 10.000000"):
        acceleration_guard.validate_command(
            RobotCommand(
                groups={
                    "left_arm": np.array([0.1, 0.5]),
                    "right_arm": np.array([0.1, 0.5]),
                },
                monotonic_ns=100_000_000,
                plan_id="unsafe-acceleration",
            )
        )


def test_mpc_executor_tracks_reference_with_bounded_first_step() -> None:
    executor = MPCExecutor(
        MPCConfig(
            horizon_steps=10,
            dynamics_a=0.85,
            tracking_weight=10.0,
            command_delta_weight=1.0,
            max_velocity=2.0,
            max_acceleration=8.0,
            position_limit_abs=3.0,
        ),
        control_dt_s=0.01,
    )
    state = _state()
    executor.reset(state)
    command = executor.step(0, state, _reference(1.0))
    assert np.all(command.groups["left_arm"] > 0)
    assert np.all(command.groups["right_arm"] < 0)
    assert np.max(np.abs(command.groups["left_arm"])) <= 0.0008 + 1e-12


@pytest.mark.parametrize('velocity,acceleration', [(0.25, 0.5), (0.8, 3.0), (1.0, 4.0)])
@pytest.mark.parametrize('target', [0.001, 0.1, -0.1, 2.0])
def test_braking_tracker_reaches_static_target_without_overshoot(velocity, acceleration, target):
    executor = SmoothExecutor(SmoothConfig(
        tracking_mode='braking', max_velocity=velocity, max_acceleration=acceleration,
    ), 0.01)
    state = _state()
    executor.reset(state)
    commands = [np.zeros(2), np.zeros(2)]
    for i in range(1200):
        commands.append(executor.step(i * 10_000_000, state, _reference(target)).groups['left_arm'])
    q = np.asarray(commands)
    v = np.diff(q, axis=0) / .01
    assert np.max(np.abs(v)) <= velocity + 1e-9
    assert np.max(np.abs(np.diff(v, axis=0) / .01)) <= acceleration + 1e-8
    assert np.all(q * np.sign(target) >= -1e-12)
    assert np.max(q * np.sign(target)) <= abs(target) + 1e-12
    np.testing.assert_allclose(q[-1], target, atol=1e-6)


def test_braking_tracker_tracks_motion_across_plans_and_brakes_through_gap():
    executor = SmoothExecutor(SmoothConfig(
        tracking_mode='braking', max_velocity=.8, max_acceleration=3,
        gripper=GripperHysteresisConfig(mode='continuous', group_indices={'arm': 1}),
    ), .01)
    state = RobotState({'arm': np.array([0., .4])}, 0, 0)
    executor.reset(state)
    q = [0., 0.]
    for i in range(200):
        ref = ActionHorizon(i * 10_000_000, 10_000_000, f'plan-{i // 25}',
                            {'arm': np.array([[.3*i*.01, .4], [.3*(i+1)*.01, .4]])})
        cmd = executor.step(i * 10_000_000, state, ref)
        q.append(cmd.groups['arm'][0])
        if i > 100:
            assert abs(q[-1] - .3*(i+1)*.01) < .005
    # Deliberately lagging feedback must not reset the continuous command.
    before_gap = q[-1]
    for i in range(50):
        cmd = executor.brake_hold((200+i)*10_000_000, state)
        q.append(cmd.groups['arm'][0])
        assert cmd.groups['arm'][1] == pytest.approx(.4)
        assert cmd.plan_id is None
    assert q[-1] >= before_gap
    assert q[-1] < before_gap + .02
    assert q[-1] == pytest.approx(q[-2])
    v = np.diff(q)/.01
    assert np.max(abs(v)) <= .8 + 1e-9
    assert np.max(abs(np.diff(v)/.01)) <= 3 + 1e-8


def test_braking_tracker_handles_unreachable_reversal_without_command_jump():
    executor = SmoothExecutor(SmoothConfig(
        tracking_mode='braking', max_velocity=.8, max_acceleration=3,
    ), .01)
    state = _state()
    executor.reset(state)
    q = [0., 0.]
    for i in range(500):
        target = 1. if i < 30 else -.1
        ref = _reference(target)
        ref.plan_id = 'first' if i < 30 else 'reversal'
        q.append(executor.step(i*10_000_000, state, ref).groups['left_arm'][0])
    v = np.diff(q)/.01
    assert np.max(abs(v)) <= .8 + 1e-9
    assert np.max(abs(np.diff(v)/.01)) <= 3 + 1e-8
    assert q[-1] == pytest.approx(-.1, abs=1e-6)


def test_braking_tracker_stops_at_position_bound_even_with_outward_feedforward():
    executor = SmoothExecutor(SmoothConfig(
        tracking_mode='braking', max_velocity=.8, max_acceleration=3, position_limit_abs=1,
    ), .01)
    state = _state()
    executor.reset(state)
    q = [0., 0.]
    for i in range(500):
        ref = _reference(2., horizon_steps=2)
        ref.groups['left_arm'][1] = 3.
        q.append(executor.step(i*10_000_000, state, ref).groups['left_arm'][0])
    assert max(q) <= 1. + 1e-12
    assert q[-1] == pytest.approx(1., abs=1e-6)
    assert max(abs(np.diff(q, n=2)/.01**2)) <= 3 + 1e-8


def test_independent_hold_brakes_one_arm_freezes_gripper_and_resumes_continuously():
    config = SmoothConfig(tracking_mode='braking', max_velocity=.8, max_acceleration=3,
        gripper=GripperHysteresisConfig(mode='continuous', group_indices={'left_arm': 1, 'right_arm': 1},
                                      max_velocity=1, max_acceleration=12))
    executor = SmoothExecutor(config, .01)
    state = RobotState({name: np.array([0., .5]) for name in ('left_arm', 'right_arm')}, 0, 0)
    groups = {name: np.array([[1., .8], [1., .8]]) for name in state.groups}
    commands = []
    for i in range(100):
        hold = ('right_arm',) if 25 <= i < 70 else ()
        ref = ActionHorizon(i * 10_000_000, 10_000_000, str(i), groups, hold_groups=hold)
        commands.append(executor.step(i * 10_000_000, state, ref).groups)
    right = np.array([c['right_arm'] for c in commands])
    left = np.array([c['left_arm'] for c in commands])
    velocity = np.diff(np.r_[0., right[:, 0]]) / .01
    assert np.max(np.abs(velocity)) <= .8 + 1e-10
    assert np.max(np.abs(np.diff(np.r_[0., velocity]))) <= .03 + 1e-10
    np.testing.assert_allclose(right[25:70, 1], right[24, 1], atol=0, rtol=0)
    assert left[69, 0] > left[25, 0] + .2
    assert abs(velocity[60]) < 1e-10
    assert right[99, 0] > right[69, 0] + .1


def test_release_guard_uses_measured_pose_and_unblended_target_then_allows_opening(monkeypatch):
    import manimux.kinematics
    from manimux.config import GripperReleaseGuardConfig
    class Kin:
        num_arm_joints = 1
        state_dim = 2
        def fk(self, joints, gripper):
            p = np.eye(4); p[0,3] = joints[0]; return p
    monkeypatch.setattr(manimux.kinematics, 'build_kinematics', lambda *a, **kw: Kin())
    cfg = SmoothConfig(tracking_mode='braking', max_velocity=.6, max_acceleration=1.5,
        gripper=GripperHysteresisConfig(mode='continuous', group_indices={'right_arm':1},
                                      max_velocity=1, max_acceleration=12),
        release_guard=GripperReleaseGuardConfig(position_tolerance_m=.02))
    e = SmoothExecutor(cfg, .01)
    st = RobotState({'right_arm':np.array([0., 0.])}, 0, 0)
    ref = ActionHorizon(0, 10_000_000, 'a', {'right_arm':np.array([[0.,1.],[0.,1.]])},
                        tracking_groups={'right_arm':np.array([.1,1.])})
    cmd = e.step(0,st,ref)
    assert cmd.groups['right_arm'][1] == 0
    assert e.gripper_diagnostics['right_arm']['release_blocked']
    # A reached commanded pose alone cannot authorize release: feedback must catch up.
    e._previous['right_arm'][0] = .1
    assert e.step(10_000_000,st,ref).groups['right_arm'][1] == 0
    st.groups['right_arm'][0] = .09
    assert e.step(20_000_000,st,ref).groups['right_arm'][1] > 0
    assert not e.gripper_diagnostics['right_arm']['release_blocked']
    # Closing still follows the original continuous rule even while far from target.
    e.reset(RobotState({'right_arm':np.array([0.,1.])},0,0))
    ref.groups['right_arm'][:,1] = 0
    assert e.step(30_000_000,st,ref).groups['right_arm'][1] < 1
    # In the non-opening branch NumPy comparisons must not leak numpy.bool
    # into recorder.event(), which uses the standard JSON encoder.
    import json
    assert json.loads(json.dumps(e.gripper_diagnostics))['right_arm']['release_blocked'] is False


@pytest.mark.parametrize('use_gap', [False, True])
def test_latched_release_survives_new_closed_plans_and_waits_for_fresh_observation(monkeypatch, use_gap):
    import json
    import manimux.kinematics
    from manimux.config import GripperReleaseGuardConfig
    class Kin:
        num_arm_joints = 1
        state_dim = 2
        def fk(self, joints, gripper):
            pose = np.eye(4); pose[0,3] = joints[0]; return pose
    monkeypatch.setattr(manimux.kinematics, 'build_kinematics', lambda *a, **kw: Kin())
    cfg = SmoothConfig(tracking_mode='braking', max_velocity=.6, max_acceleration=1.5,
        gripper=GripperHysteresisConfig(mode='continuous', group_indices={'right_arm':1,'left_arm':1},
                                      max_velocity=1, max_acceleration=12),
        release_guard=GripperReleaseGuardConfig(mode='latched_release', position_tolerance_m=.02))
    ex = SmoothExecutor(cfg,.01)
    measured = {'right_arm':np.array([0.,0.]),'left_arm':np.array([0.,1.])}
    commands = []
    completed = None
    for tick in range(300):
        now = tick*10_000_000
        state = RobotState({n:a.copy() for n,a in measured.items()},now,tick)
        # Only the first chunk asks for release. Subsequent stale chunks ask
        # to move elsewhere and close, and can even fail IK for their new goal.
        goal = np.array([.12,1.]) if tick == 0 else np.array([-.4,0.])
        tracking = {'right_arm':goal,'left_arm':np.array([.3,1.])}
        ref = ActionHorizon(now,10_000_000,str(tick//10),
            {n:np.tile(a,(2,1)) for n,a in tracking.items()},
            tracking_groups=tracking, observation_time_ns=0,
            hold_groups=('right_arm',) if tick > 0 else ())
        if use_gap and tick > 0:
            command = ex.brake_hold(now,state).groups
        else:
            command = ex.step(now,state,ref).groups
        assert ex.has_pending_release
        event = ex._gripper_events['right_arm']
        np.testing.assert_allclose(event.target,[.12,1.])
        if event.phase == 'approach':
            assert command['right_arm'][1] == 0
        if commands:
            assert command['right_arm'][1] >= commands[-1][1]-1e-12
        commands.append(command['right_arm'].copy())
        for n in measured:
            measured[n] += .2*(command[n]-measured[n])
        json.dumps(ex.gripper_diagnostics)
        if event.phase == 'await_observation':
            completed = event.completed_ns
    assert completed is not None
    assert measured['right_arm'][1] > .95
    assert measured['right_arm'][0] == pytest.approx(.12,abs=.001)
    if not use_gap:
        assert measured['left_arm'][0] > .25
    velocities = np.diff(np.r_[0.,np.array(commands)[:,0]])/.01
    assert abs(velocities).max() <= .6+1e-9
    assert abs(np.diff(np.r_[0.,velocities])/.01).max() <= 1.5+1e-8
    # An inference begun before release completed is still stale even if its
    # plan ID is new. Only a post-release observation resumes model control.
    state = RobotState(measured,3_000_000_000,300)
    ref.observation_time_ns = completed
    ex.step(3_000_000_000,state,ref)
    assert ex.has_pending_release
    ref.observation_time_ns = completed+1
    ref.hold_groups = ()
    out = ex.step(3_010_000_000,state,ref)
    assert not ex.has_pending_release
    assert out.groups['right_arm'][1] < 1
    # Explicit Pause/Home/reset cancels rather than autonomously finishing.
    ex.reset(RobotState({'right_arm':np.array([0.,0.]),'left_arm':np.array([0.,1.])},0,0))
    tracking['right_arm'] = np.array([.2,1.])
    ref.tracking_groups = tracking
    ex.step(0,ex_state := RobotState({'right_arm':np.array([0.,0.]),'left_arm':np.array([0.,1.])},0,0),ref)
    assert ex.has_pending_release
    ex.reset(ex_state)
    assert not ex.has_pending_release


@pytest.mark.parametrize('use_gap', [False, True])
@pytest.mark.parametrize('contact_aperture', [0.0, 0.3])
def test_grasp_waits_for_pose_and_closure_before_lift(monkeypatch, use_gap, contact_aperture):
    import json
    import manimux.kinematics
    from manimux.config import GripperGraspGuardConfig, GripperReleaseGuardConfig

    class Kin:
        num_arm_joints = 2
        state_dim = 3

        def fk(self, joints, gripper):
            pose = np.eye(4)
            pose[0, 3] = joints[0]
            c, s = np.cos(joints[1]), np.sin(joints[1])
            pose[:2, :2] = [[c, -s], [s, c]]
            return pose

    monkeypatch.setattr(manimux.kinematics, 'build_kinematics', lambda *a, **kw: Kin())
    cfg = SmoothConfig(
        tracking_mode='braking', max_velocity=.6, max_acceleration=1.5,
        gripper=GripperHysteresisConfig(
            mode='continuous', group_indices={'right_arm': 2, 'left_arm': 2},
            max_velocity=1, max_acceleration=12),
        release_guard=GripperReleaseGuardConfig(), grasp_guard=GripperGraspGuardConfig())
    ex = SmoothExecutor(cfg, .01)
    measured = {'right_arm': np.array([0., 0., 1.]), 'left_arm': np.array([0., 0., 1.])}
    commands, phases = [], set()
    completion = None
    for tick in range(400):
        now = tick * 10_000_000
        state = RobotState({n: a.copy() for n, a in measured.items()}, now, tick)
        # First close-onset pose must survive later lifting/opening predictions.
        right = np.array([.12, .25, .8]) if tick == 0 else np.array([.7, .6, 1.])
        goals = {'right_arm': right, 'left_arm': np.array([.3, 0., 1.])}
        ref = ActionHorizon(now, 10_000_000, str(tick // 10),
                            {n: np.tile(a, (2, 1)) for n, a in goals.items()},
                            tracking_groups=goals, observation_time_ns=0,
                            hold_groups=('right_arm',) if tick > 0 else ())
        command = (ex.brake_hold(now, state) if use_gap and tick > 0
                   else ex.step(now, state, ref))
        event = ex._gripper_events['right_arm']
        assert event.kind == 'grasp' and not ex.has_pending_release
        np.testing.assert_allclose(event.target, [.12, .25, 0.])
        phases.add(event.phase)
        if event.phase == 'approach':
            assert command.groups['right_arm'][2] == 1.
        elif event.phase == 'closing':
            assert abs(measured['right_arm'][0] - .12) <= .02
            assert abs(measured['right_arm'][1] - .25) <= cfg.grasp_guard.rotation_tolerance_rad
        elif event.phase == 'await_observation':
            completion = event.completed_ns
            assert command.groups['right_arm'][2] <= .02
            assert measured['right_arm'][2] == pytest.approx(contact_aperture, abs=.02)
        commands.append(command.groups['right_arm'].copy())
        for n in measured:
            measured[n] += .2 * (command.groups[n] - measured[n])
        measured['right_arm'][2] = max(contact_aperture, measured['right_arm'][2])
        json.dumps(ex.gripper_diagnostics)
    assert {'approach', 'closing', 'await_observation'} <= phases
    assert completion is not None
    values = np.array(commands)
    assert values[:, 0].max() <= .12 + 1e-9  # No lifting while closing or awaiting observation.
    velocity = np.diff(np.vstack(([0., 0.], values[:, :2])), axis=0) / .01
    assert abs(velocity).max() <= .6 + 1e-9
    assert abs(np.diff(np.vstack(([0., 0.], velocity)), axis=0) / .01).max() <= 1.5 + 1e-8
    assert np.max(np.diff(values[:, 2])) <= 1e-12
    if not use_gap:
        assert measured['left_arm'][0] > .25
    ref.observation_time_ns = completion
    state = RobotState(measured, 4_000_000_000, 400)
    ex.step(state.monotonic_ns, state, ref)
    assert ex.has_pending_gripper_event
    ref.observation_time_ns = completion + 1
    ref.hold_groups = ()
    goals['right_arm'] = np.array([.7, .6, 0.])
    ref.groups['right_arm'][:] = goals['right_arm']
    ref.tracking_groups = goals
    command = ex.step(state.monotonic_ns + 10_000_000, state, ref)
    assert not ex.has_pending_gripper_event
    assert command.groups['right_arm'][0] > .12
    ex.reset(RobotState({'right_arm': np.array([0., 0., 1.]),
                        'left_arm': np.array([0., 0., 1.])}, 0, 0))
    ex.step(0, RobotState(measured, 0, 0), ref)
    assert ex.has_pending_gripper_event
    ex.reset(state)  # Explicit Pause/Home/reset must cancel a grasp too.
    assert not ex.has_pending_gripper_event


@pytest.mark.parametrize(('event_kind', 'blocked_phase'), [
    ('grasp', 'approach'), ('grasp', 'closing'),
    ('release', 'approach'), ('release', 'opening'),
])
def test_gripper_timeout_waits_for_fresh_plan_then_bypasses_only_failed_arm(
    monkeypatch, event_kind, blocked_phase,
):
    import json
    import manimux.kinematics
    from manimux.config import GripperGraspGuardConfig, GripperReleaseGuardConfig

    class Kin:
        num_arm_joints = 1
        state_dim = 2

        def fk(self, joints, gripper):
            pose = np.eye(4)
            pose[0, 3] = joints[0]
            return pose

    monkeypatch.setattr(manimux.kinematics, 'build_kinematics', lambda *a, **kw: Kin())
    config = SmoothConfig(
        tracking_mode='braking', max_velocity=.6, max_acceleration=1.5,
        gripper=GripperHysteresisConfig(
            mode='continuous', group_indices={'right_arm': 1, 'left_arm': 1},
            max_velocity=1, max_acceleration=12),
        release_guard=GripperReleaseGuardConfig(),
        grasp_guard=GripperGraspGuardConfig(phase_timeout_s=2))
    ex = SmoothExecutor(config, .01)
    failed = None
    commands = []
    initial_aperture = 1. if event_kind == 'grasp' else 0.
    requested_aperture = .8 if event_kind == 'grasp' else 1.
    state = RobotState({n: np.array([0., initial_aperture])
                        for n in ('right_arm', 'left_arm')}, 0, 0)
    ex.reset(state)
    bypassed = ex._grasp_bypassed if event_kind == 'grasp' else ex._release_bypassed
    other_bypassed = ex._release_bypassed if event_kind == 'grasp' else ex._grasp_bypassed
    # Persistent 35 mm pose error, or a gripper that never finishes moving physically.
    goal = .035 if blocked_phase == 'approach' else 0.
    for tick in range(260):
        now = tick * 10_000_000
        goals = {'right_arm': np.array([goal, requested_aperture]),
                 'left_arm': np.array([.3, initial_aperture])}
        ref = ActionHorizon(now, 10_000_000, str(tick),
                            {n: np.tile(a, (2, 1)) for n, a in goals.items()},
                            tracking_groups=goals, observation_time_ns=0)
        command = ex.step(now, state, ref)
        commands.append(command.groups['right_arm'].copy())
        diag = ex.gripper_diagnostics['right_arm']
        json.dumps(ex.gripper_diagnostics)
        if diag[event_kind + '_phase'] == 'await_replan':
            assert diag['failure_reason'] == blocked_phase + '_timeout'
            assert diag['completed_ns'] is None
            failed = diag['failed_ns']
    assert failed == 2_000_000_000
    frozen = command.groups['right_arm'][1]
    assert bypassed == {'right_arm'} and not other_bypassed
    assert command.groups['left_arm'][0] > .25
    # Gaps, stale observations, and invalid fresh arm partitions cannot resume motion.
    for tick in range(260, 265):
        command = ex.brake_hold(tick * 10_000_000, state)
        assert command.groups['right_arm'][1] == frozen
        assert ex.gripper_diagnostics['right_arm'][event_kind + '_phase'] == 'await_replan'
    goals['right_arm'] = np.array([.2, 1. - initial_aperture])
    ref.groups['right_arm'][:] = goals['right_arm']
    ref.tracking_groups = goals
    ref.observation_time_ns = failed
    ex.step(2_700_000_000, state, ref)
    assert ex.has_pending_gripper_event
    ref.observation_time_ns = failed + 1
    ref.hold_groups = ('right_arm',)
    ex.step(2_710_000_000, state, ref)
    assert ex.has_pending_gripper_event
    ref.hold_groups = ()
    for tick in range(272, 310):
        command = ex.step(tick * 10_000_000, state, ref)
        commands.append(command.groups['right_arm'].copy())
        assert not ex.has_pending_gripper_event
        assert ex.gripper_diagnostics['right_arm'][event_kind + '_guard_bypassed'] is True
    assert command.groups['right_arm'][0] > goal + .02
    # Other arm can still use the same event synchronization.
    goals['left_arm'] = np.array([.4, requested_aperture])
    ref.groups['left_arm'][:] = goals['left_arm']
    ref.tracking_groups = goals
    ex.step(3_100_000_000, state, ref)
    assert ex._gripper_events['left_arm'].kind == event_kind
    assert bypassed == {'right_arm'} and not other_bypassed
    ex.reset(state)
    assert not ex.has_pending_gripper_event
    assert not ex._grasp_bypassed and not ex._release_bypassed


@pytest.mark.parametrize('approach_limit', [None, .25])
def test_open_gripper_approach_speed_is_per_arm_and_preserves_acceleration(monkeypatch, approach_limit):
    import json
    import manimux.kinematics
    from manimux.config import GripperGraspGuardConfig, GripperReleaseGuardConfig

    class Kin:
        num_arm_joints = 1
        state_dim = 2

        def fk(self, joints, gripper):
            pose = np.eye(4)
            pose[0, 3] = joints[0]
            return pose

    monkeypatch.setattr(manimux.kinematics, 'build_kinematics', lambda *a, **kw: Kin())
    ex = SmoothExecutor(SmoothConfig(
        tracking_mode='braking', max_velocity=.6, max_acceleration=1.5,
        gripper=GripperHysteresisConfig(mode='continuous',
            group_indices={'left_arm': 1, 'right_arm': 1}, max_velocity=1, max_acceleration=12),
        release_guard=GripperReleaseGuardConfig(),
        grasp_guard=GripperGraspGuardConfig(approach_max_velocity=approach_limit)), .01)
    measured = {'left_arm': np.array([0., 1.]), 'right_arm': np.array([0., 0.])}
    history = [np.array([0., 0.])]
    for tick in range(100):
        goals = {'left_arm': np.array([1., 1.]), 'right_arm': np.array([1., 0.])}
        ref = ActionHorizon(tick * 10_000_000, 10_000_000, str(tick),
                            {n: np.tile(q, (2, 1)) for n, q in goals.items()},
                            tracking_groups=goals, observation_time_ns=tick * 10_000_000)
        cmd = ex.step(tick * 10_000_000, RobotState(measured, tick * 10_000_000, tick), ref)
        measured = {n: q.copy() for n, q in cmd.groups.items()}
        history.append(np.array([cmd.groups['left_arm'][0], cmd.groups['right_arm'][0]]))
        json.dumps(ex.gripper_diagnostics)
    v = np.diff(history, axis=0) / .01
    assert max(abs(v[:, 0])) == pytest.approx(approach_limit or .6)
    assert max(abs(v[:, 1])) == pytest.approx(.6)
    # Opening during placement has its own pose guard and the normal arm limit.
    goals['right_arm'] = np.array([1., 1.])
    ref.groups['right_arm'][:] = goals['right_arm']
    ref.tracking_groups = goals
    cmd = ex.step(1_000_000_000, RobotState(measured, 1_000_000_000, 100), ref)
    if approach_limit is not None:
        assert ex.gripper_diagnostics['right_arm']['arm_velocity_limit_rad_s'] == .6
    assert ex._gripper_events['right_arm'].kind == 'release'
    # Resuming open-gripper ordinary tracking from a faster command must decelerate,
    # not instantly clip velocity to the new limit.
    ex._gripper_events.clear()
    ex._release_bypassed.add('right_arm')
    ex._previous['right_arm'][1] = 1.
    previous_v = ex._previous_velocity['right_arm'][0]
    cmd = ex.step(1_010_000_000, RobotState(measured, 1_010_000_000, 101), ref)
    assert abs(ex._previous_velocity['right_arm'][0] - previous_v) <= .015 + 1e-12
    assert np.max(abs(np.diff(v, axis=0) / .01)) <= 1.5 + 1e-9
