"""Real FK/Jacobian/QP and failure gates; no SDK, model or robot connection."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from manimux.config import load_config
from manimux.integrations.umi_dp_tianji.history import HistoryStrategy
from manimux.integrations.umi_dp_tianji.ik_config import bind_diff_ik_profile
from manimux.kinematics.tianji import TianjiKinematics, j67_ok
from manimux.kinematics.tianji_diff import DifferentialIKConfig, TianjiDifferentialIK

pytest.importorskip("osqp")
ROOT = Path(__file__).resolve().parents[2]
START = np.radians([50, -40, -30, -100, -65, 0, 40])


@pytest.fixture
def solver():
    kin = TianjiKinematics(end_effector="umi_follower")
    config = load_config(ROOT / "configs/umi_dp/tianji/infra/pass_ball/default.yaml")
    config.policy.options["ik_backend"] = "diff"
    bind_diff_ik_profile(config)
    return TianjiDifferentialIK(kin, DifferentialIKConfig(**config.policy.options["diff_ik"]))


def test_flange_jacobian_matches_finite_difference():
    kin = TianjiKinematics(end_effector="umi_follower")
    rng = np.random.default_rng(312)
    for _ in range(12):
        joints = START + rng.uniform(-0.4, 0.4, 7)
        pose, jacobian = kin.flange_and_jacobian(joints)
        np.testing.assert_allclose(pose, kin.flange(joints), atol=1e-14)
        for index in range(7):
            plus, minus = joints.copy(), joints.copy()
            plus[index] += 1e-6
            minus[index] -= 1e-6
            a, b = kin.flange(plus), kin.flange(minus)
            translation = (a[:3, 3] - b[:3, 3]) / 2e-6
            angular = Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).as_rotvec() / 2e-6
            np.testing.assert_allclose(jacobian[:, index], np.r_[translation, angular], atol=3e-9)


@pytest.mark.parametrize("dt", [0.0005, 0.004, 0.016, 0.2])
def test_rate_bound_tracking_lag_and_dt_cap(solver, dt):
    target = solver.kinematics.fk(START, 0.8)
    target[:3, 3] += [0.2, -0.1, 0.15]
    result = solver.solve(target, START, dt)
    assert not result.ok and result.reason == "tracking_lag"
    assert result.pos_err_mm > solver.config.max_lag_mm
    delta = np.max(np.abs(result.joints - START))
    assert 0 < delta <= solver.config.max_velocity_rad_s * min(dt, solver.config.dt_max_s) + 1e-14
    assert np.max(np.abs(result.qdot_rad_s)) <= solver.config.max_velocity_rad_s + 1e-14


def test_flange_tool_conversion_and_reset_drop_dual_state(solver):
    target = solver.kinematics.fk(START + np.radians([0.08] * 7), 0.8)
    first = solver.solve(target, START, 0.004)
    assert first.ok and first.pos_err_mm < 0.05
    old_problem = solver._problem
    solver.reset()
    assert solver._problem is None
    assert not solver._previous_velocity.any()
    second = solver.solve(target, START, 0.004)
    assert solver._problem is not old_problem
    np.testing.assert_array_equal(first.joints, second.joints)


def test_empty_box_does_not_reuse_previous_osqp_problem(solver):
    assert solver.solve(solver.kinematics.fk(START, 0.8), START, 0.004).ok
    problem = solver._problem
    bad = START.copy()
    bad[0] = np.radians(170)
    result = solver.solve(solver.kinematics.fk(bad, 0.8), bad, 0.004)
    assert not result.ok and result.reason == "qp_infeasible"
    assert result.detail["note"] == "empty qdot box"
    assert solver._problem is problem


@pytest.mark.parametrize("j6,j7", [(54, 54), (-54, 54), (-54, -54), (54, -54)])
def test_j67_constraint_conflict_is_rejected_in_every_quadrant(solver, j6, j7):
    joints = START.copy()
    joints[5:] = np.radians([j6, j7])
    assert j67_ok(np.degrees(joints))  # Raw boundary valid but 5-degree buffer infeasible.
    result = solver.solve(solver.kinematics.fk(joints, 0.8), joints, 0.004)
    assert not result.ok and result.reason == "qp_infeasible"
    assert "osqp_status" in result.detail


def test_optional_orientation_lag_guard(solver):
    solver.config.max_lag_deg = 0.01
    target = solver.kinematics.fk(START, 0.8)
    target[:3, :3] = Rotation.from_euler("z", 5, degrees=True).as_matrix() @ target[:3, :3]
    result = solver.solve(target, START, 0.004)
    assert not result.ok and result.reason == "tracking_lag"
    assert result.rot_err_deg > solver.config.max_lag_deg


def test_report_lag_policy_keeps_the_bounded_step(solver):
    target = solver.kinematics.fk(START, 0.8)
    target[:3, 3] += [0.2, -0.1, 0.15]
    aborted = solver.solve(target, START, 0.004)
    solver.reset()
    solver.config.lag_policy = "report"
    reported = solver.solve(target, START, 0.004)
    assert not aborted.ok and aborted.reason == "tracking_lag" and aborted.lag_exceeded
    assert reported.ok and reported.reason == "ok" and reported.lag_exceeded
    np.testing.assert_array_equal(reported.joints, aborted.joints)


def test_report_lag_policy_still_rejects_backend_failures(solver):
    solver.config.lag_policy = "report"
    joints = START.copy()
    joints[5:] = np.radians([54, 54])
    result = solver.solve(solver.kinematics.fk(joints, 0.8), joints, 0.004)
    assert not result.ok and result.reason == "qp_infeasible"


@pytest.mark.parametrize("kind", ["joints", "target", "dt", "rotation"])
def test_invalid_input_never_reaches_osqp(solver, kind):
    joints = START.copy()
    target = solver.kinematics.fk(joints, 0.8)
    dt = 0.004
    if kind == "joints":
        joints[0] = np.nan
    elif kind == "target":
        target[0, 3] = np.inf
    elif kind == "rotation":
        target[0, 0] += 0.2
    else:
        dt = 0
    assert solver.solve(target, joints, dt).reason == "invalid_input"
    assert solver._problem is None


def test_nonfinite_qp_solution_rejected(solver, monkeypatch):
    target = solver.kinematics.fk(START, 0.8)
    assert solver.solve(target, START, 0.004).ok
    monkeypatch.setattr(
        solver._problem,
        "solve",
        lambda **kwargs: SimpleNamespace(info=SimpleNamespace(status_val=1), x=np.full(7, np.nan)),
    )
    assert solver.solve(target, START, 0.004).reason == "fk_mismatch"


def test_j67_postcheck_catches_solver_tolerance_or_bad_qp_solution(solver):
    joints = START.copy()
    joints[5:] = np.radians([54, 55.14])
    assert j67_ok(np.degrees(joints))
    solver._problem = SimpleNamespace(
        update=lambda **kwargs: None,
        warm_start=lambda **kwargs: None,
        solve=lambda **kwargs: SimpleNamespace(
            info=SimpleNamespace(status_val=1), x=np.full(7, solver.vmax)
        ),
    )
    result = solver.solve(solver.kinematics.fk(joints, 0.8), joints, 0.004)
    assert not result.ok and result.reason == "j67_interference"


def test_nullspace_objective_pushes_joint_toward_interior(solver):
    joints = np.degrees(START)
    joints[0] = 160
    gradient = solver._nullspace_gradient(joints)
    assert gradient[0] > 0  # -gradient decreases the positive near-limit angle.
    assert not gradient[1:].any()
    assert not solver._nullspace_gradient(np.degrees(START)).any()


def test_diff_rate_profile_binding_and_runtime_stale_rejection():
    config = load_config(ROOT / "configs/umi_dp/tianji/infra/pass_ball/default.yaml")
    config.policy.options["ik_backend"] = "diff"
    with pytest.raises(ValueError, match="max_velocity_rad_s"):
        HistoryStrategy(config)
    config.execution.motion_limits.arm.max_velocity = 0.37
    config.execution.motion_limits.arm.max_step_dt_s = 0.012
    bind_diff_ik_profile(config)
    assert config.policy.options["diff_ik"]["max_velocity_rad_s"] == 0.37
    HistoryStrategy(config)
    config.execution.motion_limits.arm.max_velocity = 0.31
    with pytest.raises(ValueError, match="conflicts with the shared motion profile"):
        HistoryStrategy(config)


def test_cannot_relax_embodiment_margin_or_disable_umi_interference(solver):
    config = solver.config.model_copy(update={"limit_margin_deg": 3.0})
    with pytest.raises(ValueError, match="cannot reduce"):
        TianjiDifferentialIK(solver.kinematics, config)
    from manimux.integrations.umi_dp_tianji.policy_plugin import UmiDpTianjiAdapter

    runtime = load_config(ROOT / "configs/umi_dp/tianji/infra/pass_ball/default.yaml")
    runtime.robot.driver = "mock"
    runtime.policy.options["ik_backend"] = "diff"
    bind_diff_ik_profile(runtime)
    runtime.policy.options["diff_ik"]["check_j67"] = False
    with pytest.raises(ValueError, match="requires the J6/J7 constraint"):
        UmiDpTianjiAdapter(runtime.robot, runtime.policy)


@pytest.mark.parametrize("horizon,offset", [(16, 1 / 30), (64, 0.1)])
def test_real_diff_adapter_chunk_timing_and_atomic_rejection(horizon, offset):
    from manimux.integrations.umi_dp_tianji.history import WindowSnapshot
    from manimux.integrations.umi_dp_tianji.policy_plugin import UmiDpTianjiAdapter, matrix_pose
    from manimux.types import (
        ActionContext,
        InferenceRequest,
        ObservationSnapshot,
        RobotState,
        SensorFrame,
    )

    config = load_config(ROOT / "configs/umi_dp/tianji/infra/pass_ball/default.yaml")
    config.robot.driver = "mock"
    config.policy.horizon_steps = horizon
    config.policy.options.update(ik_backend="diff", first_action_offset_s=offset)
    bind_diff_ik_profile(config)
    adapter = UmiDpTianjiAdapter(config.robot, config.policy)
    state = RobotState({side + "_arm": np.r_[START, 0.8] for side in ("left", "right")}, 10**9, 1)
    previous = RobotState({key: value.copy() for key, value in state.groups.items()}, 900000000, 0)
    frames = {
        name: SensorFrame(name, np.zeros((8, 8, 3), np.uint8), 10**9, 1)
        for name in ("left_wrist", "right_wrist", "left_wrist_prev", "right_wrist_prev")
    }
    window = WindowSnapshot(state, frames, ObservationSnapshot(previous, frames))
    actions = []
    for index in range(horizon):
        target_joints = START.copy()
        target_joints[0] += np.radians(0.04 * (index + 1))
        action = {}
        for side in ("left", "right"):
            action[side + "_ee_pose"] = matrix_pose(adapter.kin[side].fk(target_joints, 0.8))
            action[side + "_ee_joint_state"] = np.array([0.8])
        actions.append(action)
    adapter.prepare_request(InferenceRequest("test", 1, 10**9, 2 * 10**9, window))
    chunk = adapter.decode_action({"actions": actions}, ActionContext(1, 10**9, 10**9))
    assert chunk.horizon_steps == horizon
    assert chunk.observation_time_ns == 10**9 + round(offset * 1e9)
    assert chunk.metadata["ik_backend"] == "diff"
    for side in ("left", "right"):
        assert adapter.kin[side]._sdk is None
        delta = np.diff(np.vstack([START, chunk.groups[side + "_arm"][:, :7]]), axis=0)
        dt = config.policy.effective_action_dt_s
        assert np.max(np.abs(delta)) <= config.execution.motion_limits.arm.max_velocity * dt
    # Failure in the right arm's last row must not return the completed left arm.
    adapter.prepare_request(InferenceRequest("test", 2, 10**9, 2 * 10**9, window))
    actions[-1]["right_ee_pose"][0] += 0.3
    with pytest.raises(ValueError, match="tracking_lag; rejecting the entire chunk"):
        adapter.decode_action({"actions": actions}, ActionContext(2, 10**9, 10**9))
    assert 2 not in adapter.anchors
    # The report policy keeps that bounded, lagging chunk and records the lag.
    config.policy.options["diff_ik"]["lag_policy"] = "report"
    reporting = UmiDpTianjiAdapter(config.robot, config.policy)
    reporting.prepare_request(InferenceRequest("test", 3, 10**9, 2 * 10**9, window))
    chunk = reporting.decode_action({"actions": actions}, ActionContext(3, 10**9, 10**9))
    lag = chunk.metadata["diff_ik_lag"]
    assert lag["right"]["lag_exceedances"] > 0
    assert lag["right"]["worst_lag_mm"] > reporting.diff_solvers["right"].config.max_lag_mm
    assert lag["left"]["lag_exceedances"] == 0
