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
