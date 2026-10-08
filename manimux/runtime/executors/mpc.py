from __future__ import annotations

import numpy as np

from manimux.runtime.executors.base import ExecutorError
from manimux.runtime.executors.limits import ScalarLimits, limit_step
from manimux.types import (
    ActionHorizon,
    GroupVector,
    RobotCommand,
    RobotState,
    copy_group_vector,
)


class MPCExecutor:
    """Small linear receding-horizon controller for the V1 simulation path."""

    def __init__(self, config: dict, control_dt_s: float) -> None:
        self._config = config
        self._dt_s = control_dt_s
        self._limits = ScalarLimits(
            max_velocity=config["max_velocity"],
            max_acceleration=config["max_acceleration"],
            position_limit_abs=config["position_limit_abs"],
        )
        self._previous: GroupVector | None = None
        self._previous_velocity: GroupVector | None = None

    @property
    def horizon_steps(self) -> int:
        return self._config["horizon_control_steps"]

    def reset(self, state: RobotState) -> None:
        self._previous = copy_group_vector(state.groups)
        self._previous_velocity = {
            name: np.zeros_like(value) for name, value in state.groups.items()
        }

    def _solve_group(
        self,
        current: np.ndarray,
        previous_command: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        horizon = min(self._config["horizon_control_steps"], reference.shape[0])
        reference = reference[:horizon]
        a = self._config["dynamics_a"]
        b_matrix = np.zeros((horizon, horizon), dtype=np.float64)
        for row in range(horizon):
            for column in range(row + 1):
                b_matrix[row, column] = (1.0 - a) * a ** (row - column)
        state_offset = np.stack([a ** (step + 1) * current for step in range(horizon)])
        difference = np.eye(horizon, dtype=np.float64)
        difference[1:, :-1] -= np.eye(horizon - 1, dtype=np.float64)
        boundary = np.zeros((horizon, current.size), dtype=np.float64)
        boundary[0] = previous_command

        tracking = self._config["tracking_weight"]
        delta = self._config["command_delta_weight"]
        system = (
            tracking * (b_matrix.T @ b_matrix)
            + delta * (difference.T @ difference)
            + 1e-6 * np.eye(horizon)
        )
        rhs = tracking * b_matrix.T @ (reference - state_offset) + delta * difference.T @ boundary
        try:
            return np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError as exc:
            raise ExecutorError("linear MPC solve failed") from exc

    def step(
        self,
        now_ns: int,
        state: RobotState,
        reference: ActionHorizon,
    ) -> RobotCommand:
        if self._previous is None or self._previous_velocity is None:
            self.reset(state)
        target: GroupVector = {}
        for name, values in reference.groups.items():
            solution = self._solve_group(state.groups[name], self._previous[name], values)
            target[name] = solution[0]
        output, velocities = limit_step(
            target,
            self._previous,
            self._previous_velocity,
            dt_s=self._dt_s,
            limits=self._limits,
        )
        self._previous = copy_group_vector(output)
        self._previous_velocity = copy_group_vector(velocities)
        return RobotCommand(groups=output, monotonic_ns=now_ns, plan_id=reference.plan_id)


def mpc_parameters(**options) -> dict:
    """保留 MPC 时域、动力学与代价权重的默认值。"""

    if "horizon_steps" in options:
        raise ValueError("unsupported MPC field: horizon_steps")
    values = {
        "max_velocity": 2.0,
        "max_acceleration": 8.0,
        "position_limit_abs": 3.14,
        "horizon_control_steps": 15,
        "dynamics_a": 0.85,
        "tracking_weight": 10.0,
        "command_delta_weight": 1.0,
        **options,
    }
    return values
