"""Tianji velocity QP, ported from tianji-control's algos/diff_ik.py.

Public poses/joints use metres/radians. Internally mm/degrees preserve the
reference QP weights, regularization and nullspace objective. This is a rate
controller with tracking lag, not an analytic inverse with convergence bounds.
See docs/tianji-diff-ik.md for source revision, guard semantics and validation.
No model, robot connection or closed-source kinematics library is loaded.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy import sparse
from scipy.spatial.transform import Rotation

from manimux.kinematics.tianji import BD67_REAL, TianjiKinematics, j67_ok


class DifferentialIKConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    # Bound from the effective shared executor profile; never a second rate constant.
    max_velocity_rad_s: float = Field(gt=0)
    dt_max_s: float = Field(gt=0)
    w_pos: float = Field(default=1.0, gt=0)
    w_rot: float = Field(default=1.0, gt=0)
    lam: float = Field(default=0.001, ge=0)
    limit_margin_deg: float | None = Field(default=None, gt=0)
    check_j67: bool = True
    j67_margin_deg: float | None = Field(default=None, gt=0)
    mu_nullspace: float = Field(default=1000.0, ge=0)
    nullspace_activation_deg: float | None = Field(default=25.0, gt=0)
    nullspace_weights: list[float] | None = None
    max_lag_mm: float = Field(default=5.0, gt=0)
    max_lag_deg: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_weights(self):
        if self.nullspace_weights is not None and (
            len(self.nullspace_weights) != 7
            or any(not math.isfinite(value) or value < 0 for value in self.nullspace_weights)
        ):
            raise ValueError("nullspace_weights must contain seven finite nonnegative values")
        return self


@dataclass(slots=True)
class DifferentialIKResult:
    ok: bool
    reason: str
    joints: np.ndarray | None = None  # radians
    qdot_rad_s: np.ndarray | None = None
    pos_err_mm: float | None = None  # flange tracking lag, as in CalibWrist
    rot_err_deg: float | None = None  # max wrapped xyz Euler-coordinate difference
    max_step_deg: float | None = None
    min_margin_deg: float | None = None
    solve_time_ms: float | None = None
    detail: dict = field(default_factory=dict)


def _upper_csc(matrix):
    rows, cols = np.triu_indices(7)
    return sparse.csc_matrix((matrix[rows, cols], (rows, cols)), shape=(7, 7))


def _constraint_matrix(c6, c7):
    return sparse.csc_matrix(
        ([1.0] * 7 + [c6, c7], (list(range(7)) + [7, 7], list(range(7)) + [5, 6])),
        shape=(8, 7),
    )


def _interference_row(joints, margin, dt):
    j6, j7 = joints[5:7]
    quadrant = 0 if j6 >= 0 and j7 >= 0 else 1 if j6 < 0 <= j7 else 2 if j6 < 0 else 3
    a0, a1, a2 = BD67_REAL[quadrant]
    boundary = a0 * j6 * j6 + a1 * j6 + a2
    slope = 2.0 * a0 * j6 + a1
    if j7 >= 0:
        gap, c6, c7 = j7 - boundary, -slope, 1.0
    else:
        gap, c6, c7 = boundary - j7, slope, -1.0
    return c6, c7, (-margin - gap) / dt


class TianjiDifferentialIK:
    """One solver per arm. Calls are sequential; ``reset`` drops all QP state.

    An entire speculative chunk starts with reset, so OSQP primal/dual state
    from a rejected chunk or previous engagement cannot affect the next one.
    Within a chunk the reference warm-start/update sequence is preserved.
    """

    def __init__(self, kinematics: TianjiKinematics, config: DifferentialIKConfig):
        try:
            import osqp
        except ImportError as exc:
            raise ImportError("Differential IK needs pip install -e '.[tianji-diff-ik]'") from exc
        self._osqp = osqp
        self.kinematics = kinematics
        self.config = config
        lower, upper = kinematics.joint_position_limits()
        self.lower, self.upper = np.degrees(lower), np.degrees(upper)
        self.margin = (
            kinematics.limit_margin_deg
            if config.limit_margin_deg is None
            else config.limit_margin_deg
        )
        if self.margin < kinematics.limit_margin_deg:
            raise ValueError("Differential IK cannot reduce the embodiment joint-limit margin")
        if np.any(self.lower + self.margin >= self.upper - self.margin):
            raise ValueError("Differential IK margin leaves no joint range")
        self.j67_margin = self.margin if config.j67_margin_deg is None else config.j67_margin_deg
        self.vmax = math.degrees(config.max_velocity_rad_s)
        self.weights = [1.0] * 7 if config.nullspace_weights is None else config.nullspace_weights
        self.W = np.diag([config.w_pos] * 3 + [config.w_rot] * 3)
        self._p_rows = np.concatenate([np.arange(j + 1) for j in range(7)])
        self._p_cols = np.concatenate([np.full(j + 1, j) for j in range(7)])
        probe = np.arange(49, dtype=float).reshape(7, 7)
        if not np.array_equal(_upper_csc(probe).data, probe[self._p_rows, self._p_cols]):
            raise RuntimeError("Unexpected sparse upper-triangle layout")
        marker = _constraint_matrix(-2.0, -3.0)
        self._a_data = marker.data.copy()
        self._a_c6 = int(np.flatnonzero(marker.data == -2.0)[0])
        self._a_c7 = int(np.flatnonzero(marker.data == -3.0)[0])
        tool = kinematics.tool_transform
        self._tool_inverse = None if tool is None else np.linalg.inv(tool)
        self.reset()

    def reset(self):
        self._problem = None
        self._previous_velocity = np.zeros(7)

    def _cost(self, joints):
        band = self.config.nullspace_activation_deg
        if band is None:
            mid, span = (self.lower + self.upper) / 2, self.upper - self.lower
            return (
                sum(
                    weight * ((value - center) / width) ** 2
                    for value, center, width, weight in zip(
                        joints, mid, span, self.weights, strict=True
                    )
                )
                / 14.0
            )
        margins = np.minimum(self.upper - joints, joints - self.lower)
        return (
            sum(
                weight * ((band - margin) / band) ** 2
                for margin, weight in zip(margins, self.weights, strict=True)
                if margin < band
            )
            / 14.0
        )

    def _nullspace_gradient(self, joints):
        band = self.config.nullspace_activation_deg
        if (
            band is not None
            and np.min(np.minimum(self.upper - joints, joints - self.lower)) >= band
        ):
            return np.zeros(7)
        gradient = np.zeros(7)
        for index in range(7):
            plus, minus = joints.copy(), joints.copy()
            plus[index] += 1e-4
            minus[index] -= 1e-4
            gradient[index] = (self._cost(plus) - self._cost(minus)) / 2e-4
        return gradient

    def solve(self, target_tcp, previous_joints, dt_s) -> DifferentialIKResult:
        start = time.perf_counter()

        def failure(reason, **detail):
            return DifferentialIKResult(
                False, reason, solve_time_ms=(time.perf_counter() - start) * 1000, detail=detail
            )

        previous = np.asarray(previous_joints, dtype=float)
        target = np.asarray(target_tcp, dtype=float)
        if (
            previous.shape != (7,)
            or not np.isfinite(previous).all()
            or target.shape != (4, 4)
            or not np.isfinite(target).all()
            or not math.isfinite(dt_s)
            or dt_s <= 0
        ):
            return failure("invalid_input")
        if not np.allclose(target[3], [0, 0, 0, 1], atol=1e-8) or not (
            np.allclose(target[:3, :3].T @ target[:3, :3], np.eye(3), atol=1e-7)
            and np.isclose(np.linalg.det(target[:3, :3]), 1.0, atol=1e-7)
        ):
            return failure("invalid_input", note="target must be a rigid transform")
        if self._tool_inverse is not None:
            target = target @ self._tool_inverse
        dt = min(max(dt_s, 1e-6), self.config.dt_max_s)
        joints = np.degrees(previous)
        current, jacobian = self.kinematics.flange_and_jacobian(previous)
        jacobian = jacobian.copy()
        jacobian[:3] *= 1000 * math.pi / 180  # m/rad -> mm/degree
        rotation_error = Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec(
            degrees=True
        )
        desired = np.r_[(target[:3, 3] - current[:3, 3]) * 1000, rotation_error] / dt
        quadratic = 2 * (jacobian.T @ self.W @ jacobian + self.config.lam * np.eye(7))
        linear = -2 * (jacobian.T @ self.W @ desired)
        if self.config.mu_nullspace > 0:
            linear += self.config.mu_nullspace * dt * self._nullspace_gradient(joints)
        lower = np.maximum(-self.vmax, (self.lower + self.margin - joints) / dt)
        upper = np.minimum(self.vmax, (self.upper - self.margin - joints) / dt)
        bad = np.flatnonzero(lower > upper)
        if bad.size:
            # OSQP update(l>u) otherwise silently retains the previous problem.
            return failure("qp_infeasible", joint=int(bad[0]), note="empty qdot box")
        if self.config.check_j67:
            c6, c7, bound = _interference_row(joints, self.j67_margin, dt)
            self._a_data[self._a_c6], self._a_data[self._a_c7] = c6, c7
            full_lower, full_upper = np.r_[lower, -np.inf], np.r_[upper, bound]
        else:
            full_lower, full_upper = lower, upper
        if self._problem is None:
            self._problem = self._osqp.OSQP()
            constraints = (
                _constraint_matrix(c6, c7) if self.config.check_j67 else sparse.eye(7, format="csc")
            )
            self._problem.setup(
                P=_upper_csc(quadratic),
                q=linear,
                A=constraints,
                l=full_lower,
                u=full_upper,
                verbose=False,
                polishing=False,
                warm_starting=True,
            )
        else:
            updates = {"Ax": self._a_data} if self.config.check_j67 else {}
            self._problem.update(
                Px=quadratic[self._p_rows, self._p_cols],
                q=linear,
                l=full_lower,
                u=full_upper,
                **updates,
            )
        self._problem.warm_start(x=self._previous_velocity)
        solved = self._problem.solve(raise_error=False)
        if solved.info.status_val not in (1, 2):
            return failure("qp_infeasible", osqp_status=solved.info.status)
        if solved.x is None or not np.isfinite(solved.x).all():
            return failure("fk_mismatch", note="non-finite QP velocity")
        velocity = np.clip(solved.x, lower, upper)
        self._previous_velocity = velocity.copy()
        next_deg = joints + velocity * dt
        next_rad = np.radians(next_deg)
        actual = self.kinematics.flange(next_rad)
        pos_error = float(np.linalg.norm((actual[:3, 3] - target[:3, 3]) * 1000))
        actual_euler = Rotation.from_matrix(actual[:3, :3]).as_euler("xyz", degrees=True)
        target_euler = Rotation.from_matrix(target[:3, :3]).as_euler("xyz", degrees=True)
        rot_error = float(np.max(np.abs((actual_euler - target_euler + 180) % 360 - 180)))
        margin = float(np.min(np.minimum(self.upper - next_deg, next_deg - self.lower)))
        reason = "ok"
        if margin < self.margin:
            reason = "joint_limit"
        elif self.config.check_j67 and not j67_ok(next_deg):
            reason = "j67_interference"
        elif pos_error > self.config.max_lag_mm or (
            self.config.max_lag_deg is not None and rot_error > self.config.max_lag_deg
        ):
            reason = "tracking_lag"
        return DifferentialIKResult(
            reason == "ok",
            reason,
            next_rad,
            np.radians(velocity),
            pos_error,
            rot_error,
            float(np.max(np.abs(next_deg - joints))),
            margin,
            (time.perf_counter() - start) * 1000,
        )
