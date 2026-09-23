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
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy import sparse
from scipy.spatial.transform import Rotation

from manimux.embodiments.arm.tianji.kinematics import BD67_REAL, TianjiArmKinematics, j67_ok
from manimux.kinematics.tianji import TianjiKinematics


class DifferentialIKConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    # Bound from the effective shared executor profile; never a second rate constant.
    max_velocity_rad_s: float = Field(gt=0)
    dt_max_s: float = Field(gt=0)
    w_pos: float = Field(default=1.0, gt=0)
    w_rot: float = Field(default=1.0, gt=0)
    lam: float = Field(default=0.001, ge=0)
    limit_margin_deg: float | None = Field(default=None, gt=0)
    j67_margin_deg: float | None = Field(default=None, gt=0)
    mu_nullspace: float = Field(default=1000.0, ge=0)
    nullspace_activation_deg: float | None = Field(default=25.0, gt=0)
    nullspace_weights: list[float] | None = None
    max_lag_mm: float = Field(default=5.0, gt=0)
    max_lag_deg: float | None = Field(default=None, gt=0)
    # abort turns a solved step whose target residual exceeds max_lag_* into a
    # tracking_lag failure; report keeps the bounded step and only flags it, as
    # CalibWrist real_run's diff_lag_policy. Backend failures reject in both.
    lag_policy: Literal["abort", "report"] = "abort"

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
    lag_exceeded: bool = False  # residual over max_lag_*; still ok under lag_policy report


def _upper_csc(matrix):
    rows, cols = np.triu_indices(7)
    return sparse.csc_matrix((matrix[rows, cols], (rows, cols)), shape=(7, 7))


def _constraint_matrix(c6, c7):
    return sparse.csc_matrix(
        ([1.0] * 7 + [c6, c7], (list(range(7)) + [7, 7], list(range(7)) + [5, 6])),
        shape=(8, 7),
    )


def rotation_vector(matrix):
    """3x3 rotation -> rotation vector (rad): scipy's Shepperd quaternion route in plain math.

    A per-step scipy Rotation costs about 20 us; this agrees with it to ~1e-13 deg.
    """
    (m00, m01, m02), (m10, m11, m12), (m20, m21, m22) = np.asarray(matrix, dtype=float).tolist()
    trace = m00 + m11 + m22
    largest = max(trace, m00, m11, m22)
    if largest == trace:
        w, x, y, z = 1 + trace, m21 - m12, m02 - m20, m10 - m01
    elif largest == m00:
        x, y, z, w = 1 + 2 * m00 - trace, m01 + m10, m02 + m20, m21 - m12
    elif largest == m11:
        y, x, z, w = 1 + 2 * m11 - trace, m01 + m10, m12 + m21, m02 - m20
    else:
        z, x, y, w = 1 + 2 * m22 - trace, m02 + m20, m12 + m21, m10 - m01
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if w < 0:
        norm = -norm
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    angle = 2 * math.atan2(math.sqrt(x * x + y * y + z * z), w)
    if angle > 1e-3:
        scale = angle / math.sin(angle / 2)
    else:
        scale = 2 + angle * angle / 12 + 7 * angle**4 / 2880
    return np.array([scale * x, scale * y, scale * z])


def rotation_matrix(vector):
    """Rotation vector (rad) -> 3x3 rotation, through the quaternion as scipy does."""
    x, y, z = (float(value) for value in vector)
    angle = math.sqrt(x * x + y * y + z * z)
    if angle > 1e-3:
        scale = math.sin(angle / 2) / angle
    else:
        scale = 0.5 - angle * angle / 48 + angle**4 / 3840
    w, x, y, z = math.cos(angle / 2), scale * x, scale * y, scale * z
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def euler_xyz_deg(matrix):
    """Extrinsic xyz Euler angles (deg), as scipy's ``as_euler("xyz")``.

    Near gimbal lock scipy picks a particular split between the first and third
    angle; defer to it there so the residual keeps its original definition.
    """
    m = np.asarray(matrix, dtype=float)
    if 1.0 - abs(m[2, 0]) < 1e-6:
        return Rotation.from_matrix(m).as_euler("xyz", degrees=True)
    return np.array(
        [
            math.degrees(math.atan2(m[2, 1], m[2, 2])),
            math.degrees(math.asin(-m[2, 0])),
            math.degrees(math.atan2(m[1, 0], m[0, 0])),
        ]
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

    def __init__(self, kinematics: TianjiArmKinematics, config: DifferentialIKConfig):
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
        # 旧 TCP 接口在此去除工具偏移；新整机由公共组合层先转换为法兰目标，
        # 传来的 arm 求解器不再持有末端信息，因此不能再次转换。
        tool = kinematics.tool_transform if isinstance(kinematics, TianjiKinematics) else None
        self._tool_inverse = None if tool is None else np.linalg.inv(tool)
        self.reset()

    # np.allclose(R.T @ R, I, atol=1e-7) with its default rtol, as a precomputed bound.
    _ORTHONORMAL_TOL = 1e-7 + 1e-5 * np.eye(3)

    def reset(self):
        self._problem = None
        self._previous_velocity = np.zeros(7)
        # Flange pose and Jacobian at the joints the last step produced. A chunk
        # seeds its next step with exactly those joints, so they are reused once.
        self._next_frame = None

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
        # np.allclose/np.isclose with their default rtol=1e-5, written out: the same
        # accept/reject decisions without their per-call overhead.
        bottom, rotation = target[3], target[:3, :3]
        (m00, m01, m02), (m10, m11, m12), (m20, m21, m22) = rotation.tolist()
        determinant = (
            m00 * (m11 * m22 - m12 * m21)
            - m01 * (m10 * m22 - m12 * m20)
            + m02 * (m10 * m21 - m11 * m20)
        )
        if not (
            abs(bottom[0]) <= 1e-8
            and abs(bottom[1]) <= 1e-8
            and abs(bottom[2]) <= 1e-8
            and abs(bottom[3] - 1.0) <= 1e-8 + 1e-5
        ) or not (
            np.all(np.abs(rotation.T @ rotation - np.eye(3)) <= self._ORTHONORMAL_TOL)
            and abs(determinant - 1.0) <= 1e-7 + 1e-5
        ):
            return failure("invalid_input", note="target must be a rigid transform")
        if self._tool_inverse is not None:
            target = target @ self._tool_inverse
        dt = min(max(dt_s, 1e-6), self.config.dt_max_s)
        joints = np.degrees(previous)
        cached = self._next_frame
        if cached is not None and np.array_equal(cached[0], previous):
            current, jacobian = cached[1], cached[2]
        else:
            current, jacobian = self.kinematics.flange_and_jacobian(previous)
        jacobian = jacobian.copy()
        jacobian[:3] *= 1000 * math.pi / 180  # m/rad -> mm/degree
        rotation_error = np.degrees(rotation_vector(target[:3, :3] @ current[:3, :3].T))
        desired = np.empty(6)
        desired[:3] = (target[:3, 3] - current[:3, 3]) * 1000
        desired[3:] = rotation_error
        desired /= dt
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
        c6, c7, bound = _interference_row(joints, self.j67_margin, dt)
        self._a_data[self._a_c6], self._a_data[self._a_c7] = c6, c7
        full_lower, full_upper = np.empty(8), np.empty(8)
        full_lower[:7], full_lower[7] = lower, -np.inf
        full_upper[:7], full_upper[7] = upper, bound
        if self._problem is None:
            self._problem = self._osqp.OSQP()
            self._problem.setup(
                P=_upper_csc(quadratic),
                q=linear,
                A=_constraint_matrix(c6, c7),
                l=full_lower,
                u=full_upper,
                verbose=False,
                polishing=False,
                warm_starting=True,
            )
        else:
            self._problem.update(
                Px=quadratic[self._p_rows, self._p_cols],
                q=linear,
                l=full_lower,
                u=full_upper,
                Ax=self._a_data,
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
        # flange_and_jacobian's pose is bit-identical to flange(); keep its Jacobian
        # for the next step, which starts from exactly these joints.
        actual, next_jacobian = self.kinematics.flange_and_jacobian(next_rad)
        self._next_frame = (next_rad, actual, next_jacobian)
        pos_error = float(np.linalg.norm((actual[:3, 3] - target[:3, 3]) * 1000))
        actual_euler = euler_xyz_deg(actual[:3, :3])
        target_euler = euler_xyz_deg(target[:3, :3])
        rot_error = float(np.max(np.abs((actual_euler - target_euler + 180) % 360 - 180)))
        margin = float(np.min(np.minimum(self.upper - next_deg, next_deg - self.lower)))
        lag_exceeded = pos_error > self.config.max_lag_mm or (
            self.config.max_lag_deg is not None and rot_error > self.config.max_lag_deg
        )
        reason = "ok"
        if margin < self.margin:
            reason = "joint_limit"
        elif not j67_ok(next_deg):
            reason = "j67_interference"
        elif lag_exceeded and self.config.lag_policy == "abort":
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
            lag_exceeded=lag_exceeded,
        )
