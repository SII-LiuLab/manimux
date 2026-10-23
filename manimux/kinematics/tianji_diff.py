"""Tianji velocity QP, ported from tianji-control's algos/diff_ik.py.

Public poses/joints use metres/radians. Internally mm/degrees preserve the
reference QP weights, regularization and nullspace objective. This is a rate
controller with tracking lag, not an analytic inverse with convergence bounds.
See .agents/skills/manimux-development/references/tianji-diff-ik.md for source
revision, guard semantics and validation.
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

    # Resolved from the executor profile or an explicitly tighter solver cap.
    max_velocity_rad_s: float = Field(gt=0)
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


class StepDifferentialIKConfig(DifferentialIKConfig):
    # Step-only: acceleration history must come from the executed command path.
    max_acceleration_rad_s2: float | None = Field(default=None, gt=0)
    elbow_guard_deg: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_elbow_guard(self):
        if self.elbow_guard_deg is not None and self.max_acceleration_rad_s2 is None:
            raise ValueError("elbow_guard_deg requires a step acceleration limit")
        return self


def resolve_diff_ik_config(motion_limits, tuning):
    """Inherit the executor rate unless tuning requests a tighter QP cap."""
    arm = (motion_limits or {}).get("arm") or {}
    maximum = arm.get("max_velocity")
    if maximum is None or not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("diffIK requires finite positive executor.motion_limits.arm.max_velocity")
    tuning = dict(tuning or {})
    tuning.setdefault("max_velocity_rad_s", maximum)
    config = DifferentialIKConfig.model_validate(tuning)
    if config.max_velocity_rad_s > maximum:
        raise ValueError(
            "diffIK max_velocity_rad_s cannot exceed executor.motion_limits.arm.max_velocity"
        )
    return config


def step_diff_config(motion_limits, tuning):
    """Resolve QP limits no higher than the independent executor limits."""
    tuning = dict(tuning or {})
    arm = (motion_limits or {}).get("arm") or {}
    acceleration = tuning.pop("max_acceleration_rad_s2", arm.get("max_acceleration"))
    elbow_guard = tuning.pop("elbow_guard_deg", None)
    base = resolve_diff_ik_config(motion_limits, tuning)
    config = StepDifferentialIKConfig(
        **base.model_dump(), max_acceleration_rad_s2=acceleration, elbow_guard_deg=elbow_guard,
    )
    executor_acceleration = arm.get("max_acceleration")
    if executor_acceleration is not None and (
        acceleration is None or acceleration > executor_acceleration
    ):
        raise ValueError("step diffIK acceleration must not exceed the executor acceleration limit")
    return config


class TianjiStepDifferentialIK:
    """One rate-controlled update from the last command, with composed TCP geometry.

    This reports tracking lag; it does not claim exact inverse-kinematics
    convergence. The caller owns atomic rejection, deadlines and command holds.
    """

    def __init__(self, models, config, dt_s):
        from manimux.kinematics.composed import ComposedManipulatorKinematics

        if not math.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("step diffIK requires a positive finite control period")
        self.models = models
        self.dt_s = dt_s
        self.solvers = {}
        for name, model in models.items():
            if not (
                isinstance(model, ComposedManipulatorKinematics)
                and isinstance(model.arm, TianjiArmKinematics)
                and len(model.coordinates) == 8
            ):
                raise ValueError(
                    "step diffIK requires composed Tianji kinematics with one tool coordinate"
                )
            self.solvers[name] = TianjiDifferentialIK(model.arm, config)

    def reset(self):
        for solver in self.solvers.values():
            solver.reset()

    def solve(self, name, target_tcp, command_seed, gripper, previous_velocity=None):
        command_seed = np.asarray(command_seed, dtype=float)
        if (command_seed.shape != (8,) or not np.isfinite(command_seed).all()
                or not np.isfinite(gripper) or not 0 <= gripper <= 1):
            return DifferentialIKResult(False, "invalid_input")
        flange = self.models[name].flange_target(target_tcp, np.array([gripper]))
        return self.solvers[name].solve(
            flange, command_seed[:7], self.dt_s, previous_velocity=previous_velocity,
        )


def _upper_csc(matrix):
    rows, cols = np.triu_indices(7)
    return sparse.csc_matrix((matrix[rows, cols], (rows, cols)), shape=(7, 7))


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


def _interference_constraints():
    """Exact global halfplanes for the measured convex, linear wrist boundary.

    Constrain the next position in every quadrant, including steps that cross
    J6/J7 zero. Reject incompatible future tables instead of linearizing them
    silently with a different feasible region.
    """
    table = np.asarray(BD67_REAL, dtype=float)
    if not (
        table.shape == (4, 3)
        and np.isfinite(table).all()
        and np.all(table[:, 0] == 0)
        and table[0, 2] == table[1, 2] > 0
        and table[2, 2] == table[3, 2] < 0
        and table[0, 1] <= 0 <= table[1, 1]
        and table[2, 1] <= 0 <= table[3, 1]
    ):
        raise ValueError("Differential IK requires convex linear J6/J7 boundaries")
    signs = np.array([1.0, 1.0, -1.0, -1.0])
    matrix = np.zeros((4, 7))
    matrix[:, 5] = -signs * table[:, 1]
    matrix[:, 6] = signs
    return matrix, signs * table[:, 2]


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
        # Allow only float64 roundoff in q + dt*qdot and the margin subtraction.
        # This is independent of OSQP tolerances and does not change the QP box.
        self._limit_roundoff_deg = (
            16 * np.finfo(float).eps
            * max(1.0, float(np.max(np.abs([self.lower, self.upper]))))
        )
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
        acceleration = getattr(config, "max_acceleration_rad_s2", None)
        self.amax = None if acceleration is None else math.degrees(acceleration)
        self.elbow_guard_deg = getattr(config, "elbow_guard_deg", None)
        self.elbow_extension_deg = (
            None if self.elbow_guard_deg is None else math.degrees(kinematics.elbow_extension_rad)
        )
        self.clear_elbow_branch()
        self.weights = [1.0] * 7 if config.nullspace_weights is None else config.nullspace_weights
        self.W = np.diag([config.w_pos] * 3 + [config.w_rot] * 3)
        self._p_rows = np.concatenate([np.arange(j + 1) for j in range(7)])
        self._p_cols = np.concatenate([np.full(j + 1, j) for j in range(7)])
        probe = np.arange(49, dtype=float).reshape(7, 7)
        if not np.array_equal(_upper_csc(probe).data, probe[self._p_rows, self._p_cols]):
            raise RuntimeError("Unexpected sparse upper-triangle layout")
        self._j67_matrix, self._j67_bound = _interference_constraints()
        self._constraint_matrix = sparse.vstack(
            [sparse.eye(7), sparse.csc_matrix(self._j67_matrix)], format="csc"
        )
        # The legacy TCP interface removes tool offsets here; new robots first
        # convert to flange targets in the composition layer.
        # The arm solver no longer holds tool information and must not convert again.
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

    def clear_elbow_branch(self):
        """Only a lifecycle reanchor may choose a new branch; QP reset preserves it."""
        self.elbow_side = None
        self._command_lower = self.lower + self.margin
        self._command_upper = self.upper - self.margin

    def ensure_elbow_branch(self, seed):
        """Select once from a stationary execution anchor, never from a target.

        A seed inside the guard is rejected, not projected or automatically moved.
        Physical joint limits continue to own the nullspace objective.
        """
        if self.elbow_guard_deg is None or self.elbow_side is not None:
            return
        seed = np.asarray(seed, dtype=float)
        if seed.shape != (7,) or not np.isfinite(seed).all():
            raise ValueError("elbow guard requires seven finite seed joints")
        angle = math.degrees(seed[3])
        distance = angle - self.elbow_extension_deg
        if abs(distance) < self.elbow_guard_deg + 1e-3:
            raise ValueError(
                "elbow guard: initial J4 is inside the extension protection zone; "
                "Home or reposition before starting"
            )
        lower, upper = self.lower + self.margin, self.upper - self.margin
        side = -1 if distance < 0 else 1
        if side < 0:
            upper[3] = min(upper[3], self.elbow_extension_deg - self.elbow_guard_deg)
        else:
            lower[3] = max(lower[3], self.elbow_extension_deg + self.elbow_guard_deg)
        if (np.any(lower >= upper) or np.any(np.degrees(seed) < lower)
                or np.any(np.degrees(seed) > upper)):
            raise ValueError("elbow guard: initial command is outside the selected safe region")
        self._command_lower, self._command_upper = lower, upper
        self.elbow_side = side

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

    def braking_time(self, dt_s):
        """A convex stopping envelope, in seconds, not an execution delay.

        If q + T*v is safe, executing dt*v leaves the feasible backup
        (1-dt/T)*v. T >= vmax/amax bounds that backup's acceleration.
        """
        return dt_s if self.amax is None else max(dt_s, self.vmax / self.amax)

    def command_violation(self, previous, command, previous_velocity, dt_s):
        """Validate the final command, including all four wrist halfplanes.

        Return a reason or None. Only float64 roundoff is allowed; OSQP's
        convergence tolerances never enlarge the command's feasible region.
        """
        if not math.isfinite(dt_s) or dt_s <= 0:
            return "invalid_control_period"
        arrays = [np.asarray(x, dtype=float) for x in (previous, command, previous_velocity)]
        if any(x.shape != (7,) or not np.isfinite(x).all() for x in arrays):
            return "invalid_command_state"
        q, next_q, old_v = map(np.degrees, arrays)
        if self.elbow_guard_deg is not None and self.elbow_side is None:
            return "elbow_branch_uninitialized"
        velocity = (next_q - q) / dt_s
        tol = self._limit_roundoff_deg
        if np.max(np.abs(velocity)) > self.vmax + 2 * tol / dt_s:
            return "velocity_limit"
        if (self.amax is not None
                and np.max(np.abs(velocity - old_v)) > self.amax * dt_s + 2 * tol / dt_s):
            return "acceleration_limit"
        for point in (q, next_q, q + self.braking_time(dt_s) * velocity):
            if (np.any(point < self._command_lower - tol)
                    or np.any(point > self._command_upper + tol)):
                return "joint_stopping_limit"
            if np.any(self._j67_matrix @ point > self._j67_bound - self.j67_margin + 3 * tol):
                return "wrist_stopping_limit"
        return None

    def brake(self, previous, previous_velocity, dt_s):
        """Certified ordinary-failure backup; never silently reset moving history."""
        if self.amax is None:
            raise ValueError("braking requires a step acceleration limit")
        self.ensure_elbow_branch(previous)
        old_v = np.asarray(previous_velocity, dtype=float)
        # Stop outright when possible, otherwise use the invariant backup.
        velocity = (np.zeros(7) if np.max(np.abs(np.degrees(old_v))) <= self.amax * dt_s
                    else (1 - dt_s / self.braking_time(dt_s)) * old_v)
        command = np.asarray(previous, dtype=float) + dt_s * velocity
        reason = self.command_violation(previous, command, old_v, dt_s)
        if reason is not None:
            raise ValueError(f"step diffIK has no certified braking command: {reason}")
        return command

    def solve(
        self, target_tcp, previous_joints, dt_s, *, previous_velocity=None,
    ) -> DifferentialIKResult:
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
        dt = dt_s if self.amax is not None else max(dt_s, 1e-6)
        joints = np.degrees(previous)
        try:
            self.ensure_elbow_branch(previous)
        except ValueError as exc:
            return failure("invalid_input", note=str(exc))
        if self.amax is not None:
            old_v = np.asarray(previous_velocity, dtype=float)
            if old_v.shape != (7,) or not np.isfinite(old_v).all():
                return failure(
                    "invalid_input", note="acceleration-limited IK requires command velocity",
                )
            # Do not attempt recovery from a command already outside the safe polytope.
            if self.command_violation(previous, previous, np.zeros(7), dt) is not None:
                return failure("invalid_input", note="command seed outside safe position polytope")
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
        horizon = self.braking_time(dt)
        # Reserve 0.001 degrees of position and 0.001 degrees/s of velocity
        # change for numerical QP residuals. Final validation uses exact bounds.
        reserve = 1e-3 if self.amax is not None else 0.0
        rate_reserve = (min(reserve, self.vmax * 1e-4, self.amax * dt * 1e-4)
                        if self.amax is not None else 0.0)
        lower = np.maximum(
            -self.vmax + rate_reserve, (self._command_lower + reserve - joints) / horizon,
        )
        upper = np.minimum(
            self.vmax - rate_reserve, (self._command_upper - reserve - joints) / horizon,
        )
        if self.amax is not None:
            lower = np.maximum(lower, np.degrees(old_v) - self.amax * dt + rate_reserve)
            upper = np.minimum(upper, np.degrees(old_v) + self.amax * dt - rate_reserve)
        bad = np.flatnonzero(lower > upper)
        if bad.size:
            # OSQP update(l>u) otherwise silently retains the previous problem.
            return failure("qp_infeasible", joint=int(bad[0]), note="empty qdot box")
        full_lower, full_upper = np.empty(11), np.empty(11)
        full_lower[:7], full_lower[7:] = lower, -np.inf
        full_upper[:7] = upper
        full_upper[7:] = (
            self._j67_bound - self.j67_margin - reserve - self._j67_matrix @ joints
        ) / horizon
        if self._problem is None:
            self._problem = self._osqp.OSQP()
            self._problem.setup(
                P=_upper_csc(quadratic),
                q=linear,
                A=self._constraint_matrix,
                l=full_lower,
                u=full_upper,
                verbose=False,
                polishing=False,
                warm_starting=True,
                **({"eps_abs": 1e-7, "eps_rel": 1e-7, "max_iter": 4000}
                   if self.amax is not None else {}),
            )
        else:
            self._problem.update(
                Px=quadratic[self._p_rows, self._p_cols],
                q=linear,
                l=full_lower,
                u=full_upper,
            )
        self._problem.warm_start(x=self._previous_velocity)
        solved = self._problem.solve(raise_error=False)
        if solved.info.status_val not in (1, 2):
            reason = "qp_infeasible"
            if self.amax is not None and solved.info.status_val not in (3, 4):
                reason = "qp_solver_failure"
            return failure(reason, osqp_status=solved.info.status, iterations=solved.info.iter)
        if solved.x is None or not np.isfinite(solved.x).all():
            return failure("fk_mismatch", note="non-finite QP velocity")
        # Independent clipping can break coupled wrist constraints. Acceleration-
        # aware step IK accepts only a jointly feasible result or takes its backup.
        velocity = solved.x.copy() if self.amax is not None else np.clip(solved.x, lower, upper)
        self._previous_velocity = velocity.copy()
        next_deg = joints + velocity * dt
        next_rad = np.radians(next_deg)
        if self.amax is not None:
            violation = self.command_violation(previous, next_rad, old_v, dt)
            if violation is not None:
                return failure("constraint_residual", constraint=violation,
                               osqp_status=solved.info.status)
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
        if margin < self.margin - self._limit_roundoff_deg:
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
