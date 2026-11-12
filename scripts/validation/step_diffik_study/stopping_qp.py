"""Study-only acceleration-aware step QP with a conservative braking envelope.

Constrain q + T*v to the existing safe position polytope, T >= vmax/amax.
After executing dt*v, the backup velocity (1-dt/T)*v preserves this envelope
and respects acceleration bounds. Its geometric braking path stays inside the
convex polytope; once slow enough it can stop outright. No future targets used.
No production configuration, runtime, boundary or controller is modified.
"""

import math
import time

import numpy as np
import osqp
from scipy import sparse
from tcp_objective import TcpStepDifferentialIK

from manimux.embodiments.arm.tianji.kinematics import BD67_REAL
from manimux.kinematics.tianji_diff import (
    DifferentialIKResult,
    TianjiStepDifferentialIK,
    _upper_csc,
    euler_xyz_deg,
    rotation_vector,
    step_diff_config,
)
from manimux.runtime.execution import ExecutionPipeline


class StoppingSolver:
    """Use degrees internally, matching the original QP objective and geometry."""

    def __init__(self, original, acceleration, dt, stop_tail, dt_cap):
        self.original = original
        self.dt = dt
        self.acceleration = math.degrees(acceleration)
        self.speed = original.vmax * min(1.0, (dt_cap or dt) / dt)
        self.n = 1
        self.braking_time = max(dt, self.speed / self.acceleration) if stop_tail else dt
        self.stop_tail = stop_tail
        self.problem = None
        self.solution = None
        self.telemetry = []
        # The recorded table is exactly the intersection of these four planes.
        assert np.array_equal(
            np.asarray(BD67_REAL),
            np.asarray(
                [
                    [0, -1.025, 110.5],
                    [0, 1.025, 110.5],
                    [0, -1.025, -110.5],
                    [0, 1.025, -110.5],
                ]
            ),
        )
        self.planes = np.zeros((4, 7))
        self.planes[:, 5:7] = [[1.025, 1], [-1.025, 1], [1.025, -1], [-1.025, -1]]
        self.matrix = sparse.vstack([sparse.eye(7), self.planes], format="csc")

    def reset(self):
        self.problem = None
        self.solution = None

    def solve(self, target, seed, previous_velocity):
        started = time.perf_counter()
        base = self.original
        config = base.config
        dt = self.dt
        q = np.degrees(seed)
        previous_velocity = np.degrees(previous_velocity)
        current, jacobian = base.kinematics.flange_and_jacobian(seed)
        jacobian = jacobian.copy()
        jacobian[:3] *= 1000 * math.pi / 180
        desired = (
            np.r_[
                (target[:3, 3] - current[:3, 3]) * 1000,
                np.degrees(rotation_vector(target[:3, :3] @ current[:3, :3].T)),
            ]
            / dt
        )
        first_p = 2 * (jacobian.T @ base.W @ jacobian + config.lam * np.eye(7))
        first_q = -2 * jacobian.T @ base.W @ desired
        if config.mu_nullspace > 0:
            first_q += config.mu_nullspace * dt * base._nullspace_gradient(q)
        p = _upper_csc(first_p)
        linear = first_q
        rate_reserve = 1e-3  # degrees/s, reserved for numerical solver residuals
        # Numerical reserve tightens existing margins; no boundary is relaxed.
        reserve = 1e-3
        braking_time = self.braking_time
        # Use the same slightly tightened rate in the braking certificate.
        if self.stop_tail:
            braking_time = max(
                dt, (self.speed - rate_reserve) / (self.acceleration - rate_reserve / dt)
            )
        lower_box = np.maximum.reduce(
            [
                np.full(7, -self.speed + rate_reserve),
                previous_velocity - self.acceleration * dt + rate_reserve,
                (base.lower + base.margin + reserve - q) / braking_time,
            ]
        )
        upper_box = np.minimum.reduce(
            [
                np.full(7, self.speed - rate_reserve),
                previous_velocity + self.acceleration * dt - rate_reserve,
                (base.upper - base.margin - reserve - q) / braking_time,
            ]
        )
        if np.any(lower_box > upper_box):
            return DifferentialIKResult(
                False, "qp_infeasible", detail={"note": "empty accel/position box"}
            )
        lower = np.r_[lower_box, np.full(4, -np.inf)]
        upper = np.r_[
            upper_box, (110.5 - base.j67_margin - reserve - self.planes @ q) / braking_time
        ]
        if self.problem is None:
            self.problem = osqp.OSQP()
            self.problem.setup(
                P=p,
                q=linear,
                A=self.matrix,
                l=lower,
                u=upper,
                verbose=False,
                polishing=True,
                warm_starting=True,
                eps_abs=1e-6,
                eps_rel=1e-6,
                max_iter=6000,
            )
        else:
            self.problem.update(Px=p.data, q=linear, l=lower, u=upper)
        if self.solution is not None:
            self.problem.warm_start(x=self.solution)
        result = self.problem.solve(raise_error=False)
        record = {
            "horizon_steps": self.n,
            "braking_time_s": braking_time,
            "osqp_status": result.info.status,
            "iterations": result.info.iter,
            "primal_residual": float(result.info.prim_res),
            "dual_residual": float(result.info.dual_res),
        }
        self.telemetry.append(record)
        if result.info.status_val not in (1, 2) or result.x is None:
            reason = (
                "qp_infeasible" if result.info.status_val in (3, 4) else "solver_iteration_limit"
            )
            return DifferentialIKResult(False, reason, detail=record)
        value = self.matrix @ result.x
        violation = float(max(np.max(lower - value), np.max(value - upper), 0))
        record["constraint_violation_internal_units"] = violation
        if violation > 1e-4:
            return DifferentialIKResult(False, "constraint_residual", detail=record)
        velocity = result.x[:7]
        if np.any(abs(velocity) > self.speed) or np.any(
            abs(velocity - previous_velocity) > self.acceleration * dt
        ):
            return DifferentialIKResult(False, "velocity_or_acceleration_residual", detail=record)
        next_q = q + velocity * dt
        if (
            np.any(next_q < base.lower + base.margin)
            or np.any(next_q > base.upper - base.margin)
            or np.any(self.planes @ next_q > 110.5 - base.j67_margin)
        ):
            return DifferentialIKResult(False, "joint_or_interference_margin", detail=record)
        self.solution = result.x.copy()
        next_rad = np.radians(next_q)
        actual, _ = base.kinematics.flange_and_jacobian(next_rad)
        pos_error = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]) * 1000)
        rot_error = float(
            np.max(
                abs(
                    (euler_xyz_deg(actual[:3, :3]) - euler_xyz_deg(target[:3, :3]) + 180) % 360
                    - 180
                )
            )
        )
        lag = pos_error > config.max_lag_mm or (
            config.max_lag_deg is not None and rot_error > config.max_lag_deg
        )
        return DifferentialIKResult(
            not (lag and config.lag_policy == "abort"),
            "tracking_lag" if lag and config.lag_policy == "abort" else "ok",
            joints=next_rad,
            qdot_rad_s=np.radians(velocity),
            pos_err_mm=pos_error,
            rot_err_deg=rot_error,
            max_step_deg=float(np.max(abs(next_q - q))),
            min_margin_deg=float(np.min(np.minimum(next_q - base.lower, base.upper - next_q))),
            solve_time_ms=(time.perf_counter() - started) * 1000,
            detail=record,
            lag_exceeded=lag,
        )


def pipeline_class(*, objective="flange", stop_tail=True):
    """Return an isolated pipeline factory; use the actual executor velocity seed."""

    class PilotPipeline(ExecutionPipeline):
        instances = []

        def __init__(self, executor, **kwargs):
            super().__init__(executor, **kwargs)
            motion = kwargs["motion_limits"]["arm"]
            config = step_diff_config(kwargs["motion_limits"], kwargs["ik_options"]["diff_ik"])
            dt = kwargs["control_dt_ns"] / 1e9
            cls = TcpStepDifferentialIK if objective == "tcp" else TianjiStepDifferentialIK
            original = cls(kwargs["kinematics"].models, config, dt)
            self.pilots = {}
            if motion["max_acceleration"] is not None:
                self.pilots = {
                    name: StoppingSolver(
                        solver, motion["max_acceleration"], dt, stop_tail, motion["max_step_dt_s"]
                    )
                    for name, solver in original.solvers.items()
                }
            solve_original, reset_original = original.solve, original.reset

            def solve(name, target, seed, gripper):
                if name not in self.pilots:
                    return solve_original(name, target, seed, gripper)
                if objective == "flange":
                    target = original.models[name].flange_target(target, np.array([gripper]))
                velocity = executor._previous_velocity[name][:7]
                return self.pilots[name].solve(target, seed[:7], velocity)

            def reset():
                reset_original()
                for solver in self.pilots.values():
                    solver.reset()

            original.solve, original.reset = solve, reset
            self.resolver.diff = original
            self.instances.append(self)

    return PilotPipeline
