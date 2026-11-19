"""Execution-time TCP resolution; joint references retain their existing path."""

import time
from dataclasses import replace

import numpy as np

from manimux.kinematics.poses import matrix_pose, pose_matrix
from manimux.policy_adapter.pose_output import POSE_SPACE
from manimux.runtime.pose_limits import pose_speed_limit_parameters


def ik_parameters(**options):
    unknown = set(options) - {
        "mode", "backend", "diff_ik", "budget_ms", "pose_speed_limit", "rl_ik",
        "final_command_deadline",
    }
    if unknown:
        raise ValueError(f"unknown execution.ik fields: {sorted(unknown)}")
    values = {
        "mode": "chunk", "backend": "analytic", "diff_ik": None,
        "budget_ms": None, "pose_speed_limit": None, "rl_ik": None,
        "final_command_deadline": False, **options,
    }
    if values["mode"] not in {"chunk", "step"}:
        raise ValueError("execution.ik.mode must be chunk or step")
    if values["backend"] not in {"analytic", "diff", "rl"}:
        raise ValueError("execution.ik.backend must be analytic, diff or rl")
    if values["backend"] in {"diff", "rl"} and values["mode"] != "step":
        raise ValueError(
            "constrained IK requires mode=step; chunk IK belongs to the adapter"
        )
    if values["diff_ik"] is not None and values["backend"] not in {"rl", "diff"}:
        raise ValueError("execution.ik.diff_ik tuning requires backend=diff or rl")
    if values["rl_ik"] is not None and not isinstance(values["rl_ik"], dict):
        raise ValueError("rl_ik requires a mapping")
    if (values["backend"] == "rl") != (values["rl_ik"] is not None):
        raise ValueError("backend rl requires an rl_ik worker configuration")
    if values["backend"] == "rl":
        values["final_command_deadline"] = True
        device = values["rl_ik"].get("device", "cpu")
        if device not in {"cpu", "cuda"}:
            raise ValueError("rl_ik.device must be cpu or cuda")
        if device == "cuda" and values["budget_ms"] not in (1, 2):
            raise ValueError("paired CUDA F requires a shared 1 or 2 ms deadline")
        if device == "cpu" and values["budget_ms"] != 2:
            raise ValueError("the dual-arm F configuration requires a shared 2 ms budget")
    if values["final_command_deadline"] and values["backend"] not in {"rl", "diff"}:
        raise ValueError("final_command_deadline requires a constrained step backend")
    budget = values["budget_ms"]
    if budget is not None and (not np.isfinite(budget) or budget <= 0):
        raise ValueError("execution.ik.budget_ms must be positive and finite")
    values["pose_speed_limit"] = pose_speed_limit_parameters(values["pose_speed_limit"])
    if values["mode"] != "step" and values["pose_speed_limit"] is not None:
        raise ValueError("pose_speed_limit requires execution.ik.mode=step")
    return values


def step_ik_enabled(config):
    return config.get("execution", {}).get("ik", {}).get("mode", "chunk") == "step"


def validate_ik_config(config):
    if not step_ik_enabled(config):
        return
    from manimux.plugins import load_plugin

    cls = load_plugin(
        config["policy"]["adapter"]["type"], group="manimux.policy_adapter", builtins={}
    )
    if not getattr(cls, "supports_step_ik", False):
        raise ValueError(
            "execution.ik.mode=step requires a pose-producing adapter with shared TCP/solver; "
            "joint adapters need no IK mode"
        )
    if config["executor"]["type"] != "direct":
        raise ValueError(
            "step IK requires direct executor; smooth/mpc require future joint references"
        )
    infer = config["inference"]
    if infer["algorithm"] not in {"async", "serial", "rtc", "paint"} or (
        infer.get("temporal_ensemble") or {}
    ).get("enabled"):
        raise ValueError(
            "step IK does not support joint-space candidate scoring or temporal ensembling"
        )
    if infer["algorithm"] in {"rtc", "paint"} and infer["algorithm"] not in getattr(
        cls, "pose_condition_modes", ()
    ):
        raise ValueError("this adapter has no pose-space RTC/PAINT condition codec")
    if (infer.get("rtc") or {}).get("condition_source") == "executor":
        raise ValueError("step IK cannot forecast future joint commands for RTC")
    if infer["blend_mode"] != "linear" or infer["handoff"] not in {"blend", "sigmoid_smooth"}:
        raise ValueError(
            "step IK requires pose linear/SLERP blend or sigmoid_smooth, "
            "not joint cubic or waypoint handoff"
        )
    if infer["independent_group_decoding"]:
        raise ValueError("step IK requires atomic dual-arm decoding")
    if config["policy"]["action_decoding"] != "inline":
        raise ValueError(
            "step IK uses inline pose decoding; process joint decoding is not applicable"
        )
    options = config["policy"]["adapter"]
    if options.get("execute_diff_ik_substeps") or options.get("handoff_waypoint"):
        raise ValueError(
            "step IK replaces decode-time substeps; "
            "remove execute_diff_ik_substeps/handoff_waypoint"
        )
    if options.get("ik_backend", "analytic") != "analytic":
        raise ValueError(
            "step IK uses pose-only decoding: remove adapter ik_backend=diff; "
            "select execution.ik.backend=diff for execution-time differential IK"
        )
    execution = config["execution"]["ik"]
    if execution.get("backend", "analytic") in {"diff", "rl"}:
        from manimux.kinematics.tianji_diff import step_diff_config

        step_diff_config(config["executor"].get("motion_limits"), execution.get("diff_ik"))
    budget = config["execution"]["ik"]["budget_ms"]
    if budget is not None and budget >= 1000 / config["robot"]["control_hz"]:
        raise ValueError("step IK budget must leave time for the rest of the control tick")


class StepIKResolver:
    """Solve the current target from caller-selected joints; reject if any arm fails.

    A late solve is rejected, never sent. Solvers without deadline support cannot
    be preempted; timing metrics expose control-period overruns. Differential
    updates can succeed with tracking lag, as configured by their lag policy.
    """

    def __init__(self, kinematics, *, budget_ms, backend="analytic", diff_ik=None,
                 motion_limits=None, control_dt_ns=None, rl_ik=None):
        self.models = kinematics.models
        self.budget_ns = int(budget_ms * 1e6)
        self.last_result = {}
        self.backend = backend
        self.diff = None
        self.rl = None
        if backend in {"diff", "rl"}:
            from manimux.kinematics.tianji_diff import TianjiStepDifferentialIK, step_diff_config

            self.diff = TianjiStepDifferentialIK(
                self.models, step_diff_config(motion_limits, diff_ik), control_dt_ns / 1e9
            )
            arm = motion_limits["arm"]
            dt = control_dt_ns / 1e9
            effective_cap = arm["max_velocity"] * min(1.0, (arm.get("max_step_dt_s") or dt) / dt)
            if any(s.config.max_velocity_rad_s > effective_cap for s in self.diff.solvers.values()):
                raise ValueError(
                    "step diffIK velocity exceeds executor's effective max_step_dt_s cap"
                )

        if backend == "rl":
            if control_dt_ns != 10_000_000 or not self.acceleration_limited:
                raise ValueError("F execution requires acceleration limits and 100 Hz control")
            if any(s.config.lag_policy != "report" for s in self.diff.solvers.values()):
                raise ValueError("RL step execution supports lag_policy=report")
            options = dict(rl_ik)
            device = options.pop("device", "cpu")
            if device == "cuda":
                from manimux.runtime.rl_step_cuda import GPURefiner

                self.rl = GPURefiner(options)
                if self.rl.reserve_ns >= self.budget_ns:
                    raise ValueError("paired CUDA merge reserve must fit the tick budget")
            elif device == "cpu":
                from manimux.runtime.rl_step import DualArmRefiner

                self.rl = DualArmRefiner(options)
            else:
                raise ValueError(f"unknown paired F device: {device}")

    def close(self):
        if self.rl is not None:
            self.rl.close()

    @property
    def acceleration_limited(self):
        return self.diff is not None and all(s.amax is not None for s in self.diff.solvers.values())

    def braking_groups(self, seed, velocities):
        return {
            name: np.r_[
                solver.brake(seed[name][:7], velocities[name][:7], self.diff.dt_s), seed[name][7],
            ]
            for name, solver in self.diff.solvers.items()
        }

    def validate_command(self, seed, command, velocities):
        for name, solver in self.diff.solvers.items():
            reason = solver.command_violation(
                seed[name][:7], command[name][:7], velocities[name][:7], self.diff.dt_s,
            )
            if reason is not None:
                raise ValueError(f"final step diffIK command rejected: {name}:{reason}")

    def reset(self, *, reanchor=False):
        if self.rl is not None:
            self.rl.history.clear()
        if self.diff is not None:
            self.diff.reset()
            if reanchor:
                for solver in self.diff.solvers.values():
                    if getattr(solver.config, "elbow_guard_deg", None) is not None:
                        solver.clear_elbow_branch()

    def poses_from_joints(self, groups):
        return {
            name: np.r_[matrix_pose(self.models[name].fk(q)), q[-1]] for name, q in groups.items()
        }

    def warmup(self, groups):
        # Initialize solvers before Start, without sending robot commands.
        if self.diff is not None:
            self.reset(reanchor=True)
            for name, q in groups.items():
                solver = self.diff.solvers[name]
                if getattr(solver.config, "elbow_guard_deg", None) is not None:
                    solver.ensure_elbow_branch(q[:7])
                if self.rl is None:
                    self.diff.solve(name, self.models[name].fk(q), q, q[-1], np.zeros(7))
            if self.rl is not None:
                self.rl.warmup(self.models, groups)
            return
        for name, q in groups.items():
            model = self.models[name]
            fixed = {model.coordinates[-1].name: float(q[-1])}
            bounded = getattr(model, "ik_deadline", None)
            if bounded is None:
                model.ik(model.fk(q), q, fixed_coordinates=fixed)
            else:
                # Warm the same bounded solver used by control ticks. Startup
                # initialization has a separate allowance and sends no commands.
                bounded(
                    model.fk(q),
                    q,
                    fixed_coordinates=fixed,
                    deadline_ns=time.monotonic_ns() + max(self.budget_ns, 5_000_000_000),
                )

    def _refine_pair(self, reference, seed, measured, command_velocity, deadline):
        output = {}
        failure = None
        candidates, diagnostics, late = self.rl.propose(reference, seed, measured, deadline)
        dt = self.diff.dt_s
        for name, q in candidates.items():
            solver = self.diff.solvers[name]
            start = seed[name][:7]
            old_v = command_velocity[name][:7]
            joint, fraction = solver.constrain_increment(start, q, old_v, dt)
            diagnostics[name]["wrist_fraction"] = fraction
            reason = solver.command_violation(start, joint, old_v, dt)
            diagnostics[name]["constraint"] = reason
            if reason is None:
                output[name] = np.r_[joint, reference.groups[name][0, 7]]
            else:
                failure = f"{name}:{reason}"
        if set(output) != set(seed):
            failure = failure or "incomplete_pair"
        return output, diagnostics, late, failure

    def resolve(self, reference, seed, *, deadline_ns=None, command_velocity=None, measured=None):
        if reference.action_space != POSE_SPACE or set(reference.groups) != set(seed):
            raise ValueError("step IK requires a complete pose reference matching seed groups")
        started = time.monotonic_ns()
        deadline = min(started + self.budget_ns, deadline_ns or started + self.budget_ns)
        output = {}
        diagnostics = {}
        failure = None
        if self.rl is not None:
            output, diagnostics, late, failure = self._refine_pair(
                reference, seed, measured, command_velocity, deadline,
            )
        for name, rows in (reference.groups.items() if self.rl is None else ()):
            if time.monotonic_ns() >= deadline:
                failure = "budget_exceeded"
                break
            model = self.models[name]
            target = rows[0]
            if self.diff is not None:
                result = self.diff.solve(
                    name, pose_matrix(target[:7]), seed[name], target[7],
                    None if command_velocity is None else command_velocity[name][:7],
                )
                diagnostics[name] = {
                    "pos_err_mm": result.pos_err_mm, "rot_err_deg": result.rot_err_deg,
                    "lag_exceeded": result.lag_exceeded, "max_step_deg": result.max_step_deg,
                    **({"failure_detail": result.detail} if not result.ok else {}),
                }
                if not result.ok:
                    failure = f"{name}:{result.reason}"
                    break
                q = np.r_[result.joints, target[7]]
                if q.shape != seed[name].shape or not np.isfinite(q).all():
                    failure = f"{name}:invalid_solution"
                    break
                output[name] = q
                continue
            fixed = {model.coordinates[-1].name: float(target[7])}
            bounded = getattr(model, "ik_deadline", None)
            if bounded is None:
                result = model.ik(pose_matrix(target[:7]), seed[name], fixed_coordinates=fixed)
            else:
                result = bounded(
                    pose_matrix(target[:7]),
                    seed[name],
                    fixed_coordinates=fixed,
                    deadline_ns=deadline,
                )
            if not result.converged:
                failure = f"{name}:{result.reason}"
                break
            q = np.asarray(result.joints, dtype=float)
            if q.shape != seed[name].shape or not np.isfinite(q).all():
                failure = f"{name}:invalid_solution"
                break
            output[name] = q.copy()
        elapsed = time.monotonic_ns() - started
        if time.monotonic_ns() > deadline:
            failure = "budget_exceeded"
        self.last_result = {
            "backend": self.backend,
            "elapsed_ms": elapsed / 1e6,
            "success": failure is None,
            "reason": failure or ("ok" if self.diff is not None else "converged"),
            **({"rl_ik": diagnostics, "late_workers": late} if self.rl is not None else
               {"diff_ik": diagnostics} if self.diff is not None else {}),
        }
        if failure:
            if self.rl is None:
                self.reset()
            return None
        return replace(
            reference,
            groups={name: q[None, :] for name, q in output.items()},
            tracking_groups=None,
            hold_groups=(),
            action_space="joint_position",
        )
