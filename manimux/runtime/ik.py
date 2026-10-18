"""Execution-time TCP resolution; joint references retain their existing path."""

import time
from dataclasses import replace

import numpy as np

from manimux.kinematics.poses import matrix_pose, pose_matrix
from manimux.policy_adapter.pose_output import POSE_SPACE
from manimux.runtime.pose_limits import pose_speed_limit_parameters


def ik_parameters(**options):
    unknown = set(options) - {"mode", "backend", "diff_ik", "budget_ms", "pose_speed_limit"}
    if unknown:
        raise ValueError(f"unknown execution.ik fields: {sorted(unknown)}")
    values = {
        "mode": "chunk", "backend": "analytic", "diff_ik": None,
        "budget_ms": None, "pose_speed_limit": None, **options,
    }
    if values["mode"] not in {"chunk", "step"}:
        raise ValueError("execution.ik.mode must be chunk or step")
    if values["backend"] not in {"analytic", "diff"}:
        raise ValueError("execution.ik.backend must be analytic or diff")
    if values["backend"] == "diff" and values["mode"] != "step":
        raise ValueError("execution.ik.backend=diff requires mode=step; chunk IK belongs to the adapter")
    if values["diff_ik"] is not None and values["backend"] != "diff":
        raise ValueError("execution.ik.diff_ik requires backend=diff")
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
    if infer["blend_mode"] != "linear" or infer["handoff"] != "blend":
        raise ValueError(
            "step IK requires pose linear/SLERP blending, not joint cubic or waypoint handoff"
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
    if execution.get("backend", "analytic") == "diff":
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
                 motion_limits=None, control_dt_ns=None):
        self.models = kinematics.models
        self.budget_ns = int(budget_ms * 1e6)
        self.last_result = {}
        self.backend = backend
        self.diff = None
        if backend == "diff":
            from manimux.kinematics.tianji_diff import TianjiStepDifferentialIK, step_diff_config

            self.diff = TianjiStepDifferentialIK(
                self.models, step_diff_config(motion_limits, diff_ik), control_dt_ns / 1e9
            )

    def reset(self):
        if self.diff is not None:
            self.diff.reset()

    def poses_from_joints(self, groups):
        return {
            name: np.r_[matrix_pose(self.models[name].fk(q)), q[-1]] for name, q in groups.items()
        }

    def warmup(self, groups):
        # Initialize solvers before Start, without sending robot commands.
        if self.diff is not None:
            self.reset()
            for name, q in groups.items():
                self.diff.solve(name, self.models[name].fk(q), q, q[-1])
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

    def resolve(self, reference, seed, *, deadline_ns=None):
        if reference.action_space != POSE_SPACE or set(reference.groups) != set(seed):
            raise ValueError("step IK requires a complete pose reference matching seed groups")
        started = time.monotonic_ns()
        deadline = min(started + self.budget_ns, deadline_ns or started + self.budget_ns)
        output = {}
        diagnostics = {}
        failure = None
        for name, rows in reference.groups.items():
            if time.monotonic_ns() >= deadline:
                failure = "budget_exceeded"
                break
            model = self.models[name]
            target = rows[0]
            if self.diff is not None:
                result = self.diff.solve(name, pose_matrix(target[:7]), seed[name], target[7])
                diagnostics[name] = {
                    "pos_err_mm": result.pos_err_mm, "rot_err_deg": result.rot_err_deg,
                    "lag_exceeded": result.lag_exceeded, "max_step_deg": result.max_step_deg,
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
            **({"diff_ik": diagnostics} if self.diff is not None else {}),
        }
        if failure:
            self.reset()
            return None
        return replace(
            reference,
            groups={name: q[None, :] for name, q in output.items()},
            tracking_groups=None,
            hold_groups=(),
            action_space="joint_position",
        )
