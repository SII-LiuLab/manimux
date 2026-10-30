"""Execution-time TCP resolution; joint references retain their existing path."""

import time
from dataclasses import replace

import numpy as np

from manimux.kinematics.failure import DEFAULT_FAILURE_POLICY, validate_failure_policy
from manimux.kinematics.poses import matrix_pose, pose_matrix
from manimux.policy_adapter.pose_output import POSE_SPACE
from manimux.runtime.pose_limits import pose_speed_limit_parameters


def ik_parameters(**options):
    unknown = set(options) - {"mode", "budget_ms", "pose_speed_limit", "failure_policy"}
    if unknown:
        raise ValueError(f"unknown execution.ik fields: {sorted(unknown)}")
    values = {
        "mode": "chunk",
        "budget_ms": None,
        "pose_speed_limit": None,
        "failure_policy": DEFAULT_FAILURE_POLICY,
        **options,
    }
    if values["mode"] not in {"chunk", "step"}:
        raise ValueError("execution.ik.mode must be chunk or step")
    validate_failure_policy(values["failure_policy"])
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
            "execution.ik.mode=step requires a pose-producing adapter; joint adapters need no IK mode"
        )
    if config["policy"]["adapter"].get("ik_backend") == "dls":
        raise ValueError("ik_backend: dls is chunk IK; it requires execution.ik.mode=chunk")
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
            "step IK replaces decode-time substeps; remove execute_diff_ik_substeps/waypoint_handoff"
        )
    if options.get("ik_backend", "analytic") != "analytic":
        raise ValueError(
            "step IK currently requires the robot native solver (ik_backend: analytic); decode-time diffIK is not interchangeable"
        )
    budget = config["execution"]["ik"]["budget_ms"]
    if budget is not None and budget >= 1000 / config["robot"]["control_hz"]:
        raise ValueError("step IK budget must leave time for the rest of the control tick")


class StepIKResolver:
    """Solve the current target from measured joints; return None if any arm fails.

    A late native solve is rejected, never sent. Native solvers without deadline
    support cannot be preempted; timing metrics expose control-period overruns.
    """

    def __init__(self, kinematics, *, budget_ms):
        self.models = kinematics.models
        self.budget_ns = int(budget_ms * 1e6)
        self.last_result = {}

    def poses_from_joints(self, groups):
        return {
            name: np.r_[matrix_pose(self.models[name].fk(q)), q[-1]] for name, q in groups.items()
        }

    def warmup(self, groups):
        # Initialize native solvers before Start, without sending robot commands.
        for name, q in groups.items():
            model = self.models[name]
            model.ik(model.fk(q), q, fixed_coordinates={model.coordinates[-1].name: float(q[-1])})

    def resolve(self, reference, measured, *, deadline_ns=None):
        if reference.action_space != POSE_SPACE or set(reference.groups) != set(measured):
            raise ValueError("step IK requires a complete pose reference matching measured groups")
        started = time.monotonic_ns()
        deadline = min(started + self.budget_ns, deadline_ns or started + self.budget_ns)
        output = {}
        failure = None
        for name, rows in reference.groups.items():
            if time.monotonic_ns() >= deadline:
                failure = "budget_exceeded"
                break
            model = self.models[name]
            target = rows[0]
            fixed = {model.coordinates[-1].name: float(target[7])}
            bounded = getattr(model, "ik_deadline", None)
            if bounded is None:
                result = model.ik(pose_matrix(target[:7]), measured[name], fixed_coordinates=fixed)
            else:
                result = bounded(
                    pose_matrix(target[:7]),
                    measured[name],
                    fixed_coordinates=fixed,
                    deadline_ns=deadline,
                )
            if not result.converged:
                failure = f"{name}:{result.reason}"
                break
            q = np.asarray(result.joints, dtype=float)
            if q.shape != measured[name].shape or not np.isfinite(q).all():
                raise ValueError(f"step IK returned an invalid solution for {name}")
            output[name] = q.copy()
        elapsed = time.monotonic_ns() - started
        if time.monotonic_ns() > deadline:
            failure = "budget_exceeded"
        self.last_result = {
            "elapsed_ms": elapsed / 1e6,
            "success": failure is None,
            "reason": failure or "converged",
        }
        if failure:
            return None
        return replace(
            reference,
            groups={name: q[None, :] for name, q in output.items()},
            tracking_groups=None,
            hold_groups=(),
            action_space="joint_position",
        )
