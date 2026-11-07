"""Execution-time TCP resolution; joint references retain their existing path."""

import time
from collections.abc import Mapping
from dataclasses import replace

import numpy as np

from manimux.kinematics.failure import DEFAULT_FAILURE_POLICY, validate_failure_policy
from manimux.kinematics.poses import matrix_pose, pose_matrix
from manimux.policy_adapter.pose_output import POSE_SPACE
from manimux.runtime.pose_limits import pose_speed_limit_parameters


def ik_parameters(**options):
    unknown = set(options) - {
        "mode",
        "budget_ms",
        "pose_speed_limit",
        "failure_policy",
        "solver",
        "differential",
    }
    if unknown:
        raise ValueError(f"unknown execution.ik fields: {sorted(unknown)}")
    values = {
        "mode": "chunk",
        "budget_ms": None,
        "pose_speed_limit": None,
        "failure_policy": DEFAULT_FAILURE_POLICY,
        "solver": "native",
        "differential": None,
        **options,
    }
    if values["mode"] not in {"chunk", "step"}:
        raise ValueError("execution.ik.mode must be chunk or step")
    if values["solver"] not in {"native", "differential"}:
        raise ValueError("execution.ik.solver must be native or differential")
    if values["solver"] == "differential":
        if values["mode"] != "step" or not isinstance(values["differential"], Mapping):
            raise ValueError(
                "differential IK requires mode=step and a differential options mapping"
            )
    elif values["differential"] is not None:
        raise ValueError("differential options require execution.ik.solver=differential")
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
            "execution.ik.mode=step requires a pose-producing adapter; "
            "joint adapters need no IK mode"
        )
    # Step IK packs one pose row and currently appends one gripper coordinate
    # per group in the control loop. Keep that contract explicit so a bare-arm
    # deployment cannot treat J7 as an aperture and fail only after connecting.
    from manimux.embodiments.layout import group_layouts

    layouts = config["robot"].get("group_layouts")
    if layouts is None:
        layouts = group_layouts(config["robot"]["group_dims"], config["policy"]["adapter"])
    if any(layout["gripper_dofs"] != 1 for layout in layouts.values()):
        raise ValueError(
            "execution.ik.mode=step requires one normalized gripper coordinate per group; "
            "use execution.ik.mode=chunk for bare arms"
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
            "step IK replaces decode-time substeps; "
            "remove execute_diff_ik_substeps/waypoint_handoff"
        )
    if options.get("ik_backend", "analytic") != "analytic":
        raise ValueError(
            "step IK requires ik_backend: analytic (no decode-time solver); "
            "select execution-time differential IK with execution.ik.solver"
        )
    budget = config["execution"]["ik"]["budget_ms"]
    if budget is not None and budget >= 1000 / config["robot"]["control_hz"]:
        raise ValueError("step IK budget must leave time for the rest of the control tick")
    if config["execution"]["ik"].get("solver") == "differential":
        from manimux.embodiments.robot.base import RobotModel

        # Offline capability/configuration check before any controller connects.
        models = RobotModel.from_config(config["robot"]["config"]).kinematics.models
        for model in models.values():
            model.create_differential_ik(**config["execution"]["ik"]["differential"])


class StepIKResolver:
    """Resolve one target from measured state; return None if any arm fails.

    Differential steps also use the separate previous command and nominal dt.
    A late solve is rejected, never sent. Native solvers without deadline
    support cannot be preempted; timing metrics expose control-period overruns.
    """

    def __init__(self, kinematics, *, budget_ms, differential=None, control_dt_s=None):
        self.models = kinematics.models
        self.budget_ns = int(budget_ms * 1e6)
        self.last_result = {}
        self.control_dt_s = control_dt_s
        self.differential = None
        self.command_bounds = {}
        if differential is not None:
            if control_dt_s is None or not np.isfinite(control_dt_s) or control_dt_s <= 0:
                raise ValueError("differential IK requires a positive control period")
            self.differential = {
                name: model.create_differential_ik(**differential)
                for name, model in self.models.items()
            }

    def reset(self):
        self.command_bounds = {}
        if self.differential is not None:
            for solver in self.differential.values():
                solver.reset()

    def command_violation(self, groups):
        """Executor acceleration limiting must not bypass the differential bounds."""
        for name, bounds in self.command_bounds.items():
            q = np.asarray(groups[name])
            if np.any(q < bounds[:, 0] - 1e-10) or np.any(q > bounds[:, 1] + 1e-10):
                return f"{name}:executor_outside_differential_bounds"
        return None

    def poses_from_joints(self, groups):
        return {
            name: np.r_[matrix_pose(self.models[name].fk(q)), q[-1]] for name, q in groups.items()
        }

    def warmup(self, groups):
        # Initialize the selected solvers before Start, without robot commands.
        self.reset()
        for name, q in groups.items():
            model = self.models[name]
            fixed = {model.coordinates[-1].name: float(q[-1])}
            if self.differential is None:
                model.ik(model.fk(q), q, fixed_coordinates=fixed)
            else:
                self.differential[name].solve(
                    model.fk(q), q, self.control_dt_s, measured=q, fixed_coordinates=fixed
                )
        self.reset()

    def resolve(self, reference, measured, *, deadline_ns=None, previous=None):
        if reference.action_space != POSE_SPACE or set(reference.groups) != set(measured):
            raise ValueError("step IK requires a complete pose reference matching measured groups")
        if self.differential is not None and (previous is None or set(previous) != set(measured)):
            raise ValueError("differential IK requires the separate last command for every group")
        started = time.monotonic_ns()
        deadline = min(started + self.budget_ns, deadline_ns or started + self.budget_ns)
        output = {}
        self.command_bounds = {}
        failure = None
        group_status = {}
        for name, rows in reference.groups.items():
            if time.monotonic_ns() >= deadline:
                failure = "budget_exceeded"
                break
            model = self.models[name]
            target = rows[0]
            fixed = {model.coordinates[-1].name: float(target[7])}
            bounded = getattr(model, "ik_deadline", None)
            if self.differential is not None:
                result = self.differential[name].solve(
                    pose_matrix(target[:7]),
                    previous[name],
                    self.control_dt_s,
                    measured=measured[name],
                    fixed_coordinates=fixed,
                )
                accepted = result.ok
                group_status[name] = {
                    "accepted": accepted,
                    "limited": result.limited,
                    "reason": result.reason,
                }
            elif bounded is None:
                result = model.ik(pose_matrix(target[:7]), measured[name], fixed_coordinates=fixed)
                accepted = result.converged
            else:
                result = bounded(
                    pose_matrix(target[:7]),
                    measured[name],
                    fixed_coordinates=fixed,
                    deadline_ns=deadline,
                )
                accepted = result.converged
            if not accepted:
                failure = f"{name}:{result.reason}"
                break
            q = np.asarray(result.joints, dtype=float)
            if q.shape != measured[name].shape or not np.isfinite(q).all():
                raise ValueError(f"step IK returned an invalid solution for {name}")
            output[name] = q.copy()
            if self.differential is not None:
                self.command_bounds[name] = result.position_bounds
        elapsed = time.monotonic_ns() - started
        if time.monotonic_ns() > deadline:
            failure = "budget_exceeded"
        self.last_result = {
            "elapsed_ms": elapsed / 1e6,
            "success": failure is None,
            "reason": failure or ("bounded_step" if self.differential is not None else "converged"),
            "solver": "differential" if self.differential is not None else "native",
            "groups": group_status,
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
