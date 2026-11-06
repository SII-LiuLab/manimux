#!/usr/bin/env python3
"""Independent trace checks, common pre-hold comparison and reset test."""

import json

import numpy as np
import run as study
from stopping_qp import pipeline_class
from test_stopping_qp import OUT

from manimux.types import ActionHorizon, RobotState


def first_hold(trace):
    indices = np.flatnonzero(~np.isin(trace["reason"], ["ok", "gap"]))
    return int(indices[0]) if len(indices) else len(trace["reason"])


def summarize_difference(base, candidate):
    return {
        key: [
            candidate["all"][field][stage][g][quantity][stat]
            / max(base["all"][field][stage][g][quantity][stat], 1e-12)
            for g in study.GROUPS
        ]
        for key, field, stage, quantity, stat in (
            ("position", "errors", "target_to_command", "translation_mm", "rms"),
            ("rotation", "errors", "target_to_command", "rotation_deg", "rms"),
            ("linear_jerk", "motion", "command", "linear_jerk", "rms"),
            ("angular_jerk", "motion", "command", "angular_jerk", "rms"),
        )
    }


def check_reset(data, model):
    config = data["c"]
    cls = pipeline_class(objective="tcp", stop_tail=True)
    executor = study.DirectExecutor(config["executor"]["motion_limits"], 0.01)
    pipe = cls(
        executor,
        kinematics=model.kinematics,
        ik_options=config["execution"]["ik"],
        control_dt_ns=10_000_000,
        motion_limits=config["executor"]["motion_limits"],
    )
    tick = next(iter(data["events"]))
    previous = {g: data["arrays"]["command"][0, j].copy() for j, g in enumerate(study.GROUPS)}
    state = RobotState(previous, int(data["times"][tick]), tick)
    executor.reset(state)
    reference = ActionHorizon(
        int(data["times"][tick]),
        10_000_000,
        str(data["ids"][tick]),
        {g: data["target"][tick, j][None] for j, g in enumerate(study.GROUPS)},
        action_space="absolute_tcp_pose",
    )
    pipe.resolver.budget_ns = 10**18
    result = pipe.step(state.monotonic_ns, state, reference, last_command=previous)
    assert result.ik_status["success"]
    previous = result.command.groups
    pipe.resolver.budget_ns = 1
    result = pipe.step(state.monotonic_ns + 10_000_000, state, reference, last_command=previous)
    assert result.ik_status["reason"] == "budget_exceeded"
    assert all(np.array_equal(result.command.groups[g], previous[g]) for g in study.GROUPS)
    assert all(
        solver.problem is None and solver.solution is None for solver in pipe.pilots.values()
    )
    assert all(np.count_nonzero(v) == 0 for v in executor._previous_velocity.values())
    return True


def main():
    cohort = [r for r in json.loads((study.OUT / "cohort.json").read_text()) if r["included"]]
    results = []
    reset_checked = False
    for row in cohort:
        paths = sorted((OUT / "episodes" / row["id"]).glob("*/checks.json"))
        if not paths:
            continue
        data = study.load(row)
        model = study.make_model(data["c"])
        base = dict(np.load(study.OUT / "episodes" / row["id"] / "baseline.npz"))
        arm = data["c"]["executor"]["motion_limits"]["arm"]
        dt = 1 / data["c"]["robot"]["control_hz"]
        if not reset_checked and arm["max_acceleration"] is not None:
            reset_checked = check_reset(data, model)
        for path in paths:
            checks = json.loads(path.read_text())
            trace = dict(np.load(path.parent / "trace.npz"))
            cut = min(first_hold(base), first_hold(trace))
            prefix = dict(data)
            for key in ("times", "target", "ids"):
                prefix[key] = data[key][:cut]
            before = study.measure(prefix, {k: v[:cut] for k, v in base.items()}, model)
            after = study.measure(prefix, {k: v[:cut] for k, v in trace.items()}, model)
            item = {
                "episode": row["id"],
                "task": row["task"],
                "status": row["status"],
                "split": row["split"],
                "stratum": checks["stratum"],
                "candidate": path.parent.name,
                "common_prefix_ticks": cut,
                "common_prefix_ratios": summarize_difference(before, after),
            }
            if checks["candidate"]["objective"] == "flange" and arm["max_acceleration"] is None:
                item["no_acceleration_bit_exact"] = all(
                    np.array_equal(trace[k], base[k]) for k in ["reference", "command"]
                )
                assert item["no_acceleration_bit_exact"]
            if checks["candidate"]["stop_tail"] and arm["max_acceleration"] is not None:
                active = trace["reason"] != "gap"
                command = trace["command"][:, :, :7]
                previous = np.concatenate([command[:1], command[:-1]])
                velocity = (command - previous) / dt
                rate_reserve = np.radians(1e-3)
                speed = arm["max_velocity"] * min(1, arm["max_step_dt_s"] / dt)
                time = max(
                    dt, (speed - rate_reserve) / (arm["max_acceleration"] - rate_reserve / dt)
                )
                terminal = previous + time * velocity
                minima = []
                for j, g in enumerate(study.GROUPS):
                    lower, upper = model.kinematics.models[g].arm.joint_position_limits()
                    margin = np.radians(data["c"]["execution"]["ik"]["diff_ik"]["limit_margin_deg"])
                    q = terminal[active, j]
                    joint_clearance = np.min(np.minimum(q - lower - margin, upper - margin - q))
                    j67_clearance = np.min(
                        110.5
                        - data["c"]["execution"]["ik"]["diff_ik"]["j67_margin_deg"]
                        - 1.025 * abs(np.degrees(q[:, 5]))
                        - abs(np.degrees(q[:, 6]))
                    )
                    minima.append(
                        {
                            "arm": g,
                            "joint_clearance_rad": float(joint_clearance),
                            "j67_clearance_deg": float(j67_clearance),
                        }
                    )
                    assert joint_clearance >= -1e-9 and j67_clearance >= -1e-7
                item["braking_envelope_checks"] = minima
                assert checks["reference_command_max_abs_joint_rad"] < 1e-9
                assert (
                    checks["nominal_command_max_acceleration_rad_s2"]
                    <= arm["max_acceleration"] + 1e-6
                )
                assert checks["nominal_command_max_speed_rad_s"] <= speed + 1e-9
                assert not checks["command_safety_violations"]
            results.append(item)
    summary = []
    for task in ("pack_plate", "pack_bottle"):
        for split in ("development", "holdout"):
            for candidate in ("acceleration_qp", "stopping_qp", "tcp_stopping_qp"):
                selected = [
                    r
                    for r in results
                    if r["task"] == task
                    and r["candidate"] == candidate
                    and ((r["split"] == "holdout") == (split == "holdout"))
                ]
                if selected:
                    summary.append(
                        {
                            "task": task,
                            "split": split,
                            "candidate": candidate,
                            "mean_ratios": {
                                k: float(np.mean([r["common_prefix_ratios"][k] for r in selected]))
                                for k in selected[0]["common_prefix_ratios"]
                            },
                            "worst_ratios": {
                                k: float(np.max([r["common_prefix_ratios"][k] for r in selected]))
                                for k in selected[0]["common_prefix_ratios"]
                            },
                        }
                    )
    study.save(
        OUT / "verification.json",
        {
            "atomic_budget_hold_and_reset": reset_checked,
            "scope": "Offline command checks; no changed-state or real deadline claim.",
            "common_pre_hold_summary": summary,
            "episodes": results,
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
