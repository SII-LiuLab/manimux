#!/usr/bin/env python3
"""Fixed user-requested pi rad/s and 80 rad/s^2 offline step comparison.

Single factors precede joint changes. This does not change production configs or
historical guards. Full generated trajectories after a historical guard rejection
are conditional command-generation diagnostics, not executable historical runs.
No changed measured state, hardware controller or policy service is simulated.
"""

import argparse
import json
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy

import numpy as np
import run as study
from stopping_qp import pipeline_class

from manimux.runtime.execution import ExecutionPipeline

OUT = study.OUT / "pi80_review"
CHOICES = {
    "velocity_pi_only": {"velocity": math.pi, "objective": "flange", "braking": False},
    "acceleration_80_only": {"acceleration": 80.0, "objective": "flange", "braking": False},
    "legacy_flange_pi80": {
        "velocity": math.pi,
        "acceleration": 80.0,
        "objective": "flange",
        "braking": False,
    },
    "legacy_tcp_pi80": {
        "velocity": math.pi,
        "acceleration": 80.0,
        "objective": "tcp",
        "braking": False,
    },
    "braking_tcp_pi80": {
        "velocity": math.pi,
        "acceleration": 80.0,
        "objective": "tcp",
        "braking": True,
    },
}


def trial(args):
    row, name = args
    destination = OUT / "episodes" / row["id"] / name
    if (destination / "checks.json").exists():
        return row["id"], name, "cached"
    choice = CHOICES[name]
    data = study.load(row)
    frozen_motion = deepcopy(data["c"]["executor"]["motion_limits"]["arm"])
    data["c"] = deepcopy(data["c"])
    motion = data["c"]["executor"]["motion_limits"]["arm"]
    if "velocity" in choice:
        motion["max_velocity"] = choice["velocity"]
    if "acceleration" in choice:
        motion["max_acceleration"] = choice["acceleration"]
    if choice["braking"]:
        cls = pipeline_class(objective=choice["objective"], stop_tail=True)
        study.ExecutionPipeline = cls
        changes = {}
    else:
        study.ExecutionPipeline = ExecutionPipeline
        changes = {"study.objective": choice["objective"]}
    trace, failures, model = study.simulate(data, changes)
    metric = study.measure(data, trace, model)
    active = trace["reason"] != "gap"
    dt = 1 / data["c"]["robot"]["control_hz"]
    command = trace["command"][:, :, :7]
    previous = np.concatenate([command[:1], command[:-1]])
    velocity = (command - previous) / dt
    acceleration = np.diff(velocity, axis=0, prepend=np.zeros_like(velocity[:1])) / dt
    historical_guard = np.array(
        [data["c"]["executor"]["command_safety"]["max_velocity"][g][:7] for g in study.GROUPS]
    )
    guard_mask = active & np.any(abs(velocity) > historical_guard + 1e-9, axis=(1, 2))
    checks = {
        "task": row["task"],
        "status": row["status"],
        "split": row["split"],
        "stratum": data["info"]["stratum"],
        "choice": choice,
        "frozen_motion": frozen_motion,
        "candidate_motion": motion,
        "manifest_sha256": data["info"]["manifest_sha"],
        "stream_sha256": data["info"]["stream_sha256"],
        "source_sha256": {
            p: study.sha(study.ROOT / "scripts/validation/step_diffik_study" / p)
            for p in ["check_pi80.py", "run.py", "stopping_qp.py", "tcp_objective.py"]
        },
        "reasons": metric["counts"]["reasons"],
        "historical_guard_rejected_ticks": int(guard_mask.sum()),
        "first_historical_velocity_guard_rejection": int(np.flatnonzero(guard_mask)[0])
        if guard_mask.any()
        else None,
        "historical_command_safety_violations": metric["command_safety_violations"],
        "pi_velocity_excess_ticks": int(
            (active & np.any(abs(velocity) > math.pi + 1e-9, axis=(1, 2))).sum()
        ),
        "acceleration_80_excess_ticks_including_hold": int(
            (active & np.any(abs(acceleration) > 80 + 1e-6, axis=(1, 2))).sum()
        ),
        "max_nominal_speed_rad_s": float(np.max(abs(velocity[active]))),
        "max_nominal_acceleration_rad_s2": float(np.max(abs(acceleration[active]))),
        "reference_command_max_joint_rad": float(
            np.max(abs(trace["reference"][active, :, :7] - command[active]))
        ),
        "joint_bounds": {},
    }
    for j, g in enumerate(study.GROUPS):
        lower, upper = model.kinematics.models[g].arm.joint_position_limits()
        margin = data["c"]["execution"]["ik"]["diff_ik"]["limit_margin_deg"]
        q = command[active, j]
        checks["joint_bounds"][g] = {
            "min_inner_margin_deg": float(
                np.min(np.degrees(np.minimum(q - lower, upper - q))) - margin
            ),
            "min_j67_buffer_deg": float(
                np.min(110.5 - 1.025 * abs(np.degrees(q[:, 5])) - abs(np.degrees(q[:, 6])))
            ),
        }
    # Common pre-first-hold window avoids rewarding a baseline's long zero-motion tail.
    base = dict(np.load(study.OUT / "episodes" / row["id"] / "baseline.npz"))

    def first_hold(t):
        indices = np.flatnonzero(~np.isin(t["reason"], ["ok", "gap"]))
        return int(indices[0]) if len(indices) else len(t["reason"])

    cut = min(first_hold(base), first_hold(trace))
    prefix = dict(data)
    for key in ("times", "target", "ids"):
        prefix[key] = data[key][:cut]
    base_prefix = study.measure(prefix, {k: v[:cut] for k, v in base.items()}, model)
    candidate_prefix = study.measure(prefix, {k: v[:cut] for k, v in trace.items()}, model)
    from verify_stopping_qp import summarize_difference

    checks["common_pre_first_hold_ticks"] = cut
    checks["common_pre_first_hold_ratios"] = summarize_difference(base_prefix, candidate_prefix)
    study.save(destination / "metrics.json", metric)
    study.save(destination / "checks.json", checks)
    study.save(destination / "failures.json", failures)
    np.savez_compressed(destination / "trace.npz", **trace)
    return row["id"], name, checks["reasons"], checks["historical_guard_rejected_ticks"]


def report():
    from verify_stopping_qp import summarize_difference

    cohort = [x for x in json.loads((study.OUT / "cohort.json").read_text()) if x["included"]]
    reports = []
    for task in ("pack_plate", "pack_bottle"):
        for split in ("development", "historical_holdout_regression"):
            rows = [
                r
                for r in cohort
                if r["task"] == task and ((r["split"] == "holdout") == (split != "development"))
            ]
            for name in CHOICES:
                entries, totals = [], Counter()
                for row in rows:
                    dest = OUT / "episodes" / row["id"] / name
                    if not (dest / "checks.json").exists():
                        continue
                    check = json.loads((dest / "checks.json").read_text())
                    base = json.loads(
                        (
                            study.OUT / "episodes" / row["id"] / "baseline_sim_metrics.json"
                        ).read_text()
                    )
                    candidate = json.loads((dest / "metrics.json").read_text())
                    ratios = summarize_difference(base, candidate)
                    totals.update(check["reasons"])
                    entries.append(
                        {
                            "episode": row["id"],
                            "status": row["status"],
                            "stratum": check["stratum"],
                            "ratios": ratios,
                            "prefix_ratios": check["common_pre_first_hold_ratios"],
                            "historical_guard_rejected_ticks": check[
                                "historical_guard_rejected_ticks"
                            ],
                            "checks": str(dest / "checks.json"),
                        }
                    )
                if entries:
                    reports.append(
                        {
                            "task": task,
                            "split": split,
                            "candidate": name,
                            "reasons": dict(totals),
                            "historical_guard_rejected_ticks": sum(
                                r["historical_guard_rejected_ticks"] for r in entries
                            ),
                            "mean_ratios": {
                                k: float(np.mean([r["ratios"][k] for r in entries]))
                                for k in entries[0]["ratios"]
                            },
                            "common_pre_first_hold_mean_ratios": {
                                k: float(np.mean([r["prefix_ratios"][k] for r in entries]))
                                for k in entries[0]["ratios"]
                            },
                            "episodes": entries,
                        }
                    )
    study.save(OUT / "summary.json", reports)
    for row in reports:
        print({k: v for k, v in row.items() if k != "episodes"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["single", "joint", "report"])
    args = parser.parse_args()
    if args.phase == "report":
        report()
        return
    study.save(
        OUT / "fixed_design.json",
        {
            "choices": CHOICES,
            "selection": "User-specified fixed rate check; no candidate tuning or selection.",
            "holdout": "Previously viewed holdout sessions are regression only in this follow-up.",
            "guards": (
                "Historical guards unchanged; rejection points reported. "
                "Full post-rejection traces are conditional generator output."
            ),
            "pose_limits_handoff_physical_bounds": "Historical values unchanged.",
        },
    )
    rows = [r for r in json.loads((study.OUT / "cohort.json").read_text()) if r["included"]]
    names = list(CHOICES)[:2] if args.phase == "single" else list(CHOICES)[2:]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(trial, (row, name)) for row in rows for name in names]
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
