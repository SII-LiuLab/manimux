#!/usr/bin/env python3
# ruff: noqa: E402, E501
"""Offline step DiffIK study. Never instantiate a robot or policy client.

Run from repo root: .venv/bin/python scripts/validation/step_diffik_study/run.py baseline
All derived files live under data/analysis/step_diffik_study/20261019.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

import numpy as np
import zarr
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = Path(os.environ.get("STEP_DIFFIK_SOURCE_ROOT", str(ROOT)))
sys.path.insert(0, str(CODE_ROOT))
from manimux.embodiments.robot.base import RobotModel
from manimux.kinematics.poses import matrix_pose
from manimux.runtime.execution import ExecutionPipeline
from manimux.runtime.executors.direct import DirectExecutor
from manimux.runtime.timeline import _ActivePlan, _sample_plan
from manimux.types import ActionHorizon, RobotState

SOURCE = ROOT / "data/experiments/umi-dp-bottle-plate-scratch-tf32-20261017"
OUT = Path(
    os.environ.get("STEP_DIFFIK_OUTPUT", str(ROOT / "data/analysis/step_diffik_study/20261019"))
)
GROUPS = ["left_arm", "right_arm"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def ident(path):
    return path.parent.name + "__" + path.name


def make_model(c):
    path = Path(c["robot"]["config"])
    if path.is_relative_to(ROOT):
        path = CODE_ROOT / path.relative_to(ROOT)
    return RobotModel.from_config(path)


def discover():
    cohort = []
    for task in ["pack_plate", "pack_bottle"]:
        paths = sorted(
            SOURCE.glob(f"{task}_*async_step/session-*/rollout*/meta.json"), reverse=True
        )
        count = 0
        sessions = sorted(
            {
                p.parent.parent.name
                for p in paths
                if json.loads((p.parent / "result.json").read_text())["steps"] > 0
            }
        )
        for p in paths:
            p = p.parent
            result = json.loads((p / "result.json").read_text())
            included = result["steps"] > 0 and count < 20
            if included:
                count += 1
            cohort.append(
                dict(
                    path=str(p),
                    id=ident(p),
                    task=task,
                    session=p.parent.name,
                    included=included,
                    reason="eligible" if included else "zero_ticks_or_cohort_cap",
                    status="complete"
                    if result["run_completed"] and not p.name.endswith(".partial")
                    else "aborted",
                    split=(
                        "development_no_independent_holdout"
                        if len(sessions) == 1
                        else "holdout"
                        if p.parent.name == sessions[-1]
                        else "development"
                    ),
                    result=result,
                    success_label=json.loads((p / "evaluation/human-label.json").read_text())
                    if (p / "evaluation/human-label.json").exists()
                    else None,
                )
            )
    return cohort


def load(row):
    path = Path(row["path"])
    manifest = json.loads((path.parent / "session-manifest.json").read_text())
    c = manifest["config"]
    assert c["execution"]["ik"]["mode"] == "step" and c["execution"]["ik"]["backend"] == "diff"
    assert c["inference"]["blend_policy_steps"] == 0
    z = zarr.open(str(path / "data.zarr"), "r")
    events = [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]
    n = len(z["ticks/monotonic_ns"])
    times = z["ticks/monotonic_ns"][:]
    plans, audits = {}, []
    for k in z["plans"]:
        p = z["plans"][k]
        a = dict(p["committed"].attrs)
        groups = {g: p["committed"][g][:] for g in GROUPS}
        plans[a["plan_id"]] = _ActivePlan(
            a["plan_id"],
            p.attrs["request_seq"],
            a["start_time_ns"],
            a["dt_ns"],
            groups,
            action_space="absolute_tcp_pose",
        )
        trim = a["time_trimmed_steps"]
        skip = a["handoff_skipped_steps"]
        source_offset = p["infra_output"].attrs.get("source_offset_steps", 0)
        match = all(
            np.array_equal(
                p["infra_output"][g][trim + skip : trim + skip + len(groups[g])], groups[g]
            )
            for g in GROUPS
        )
        audits.append(
            dict(
                plan_id=a["plan_id"],
                time_trim=trim,
                handoff_skip=skip,
                raw_to_committed_exact=match,
                clock_exact=a["start_time_ns"]
                == a["observation_time_ns"] + (source_offset + trim) * a["dt_ns"],
            )
        )
    step_events = {
        e["step"]: e
        for e in events
        if e["kind"] in ["step_diff_ik", "step_ik_hold"] and e["step"] < n
    }
    target = np.full((n, 2, 8), np.nan)
    ids = z["ticks/plan_id"][:]
    for i, e in step_events.items():
        sample = _sample_plan(plans[ids[i]], int(times[i]))
        assert sample is not None, (path, i, ids[i])
        target[i] = [sample[g] for g in GROUPS]
        if "monotonic_ns" in e:
            assert e["monotonic_ns"] == times[i]
    arrays = {
        key: np.stack([z[f"ticks/{key}/{g}"][:] for g in GROUPS], axis=1)
        for key in ["state", "command", "scheduled"]
    }
    source_files = manifest["source_provenance"]["repositories"]["manimux"]["files"]
    names = [
        "manimux/runtime/edge.py",
        "manimux/runtime/ik.py",
        "manimux/runtime/execution.py",
        "manimux/runtime/pose_limits.py",
        "manimux/runtime/timeline.py",
        "manimux/runtime/executors/direct.py",
        "manimux/runtime/executors/limits.py",
        "manimux/kinematics/tianji_diff.py",
        "manimux/kinematics/composed.py",
        "manimux/kinematics/poses.py",
        "manimux/embodiments/arm/tianji/kinematics.py",
        "manimux/embodiments/robot/base.py",
        "manimux/embodiments/end_effector/taccap/geometry.py",
        "manimux/configs/embodiment/robot/tianji_taccap_pack_plate.yaml",
        "manimux/configs/embodiment/arm/tianji_left.yaml",
        "manimux/configs/embodiment/arm/tianji_right.yaml",
        "manimux/configs/embodiment/end_effector/taccap.yaml",
    ]
    provenance = {
        name: dict(
            recorded=source_files.get(name, {}).get("sha256"),
            current=sha(CODE_ROOT / name) if (CODE_ROOT / name).exists() else None,
        )
        for name in names
    }
    info = dict(
        row,
        config=c,
        git_sha=manifest["git_sha"],
        entry_config_sha=manifest["config_sha256"],
        resolved_config_sha=hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest(),
        inputs_sha={k: sha(path / k) for k in ["meta.json", "result.json", "events.jsonl"]},
        manifest_sha=sha(path.parent / "session-manifest.json"),
        source_files=provenance,
        plan_audit=audits,
        event_counts=dict(Counter(e["kind"] for e in events)),
        historical_hold_reasons=dict(
            Counter(e["reason"] for e in step_events.values() if not e["success"])
        ),
        events_without_recorded_tick=[
            e for e in events if e["kind"] in ["step_diff_ik", "step_ik_hold"] and e["step"] >= n
        ],
        stream_sha256={
            k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
            for k, v in dict(arrays, target=target, times=times).items()
        },
        terminal_events=[e for e in events if e["kind"] in ["episode_aborted", "episode_finished"]],
        stratum=f"v{c['executor']['motion_limits']['arm']['max_velocity']}_a{c['executor']['motion_limits']['arm']['max_acceleration']}_refill{c['inference']['refill_threshold_s']}",
    )
    return dict(
        info=info, c=c, arrays=arrays, target=target, times=times, ids=ids, events=step_events
    )


def simulate(d, changes=None, teacher=False):
    c = deepcopy(d["c"])
    changes = dict(changes or {})
    objective = changes.pop("study.objective", "flange")
    for key, value in changes.items():
        at = c
        for part in key.split(".")[:-1]:
            at = at[part]
        at[key.split(".")[-1]] = value
    model = make_model(c)
    dt = 1 / c["robot"]["control_hz"]
    n = len(d["times"])
    executor = DirectExecutor(c["executor"]["motion_limits"], dt)
    pipe = ExecutionPipeline(
        executor,
        kinematics=model.kinematics,
        ik_options=c["execution"]["ik"],
        control_dt_ns=round(dt * 1e9),
        motion_limits=c["executor"]["motion_limits"],
    )
    if objective == "tcp":
        from tcp_objective import TcpStepDifferentialIK

        from manimux.kinematics.tianji_diff import step_diff_config

        pipe.resolver.diff = TcpStepDifferentialIK(
            model.kinematics.models,
            step_diff_config(c["executor"]["motion_limits"], c["execution"]["ik"]["diff_ik"]),
            dt,
        )
    elif objective != "flange":
        raise ValueError(objective)
    # Deterministic mathematical replay: host scheduling must not invent deadline failures.
    # Recorded budget failures are injected below; local elapsed times are reference only.
    pipe.resolver.budget_ns = 10**18
    command = np.empty((n, 2, 8))
    reference = np.empty_like(command)
    limited = np.full_like(command, np.nan)
    fractions = np.ones((n, 2))
    elapsed = np.zeros(n)
    reason = np.full(n, "gap", dtype="U80")
    saturation = np.zeros((n, 2, 7), bool)
    lag = np.zeros((n, 2), bool)
    details = []
    previous = {g: d["arrays"]["command"][0, j].copy() for j, g in enumerate(GROUPS)}
    executor.reset(RobotState(previous, int(d["times"][0]), 0))
    if pipe.pose_limiter:
        apply = pipe.pose_limiter.apply

        def capture(ref, seed):
            result, fr = apply(ref, seed)
            for j, g in enumerate(GROUPS):
                limited[index, j] = result.groups[g][0]
                fractions[index, j] = fr.get(g, 1.0)
            return result, fr

        pipe.pose_limiter.apply = capture
    solve = pipe.resolver.diff.solve

    def capture_solve(name, target, seed, gripper):
        r = solve(name, target, seed, gripper)
        j = GROUPS.index(name)
        if r.qdot_rad_s is not None:
            saturation[index, j] = np.abs(r.qdot_rad_s) >= c["executor"]["motion_limits"]["arm"][
                "max_velocity"
            ] * (1 - 1e-5)
        lag[index, j] = r.lag_exceeded
        if not r.ok and (
            not details or details[-1]["reason"] != r.reason or details[-1]["index"] != index - 1
        ):
            details.append(
                dict(index=index, group=name, reason=r.reason, detail=r.detail, seed=seed.tolist())
            )
        return r

    pipe.resolver.diff.solve = capture_solve
    for index, now in enumerate(d["times"]):
        if teacher and index:
            previous = {
                g: d["arrays"]["command"][index - 1, j].copy() for j, g in enumerate(GROUPS)
            }
            executor._previous = deepcopy(previous)
            executor._previous_velocity = {
                g: (
                    (d["arrays"]["command"][index - 1, j] - d["arrays"]["command"][index - 2, j])
                    / dt
                    if index > 1
                    else np.zeros(8)
                )
                for j, g in enumerate(GROUPS)
            }
        state = RobotState(deepcopy(previous), int(now), index)
        if index not in d["events"]:
            pipe.reset()
            executor.reset(state)
            command[index] = reference[index] = [previous[g] for g in GROUPS]
            continue
        ref = ActionHorizon(
            start_time_ns=int(now),
            dt_ns=round(dt * 1e9),
            plan_id=str(d["ids"][index]),
            groups={g: d["target"][index, j][None, :] for j, g in enumerate(GROUPS)},
            action_space="absolute_tcp_pose",
        )
        if pipe.pose_limiter is None:
            limited[index] = d["target"][index]
        started = time.perf_counter()
        result = pipe.step(int(now), state, ref, last_command=previous)
        elapsed[index] = (time.perf_counter() - started) * 1000
        reason[index] = result.ik_status["reason"]
        if d["events"][index]["reason"] == "budget_exceeded":
            pipe.reset()
            executor.reset(state)
            command[index] = reference[index] = [previous[g] for g in GROUPS]
            reason[index] = "historical_budget_exceeded"
        else:
            command[index] = [result.command.groups[g] for g in GROUPS]
            reference[index] = [result.reference.groups[g][0] for g in GROUPS]
        previous = {g: command[index, j].copy() for j, g in enumerate(GROUPS)}
    return (
        dict(
            command=command,
            reference=reference,
            limited=limited,
            fraction=fractions,
            elapsed_ms=elapsed,
            reason=reason,
            saturation=saturation,
            lag=lag,
        ),
        details,
        model,
    )


def stats(values):
    a = np.asarray(values)
    a = a[np.isfinite(a)]
    if not len(a):
        return None
    return dict(
        n=len(a),
        rms=float(np.sqrt(np.mean(a * a))),
        mean=float(np.mean(a)),
        p95=float(np.percentile(a, 95)),
        max=float(np.max(a)),
    )


def poses(model, q):
    out = np.full(q.shape, np.nan)
    for i, j in np.argwhere(np.isfinite(q[:, :, 0])):
        out[i, j, :7] = matrix_pose(model.kinematics.models[GROUPS[j]].fk(q[i, j]))
        out[i, j, 7] = q[i, j, 7]
    return out


def error(a, b):
    translation = np.linalg.norm(a[:, :, :3] - b[:, :, :3], axis=-1) * 1000
    rotation = np.degrees(
        2 * np.arccos(np.clip(np.abs(np.sum(a[:, :, 3:7] * b[:, :, 3:7], axis=-1)), 0, 1))
    )
    return translation, rotation


def derivatives(p, t):
    dt = np.diff(t) / 1e9
    xyz = np.diff(p[:, :, :3], axis=0) / dt[:, None, None]
    w = np.full_like(xyz, np.nan)
    for j in range(2):
        valid = np.isfinite(p[:-1, j, 0]) & np.isfinite(p[1:, j, 0])
        a = p[:-1, j, 3:7][valid]
        b = p[1:, j, 3:7][valid]
        ra = Rotation.from_quat(a[:, [1, 2, 3, 0]])
        rb = Rotation.from_quat(b[:, [1, 2, 3, 0]])
        w[valid, j] = (rb * ra.inv()).as_rotvec() / dt[valid, None]
    result = {}
    for name, v in [("linear", xyz), ("angular", w)]:
        a = np.diff(v, axis=0) / ((dt[1:] + dt[:-1]) / 2)[:, None, None]
        j = np.diff(a, axis=0) / ((dt[2:] + 2 * dt[1:-1] + dt[:-2]) / 4)[:, None, None]
        for order, x in [("speed", v), ("acceleration", a), ("jerk", j)]:
            pad = np.full((len(t), 2), np.nan)
            pad[len(t) - len(x) :] = np.linalg.norm(x, axis=-1)
            result[f"{name}_{order}"] = pad
    return result


def measure(d, r, model, historical=False):
    active = np.isfinite(d["target"][:, 0, 0])
    t = d["times"]
    seam = np.zeros(len(t), bool)
    changes = np.flatnonzero(
        active & (np.r_[True, d["ids"][1:] != d["ids"][:-1]] | ~np.r_[False, active[:-1]])
    )
    for i in changes:
        seam |= (t >= t[i]) & (t < t[i] + 100_000_000)
    stages = {
        "target": d["target"],
        "limited": r["limited"],
        "reference": poses(model, r["reference"]),
        "command": poses(model, r["command"]),
    }
    if historical:
        stages["state"] = poses(model, d["arrays"]["state"])
    pairs = [
        ("target", "limited"),
        ("limited", "reference"),
        ("reference", "command"),
        ("target", "command"),
    ]
    if historical:
        pairs += [("command", "state"), ("target", "state")]
    errors = {f"{a}_to_{b}": error(stages[a], stages[b]) for a, b in pairs}
    motions = {k: derivatives(v, t) for k, v in stages.items()}
    result = {}
    for region, mask in [
        ("all", active),
        ("handoff_100ms", active & seam),
        ("stable", active & ~seam),
        ("stable_success", active & ~seam & (r["reason"] == "ok")),
        ("ik_hold", active & (r["reason"] != "ok")),
    ]:
        result[region] = dict(ticks=int(mask.sum()), errors={}, motion={})
        for name, (pos, rot) in errors.items():
            result[region]["errors"][name] = {
                g: dict(translation_mm=stats(pos[mask, j]), rotation_deg=stats(rot[mask, j]))
                for j, g in enumerate(GROUPS)
            }
        for name, der in motions.items():
            result[region]["motion"][name] = {
                g: {k: stats(v[mask, j]) for k, v in der.items()} for j, g in enumerate(GROUPS)
            }
    result["counts"] = dict(
        reasons=dict(Counter(r["reason"])),
        pose_limited={g: int((r["fraction"][active, j] < 1).sum()) for j, g in enumerate(GROUPS)},
        diff_velocity_saturated_by_joint={
            g: r["saturation"][active, j].sum(axis=0).tolist() for j, g in enumerate(GROUPS)
        },
        diff_lag_flags={g: int(r["lag"][active, j].sum()) for j, g in enumerate(GROUPS)},
        executor_shaped={
            g: int(
                (
                    np.max(
                        np.abs(r["reference"][active, j, :7] - r["command"][active, j, :7]), axis=1
                    )
                    > 1e-9
                ).sum()
            )
            for j, g in enumerate(GROUPS)
        },
        offline_elapsed_ms=stats(r["elapsed_ms"][active]),
        tick_dt_ms=stats(np.diff(t) / 1e6),
    )
    safety = d["c"]["executor"]["command_safety"]
    violations = []
    for j, g in enumerate(GROUPS):
        q = r["command"][:, j]
        velocity = np.diff(q, axis=0) * d["c"]["robot"]["control_hz"]
        for label, mask in [
            (
                "position",
                np.any(
                    (q < np.array(safety["position_lower"][g]) - 1e-3)
                    | (q > np.array(safety["position_upper"][g]) + 1e-3),
                    axis=1,
                ),
            ),
            (
                "velocity",
                np.any(np.abs(velocity) > np.array(safety["max_velocity"][g]) + 1e-9, axis=1),
            ),
        ]:
            indices = np.flatnonzero(mask)
            if len(indices):
                violations.append(
                    dict(
                        group=g,
                        guard=label,
                        first_tick=int(indices[0] + (label == "velocity")),
                        count=len(indices),
                    )
                )
    result["command_safety_violations"] = violations
    return result


def run_baseline(rows):
    for row in rows:
        if not row["included"]:
            continue
        d = load(row)
        dest = OUT / "episodes" / row["id"]
        save(dest / "identity.json", d["info"])
        r, detail, model = simulate(d)
        teacher, _, _ = simulate(d, teacher=True)
        active = np.isfinite(d["target"][:, 0, 0])
        recorded = np.array([d["events"][i]["reason"] for i in np.flatnonzero(active)])
        parity = {
            k: dict(
                max_joint_rad=float(
                    np.max(np.abs(v["command"][:, :, :7] - d["arrays"]["command"][:, :, :7]))
                ),
                max_gripper=float(
                    np.max(np.abs(v["command"][:, :, 7] - d["arrays"]["command"][:, :, 7]))
                ),
                max_reference_rad=float(
                    np.max(
                        np.abs(
                            v["reference"][active, :, :7] - d["arrays"]["scheduled"][active, :, :7]
                        )
                    )
                ),
                verdict_mismatch=int(np.sum(v["reason"][active] != recorded)),
            )
            for k, v in [("free_run", r), ("historical_seed", teacher)]
        }
        save(dest / "baseline_parity.json", parity)
        save(dest / "baseline_failures.json", detail)
        np.savez_compressed(dest / "baseline.npz", **r, target=d["target"], times=d["times"])
        save(dest / "baseline_sim_metrics.json", measure(d, r, model))
        # Historical measured state is only paired with historical references/commands.
        h = dict(
            r,
            command=d["arrays"]["command"],
            reference=d["arrays"]["scheduled"],
            limited=teacher["limited"],
            fraction=teacher["fraction"],
            reason=np.array(
                [d["events"].get(i, {}).get("reason", "gap") for i in range(len(active))]
            ),
        )
        save(dest / "historical_metrics.json", measure(d, h, model, historical=True))
        print(row["id"], parity, flush=True)


ARM = "executor.motion_limits.arm."
IK = "execution.ik."
SINGLES = {
    "speed_030": {ARM + "max_velocity": 0.3},
    "speed_060": {ARM + "max_velocity": 0.6},
    "speed_090": {ARM + "max_velocity": 0.9},
    "accel_10": {ARM + "max_acceleration": 10.0},
    "accel_40": {ARM + "max_acceleration": 40.0},
    "accel_none": {ARM + "max_acceleration": None},
    "linear_020": {IK + "pose_speed_limit.max_linear_m_s": 0.2},
    "linear_080": {IK + "pose_speed_limit.max_linear_m_s": 0.8},
    "angular_090": {IK + "pose_speed_limit.max_angular_rad_s": 0.9},
    "angular_360": {IK + "pose_speed_limit.max_angular_rad_s": 3.6},
    "pose_none": {IK + "pose_speed_limit": None},
    "dtcap_008": {ARM + "max_step_dt_s": 0.008},
    "dtcap_010": {ARM + "max_step_dt_s": 0.010},
    "dtcap_032": {ARM + "max_step_dt_s": 0.032},
    "per_joint": {ARM + "mode": "per_joint"},
    "lambda_001": {IK + "diff_ik.lam": 0.01},
    "lambda_010": {IK + "diff_ik.lam": 0.1},
    "nullspace_0": {IK + "diff_ik.mu_nullspace": 0.0},
    "nullspace_5000": {IK + "diff_ik.mu_nullspace": 5000.0},
    "nullspace_50000": {IK + "diff_ik.mu_nullspace": 50000.0},
    "ratio_32_fixed_runtime": {"robot.options.hardware.velocity_ratio": 32},
}


def trial(args):
    row, name, changes = args
    dest = OUT / "episodes" / row["id"] / "candidates" / name
    if (dest / "metrics.json").exists():
        return row["id"], name, "cached"
    d = load(row)
    r, detail, model = simulate(d, changes)
    metric = measure(d, r, model)
    save(dest / "metrics.json", metric)
    save(dest / "parameters.json", changes)
    save(dest / "failures.json", detail)
    # Keep every candidate's endogenous command, newly solved reference and limiter input.
    np.savez_compressed(dest / "trace.npz", **r)
    return row["id"], name, metric["counts"]["reasons"]


def batch(rows, choices):
    jobs = [(row, name, changes) for row in rows for name, changes in choices.items()]
    with ProcessPoolExecutor(max_workers=2) as pool:
        pending = [pool.submit(trial, x) for x in jobs]
        for f in as_completed(pending):
            print(*f.result(), flush=True)


def refresh(row):
    d = load(row)
    dest = OUT / "episodes" / row["id"]
    model = make_model(d["c"])
    save(dest / "identity.json", d["info"])
    r = dict(np.load(dest / "baseline.npz"))
    save(dest / "baseline_sim_metrics.json", measure(d, r, model))
    h = dict(r, command=d["arrays"]["command"], reference=d["arrays"]["scheduled"])
    save(dest / "historical_metrics.json", measure(d, h, model, historical=True))
    for p in sorted((dest / "candidates").glob("*/trace.npz")):
        r = dict(np.load(p))
        save(p.parent / "metrics.json", measure(d, r, model))
    return row["id"]


def score(row, name):
    dest = OUT / "episodes" / row["id"]
    b = json.loads((dest / "baseline_sim_metrics.json").read_text())
    m = (
        b
        if name == "baseline"
        else json.loads((dest / "candidates" / name / "metrics.json").read_text())
    )
    ratios = {}
    for quantity, source, key, stat in [
        ("position", "errors", "translation_mm", "rms"),
        ("position_p95", "errors", "translation_mm", "p95"),
        ("rotation", "errors", "rotation_deg", "rms"),
        ("rotation_p95", "errors", "rotation_deg", "p95"),
        ("speed", "motion", "linear_speed", "p95"),
        ("acceleration", "motion", "linear_acceleration", "rms"),
        ("jerk", "motion", "linear_jerk", "rms"),
        ("angular_speed", "motion", "angular_speed", "p95"),
        ("angular_acceleration", "motion", "angular_acceleration", "rms"),
        ("angular_jerk", "motion", "angular_jerk", "rms"),
    ]:
        stage = "target_to_command" if source == "errors" else "command"
        ratios[quantity] = [
            m["all"][source][stage][g][key][stat] / max(b["all"][source][stage][g][key][stat], 1e-9)
            for g in GROUPS
        ]

    def holds(x):
        return sum(v for k, v in x["counts"]["reasons"].items() if k not in ["ok", "gap"])

    return dict(
        episode=row["id"],
        status=row["status"],
        stratum=json.loads((dest / "identity.json").read_text())["stratum"],
        ratios=ratios,
        holds=holds(m),
        baseline_holds=holds(b),
        safety=m.get("command_safety_violations", []),
    )


def ranking(rows, names):
    ranked = []
    for name in names:
        entries = [score(row, name) for row in rows]
        means = {
            k: float(np.mean([v for x in entries for v in x["ratios"][k]]))
            for k in entries[0]["ratios"]
        }
        worst = {k: max(v for x in entries for v in x["ratios"][k]) for k in entries[0]["ratios"]}
        safe = not any(e["safety"] for e in entries)
        holds = sum(e["holds"] for e in entries)
        baseholds = sum(e["baseline_holds"] for e in entries)
        passes = (
            safe
            and all(e["holds"] <= e["baseline_holds"] for e in entries)
            and all(
                worst[k] <= 1.05 for k in ["position", "position_p95", "rotation", "rotation_p95"]
            )
            and all(
                worst[k] <= 1.10
                for k in [
                    "speed",
                    "acceleration",
                    "jerk",
                    "angular_speed",
                    "angular_acceleration",
                    "angular_jerk",
                ]
            )
        )
        ranked.append(
            dict(
                candidate=name,
                passes=passes,
                safe=safe,
                holds=holds,
                baseline_holds=baseholds,
                mean_ratios=means,
                worst_ratios=worst,
                episodes=entries,
                objective=(means["position"] + means["rotation"]) / 2,
            )
        )
    return sorted(ranked, key=lambda x: (not x["passes"], x["objective"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=[
            "baseline",
            "single",
            "rank-single",
            "joint",
            "tcp",
            "tcp-joint",
            "holdout",
            "refresh-metrics",
        ],
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = discover()
    save(OUT / "cohort.json", rows)
    save(
        OUT / "environment.json",
        dict(
            python=sys.version,
            numpy=np.__version__,
            scipy=__import__("scipy").__version__,
            osqp=__import__("osqp").__version__,
            git_head=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            script_sha=sha(__file__),
        ),
    )
    eligible = [r for r in rows if r["included"]]
    development = [r for r in eligible if r["split"] != "holdout"]
    if args.phase == "baseline":
        run_baseline(rows)
    elif args.phase == "single":
        save(OUT / "single_factor_space.json", SINGLES)
        save(
            OUT / "selection_protocol.json",
            dict(
                written_before_search=True,
                split="Latest plate session held out entirely; bottle has one session and no independent holdout.",
                selection="Equal episode and arm weighting. Require every development episode/arm position and rotation RMS/p95 <=1.05 baseline, command linear/angular speed p95 and acceleration/jerk RMS <=1.10; holds must not increase per episode and command guards must pass. Minimize mean position/rotation error ratio among survivors. Keep complete/aborted/stratum results separate.",
                joint="Combine at most three promising distinct single factors from development; freeze selection before holdout.",
                timing="Fixed recorded target and timestamps; nominal IK/executor dt 10ms. Historical deadline holds injected, local elapsed never used as real-time evidence.",
                locked=[
                    "handoff_skip_steps",
                    "physical_joint_bounds",
                    "IK_margins",
                    "lag_policy",
                    "policy_targets",
                    "timeline",
                    "gripper_limits",
                ],
                unsupported="No plant/contact/controller-feedback simulation; no candidate state or task-success estimates.",
            ),
        )
        batch(development, SINGLES)
    elif args.phase == "rank-single":
        for task in ["pack_plate", "pack_bottle"]:
            save(
                OUT / f"{task}_single_ranking.json",
                ranking(
                    [r for r in development if r["task"] == task], ["baseline"] + list(SINGLES)
                ),
            )
    elif args.phase == "joint":
        choices = json.loads((OUT / "joint_factor_space.json").read_text())
        for task, space in choices.items():
            batch([r for r in development if r["task"] == task], space)
        for task, space in choices.items():
            save(
                OUT / f"{task}_joint_ranking.json",
                ranking(
                    [r for r in development if r["task"] == task],
                    ["baseline"] + list(SINGLES) + list(space),
                ),
            )
    elif args.phase == "tcp":
        space = {"tcp_objective": {"study.objective": "tcp"}}
        save(OUT / "tcp_single_factor_space.json", space)
        batch(development, space)
    elif args.phase == "tcp-joint":
        choices = json.loads((OUT / "tcp_joint_factor_space.json").read_text())
        old = json.loads((OUT / "joint_factor_space.json").read_text())
        for task, space in choices.items():
            batch([r for r in development if r["task"] == task], space)
        for task, space in choices.items():
            save(
                OUT / f"{task}_final_development_ranking.json",
                ranking(
                    [r for r in development if r["task"] == task],
                    ["baseline"]
                    + list(SINGLES)
                    + list(old[task])
                    + ["tcp_objective"]
                    + list(space),
                ),
            )
    elif args.phase == "holdout":
        selected = json.loads((OUT / "frozen_selection.json").read_text())
        for task, space in selected.items():
            batch([r for r in eligible if r["task"] == task and r["split"] == "holdout"], space)
    elif args.phase == "refresh-metrics":
        with ProcessPoolExecutor(max_workers=2) as pool:
            for result in pool.map(refresh, eligible):
                print(result, flush=True)


if __name__ == "__main__":
    main()
