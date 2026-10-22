"""Offline chunk/step IK timing on FK-reconstructed recorded YAM joint plans.

Uses RobotModel only: no controllers, robot connection, or hardware commands.
Recorded measured joints initialize each plan. Subsequent measured states are an
ideal one-tick follower of successful commands, not a physical dynamics replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import zarr

from manimux.cli import load_config, prepare_experiment
from manimux.embodiments.robot.base import RobotModel
from manimux.kinematics.poses import matrix_pose
from manimux.policy_adapter.openwam.yam import SEMANTICS, OpenWAMYamAdapter
from manimux.policy_adapter.pose_output import POSE_SPACE
from manimux.runtime.execution import ExecutionPipeline
from manimux.runtime.executors.direct import DirectExecutor
from manimux.runtime.timeline import ActionTimeline
from manimux.types import ActionContext, RobotState

GROUPS = ("left_arm", "right_arm")


def stats(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {"n": 0}
    return dict(
        n=len(a),
        mean=float(a.mean()),
        p50=float(np.median(a)),
        p95=float(np.percentile(a, 95)),
        p99=float(np.percentile(a, 99)),
        max=float(a.max()),
    )


def copied(groups):
    return {g: v.copy() for g, v in groups.items()}


def poses(kin, joints):
    return {g: np.r_[matrix_pose(kin.models[g].fk(q)), q[-1]] for g, q in joints.items()}


def errors(kin, command, target):
    from scipy.spatial.transform import Rotation

    from manimux.kinematics.poses import pose_matrix

    pos, rot = [], []
    for g in GROUPS:
        actual = kin.models[g].fk(command[g])
        desired = pose_matrix(target[g][:7])
        pos.append(np.linalg.norm(actual[:3, 3] - desired[:3, 3]) * 1000)
        rot.append(Rotation.from_matrix(actual[:3, :3].T @ desired[:3, :3]).magnitude())
    return float(max(pos)), float(max(rot))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--budget-ms", type=float, default=5.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = prepare_experiment(
        **load_config(
            "manimux/configs/experiments/openwam-custom/put_bottles/serial_step30000_step_ik.yaml"
        )
    )
    kin = RobotModel.from_config(cfg["robot"]["config"]).kinematics
    adapters = [OpenWAMYamAdapter(cfg["robot"], cfg["policy"], kinematics=kin) for _ in range(2)]
    adapters[1].decode_pose_only = True
    data = zarr.open_group(str(args.source), mode="r")
    names = sorted(data["plans"].group_keys())
    indices = np.unique(np.linspace(0, len(names) - 1, args.samples).astype(int))
    times = data["ticks/monotonic_ns"][:]
    measured = {g: data[f"ticks/state/{g}"][:] for g in GROUPS}
    fixtures = []
    for index in indices:
        name = names[index]
        group = data["plans"][name]["canonical_raw"]
        if group.attrs["action_space"] != "joint_position":
            raise ValueError("Expected historical decoded joint trajectories")
        t = group.attrs["observation_time_ns"]
        tick = max(0, min(len(times) - 1, int(np.searchsorted(times, t, side="right") - 1)))
        seed = {g: measured[g][tick].copy() for g in GROUPS}
        joints = {g: group[g][:] for g in GROUPS}
        rows = {
            g: np.stack([np.r_[matrix_pose(kin.models[g].fk(q)), q[-1]] for q in joints[g]])
            for g in GROUPS
        }
        fixtures.append((name, seed, rows))
    # Warm both native legacy and bounded solver paths before timing.
    for g in GROUPS:
        model = kin.models[g]
        q = fixtures[0][1][g]
        for _ in range(5):
            target = model.fk(q)
            model.ik(target, q, fixed_coordinates={"gripper": float(q[-1])})
            model.ik_deadline(
                target,
                q,
                fixed_coordinates={"gripper": float(q[-1])},
                deadline_ns=time.monotonic_ns() + 1_000_000_000,
            )
    results = []
    fixture_hash = hashlib.sha256()
    for name, seed, rows in fixtures:
        for g in GROUPS:
            fixture_hash.update(seed[g].tobytes())
            fixture_hash.update(rows[g].tobytes())
    # Randomize run order deterministically to reduce run-order timing bias.
    jobs = [
        (i, r, mode)
        for i in range(len(fixtures))
        for r in range(args.repeats)
        for mode in ("chunk", "step_5ms", "step_100ms")
    ]
    np.random.default_rng(42).shuffle(jobs)
    control_dt = 10_000_000
    for n, (i, repeat, mode) in enumerate(jobs):
        name, seed, rows = fixtures[i]
        adapter = adapters[0 if mode == "chunk" else 1]
        context = ActionContext(n + 1, 0, 0, ik_seed=RobotState(seed, 0, 0))
        adapter.anchors[n + 1] = copied(seed)
        raw = {"format": "pose", "action_semantics": SEMANTICS, "actions": rows}
        start = time.perf_counter_ns()
        try:
            chunk = adapter.decode_action(raw, context)
        except ValueError as exc:
            results.append(
                dict(
                    plan=name,
                    repeat=repeat,
                    mode=mode,
                    decode_ms=(time.perf_counter_ns() - start) / 1e6,
                    error=str(exc),
                )
            )
            continue
        decode_ms = (time.perf_counter_ns() - start) / 1e6
        is_step = mode != "chunk"
        options = (
            {"mode": "step", "budget_ms": args.budget_ms if mode == "step_5ms" else 100.0}
            if is_step
            else None
        )
        pipe = ExecutionPipeline(
            DirectExecutor(), kinematics=kin, ik_options=options, control_dt_ns=control_dt
        )
        pipe.executor.reset(RobotState(copied(seed), 0, 0))
        timeline = ActionTimeline(
            dict.fromkeys(GROUPS, 8 if is_step else 7),
            action_space=POSE_SPACE if is_step else "joint_position",
        )
        accepted = timeline.commit(
            chunk,
            now_ns=0,
            max_plan_age_ns=10**9,
            current_command=poses(kin, seed) if is_step else seed,
            blend_steps=0,
        )
        assert accepted.accepted
        # Same pose timeline is used only to compute errors outside timed sections.
        pose_timeline = ActionTimeline(dict.fromkeys(GROUPS, 8), action_space=POSE_SPACE)
        pose_chunk = replace(chunk, groups=rows, action_space=POSE_SPACE)
        assert pose_timeline.commit(
            pose_chunk,
            now_ns=0,
            max_plan_age_ns=10**9,
            current_command=poses(kin, seed),
            blend_steps=0,
        ).accepted
        last = copied(seed)
        elapsed, statuses, pe, re = [], [], [], []
        # Execute the selected experiment's first 20 policy intervals at 100 Hz.
        for tick, now in enumerate(range(0, 20 * chunk.dt_ns, control_dt)):
            state = RobotState(copied(last), now, tick)
            t0 = time.perf_counter_ns()
            reference = timeline.reference_horizon(
                now_ns=now, dt_ns=control_dt, horizon_steps=pipe.horizon_steps
            )
            output = pipe.step(now, state, reference, last_command=last)
            elapsed.append((time.perf_counter_ns() - t0) / 1e6)
            statuses.append(output.ik_status or {"success": True, "reason": "joint_reference"})
            expected = pose_timeline.reference_horizon(
                now_ns=now, dt_ns=control_dt, horizon_steps=1
            )
            p, r = errors(kin, output.command.groups, {g: v[0] for g, v in expected.groups.items()})
            pe.append(p)
            re.append(r)
            last = copied(output.command.groups)
        results.append(
            dict(
                plan=name,
                repeat=repeat,
                mode=mode,
                decode_ms=decode_ms,
                tick_ms=elapsed,
                status=statuses,
                error_mm=pe,
                error_rad=re,
            )
        )
        if (n + 1) % 15 == 0:
            print(json.dumps({"completed": n + 1, "total": len(jobs)}), flush=True)
    summary = {}
    for mode in ("chunk", "step_5ms", "step_100ms"):
        subset = [r for r in results if r["mode"] == mode]
        valid = [r for r in subset if "error" not in r]
        ticks = [t for r in valid for t in r["tick_ms"]]
        statuses = [s for r in valid for s in r["status"]]
        pos = [e for r in valid for e, s in zip(r["error_mm"], r["status"]) if s["success"]]
        rot = [e for r in valid for e, s in zip(r["error_rad"], r["status"]) if s["success"]]
        summary[mode] = dict(
            runs=len(subset),
            rejected_chunks=len(subset) - len(valid),
            decode_ms=stats([r["decode_ms"] for r in subset]),
            tick_ms=stats(ticks),
            first_tick_ms=stats([r["tick_ms"][0] for r in valid]),
            subsequent_tick_ms=stats([t for r in valid for t in r["tick_ms"][1:]]),
            failure_reasons=dict(Counter(s["reason"] for s in statuses if not s["success"])),
            successful_ticks=sum(s["success"] for s in statuses),
            total_ticks=len(statuses),
            over_5ms=sum(t > 5 for t in ticks),
            over_10ms=sum(t > 10 for t in ticks),
            successful_command_error_mm=stats(pos),
            successful_command_error_rad=stats(rot),
            cpu_ms_per_executed_segment=stats([r["decode_ms"] + sum(r["tick_ms"]) for r in valid]),
        )
    metadata = dict(
        source=str(args.source.resolve()),
        samples=len(fixtures),
        repeats=args.repeats,
        selected_plans=[f[0] for f in fixtures],
        fixture_sha256=fixture_hash.hexdigest(),
        platform=platform.platform(),
        cpu=Path("/proc/cpuinfo").read_text().split("model name")[1].split("\n")[0],
        limitations=[
            "FK of saved decoded joints, not original model poses",
            "Ideal one-tick follower after recorded initial measured state; no hardware/dynamics",
            "100 Hz time coordinates; wall-clock execution is unpaced",
            "No camera, inference, hardware, logging, chunk blending or scheduling overhead",
            "Step 100 ms is an offline diagnostic, not a deployment setting",
        ],
        policy_horizon=32,
        policy_dt_ns=chunk.dt_ns,
        executed_policy_intervals=20,
        control_dt_ns=control_dt,
        step_budget_ms=args.budget_ms,
    )
    (args.output / "samples.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.output / "summary.json").write_text(
        json.dumps({"metadata": metadata, "results": summary}, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
