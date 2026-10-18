#!/usr/bin/env python3
"""Build a standalone HTML comparison from explicit DP rollout roots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from manimux.embodiments.robot import RobotModel  # noqa: E402
from manimux.evaluation.chunk_overlap import (  # noqa: E402
    METRICS,
    measure_overlap,
    measure_seam,
    summarize,
)
from manimux.evaluation.trajectory import ReferenceTrajectory  # noqa: E402

VIEWS = {
    "canonical_full": "Decoded · full overlap",
    "canonical_future": "Decoded · after switch",
    "committed_future": "Committed · full overlap",
    "committed_fixed": "Committed · fixed window",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_ref(group, index, stage, hasher):
    g = group[stage]
    attrs = dict(g.attrs)
    if attrs.get("action_space") != "joint_position":
        raise ValueError(f"{index}/{stage}: expected joint_position")
    dt = int(attrs["dt_ns"])
    start = (
        int(attrs["start_time_ns"])
        if stage == "committed"
        else int(attrs["observation_time_ns"]) + int(attrs.get("source_offset_steps", 0)) * dt
    )
    arrays = {name: np.asarray(g[name][:], dtype="<f8") for name in g.array_keys()}
    hasher.update(json.dumps([index, stage, attrs, dict(group.attrs)], sort_keys=True).encode())
    for name, a in sorted(arrays.items()):
        hasher.update(json.dumps([name, a.shape]).encode())
        hasher.update(a.tobytes())
    return ReferenceTrajectory(
        index, str(group.attrs["plan_id"]), int(group.attrs["request_seq"]), start, dt, arrays
    )


def tick_evidence(ticks):
    times = np.asarray(ticks["monotonic_ns"][:], dtype=np.int64)
    ids = np.asarray(ticks["plan_id"][:]).astype(str)
    if len(times) != len(ids) or np.any(np.diff(times) <= 0):
        raise ValueError("invalid tick clocks")
    intervals = np.diff(times)
    gap_limit = max(30_000_000, float(np.median(intervals)) * 3) if len(intervals) else 0
    transitions = {}
    for i in np.flatnonzero(ids[1:] != ids[:-1]) + 1:
        key = (ids[i - 1], ids[i])
        transitions.setdefault(key, []).append(
            {
                "time_ns": int(times[i]),
                "gap_ms": float((times[i] - times[i - 1]) / 1e6),
                "continuous_ticks": bool(times[i] - times[i - 1] <= gap_limit),
            }
        )
    return times, ids, transitions


def episode_info(path, method):
    meta = read_json(path / "meta.json")
    manifest = read_json(path.parent / "session-manifest.json")
    c = manifest["config"]
    b = meta.get("policy_backend", {}).get("model", {})
    inf = c["inference"]
    alg = inf.get("algorithm")
    # Historical manifests retain the old Async selector; do not rewrite recordings.
    expected = {"train_time_rtc"} if method == "train" else {"async", "manimux"}
    partial = path.name.endswith(".partial") or not (path / "result.json").is_file()
    reason = None
    if alg not in expected:
        reason = f"algorithm_mismatch:{alg}"
    elif meta.get("horizon_steps") != 32:
        reason = "not_H32"
    elif partial:
        reason = "partial_recording"
    result = read_json(path / "result.json") if (path / "result.json").exists() else {}
    skip = inf.get("handoff_skip_steps", 0)
    return {
        "id": path.parent.name + "/" + path.name,
        "path": str(path),
        "method": method,
        "partial": partial,
        "excluded_reason": reason,
        "algorithm": alg,
        "horizon": meta.get("horizon_steps"),
        "task": meta.get("task"),
        "checkpoint": b.get("checkpoint_path"),
        "checkpoint_sha256": b.get("checkpoint_sha256"),
        "robot_config": c["robot"]["config"],
        "action_dt_s": meta.get("action_dt_s"),
        "blend_steps": meta.get("blend_steps"),
        "handoff": inf.get("handoff"),
        "skip": skip,
        "variant": f"{inf.get('handoff')} / skip={skip}",
        "chunk_steps": inf.get("chunk_policy_steps", inf.get("chunk_steps")),
        "expected_decode_s": inf.get("expected_decode_s"),
        "num_inference_steps": b.get("num_inference_steps"),
        "execute_diff_ik_substeps": c["policy"].get("adapter", {}).get("execute_diff_ik_substeps"),
        "terminal_reason": result.get("terminal_reason"),
        "layout_id": meta.get("layout_id"),
        "repeat_id": meta.get("repeat_id"),
        "plan_count": None,
        "tick_count": None,
        "backend": b,
        "inference": inf,
        "source_hashes": {
            name: digest(path / name)
            for name in ("meta.json", "events.jsonl", "result.json")
            if (path / name).exists()
        },
        "manifest_sha256": digest(path.parent / "session-manifest.json"),
        "resolved_config_sha256": hashlib.sha256(
            json.dumps(c, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def analyze_episode(info, models, args):
    path = Path(info["path"])
    root = zarr.open_group(str(path / "data.zarr"), mode="r")
    names = sorted(root["plans"].group_keys(), key=int)
    info["plan_count"] = len(names)
    times, tick_ids, transitions = tick_evidence(root["ticks"])
    info["tick_count"] = len(times)
    info["duration_s"] = float((times[-1] - times[0]) / 1e9) if len(times) else 0
    if len(names) < 2:
        info["excluded_reason"] = "fewer_than_two_plans"
        return []
    hasher = hashlib.sha256()
    hasher.update(times.tobytes())
    hasher.update(json.dumps(tick_ids.tolist()).encode())
    plans = []
    events = [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]
    submissions = {
        e["request_seq"]: e
        for e in events
        if e.get("kind") == "inference_submitted" and "request_seq" in e
    }
    info["conditioned_requests"] = sum(bool(e.get("conditioned")) for e in submissions.values())
    info["submitted_requests"] = len(submissions)
    info["prefix_lengths"] = dict(Counter(str(e.get("prefix_steps")) for e in submissions.values()))
    info["prefix_out_of_support_requests"] = sum(
        e.get("prefix_in_training_support") is False for e in submissions.values()
    )
    for name in names:
        g = root["plans"][name]
        refs = {
            stage: load_ref(g, int(name), stage, hasher) for stage in ("canonical_raw", "committed")
        }
        refs["metadata"] = dict(g["canonical_raw"].attrs).get("metadata", {})
        refs["hold"] = dict(g["canonical_raw"].attrs).get("hold_from_step", {})
        plans.append(refs)
    info["selected_arrays_sha256"] = hasher.hexdigest()
    rows = []
    seen = set(tick_ids)
    for old, new in zip(plans, plans[1:], strict=False):
        o, n = old["committed"], new["committed"]
        evidence = transitions.get((o.plan_id, n.plan_id), [])
        if len(evidence) == 1 and evidence[0]["continuous_ticks"]:
            eligibility = "observed_transition"
        elif evidence:
            eligibility = "tick_gap_or_repeated_transition"
        elif o.plan_id not in seen or n.plan_id not in seen:
            eligibility = "plan_not_observed"
        else:
            eligibility = "not_adjacent_in_ticks"
        submission = submissions.get(n.request_seq, {})
        for arm, model in models.items():
            seam = measure_seam(o, n, arm, model)
            for view in VIEWS:
                stage = "canonical_raw" if view.startswith("canonical") else "committed"
                start = None if view == "canonical_full" else n.start_ns
                duration = round(args.fixed_ms * 1e6) if view == "committed_fixed" else None
                metric = measure_overlap(
                    old[stage],
                    new[stage],
                    arm,
                    model,
                    start_ns=start,
                    duration_ns=duration,
                    edge_fraction=args.edge_fraction,
                    stationary_speed_mm_s=args.stationary_speed_mm_s,
                )
                # Explicit holds can look perfectly consistent without being a
                # continuous replan. Keep the row but exclude it from the default view.
                row_eligibility = (
                    "explicit_hold" if arm in old["hold"] or arm in new["hold"] else eligibility
                )
                rows.append(
                    {
                        "episode": info["id"],
                        "method": info["method"],
                        "variant": info["variant"],
                        "arm": arm,
                        "view": view,
                        "old_chunk": o.index,
                        "new_chunk": n.index,
                        "old_plan_id": o.plan_id,
                        "new_plan_id": n.plan_id,
                        "old_request_seq": o.request_seq,
                        "new_request_seq": n.request_seq,
                        "switch_ns": n.start_ns,
                        "eligibility": row_eligibility,
                        "tick_transition": evidence,
                        "conditioned": submission.get("conditioned"),
                        "prefix_steps": submission.get("prefix_steps"),
                        "prefix_in_training_support": submission.get("prefix_in_training_support"),
                        "switch_from_overlap_start_ms": (n.start_ns - metric["start_ns"]) / 1e6
                        if metric.get("start_ns") is not None
                        else None,
                        **seam,
                        **metric,
                    }
                )
    info["summary"] = {
        view: {
            arm: summarize(
                [
                    r
                    for r in rows
                    if r["view"] == view
                    and r["arm"] == arm
                    and r["eligibility"] == "observed_transition"
                ],
                (*METRICS, "seam_mm"),
            )
            for arm in models
        }
        for view in VIEWS
    }
    return rows


def aggregate(infos):
    result = {}
    for method in ("baseline", "train"):
        result[method] = {}
        for view in VIEWS:
            result[method][view] = {}
            for arm in ("left_arm", "right_arm"):
                summaries = [
                    i["summary"][view][arm]
                    for i in infos
                    if i["method"] == method and "summary" in i
                ]
                metrics = {}
                for key in (*METRICS, "seam_mm"):
                    means = [s["metrics"][key]["mean"] for s in summaries if s["metrics"][key]["n"]]
                    p95s = [s["metrics"][key]["p95"] for s in summaries if s["metrics"][key]["n"]]
                    maxima = [s["metrics"][key]["max"] for s in summaries if s["metrics"][key]["n"]]
                    metrics[key] = {
                        "episodes": len(means),
                        "boundaries": sum(s["metrics"][key]["n"] for s in summaries),
                        "mean_of_episode_means": float(np.mean(means)) if means else None,
                        "mean_of_episode_p95": float(np.mean(p95s)) if p95s else None,
                        "worst": max(maxima) if maxima else None,
                    }
                result[method][view][arm] = metrics
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--edge-fraction", type=float, default=0.2)
    parser.add_argument("--fixed-ms", type=float, default=100)
    parser.add_argument("--stationary-speed-mm-s", type=float, default=0.1)
    args = parser.parse_args()
    if not 0 < args.edge_fraction <= 0.5 or not np.isfinite(args.fixed_ms) or args.fixed_ms <= 0:
        parser.error("require 0 < edge-fraction <= 0.5 and positive finite fixed-ms")
    if not np.isfinite(args.stationary_speed_mm_s) or args.stationary_speed_mm_s < 0:
        parser.error("stationary-speed-mm-s must be finite and nonnegative")
    out = args.out.resolve()
    paths = []
    for method, root in (("baseline", args.baseline_root), ("train", args.train_root)):
        root = root.resolve()
        if not root.is_dir():
            parser.error(f"missing root: {root}")
        if out == root or root in out.parents:
            parser.error("output must be outside both recording roots")
        paths.extend((method, p.parent) for p in sorted(root.glob("*/rollout-*/meta.json")))
    if out.exists():
        parser.error("choose a new output directory to preserve previous reports")
    infos, rows, model_cache = [], [], {}
    for method, path in paths:
        info = episode_info(path, method)
        infos.append(info)
        if info["excluded_reason"]:
            continue
        config = Path(info["robot_config"])
        try:
            if config not in model_cache:
                model_cache[config] = dict(RobotModel.from_config(config).kinematics.models)
            models = {
                k: v for k, v in model_cache[config].items() if k in ("left_arm", "right_arm")
            }
            if len(models) != 2:
                raise ValueError("requires explicit dual-arm offline FK models")
            info["fk_config_sha256"] = digest(config)
            new_rows = analyze_episode(info, models, args)
            rows.extend(new_rows)
            print(
                f"{method}: {info['id']} — {info['plan_count']} plans, {len(new_rows)} rows",
                flush=True,
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            info["excluded_reason"] = f"processing_error:{type(exc).__name__}:{exc}"
            print(f"ERROR {info['id']}: {exc}", file=sys.stderr)
    if not rows:
        parser.error("no analyzable rollout pairs")
    labels = dict(METRICS)
    labels["head_mse_mm2"] = f"First {args.edge_fraction:.0%} samples XYZ MSE (mm²)"
    labels["tail_mse_mm2"] = f"Last {args.edge_fraction:.0%} samples XYZ MSE (mm²)"
    labels["seam_mm"] = "Committed switch position gap (mm)"
    views = {**VIEWS, "committed_fixed": f"Committed · first {args.fixed_ms:g} ms"}
    report = {
        "schema": "dp-chunk-overlap-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "formal_evaluation": False,
        "metric_labels": labels,
        "view_labels": views,
        "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "episodes": infos,
        "rows": rows,
        "summary": aggregate(infos),
        "notes": [
            "Baseline = DP Async; train = train-time RTC. No ordinary RTC included.",
            "All H32 finalized attempts in both roots; partial and empty runs inventoried.",
            "canonical_raw is post-adapter joint reference (waypoint / IK may modify it),"
            " not native DP output. Native EE chunk/prefix residuals are not recorded.",
            "Source clock = observation_time_ns + source_offset_steps * dt_ns; the recorded"
            " observation_time_ns includes first-action offset. Never add that offset twice.",
            "XYZ MSE = mean over samples AND 3 coordinates (mm²); distance RMSE = sqrt(3*MSE)."
            " Edge windows use ceil(fraction*N) samples. Cosine uses flattened velocity sequences.",
            "Default tick-adjacent candidates have one observed consecutive plan transition without"
            " a >3x median tick gap (minimum 30 ms). This is not formal pause/hold validation.",
            "Rollouts receive equal weight. P95 is computed within rollout, then averaged."
            " No pooled significance test; rollouts within a session need not be independent.",
            "Checkpoints, dates, handoff mode, skip, history/timing and initial layouts differ."
            " This is an exploratory system comparison, not a causal ablation of train-time RTC.",
            "XYZ excludes orientation/gripper. Joint MSE excludes the final gripper coordinate."
            " Saved committed rows may omit dense executor substeps; no measured smoothness claim.",
            "FK uses the current manifest-referenced assembly. Hashes cannot prove historical"
            " asset equivalence. Source metadata/arrays are hashed; raw recordings are unchanged.",
        ],
    }
    source_files = [
        Path(__file__),
        ROOT / "manimux/evaluation/chunk_overlap.py",
        ROOT / "manimux/evaluation/trajectory.py",
        ROOT / "manimux/evaluation/chunk_overlap_template.html",
    ]
    provenance = {
        "command": [sys.executable, *sys.argv],
        "source_sha256": {str(p.resolve()): digest(p) for p in source_files},
        "python": sys.version,
        "numpy": np.__version__,
        "zarr": zarr.__version__,
    }
    out.mkdir(parents=True)
    for name, obj in (
        ("episode-metrics.json", report),
        ("summary.json", report["summary"]),
        ("provenance.json", provenance),
    ):
        (out / name).write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n")
    (out / "cohort.jsonl").write_text(
        "".join(
            json.dumps({k: v for k, v in i.items() if k != "summary"}, ensure_ascii=False) + "\n"
            for i in infos
        )
    )
    fields = [k for k in rows[0] if k not in ("curve", "tick_transition")]
    with (out / "boundary-metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    # The HTML needs only display precision for dense traces; JSON/CSV keep full precision.
    for row in rows:
        if row["curve"]:
            row["curve"] = {k: np.round(v, 4).tolist() for k, v in row["curve"].items()}
    payload = json.dumps(report, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    payload = (
        payload.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    )
    template = source_files[-1].read_text()
    (out / "index.html").write_text(template.replace("__REPORT_DATA__", payload))
    print(f"Report: {out / 'index.html'}")


if __name__ == "__main__":
    main()
