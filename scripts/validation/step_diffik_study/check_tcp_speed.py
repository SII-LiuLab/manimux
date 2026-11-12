"""Fixed TCP speed ablation with TCP objective and pi/80 joint limits.

Full recurrent command-generation replay; unchanged historical guard thresholds.
Old holdout is regression only. Do not infer physical feedback or deployment safety.
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import run as study
from reference_layer_review import errors
from verify_stopping_qp import summarize_difference

OUT = study.OUT / "tcp_speed_pi80_review"
BASE = study.OUT / "pi80_review/episodes"
CHOICES = {
    "linear_080": {"max_linear_m_s": 0.8, "max_angular_rad_s": 1.8},
    "angular_360": {"max_linear_m_s": 0.4, "max_angular_rad_s": 3.6},
    "both_double": {"max_linear_m_s": 0.8, "max_angular_rad_s": 3.6},
}


def trial(args):
    row, name = args
    path = OUT / "episodes" / row["id"] / name
    if (path / "checks.json").exists():
        return row["id"], name, "cached"
    data = study.load(row)
    motion = data["c"]["executor"]["motion_limits"]["arm"]
    motion.update(max_velocity=math.pi, max_acceleration=80.0)
    data["c"]["execution"]["ik"]["pose_speed_limit"] = CHOICES[name]
    trace, failures, model = study.simulate(data, {"study.objective": "tcp"})
    metric = study.measure(data, trace, model)
    baseline = json.loads((BASE / row["id"] / "legacy_tcp_pi80/metrics.json").read_text())
    active = trace["reason"] != "gap"
    reference_error = errors(model, trace["reference"], data["target"])
    command = trace["command"][:, :, :7]
    velocity = np.diff(command, axis=0, prepend=command[:1]) * 100
    acceleration = np.diff(velocity, axis=0, prepend=np.zeros_like(velocity[:1])) * 100
    check = {
        "episode": row["id"],
        "task": row["task"],
        "status": row["status"],
        "split": row["split"],
        "stratum": data["info"]["stratum"],
        "candidate": name,
        "pose_limits": CHOICES[name],
        "manifest_sha256": data["info"]["manifest_sha"],
        "stream_sha256": data["info"]["stream_sha256"],
        "source_sha256": {
            n: study.sha(study.ROOT / "scripts/validation/step_diffik_study" / n)
            for n in ["check_tcp_speed.py", "run.py", "tcp_objective.py"]
        },
        "counts": metric["counts"],
        "ratios_vs_tcp_pi80": summarize_difference(baseline, metric),
        "guard_violations": metric["command_safety_violations"],
        "acceleration_80_excess_ticks": int(
            (active & np.any(abs(acceleration) > 80 + 1e-6, axis=(1, 2))).sum()
        ),
        "max_acceleration_rad_s2": float(abs(acceleration[active]).max()),
        "rms": {
            "target_to_reference_" + kind: float(np.mean(np.sqrt(np.mean(v[active] ** 2, axis=0))))
            for kind, v in reference_error.items()
        },
    }
    for pair, groups in metric["all"]["errors"].items():
        for kind in ["translation_mm", "rotation_deg"]:
            check["rms"][pair + "_" + kind] = float(
                np.mean([groups[g][kind]["rms"] for g in study.GROUPS])
            )
    # Equal prefix ending before either candidate or the pi/80 baseline first holds.
    base_trace = dict(np.load(BASE / row["id"] / "legacy_tcp_pi80/trace.npz"))

    def first_hold(t):
        inds = np.flatnonzero(~np.isin(t["reason"], ["ok", "gap"]))
        return int(inds[0]) if len(inds) else len(t["reason"])

    cut = min(first_hold(trace), first_hold(base_trace))
    prefix = dict(data)
    for k in ["times", "target", "ids"]:
        prefix[k] = data[k][:cut]
    check["common_prefix_ticks"] = cut
    check["common_prefix_ratios"] = summarize_difference(
        study.measure(prefix, {k: v[:cut] for k, v in base_trace.items()}, model),
        study.measure(prefix, {k: v[:cut] for k, v in trace.items()}, model),
    )
    study.save(path / "metrics.json", metric)
    study.save(path / "checks.json", check)
    study.save(path / "failures.json", failures)
    np.savez_compressed(path / "trace.npz", **trace)
    return row["id"], name, check["counts"]["reasons"]


def report():
    checks = [json.loads(p.read_text()) for p in OUT.glob("episodes/*/*/checks.json")]
    groups = defaultdict(list)
    strata = defaultdict(list)
    for c in checks:
        key = c["task"], c["split"], c["candidate"]
        groups[key].append(c)
        strata[(*key, c["stratum"], c["status"])].append(c)

    def aggregate(key, entries):
        reasons = Counter()
        for e in entries:
            reasons.update(e["counts"]["reasons"])
        return {
            "group": key,
            "episodes": [e["episode"] for e in entries],
            "reasons": dict(reasons),
            "mean_rms": {
                k: float(np.mean([e["rms"][k] for e in entries])) for k in entries[0]["rms"]
            },
            "mean_ratios_vs_tcp_pi80": {
                k: float(np.mean([e["ratios_vs_tcp_pi80"][k] for e in entries]))
                for k in entries[0]["ratios_vs_tcp_pi80"]
            },
            "mean_common_prefix_ratios": {
                k: float(np.mean([e["common_prefix_ratios"][k] for e in entries]))
                for k in entries[0]["common_prefix_ratios"]
            },
            "position_guard_arm_ticks": sum(
                v["count"]
                for e in entries
                for v in e["guard_violations"]
                if v["guard"] == "position"
            ),
            "acceleration_80_excess_ticks": sum(e["acceleration_80_excess_ticks"] for e in entries),
        }

    summary = [aggregate(k, v) for k, v in groups.items()]
    study.save(OUT / "summary.json", summary)
    study.save(OUT / "strata.json", [aggregate(k, v) for k, v in strata.items()])
    for x in summary:
        print(x)


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
            "baseline": "legacy_tcp_pi80: linear 0.4 m/s, angular 1.8 rad/s",
            "selection": "Fixed factorial comparison, no selection or holdout tuning.",
            "guard": (
                "Historical thresholds unchanged; post-rejection full traces conditional only."
            ),
            "feedback": (
                "Candidate previous command feeds the next tick; no changed physical state."
            ),
        },
    )
    rows = [r for r in json.loads((study.OUT / "cohort.json").read_text()) if r["included"]]
    names = list(CHOICES)[:2] if args.phase == "single" else ["both_double"]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(trial, (r, n)) for r in rows for n in names]
        for f in as_completed(futures):
            print(f.result(), flush=True)


if __name__ == "__main__":
    main()
