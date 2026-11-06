#!/usr/bin/env python3
"""Summarize verified step-only baseline evidence without hardware or inference.

Run from repository root with .venv/bin/python. This checks saved replay traces;
it neither retunes candidates nor treats historical state as candidate feedback.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/analysis/step_diffik_study/20261019"


def read(path):
    return json.loads(path.read_text())


def main():
    verified = {r["episode"]: r for r in read(OUT / "verification.json")["baselines"]}
    forensic = {r["id"]: r for r in read(OUT / "baseline_forensics.json")}
    rows = []
    for episode in read(OUT / "cohort.json"):
        if not episode["included"]:
            continue
        directory = OUT / "episodes" / episode["id"]
        config = read(directory / "identity.json")["config"]
        assert config["execution"]["ik"]["mode"] == "step"
        assert config["executor"]["type"] == "direct"
        parity = verified[episode["id"]]
        assert parity["exact_command"] and parity["exact_reference"]
        trace = np.load(directory / "baseline.npz")
        metrics = read(directory / "baseline_sim_metrics.json")
        active = trace["reason"] != "gap"
        ok = trace["reason"] == "ok"
        arm_rows = {}
        for j, arm in enumerate(("left_arm", "right_arm")):
            arm_rows[arm] = {
                "successful_solve_ticks": int(ok.sum()),
                "any_joint_velocity_saturation_fraction_on_ok": float(
                    trace["saturation"][ok, j].any(axis=-1).mean()
                ),
                "reference_command_max_abs_joint_rad": float(
                    abs(trace["reference"][active, j, :7] - trace["command"][active, j, :7]).max()
                ),
                "pose_limited_ticks": metrics["counts"]["pose_limited"][arm],
                "executor_shaped_ticks": metrics["counts"]["executor_shaped"][arm],
                "errors_by_region": {
                    region: {
                        stage: value[arm] for stage, value in metrics[region]["errors"].items()
                    }
                    for region in ("all", "handoff_100ms", "stable_success", "ik_hold")
                },
                "executor_forensics": {
                    k: v
                    for k, v in forensic[episode["id"]]["per_arm"][arm].items()
                    if k.startswith("executor_")
                },
                "historical_previous_command_to_state": forensic[episode["id"]]["feedback"][arm],
            }
        rows.append(
            {
                "episode": episode["path"],
                "task": episode["task"],
                "status": episode["status"],
                "session": episode["session"],
                "executor_type": config["executor"]["type"],
                "motion_limits": config["executor"]["motion_limits"]["arm"],
                "pose_speed_limit": config["execution"]["ik"]["pose_speed_limit"],
                "control_hz": config["robot"]["control_hz"],
                "counts": metrics["counts"]["reasons"],
                "arms": arm_rows,
                "first_failure": forensic[episode["id"]]["first_failure"],
                "baseline_trace_sha256": hashlib.sha256(
                    (directory / "baseline.npz").read_bytes()
                ).hexdigest(),
            }
        )
    report = {
        "scope": "11 verified historical step baselines only; no chunk metrics",
        "limits": (
            "Errors across layers and P95 values are not additive. "
            "Limiter input is reanchored to previous command, "
            "so its gap includes accumulated downstream lag. "
            "State comparisons describe historical feedback only."
        ),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "episodes": rows,
    }
    target = OUT / "step_lag_check.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Checked {len(rows)} direct step baselines; wrote {target}")


if __name__ == "__main__":
    main()
