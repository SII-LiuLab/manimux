"""Audit pi/80 generated traces; stratify task/session/status without selecting rates."""

import json
from collections import Counter, defaultdict

import numpy as np
import run as study
from check_pi80 import CHOICES, OUT
from reference_layer_review import FK_CHECKS, errors


def main():
    rows = [r for r in json.loads((study.OUT / "cohort.json").read_text()) if r["included"]]
    details = []
    for row in rows:
        data = study.load(row)
        model = study.make_model(data["c"])
        active = np.isfinite(data["target"][:, 0, 0])
        for name in ["baseline", *CHOICES]:
            path = OUT / "episodes" / row["id"] / name
            if name == "baseline":
                trace_path = study.OUT / "episodes" / row["id"] / "baseline.npz"
                metric_path = study.OUT / "episodes" / row["id"] / "baseline_sim_metrics.json"
                check = {}
            else:
                trace_path, metric_path = path / "trace.npz", path / "metrics.json"
                check = json.loads((path / "checks.json").read_text())
            trace = dict(np.load(trace_path))
            metric = json.loads(metric_path.read_text())
            ref_error = errors(model, trace["reference"], data["target"])
            entry = {
                "episode": row["id"],
                "task": row["task"],
                "status": row["status"],
                "split": row["split"],
                "stratum": data["info"]["stratum"],
                "candidate": name,
                "reasons": metric["counts"]["reasons"],
                "checks": check,
                "rms": {},
                "worst_inner_boundary": {},
            }
            for kind, values in ref_error.items():
                entry["rms"]["target_to_reference_" + kind] = float(
                    np.mean(np.sqrt(np.mean(values[active] ** 2, axis=0)))
                )
            for pair, groups in metric["all"]["errors"].items():
                for kind in ["translation_mm", "rotation_deg"]:
                    entry["rms"][pair + "_" + kind] = float(
                        np.mean([groups[g][kind]["rms"] for g in study.GROUPS])
                    )
            for g_idx, group in enumerate(study.GROUPS):
                low, high = model.kinematics.models[group].arm.joint_position_limits()
                q = trace["command"][:, g_idx, :7]
                margin = data["c"]["execution"]["ik"]["diff_ik"]["limit_margin_deg"]
                distance = np.degrees(np.minimum(q - low, high - q)) - margin
                distance[~active] = np.inf
                tick, joint = np.unravel_index(np.argmin(distance), distance.shape)
                entry["worst_inner_boundary"][group] = {
                    "tick": int(tick),
                    "joint_one_based": int(joint + 1),
                    "inner_margin_deg": float(distance[tick, joint]),
                    "command_deg": float(np.degrees(q[tick, joint])),
                    "reference_deg": float(
                        np.degrees(trace["reference"][tick, g_idx, joint])
                    ),
                    "physical_lower_deg": float(np.degrees(low[joint])),
                    "physical_upper_deg": float(np.degrees(high[joint])),
                }
            details.append(entry)
    grouped = defaultdict(list)
    for d in details:
        grouped[(d["task"], d["split"], d["stratum"], d["status"], d["candidate"])].append(d)
    strata = []
    for (task, split, stratum, status, name), entries in grouped.items():
        reasons = Counter()
        for e in entries:
            reasons.update(e["reasons"])
        strata.append(
            {
                "task": task,
                "split": split,
                "stratum": stratum,
                "status": status,
                "candidate": name,
                "episodes": [e["episode"] for e in entries],
                "reasons": dict(reasons),
                "mean_episode_arm_rms": {
                    k: float(np.mean([e["rms"][k] for e in entries]))
                    for k in entries[0]["rms"]
                },
            }
        )
    study.save(
        OUT / "verification.json",
        {
            "traces": len(details),
            "fk_max_matrix_difference": max(FK_CHECKS),
            "strata": strata,
            "episodes": details,
            "interpretation": (
                "All trajectories after historical guard rejection are conditional. "
                "No changed physical state is predicted. Old holdout is regression only. "
                "No sample is selected by outcome; complete does not mean success."
            ),
        },
    )
    print("audited", len(details), "traces; FK max difference", max(FK_CHECKS))


if __name__ == "__main__":
    main()
