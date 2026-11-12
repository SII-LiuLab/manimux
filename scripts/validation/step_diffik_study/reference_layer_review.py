#!/usr/bin/env python3
"""Measure TCP target-to-reference in existing recurrent step candidate replays.

This is postprocessing, not a new fixed-reference counterfactual or selection.
Only development sessions are included. No robot/controller is constructed.
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import run as study
from scipy.spatial.transform import Rotation

OUT = study.OUT
FK_CHECKS = []
NAMES = (
    "speed_060",
    "speed_090",
    "accel_none",
    "linear_080",
    "pose_none",
    "ratio_32_fixed_runtime",
    "tcp_objective",
    "tcp_accel40_angular090",
    "tcp_angular090_speed060",
)


def fk_batch(model, joints):
    """Modified-DH FK, checked at eight samples of every input trace."""
    matrices = np.broadcast_to(np.eye(4), (len(joints), 4, 4)).copy()
    for j, (alpha, a, d, theta0) in enumerate(model.arm._dh):
        alpha = np.radians(alpha)
        theta = np.radians(theta0) + (joints[:, j] if j < 7 else 0)
        ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
        local = np.zeros_like(matrices)
        local[:, 0, 0], local[:, 0, 1], local[:, 0, 3] = ct, -st, a * 0.001
        local[:, 1, 0], local[:, 1, 1], local[:, 1, 2] = st * ca, ct * ca, -sa
        local[:, 1, 3] = -sa * d * 0.001
        local[:, 2, 0], local[:, 2, 1], local[:, 2, 2] = st * sa, ct * sa, ca
        local[:, 2, 3], local[:, 3, 3] = ca * d * 0.001, 1
        matrices = matrices @ local
    offset = model._offset(np.array([0.0]))
    assert np.array_equal(offset, model._offset(np.array([1.0])))
    matrices = matrices @ offset
    for i in np.linspace(0, len(joints) - 1, 8, dtype=int):
        difference = float(np.max(abs(matrices[i] - model.fk(joints[i]))))
        FK_CHECKS.append(difference)
        assert difference < 1e-8
    return matrices


def errors(model, reference, target):
    matrices = np.stack(
        [fk_batch(model.kinematics.models[g], reference[:, j]) for j, g in enumerate(study.GROUPS)],
        axis=1,
    )
    mm = np.linalg.norm(matrices[:, :, :3, 3] - target[:, :, :3], axis=-1) * 1000
    valid = np.isfinite(target[:, :, 0])
    deg = np.full(valid.shape, np.nan)
    rotations = Rotation.from_quat(target[:, :, [4, 5, 6, 3]][valid]).as_matrix()
    deg[valid] = np.degrees(
        Rotation.from_matrix(
            rotations @ np.swapaxes(matrices[:, :, :3, :3][valid], -1, -2)
        ).magnitude()
    )
    return {"translation_mm": mm, "rotation_deg": deg}


def first_hold(trace):
    where = np.flatnonzero(~np.isin(trace["reason"], ["ok", "gap"]))
    return int(where[0]) if len(where) else len(trace["reason"])


def main():
    rows = []
    for episode in json.loads((OUT / "cohort.json").read_text()):
        if not episode["included"] or episode["split"] == "holdout":
            continue
        directory = OUT / "episodes" / episode["id"]
        identity = json.loads((directory / "identity.json").read_text())
        model = study.make_model(identity["config"])
        base = np.load(directory / "baseline.npz")
        baseline = errors(model, base["reference"], base["target"])
        active = base["reason"] != "gap"
        for name in NAMES:
            path = directory / "candidates" / name / "trace.npz"
            if not path.exists():
                continue
            candidate = np.load(path)
            current = errors(model, candidate["reference"], base["target"])
            prefix = active & (
                np.arange(len(active)) < min(first_hold(base), first_hold(candidate))
            )
            regions = {}
            for region, mask in (("all", active), ("common_pre_first_hold", prefix)):
                regions[region] = {
                    quantity: {
                        "baseline_rms": np.sqrt(np.mean(before[mask] ** 2, axis=0)).tolist(),
                        "candidate_rms": np.sqrt(
                            np.mean(current[quantity][mask] ** 2, axis=0)
                        ).tolist(),
                        "per_arm_ratio": (
                            np.sqrt(np.mean(current[quantity][mask] ** 2, axis=0))
                            / np.maximum(np.sqrt(np.mean(before[mask] ** 2, axis=0)), 1e-12)
                        ).tolist(),
                    }
                    for quantity, before in baseline.items()
                }
            rows.append(
                {
                    "task": episode["task"],
                    "episode": episode["id"],
                    "status": episode["status"],
                    "stratum": identity["stratum"],
                    "candidate": name,
                    "regions": regions,
                    "baseline_holds": int((~np.isin(base["reason"], ["ok", "gap"])).sum()),
                    "candidate_holds": int((~np.isin(candidate["reason"], ["ok", "gap"])).sum()),
                    "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["candidate"])].append(row)
    summary = []
    for (task, name), group in grouped.items():
        result = {"task": task, "candidate": name, "episodes": len(group)}
        for region in ("all", "common_pre_first_hold"):
            result[region] = {
                quantity: float(
                    np.mean([r["regions"][region][quantity]["per_arm_ratio"] for r in group])
                )
                for quantity in ("translation_mm", "rotation_deg")
            }
        summary.append(result)
    result = {
        "scope": "Postprocessing saved recurrent step replays; development only; no new selection",
        "summary_note": (
            "Equal episode/arm mean error ratios. Pooled summaries are descriptive; "
            "status/config strata remain explicit in rows. "
            "Hold references are held commands, not successful QP solutions."
        ),
        "public_fk_check_max_abs": max(FK_CHECKS),
        "summary": summary,
        "episodes": rows,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / "reference_layer_review.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
