#!/usr/bin/env python3
"""Attribute one saved chunk candidate's executor stages; no IK counterfactual.

Run from repository root with .venv/bin/python. Saved candidate references and
commands are inputs. Each stage uses the candidate's recurrent command history.
Local stage displacements are not independent causal ablations or additive P95s.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location(
    "chunk_audit_readonly", ROOT / "scripts/validation/chunk_diffik_audit/audit.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def main():
    directory = ROOT / "data/analysis/chunk_diffik_audit_20261019"
    key = "pack_plate_latest_async_chunk__session-20261017T071505Z-d79e604e__rollout-003"
    source = directory / "selected_traces" / f"{key}.npz"
    saved = np.load(source)
    metadata = json.loads(source.with_suffix(".json").read_text())
    row = next(
        r for r in json.loads((directory / "cohort.json").read_text())["selected"] if r["id"] == key
    )
    d = audit.load(row)
    choice = metadata["candidate"]
    reference = saved["counterfactual_reference"]
    expected = saved["counterfactual_command"]
    public, _ = audit.replay(d, reference, choice, public=True)
    assert np.max(abs(public[:, :, :7] - expected[:, :, :7])) < 1e-12
    s = dict(d["c"]["executor"]["smooth"])
    s.update(choice["smooth"])
    s["max_velocity"] = choice["velocity"]
    assert s["tracking_mode"] == "legacy" and s["mode"] == "isotropic"
    dt = d["dt"]
    alpha = dt / (dt + 1 / (2 * np.pi * s["cutoff_hz"]))
    vmax = s["max_velocity"] * min(1, (s["max_step_dt_s"] or dt) / dt)
    stages = {k: expected.copy() for k in ("filtered", "speed", "acceleration", "position")}
    flags = np.zeros((len(reference), 2, 3), dtype=bool)
    q = d["a"]["command"][0, :, :7].copy()
    v = np.zeros_like(q)

    def bound(x, limit):
        return x * np.minimum(1, limit / np.maximum(abs(x).max(axis=-1, keepdims=True), 1e-300))

    for i in range(len(reference)):
        if d["resets"][i]:
            q = d["a"]["command"][i, :, :7].copy()
            v[:] = 0
        elif not d["active"][i]:
            v[:] = 0
        else:
            filtered = q + alpha * (reference[i, :, :7] - q)
            desired = (filtered - q) / dt
            speed = bound(desired, vmax)
            acceleration = v + bound(speed - v, s["max_acceleration"] * dt)
            before_clip = q + acceleration * dt
            output = np.clip(before_clip, -s["position_limit_abs"], s["position_limit_abs"])
            for k, value in zip(
                stages, (filtered, q + speed * dt, before_clip, output), strict=True
            ):
                stages[k][i, :, :7] = value
            flags[i, :, 0] = np.any(abs(speed - desired) > 1e-10, axis=-1)
            flags[i, :, 1] = np.any(abs(acceleration - speed) > 1e-10, axis=-1)
            flags[i, :, 2] = np.any(abs(output - before_clip) > 1e-10, axis=-1)
            q, v = output, acceleration
        assert np.max(abs(q - expected[i, :, :7])) < 1e-12

    kin = audit.RobotModel.from_config(d["c"]["robot"]["config"]).kinematics
    poses = {"reference": audit.poses(kin, reference)}
    poses.update({k: audit.poses(kin, v) for k, v in stages.items()})
    mask = d["masks"]["all"]
    pairs = list(zip(list(poses)[:-1], list(poses)[1:], strict=True))
    local = {}
    for a, b in pairs:
        mm, deg = audit.err(poses[a], poses[b])
        local[f"{a}_to_{b}"] = {"mm": audit.stats(mm[mask]), "deg": audit.stats(deg[mask])}
    score = audit.score(d, kin, reference, expected, saved["original_target_tcp"])
    report = {
        "scope": (
            "Local executor-stage attribution of saved chunk candidate; "
            "not a new DiffIK parameter test, not measured feedback prediction"
        ),
        "episode": metadata["episode"],
        "candidate": choice,
        "trace_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in (
                "scripts/validation/chunk_diffik_audit/audit.py",
                "manimux/runtime/executors/smooth.py",
                "manimux/runtime/executors/limits.py",
            )
        },
        "config": s,
        "dt_s": dt,
        "alpha": alpha,
        "public_executor_match_max_rad": float(np.max(abs(public[:, :, :7] - expected[:, :, :7]))),
        "active_ticks": int(mask.sum()),
        "active_arm_ticks": int(mask.sum() * 2),
        "limited_arm_ticks": dict(
            zip(
                ("speed", "acceleration", "position"),
                flags[mask].sum(axis=(0, 1)).tolist(),
                strict=True,
            )
        ),
        "limited_arm_fraction": dict(
            zip(
                ("speed", "acceleration", "position"),
                flags[mask].mean(axis=(0, 1)).tolist(),
                strict=True,
            )
        ),
        "local_stage_displacement": local,
        "full_errors": {
            k: score[k]
            for k in (
                "target_reference_all_mm",
                "reference_command_all_mm",
                "target_command_all_mm",
            )
        },
        "interpretation": (
            "Stage TCP displacement vectors telescope at each tick; their norms/P95 do not add. "
            "Filter input includes lag accumulated by all stages. "
            "This is not unique causal attribution of accumulated lag."
        ),
    }
    out = ROOT / "data/analysis/step_diffik_study/20261019/executor_stage_breakdown.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
