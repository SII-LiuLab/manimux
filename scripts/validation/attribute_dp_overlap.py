#!/usr/bin/env python3
"""Separate model target changes from post-IK reference changes, without hardware.

Requires UMI ee_target_trace metadata. Old recordings remain explicitly missing.
The vector identity is exact; norms of its components must not be added.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import compare_dp_chunk_overlap as overlap
import numpy as np
import zarr

from manimux.evaluation.trajectory import _sample


def target_xyz(trace, stage, times_ns):
    value = trace[stage]
    poses = np.asarray(value["pose_xyz_wxyz"], dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 7 or not np.isfinite(poses).all():
        raise ValueError("invalid_target_poses")
    offsets = times_ns - int(value["start_time_ns"])
    dt = int(value["dt_ns"])
    if dt <= 0 or min(offsets) < 0 or max(offsets) > (len(poses) - 1) * dt:
        raise ValueError("target_window_not_covered")
    return _sample(poses[:, :3], offsets, dt) * 1000


def decompose(row, old_trace, new_trace):
    if old_trace is None or new_trace is None:
        return {"status": "missing", "reason": "native_targets_not_recorded"}
    for trace in (old_trace, new_trace):
        if (trace["schema"] != "umi-ee-target-trace-v1" or trace["frame"] != "arm_base"
                or trace["position_unit"] != "m"):
            return {"status": "missing", "reason": "unsupported_target_trace"}
    times = row["start_ns"] + np.rint(np.array(row["curve"]["time_ms"]) * 1e6).astype(np.int64)
    try:
        om, nm = [target_xyz(t, "model_output", times) for t in (old_trace, new_trace)]
        oi, ni = [target_xyz(t, "ik_input", times) for t in (old_trace, new_trace)]
    except ValueError as exc:
        return {"status": "missing", "reason": str(exc)}
    op, np_ = [np.asarray(row["curve"][k]) for k in ("old_xyz_mm", "new_xyz_mm")]
    parts = {"model_target_gap": nm - om,
             "adapter_target_change": (ni - nm) - (oi - om),
             "ik_reference_residual_change": (np_ - ni) - (op - oi)}
    total = np_ - op
    assert np.allclose(sum(parts.values()), total, atol=1e-9, rtol=1e-12)
    parts.update(total_gap=total, old_ik_reference_residual=op - oi,
                 new_ik_reference_residual=np_ - ni)
    result = {"status": "valid", "time_ms": row["curve"]["time_ms"], "components": {}}
    for key, values in parts.items():
        norms = np.linalg.norm(values, axis=1)
        result["components"][key] = {"xyz_mm": values.tolist(), "norm_mm": norms.tolist(),
                                     "endpoint_mm": float(norms[-1]),
                                     "mean_distance_mm": float(np.mean(norms))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixed-ms", type=float, default=100)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("choose a new output path to preserve previous analysis")
    episode = args.episode.resolve()
    if args.out.resolve().is_relative_to(episode.parent):
        parser.error("derived outputs must be outside the recorded session")
    config = overlap.read_json(episode.parent / "session-manifest.json")["config"]
    if config["inference"]["blend_policy_steps"]:
        parser.error("attribution currently requires blend_policy_steps=0")
    method = "train" if config["inference"]["algorithm"] == "train_time_rtc" else "baseline"
    info = overlap.episode_info(episode, method)
    if info["excluded_reason"] not in (None, "partial_recording"):
        parser.error(info["excluded_reason"])
    model_path = Path(info["robot_config"])
    info["fk_config_sha256"] = overlap.digest(model_path)
    models = {k: v for k, v in overlap.RobotModel.from_config(model_path).kinematics.models.items()
              if k in ("left_arm", "right_arm")}
    overlap.VIEWS = {"committed_fixed": "Fixed window"}
    source_rows = overlap.analyze_episode(info, models, SimpleNamespace(
        fixed_ms=args.fixed_ms, edge_fraction=0.2, stationary_speed_mm_s=0.1))
    plans = zarr.open_group(str(episode / "data.zarr"), mode="r")["plans"]
    rows = []
    for row in source_rows:
        output = {k: row[k] for k in ("arm", "old_chunk", "new_chunk", "old_request_seq",
                                     "new_request_seq", "eligibility", "switch_ns")}
        if row["eligibility"] != "observed_transition" or row["status"] != "valid":
            output.update(status="missing", reason=row["reason"] or row["eligibility"])
        else:
            traces = [plans[f"{row[k]:06d}"]["canonical_raw"].attrs["metadata"]
                      .get("ee_target_trace", {}).get(row["arm"])
                      for k in ("old_chunk", "new_chunk")]
            output.update(decompose(row, *traces))
        rows.append(output)
    summaries = {}
    for arm in models:
        selected = [r for r in rows if r["arm"] == arm]
        valid = [r for r in selected if r["status"] == "valid"]
        summaries[arm] = {"valid_boundaries": len(valid), "candidate_boundaries": len(selected),
                          "missing": dict(Counter(r["reason"] for r in selected
                                                  if r["status"] != "valid"))}
        if valid:
            summaries[arm]["components"] = {
                key: {stat: float(np.mean([r["components"][key][stat] for r in valid]))
                      for stat in ("endpoint_mm", "mean_distance_mm")}
                for key in valid[0]["components"]}
    result = {"schema": "umi-overlap-attribution-v1", "episode": info,
              "fixed_ms": args.fixed_ms, "summary": summaries, "rows": rows,
              "notes": ["Linear XYZ targets; joint interpolation then FK for references.",
                        "IK reference residual includes interpolation, not measured robot error.",
                        "Total vector = model gap + adapter change + IK residual change.",
                        "Component norms are not additive and are not causal percentages.",
                        "Partial recordings are diagnostic; only continuous transitions qualify."],
              "source_sha256": {str(p): overlap.digest(p) for p in
                                (Path(__file__).resolve(), Path(overlap.__file__).resolve())}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(summaries, ensure_ascii=False))


if __name__ == "__main__":
    main()
