"""Time-aligned overlap diagnostics for decoded joint-reference chunks.

These metrics describe saved references, not native model outputs or physical
smoothness. In particular, canonical_raw may already contain waypoint/IK edits.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from manimux.evaluation.trajectory import FKModel, ReferenceTrajectory, _sample

METRICS = {
    "start_mse_mm2": "First point XYZ MSE (mm²)",
    "end_mse_mm2": "Last point XYZ MSE (mm²)",
    "head_mse_mm2": "First 20% samples XYZ MSE (mm²)",
    "tail_mse_mm2": "Last 20% samples XYZ MSE (mm²)",
    "mse_mm2": "Whole overlap XYZ MSE (mm²)",
    "rmse_mm": "Whole overlap distance RMSE (mm)",
    "mean_error_mm": "Mean Euclidean distance (mm)",
    "p95_error_mm": "P95 Euclidean distance (mm)",
    "shape_rmse_mm": "Start-centred shape RMSE (mm)",
    "velocity_rmse_mm_s": "Velocity RMSE (mm/s)",
    "velocity_cosine": "Velocity cosine (−1…1)",
    "joint_mse_rad2": "Arm joint MSE, gripper excluded (rad²)",
    "duration_ms": "Overlap duration (ms)",
}


def align_overlap(
    old: ReferenceTrajectory,
    new: ReferenceTrajectory,
    group: str,
    *,
    start_ns: int | None = None,
    duration_ns: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample both domains on a uniform inclusive grid; never extrapolate.

    Grid spacing is at most incoming dt. The exact overlap endpoint is included.
    Explicit start/duration require full coverage rather than shortening a window.
    """
    arrays = []
    for ref in (old, new):
        if ref.dt_ns <= 0:
            raise ValueError("invalid_dt")
        if group not in ref.groups:
            raise ValueError("missing_group")
        a = np.asarray(ref.groups[group], dtype=float)
        if a.ndim != 2 or min(a.shape) == 0:
            raise ValueError("invalid_shape")
        if not np.isfinite(a).all():
            raise ValueError("nonfinite_reference")
        arrays.append(a)
    if arrays[0].shape[1] != arrays[1].shape[1]:
        raise ValueError("dimension_mismatch")
    if new.start_ns < old.start_ns:
        raise ValueError("nonmonotonic_start")
    lo = max(old.start_ns, new.start_ns)
    hi = min(
        ref.start_ns + (len(a) - 1) * ref.dt_ns for ref, a in zip((old, new), arrays, strict=True)
    )
    if start_ns is not None:
        if start_ns < lo or start_ns > hi:
            raise ValueError("switch_not_covered")
        lo = start_ns
    if duration_ns is not None:
        if duration_ns <= 0:
            raise ValueError("invalid_duration")
        if lo + duration_ns > hi:
            raise ValueError("short_fixed_window")
        hi = lo + duration_ns
    if hi <= lo:
        raise ValueError("no_positive_overlap")
    n = math.ceil((hi - lo) / new.dt_ns) + 1
    if n > 10000:
        raise ValueError("oversized_sampling_grid")
    # Operate on small relative offsets before adding integer monotonic clocks.
    times = lo + np.rint(np.linspace(0, hi - lo, n)).astype(np.int64)
    a, b = [
        _sample(v, times - r.start_ns, r.dt_ns) for r, v in zip((old, new), arrays, strict=True)
    ]
    return times, a, b


def measure_overlap(
    old: ReferenceTrajectory,
    new: ReferenceTrajectory,
    group: str,
    model: FKModel,
    *,
    start_ns: int | None = None,
    duration_ns: int | None = None,
    edge_fraction: float = 0.2,
    stationary_speed_mm_s: float = 0.1,
    gripper_dofs: int = 1,
) -> dict:
    """Compute per-coordinate MSE and separate Euclidean distance metrics."""
    if not 0 < edge_fraction <= 0.5:
        raise ValueError("edge_fraction must be in (0, 0.5]")
    if not np.isfinite(stationary_speed_mm_s) or stationary_speed_mm_s < 0:
        raise ValueError("stationary speed must be finite and nonnegative")
    if type(gripper_dofs) is not int or gripper_dofs < 0:
        raise ValueError("gripper_dofs must be a nonnegative integer")
    result = dict.fromkeys(METRICS)
    result.update(status="missing", reason=None, cosine_reason=None, curve=None)
    try:
        times, qa, qb = align_overlap(old, new, group, start_ns=start_ns, duration_ns=duration_ns)
    except ValueError as exc:
        result.update(reason=str(exc), cosine_reason=str(exc))
        return result
    a, b = [np.asarray([model.fk(q)[:3, 3] for q in rows]) * 1000 for rows in (qa, qb)]
    if a.shape != (len(times), 3) or b.shape != a.shape or not np.isfinite([a, b]).all():
        result.update(reason="invalid_fk", cosine_reason="invalid_fk")
        return result
    diff = b - a
    dist = np.linalg.norm(diff, axis=1)
    point_mse = np.mean(diff**2, axis=1)
    edge_n = max(1, math.ceil(len(times) * edge_fraction))
    dt = np.diff(times)[:, None] / 1e9
    va, vb = np.diff(a, axis=0) / dt, np.diff(b, axis=0) / dt

    def rms(v):
        return float(np.sqrt(np.mean(np.sum(v**2, axis=1))))

    speed_a, speed_b = rms(va), rms(vb)
    cos = None
    if min(speed_a, speed_b) <= stationary_speed_mm_s:
        result["cosine_reason"] = "stationary_or_near_stationary"
    else:
        cos = float(np.clip(np.sum(va * vb) / (np.linalg.norm(va) * np.linalg.norm(vb)), -1, 1))
    arm_dims = qa.shape[1] - gripper_dofs
    result.update(
        status="valid",
        start_mse_mm2=float(point_mse[0]),
        end_mse_mm2=float(point_mse[-1]),
        head_mse_mm2=float(np.mean(point_mse[:edge_n])),
        tail_mse_mm2=float(np.mean(point_mse[-edge_n:])),
        mse_mm2=float(np.mean(point_mse)),
        rmse_mm=rms(diff),
        mean_error_mm=float(np.mean(dist)),
        p95_error_mm=float(np.percentile(dist, 95)),
        shape_rmse_mm=rms((b - b[0]) - (a - a[0])),
        velocity_rmse_mm_s=rms(vb - va),
        velocity_cosine=cos,
        joint_mse_rad2=float(np.mean((qb[:, :arm_dims] - qa[:, :arm_dims]) ** 2))
        if arm_dims > 0
        else None,
        duration_ms=float((times[-1] - times[0]) / 1e6),
        samples=len(times),
        edge_samples=edge_n,
        old_rms_speed_mm_s=speed_a,
        new_rms_speed_mm_s=speed_b,
        start_ns=int(times[0]),
        end_ns=int(times[-1]),
        curve={
            "time_ms": ((times - times[0]) / 1e6).tolist(),
            "error_mm": dist.tolist(),
            "old_xyz_mm": a.tolist(),
            "new_xyz_mm": b.tolist(),
        },
    )
    return result


def measure_seam(
    old: ReferenceTrajectory, new: ReferenceTrajectory, group: str, model: FKModel
) -> dict:
    """Position gap at the incoming committed start; endpoint holds are invalid."""
    sampled = []
    for ref in (old, new):
        a = np.asarray(ref.groups.get(group, []), dtype=float)
        if ref.dt_ns <= 0 or a.ndim != 2 or min(a.shape) == 0 or not np.isfinite(a).all():
            return {"seam_mm": None, "seam_reason": "invalid_reference"}
        offset = new.start_ns - ref.start_ns
        if offset < 0 or offset > (len(a) - 1) * ref.dt_ns:
            return {"seam_mm": None, "seam_reason": "switch_not_covered"}
        sampled.append(_sample(a, np.array([offset], dtype=np.int64), ref.dt_ns)[0])
    pa, pb = (np.asarray(model.fk(q)[:3, 3]) for q in sampled)
    if not np.isfinite([pa, pb]).all():
        return {"seam_mm": None, "seam_reason": "invalid_fk"}
    return {"seam_mm": float(np.linalg.norm(pb - pa) * 1000), "seam_reason": None}


def summarize(rows: list[dict], keys=tuple(METRICS)) -> dict:
    """Within-rollout statistics. Counts are per metric, not complete cases."""
    metrics = {}
    for key in keys:
        v = [r[key] for r in rows if r.get(key) is not None]
        metrics[key] = {
            "n": len(v),
            "mean": float(np.mean(v)) if v else None,
            "median": float(np.median(v)) if v else None,
            "p95": float(np.percentile(v, 95)) if v else None,
            "max": float(np.max(v)) if v else None,
        }
    return {
        "candidates": len(rows),
        "valid": sum(r["status"] == "valid" for r in rows),
        "reasons": dict(Counter(r["reason"] for r in rows if r.get("reason"))),
        "cosine_reasons": dict(Counter(r["cosine_reason"] for r in rows if r.get("cosine_reason"))),
        "metrics": metrics,
    }
