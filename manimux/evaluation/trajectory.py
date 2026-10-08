"""Offline, time-aligned similarity of committed joint-reference trajectories.

This module does not establish execution eligibility. Adjacent accepted plans are
candidates, not proof of a continuous physical handoff. XYZ metrics exclude tool
orientation and gripper agreement and must not be called measured-motion metrics.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class FKModel(Protocol):
    def fk(self, q: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class ReferenceTrajectory:
    index: int
    plan_id: str
    request_seq: int
    start_ns: int
    dt_ns: int
    groups: dict[str, np.ndarray]


METRICS = (
    "seam_mm",
    "position_rmse_mm",
    "endpoint_mm",
    "shape_rmse_mm",
    "velocity_rmse_mm_s",
    "velocity_cosine",
)


def _invalid(reason: str) -> dict:
    return {
        "status": "missing",
        "reason": reason,
        **dict.fromkeys(METRICS),
        "cosine_missing_reason": reason,
    }


def _sample(values: np.ndarray, offsets_ns: np.ndarray, dt_ns: int) -> np.ndarray:
    # Coverage is checked by the caller. Integer-relative time avoids loss of
    # precision at large absolute timestamps; no out-of-range endpoint padding.
    lower, remainder = np.divmod(offsets_ns, dt_ns)
    upper = np.minimum(lower + 1, len(values) - 1)
    alpha = (remainder / dt_ns)[:, None]
    return (1 - alpha) * values[lower] + alpha * values[upper]


def compare_window(
    old: ReferenceTrajectory,
    new: ReferenceTrajectory,
    group: str,
    model: FKModel,
    steps: int,
    *,
    stationary_speed_mm_s: float = 0.01,
) -> dict:
    """Compare k=1..N future points; k=0 anchors shape and velocity.

    The sample interval is the incoming committed trajectory's recorded dt.
    Both trajectories are sampled at t_new + k*dt, using their own dt for
    joint interpolation before FK. All N+1 points must be in both domains.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if not np.isfinite(stationary_speed_mm_s) or stationary_speed_mm_s < 0:
        raise ValueError("stationary_speed_mm_s must be finite and nonnegative")
    if old.dt_ns <= 0 or new.dt_ns <= 0:
        return _invalid("invalid_dt")
    arrays = []
    for label, ref in (("old", old), ("new", new)):
        if group not in ref.groups:
            return _invalid(f"{label}_missing_group")
        values = np.asarray(ref.groups[group], dtype=np.float64)
        if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
            return _invalid(f"{label}_invalid_shape")
        if not np.isfinite(values).all():
            return _invalid(f"{label}_nonfinite")
        arrays.append(values)
    if arrays[0].shape[1] != arrays[1].shape[1]:
        return _invalid("coordinate_dimension_mismatch")
    if new.start_ns < old.start_ns:
        return _invalid("nonmonotonic_start")
    end_ns = new.start_ns + steps * new.dt_ns
    for label, ref, values in zip(("old", "new"), (old, new), arrays, strict=True):
        last_ns = ref.start_ns + (len(values) - 1) * ref.dt_ns
        if new.start_ns > last_ns:
            return _invalid(f"{label}_does_not_cover_switch")
        if end_ns > last_ns:
            return _invalid(f"{label}_short_window")
    xyz = []
    offsets = np.arange(steps + 1, dtype=np.int64) * new.dt_ns
    for ref, values in zip((old, new), arrays, strict=True):
        sampled = _sample(values, offsets + (new.start_ns - ref.start_ns), ref.dt_ns)
        points = np.asarray([model.fk(q)[:3, 3] for q in sampled], dtype=np.float64)
        if points.shape != (steps + 1, 3) or not np.isfinite(points).all():
            return _invalid("invalid_fk_output")
        xyz.append(points * 1000)
    a, b = xyz

    def rms(v):
        return float(np.sqrt(np.mean(np.sum(v * v, axis=1))))

    va, vb = np.diff(a, axis=0) / (new.dt_ns / 1e9), np.diff(b, axis=0) / (new.dt_ns / 1e9)
    speed_a, speed_b = rms(va), rms(vb)
    cosine_reason = None
    cosine = None
    if speed_a <= stationary_speed_mm_s or speed_b <= stationary_speed_mm_s:
        cosine_reason = "stationary_or_near_stationary"
    else:
        cosine = float(np.clip(np.sum(va * vb) / (np.linalg.norm(va) * np.linalg.norm(vb)), -1, 1))
    return {
        "status": "valid",
        "reason": None,
        "seam_mm": float(np.linalg.norm(b[0] - a[0])),
        "position_rmse_mm": rms(b[1:] - a[1:]),
        "endpoint_mm": float(np.linalg.norm(b[-1] - a[-1])),
        "shape_rmse_mm": rms((b[1:] - b[0]) - (a[1:] - a[0])),
        "velocity_rmse_mm_s": rms(vb - va),
        "velocity_cosine": cosine,
        "cosine_missing_reason": cosine_reason,
    }


def _summarize(rows: list[dict]) -> dict:
    result = {
        "candidate_count": len(rows),
        "numeric_valid_count": sum(r["status"] == "valid" for r in rows),
        "missing_reasons": dict(Counter(r["reason"] for r in rows if r["reason"])),
        "cosine_missing_reasons": dict(
            Counter(r["cosine_missing_reason"] for r in rows if r["cosine_missing_reason"])
        ),
        "metrics": {},
    }
    for key in METRICS:
        values = [r[key] for r in rows if r[key] is not None]
        stats = {"count": len(values), "mean": None, "median": None}
        if key == "velocity_cosine":
            stats.update(p05=None, min=None)
        else:
            stats.update(p95=None, max=None)
        if values:
            stats.update(mean=float(np.mean(values)), median=float(np.median(values)))
            if key == "velocity_cosine":
                stats.update(p05=float(np.percentile(values, 5)), min=float(np.min(values)))
            else:
                stats.update(p95=float(np.percentile(values, 95)), max=float(np.max(values)))
        result["metrics"][key] = stats
    return result


def analyze_candidates(
    plans: list[ReferenceTrajectory],
    models: dict[str, FKModel],
    *,
    windows: tuple[int, ...] = (10, 20, 30),
    stationary_speed_mm_s: float = 0.01,
) -> dict:
    """Return separate per-window and common-boundary summaries for ONE episode."""
    if not windows or any(type(n) is not int or n <= 0 for n in windows):
        raise ValueError("windows must contain positive integer policy steps")
    if len(set(windows)) != len(windows):
        raise ValueError("windows must be unique")
    if not np.isfinite(stationary_speed_mm_s) or stationary_speed_mm_s < 0:
        raise ValueError("stationary speed threshold must be finite and nonnegative")
    if not models:
        raise ValueError("at least one explicit FK group is required")
    input_hash = hashlib.sha256()
    for plan in plans:
        input_hash.update(
            json.dumps(
                [plan.index, plan.plan_id, plan.request_seq, plan.start_ns, plan.dt_ns]
            ).encode()
        )
        for group, values in sorted(plan.groups.items()):
            normalized = np.asarray(values, dtype="<f8", order="C")
            input_hash.update(json.dumps([group, normalized.shape]).encode())
            input_hash.update(normalized.tobytes())
    rows = []
    for old, new in zip(plans, plans[1:], strict=False):
        for group, model in models.items():
            for n in windows:
                rows.append(
                    {
                        "old_chunk": old.index,
                        "new_chunk": new.index,
                        "old_plan_id": old.plan_id,
                        "new_plan_id": new.plan_id,
                        "old_request_seq": old.request_seq,
                        "new_request_seq": new.request_seq,
                        "group": group,
                        "window_steps": n,
                        "switch_ns": new.start_ns,
                        "sample_dt_ns": new.dt_ns,
                        "window_ms": n * new.dt_ns / 1e6,
                        "execution_eligibility": "not_checked",
                        **compare_window(
                            old, new, group, model, n, stationary_speed_mm_s=stationary_speed_mm_s
                        ),
                    }
                )
    summaries = {}
    for group in models:
        group_rows = [r for r in rows if r["group"] == group]
        valid_counts = Counter(
            (r["old_chunk"], r["new_chunk"]) for r in group_rows if r["status"] == "valid"
        )
        common = {pair for pair, count in valid_counts.items() if count == len(windows)}
        summaries[group] = {
            "by_window": {
                str(n): _summarize([r for r in group_rows if r["window_steps"] == n])
                for n in windows
            },
            "common_boundary_count": len(common),
            "common_boundary_by_window": {
                str(n): _summarize(
                    [
                        r
                        for r in group_rows
                        if r["window_steps"] == n and (r["old_chunk"], r["new_chunk"]) in common
                    ]
                )
                for n in windows
            },
        }
    return {
        "schema": "committed-reference-overlap-v1",
        "selected_reference_sha256": input_hash.hexdigest(),
        "scope": "adjacent accepted plan candidates; execution eligibility NOT checked",
        "formal_evaluation": False,
        "definition": {
            "stage": "committed joint reference, interpolated before offline FK to XYZ",
            "samples": "t_new + k * incoming_dt, k=1..N; k=0 anchors shape and velocity",
            "units": "position/shape/endpoint/seam: mm; velocity RMSE: mm/s; cosine: [-1,1]",
            "stationary_speed_mm_s": stationary_speed_mm_s,
            "cosine": "flattened velocity sequences; missing if either RMS speed <= threshold",
            "coverage": "full window required for BOTH plans; no extrapolation or padding",
            "aggregation": "single-episode candidate statistics; not pooled multi-episode scores",
            "excluded_dimensions": "orientation and gripper agreement",
        },
        "windows_policy_steps": list(windows),
        "plan_count": len(plans),
        "candidate_pair_count": max(0, len(plans) - 1),
        "rows": rows,
        "summary": summaries,
    }


def write_overlap_report(report: dict, out: Path, *, episode: Path, robot_config: Path) -> None:
    """Write diagnostics outside the immutable episode, with explicit provenance."""
    out, episode, robot_config = out.resolve(), episode.resolve(), robot_config.resolve()
    if out == episode or episode in out.parents:
        raise ValueError("analysis output must be outside the raw episode")
    report = dict(report)
    report["provenance"] = {
        "episode": str(episode),
        "robot_config": str(robot_config),
        "metric_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "robot_config_sha256": hashlib.sha256(robot_config.read_bytes()).hexdigest(),
        "fk_note": "Caller-selected offline assembly; verify against recorded embodiment/tool. "
        "Config hash does not hash referenced assets or establish historical equivalence.",
        "input_metadata_sha256": {
            name: hashlib.sha256((episode / name).read_bytes()).hexdigest()
            for name in ("meta.json", "events.jsonl", "result.json")
            if (episode / name).is_file()
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    rows = report.pop("rows")
    (out / "overlap-metrics.json").write_text(
        json.dumps({**report, "rows": rows}, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (out / "overlap-summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (out / "overlap-metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0]) if rows else ["old_chunk", "new_chunk", "group", "window_steps"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
