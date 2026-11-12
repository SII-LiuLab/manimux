"""Execution-qualified committed-reference seams, grouped by boundary class."""

from __future__ import annotations

from collections import Counter

import numpy as np

from manimux.evaluation.execution import sample
from manimux.evaluation.trajectory import compare_window

CLASSES = ("continuous", "end_hold")


def analyze(evidence, models, max_gap_s):
    times, ids = evidence["times"], evidence["ids"]
    plans = evidence["plans"]
    boundaries = {e["plan_id"]: e for e in evidence["events"] if e.get("kind") == "plan_boundary"}
    rows = []
    # Acceptance order is deliberately unused: only owners observed on ticks.
    changes = np.flatnonzero(ids[1:] != ids[:-1]) + 1
    for i in changes:
        old_id, new_id = str(ids[i - 1]), str(ids[i])
        row = dict(
            old_plan_id=old_id,
            new_plan_id=new_id,
            old_last_tick=int(i - 1),
            new_first_tick=int(i),
            status="excluded",
            reason=None,
            arms={},
        )
        rows.append(row)
        if not old_id or not new_id:
            row["reason"] = "startup_or_identity_break"
            continue
        if old_id not in plans or new_id not in plans:
            row["reason"] = "unknown_plan_identity"
            continue
        old, new = plans[old_id], plans[new_id]
        row.update(
            old_request_seq=old.request_seq,
            new_request_seq=new.request_seq,
            switch_ns=new.start_ns,
            old_action_space=old.action_space,
            new_action_space=new.action_space,
        )
        event = boundaries.get(new_id)
        if event is None or event.get("request_seq") != new.request_seq:
            row["reason"] = "missing_or_mismatched_boundary_event"
            continue
        if (
            evidence["breaks"][i]
            or not evidence["valid"][i - 1 : i + 1].all()
            or times[i] - times[i - 1] > max_gap_s * 1e9
        ):
            row["reason"] = "pause_gap_or_invalid_reference"
            continue
        if not times[i - 1] < new.start_ns <= times[i]:
            row["reason"] = "committed_start_not_bracketed_by_execution"
            continue
        if int(event.get("monotonic_ns", times[i] + 1)) > times[i]:
            row["reason"] = "boundary_event_after_execution"
            continue
        if old.start_ns > new.start_ns:
            row["reason"] = "nonmonotonic_plan_start"
            continue
        for group, model in models.items():
            end = old.start_ns + (len(old.groups[group]) - 1) * old.dt_ns
            kind = "continuous" if new.start_ns <= end else "end_hold"
            if kind == "continuous":
                outgoing = sample(old, group, new.start_ns)
                a = (
                    outgoing[:3]
                    if old.action_space == "absolute_tcp_pose"
                    else model.fk(outgoing)[:3, 3]
                )
            else:
                # Explicit hold convention: the actual last recorded held target,
                # never an assumed/clamped old endpoint.
                if times[i - 1] <= end:
                    row["arms"][group] = dict(status="missing", reason="hold_not_observed")
                    continue
                outgoing = evidence["commands"][group][i - 1]
                held_indices = np.flatnonzero(
                    (ids == old_id) & (times > end) & (times < new.start_ns)
                )
                if not len(held_indices) or not np.allclose(
                    evidence["commands"][group][held_indices], outgoing, atol=1e-7, rtol=0
                ):
                    row["arms"][group] = dict(status="missing", reason="hold_not_constant")
                    continue
                a = model.fk(outgoing)[:3, 3]
            incoming = new.groups[group][0]
            b = (
                incoming[:3]
                if new.action_space == "absolute_tcp_pose"
                else model.fk(incoming)[:3, 3]
            )
            if not np.isfinite([a, b]).all():
                raise ValueError("invalid_fk_output")
            row["arms"][group] = dict(
                status="available",
                boundary_class=kind,
                seam_mm=float(np.linalg.norm(b - a) * 1000),
                old_xyz_m=a.tolist(),
                new_xyz_m=b.tolist(),
                hold_duration_s=max(0, new.start_ns - end) / 1e9,
                overlap={str(n): compare_window(old, new, group, model, n) for n in (10, 20, 30)},
            )
        row["status"] = (
            "available"
            if any(v["status"] == "available" for v in row["arms"].values())
            else "excluded"
        )
        if row["status"] == "excluded":
            row["reason"] = "no_eligible_arm"
    summary = {}
    for group in models:
        summary[group] = {}
        for kind in CLASSES:
            selected = [r for r in rows if r["arms"].get(group, {}).get("boundary_class") == kind]
            values = [r["arms"][group]["seam_mm"] for r in selected]
            worst = max(selected, key=lambda r: r["arms"][group]["seam_mm"]) if selected else None
            summary[group][kind] = dict(
                n=len(values),
                mean=float(np.mean(values)) if values else None,
                p95=float(np.percentile(values, 95, method="linear")) if values else None,
                max=max(values) if values else None,
                worst_boundary=None
                if worst is None
                else {k: worst[k] for k in ("old_plan_id", "new_plan_id", "switch_ns")},
            )
    return dict(
        status="available",
        reason=None,
        schema="executed-reference-seams-v1",
        input_sha256=evidence["input_sha256"],
        rows=rows,
        summary=summary,
        exclusions=dict(Counter(r["reason"] for r in rows if r["reason"])),
        accepted_plan_count=len(plans),
        executed_plan_count=len(set(ids) - {""}),
    )


def aggregate(episodes):
    result = {}
    for group in sorted({g for r in episodes for g in r.get("formal_seam", {}).get("summary", {})}):
        result[group] = {}
        for kind in CLASSES:
            selected = [
                (r, r["formal_seam"]["summary"][group][kind])
                for r in episodes
                if r.get("formal_seam", {}).get("summary", {}).get(group, {}).get(kind, {}).get("n")
            ]
            worst = max(selected, key=lambda pair: pair[1]["max"]) if selected else None
            result[group][kind] = dict(
                episode_count=len(selected),
                boundary_count=sum(s["n"] for _, s in selected),
                mean=float(np.mean([s["mean"] for _, s in selected])) if selected else None,
                p95=float(np.mean([s["p95"] for _, s in selected])) if selected else None,
                max=worst[1]["max"] if worst else None,
                worst=None
                if worst is None
                else dict(episode_dir=worst[0]["episode_dir"], **worst[1]["worst_boundary"]),
            )
    return result


def aggregate_overlap(episodes):
    """Each window gets its own eligible sample set; episode means remain equal."""
    from manimux.evaluation.trajectory import METRICS

    result = {}
    groups = sorted({g for r in episodes for g in r.get("formal_seam", {}).get("summary", {})})
    for group in groups:
        result[group] = {}
        for n in (10, 20, 30):
            per_episode = []
            for episode in episodes:
                rows = [
                    r["arms"][group]["overlap"][str(n)]
                    for r in episode.get("formal_seam", {}).get("rows", [])
                    if r["arms"].get(group, {}).get("status") == "available"
                    and r["arms"][group]["overlap"][str(n)]["status"] == "valid"
                ]
                if rows:
                    per_episode.append(rows)
            metrics = {}
            for key in METRICS:
                means = [
                    float(np.mean([r[key] for r in rows if r[key] is not None]))
                    for rows in per_episode
                    if any(r[key] is not None for r in rows)
                ]
                metrics[key] = dict(
                    episode_count=len(means),
                    mean_of_episode_means=float(np.mean(means)) if means else None,
                )
            result[group][str(n)] = dict(
                numeric_boundary_count=sum(map(len, per_episode)), metrics=metrics
            )
    return result
