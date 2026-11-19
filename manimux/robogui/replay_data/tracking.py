"""Offline, timestamp-aligned tracking evidence. Never reconstruct missing model outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr
from scipy.spatial.transform import Rotation, Slerp

from manimux.embodiments.robot.base import RobotModel


def interpolate_rows(rows, phase):
    p = np.clip(np.asarray(phase), 0, len(rows) - 1)
    lo = np.floor(p).astype(int)
    hi = np.minimum(lo + 1, len(rows) - 1)
    return rows[lo] + (rows[hi] - rows[lo]) * (p - lo)[..., None]


def pose_rows(rows, phase):
    """Linear XYZ and shortest-arc SLERP; rows use qw,qx,qy,qz."""
    result = np.tile(np.eye(4), (len(phase), 1, 1))
    result[:, :3, 3] = interpolate_rows(rows[:, :3], phase)
    quats = rows[:, [4, 5, 6, 3]]
    if len(rows) == 1:
        result[:, :3, :3] = Rotation.from_quat(quats[0]).as_matrix()
    else:
        result[:, :3, :3] = Slerp(np.arange(len(rows)), Rotation.from_quat(quats))(
            np.clip(phase, 0, len(rows) - 1)
        ).as_matrix()
    return result


def source_phase(plan, stamps, *, starts_at_commit=False):
    """Map accepted execution times back to pre-retiming source rows.

    Retimed uses its exact knot clock, including removed progress and endpoint
    rounding. The bridge modifies the executable path, not this source phase.
    Unsupported waypoint remapping is deliberately left missing.
    """
    raw, infra, committed = (plan[k] for k in ("canonical_raw", "infra_output", "committed"))
    obs, dt = int(raw.attrs["observation_time_ns"]), int(raw.attrs["dt_ns"])
    metadata = dict(infra.attrs).get("metadata", {})
    retime = metadata.get("rtc_retime")
    if retime:
        knots = np.asarray(retime["source_knots_s"], dtype=float)
        start = int(retime["handoff_time_ns"])
        end = obs + (int(retime["execution_horizon"]) - 1) * int(infra.attrs["dt_ns"])
        phase_time = np.interp(float(retime["matched_source_phase"]), np.arange(len(knots)), knots)
        elapsed_s = (stamps - start) / 1e9
        if retime.get("schema") != "rtc-retime-v2":
            # Older runtime versions stretched the surviving path to the grid endpoint.
            elapsed_s = (stamps - start) / max(1, end - start) * (knots[-1] - phase_time)
        phase = np.interp(
            phase_time + elapsed_s,
            knots,
            np.arange(len(knots)),
        )
        return phase, (stamps >= start) & (stamps <= end)
    if int(raw.attrs.get("source_offset_steps", 0)) or metadata.get("ik_seed_plan_id"):
        return np.zeros(len(stamps)), np.zeros(len(stamps), dtype=bool)
    start = int(committed.attrs["start_time_ns"])
    count = next(iter(committed.arrays()))[1].shape[0]
    end = start + (count - 1) * int(committed.attrs["dt_ns"])
    trim = int(committed.attrs.get("time_trimmed_steps", 0))
    skip = int(committed.attrs.get("handoff_skipped_steps", 0))
    phase = (stamps - start) / dt + trim + skip
    if starts_at_commit and "runtime_trajectory" in raw:
        dense = raw["runtime_trajectory"]
        phase += (int(dense.attrs["start_time_ns"]) - obs) / dt
        last_source = obs + (count - 1) * dt
        dense_count = min(
            next(iter(dense.arrays()))[1].shape[0],
            (last_source - int(dense.attrs["start_time_ns"])) // int(dense.attrs["dt_ns"]) + 1,
        )
        end = start + (dense_count - 1) * int(dense.attrs["dt_ns"])
    return phase, (stamps >= start) & (stamps <= end)


def pose_error(a, b):
    pos = np.linalg.norm(a[:, :3, 3] - b[:, :3, 3], axis=1) * 1000
    trace = np.einsum("nij,nij->n", a[:, :3, :3], b[:, :3, :3])
    angle = np.degrees(np.arccos(np.clip((trace - 1) / 2, -1, 1)))
    return {"mm": pos, "deg": angle}


def segments(stamps, labels):
    """Half-open display intervals; sample endpoints remain unmodified."""
    boundaries = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    return [
        (int(stamps[a]), int(stamps[b]) if b < len(stamps) else int(stamps[-1]), str(labels[a]))
        for a, b in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]


@dataclass
class TrackingData:
    episode: Path
    stamps: np.ndarray
    poses: dict
    errors: dict
    plan_ids: np.ndarray
    active: np.ndarray
    timeline: list
    requests: list
    plans: dict
    notes: list[str]


def load_tracking(episode: Path, fallback_models=None):
    episode = Path(episode)
    root = zarr.open_group(str(episode / "data.zarr"), mode="r")
    ticks = root["ticks"]
    stamps = np.asarray(ticks["monotonic_ns"], dtype=np.int64)
    if not len(stamps) or np.any(np.diff(stamps) <= 0):
        raise ValueError("Tracking replay requires increasing, nonempty tick timestamps")
    notes = []
    episode_meta = json.loads((episode / "meta.json").read_text())
    starts_at_commit = episode_meta.get("action_start_mode") == "first_step_when_ready"
    manifest = episode.parent / "session-manifest.json"
    if manifest.is_file():
        cfg = json.loads(manifest.read_text())["config"]
        models = RobotModel.from_config(cfg["robot"]["config"]).kinematics.models
    elif fallback_models is not None:
        models = fallback_models
        notes.append("No session model snapshot: FK uses the selected display model.")
    else:
        raise ValueError("No recorded robot model or fallback model for FK")
    events_path = episode / "events.jsonl"
    events = (
        [json.loads(s) for s in events_path.read_text().splitlines()]
        if events_path.exists()
        else []
    )
    controls = [e for e in events if e["kind"] == "robogui_control_state"]
    running = np.ones(len(stamps), dtype=bool)
    if controls:
        controls.sort(key=lambda e: e["monotonic_ns"])
        ci = np.searchsorted([e["monotonic_ns"] for e in controls], stamps, side="right") - 1
        running = np.array([i >= 0 and not controls[i].get("paused", False) for i in ci])
    ids = np.asarray(ticks["plan_id"])
    models = {name: models[name] for name in ticks["state"].array_keys()}
    poses, errors, plans = {}, {}, {}
    n = len(stamps)
    for name, model in models.items():
        poses[name] = {key: np.full((n, 4, 4), np.nan) for key in ("policy", "ik")}
        for key, stream in [
            ("reference", "scheduled"),
            ("command", "command"),
            ("feedback", "state"),
        ]:
            rows = np.asarray(ticks[stream][name])
            if len(rows) != n:
                raise ValueError(f"{stream}/{name}: sample count mismatch")
            poses[name][key] = np.array([model.fk(q) for q in rows])
    active = np.zeros(n, dtype=bool)
    coverage = np.zeros(n, dtype=bool)
    if "plans" in root:
        for key in root["plans"]:
            p = root["plans"][key]
            pid = str(p.attrs["plan_id"])
            seq = int(p.attrs["request_seq"])
            raw = p["canonical_raw"]
            infra = p["infra_output"]
            meta = dict(infra.attrs).get("metadata", {})
            retime = meta.get("rtc_retime", {})
            start = int(retime.get("handoff_time_ns", p["committed"].attrs["start_time_ns"]))
            plans[pid] = {
                "seq": seq,
                "start_ns": start,
                "retimed": bool(retime),
                "matched": retime.get("matched"),
                "lag_summary": meta.get("diff_ik_lag", {}),
            }
            selected = np.flatnonzero((ids == pid) & running)
            if not len(selected):
                continue
            # Timeline coverage is distinct from availability of a source mapping.
            dense = infra.get("runtime_trajectory")
            c = p["committed"]
            phase, valid = source_phase(p, stamps[selected], starts_at_commit=starts_at_commit)
            if retime and dense is not None:
                end = int(dense.attrs["start_time_ns"]) + (
                    next(iter(dense.arrays()))[1].shape[0] - 1
                ) * int(dense.attrs["dt_ns"])
                covered = (stamps[selected] >= start) & (stamps[selected] <= end)
            else:
                # Use the committed clock for ordinary/serial playback, rather
                # than the original prediction clock that preceded inference.
                count = next(iter(c.arrays()))[1].shape[0]
                end = int(c.attrs["start_time_ns"]) + (count - 1) * int(c.attrs["dt_ns"])
                covered = (
                    valid
                    if starts_at_commit
                    else ((stamps[selected] >= start) & (stamps[selected] <= end))
                )
            coverage[selected[covered]] = True
            if raw.attrs["action_space"] != "joint_position":
                continue
            valid &= covered
            selected, phase = selected[valid], phase[valid]
            active[selected] = True
            for name, model in models.items():
                if name not in raw:
                    continue
                qrows = np.asarray(raw[name])
                ok = (phase >= 0) & (phase <= len(qrows) - 1 + 1e-6)
                ix, ph = selected[ok], phase[ok]
                if not len(ix):
                    continue
                if "runtime_trajectory" in raw:
                    d = raw["runtime_trajectory"]
                    source_ns = ph * int(raw.attrs["dt_ns"])
                    knots = meta.get("pre_ik_knots_s")
                    if knots is not None:
                        source_ns = np.interp(ph, np.arange(len(knots)), knots) * 1e9
                    pos = (int(raw.attrs["observation_time_ns"]) - int(d.attrs["start_time_ns"])
                           + source_ns) / int(d.attrs["dt_ns"])
                    rows = interpolate_rows(np.asarray(d[name]), pos)
                else:
                    rows = interpolate_rows(qrows, ph)
                poses[name]["ik"][ix] = np.array([model.fk(q) for q in rows])
                if "policy_targets" in raw and name in raw["policy_targets"]:
                    target = raw["policy_targets"]
                    if target.attrs.get("schema") != "absolute-tcp-targets-v1":
                        raise ValueError("Unknown recorded policy target schema")
                    target_rows = np.asarray(target[name])
                    if (
                        target_rows.ndim != 2
                        or target_rows.shape[1] != 8
                        or not np.isfinite(target_rows).all()
                    ):
                        raise ValueError("Invalid recorded policy targets")
                    good = ph <= len(target_rows) - 1 + 1e-6
                    poses[name]["policy"][ix[good]] = pose_rows(target_rows, ph[good])
    for name, streams in poses.items():
        errors[name] = {}
        for label, a, b in [
            ("policy_ik", "policy", "ik"),
            ("ik_feedback", "ik", "feedback"),
            ("reference_feedback", "reference", "feedback"),
            ("command_feedback", "command", "feedback"),
        ]:
            values = pose_error(streams[a], streams[b])
            for values_array in values.values():
                values_array[~coverage] = np.nan
            errors[name][label] = values
        if not np.isfinite(streams["policy"][:, 0, 3]).any():
            notes.append(
                f"{name}: pre-IK policy targets were not recorded; policy → IK is unavailable."
            )
    if any((ids != "") & running & ~active & coverage):
        notes.append(
            "Some plans have no supported pre-IK/source-phase mapping (e.g. waypoint or step IK)."
        )
    labels = np.array(
        [
            f"#{plans[pid]['seq']}" if pid in plans and cov else "Hold / no path"
            for pid, cov in zip(ids, coverage, strict=True)
        ],
        dtype=object,
    )
    labels[~running] = "Paused"
    requests = {}
    for e in events:
        seq = e.get("request_seq")
        if seq is None or seq < 0:
            continue
        row = requests.setdefault(seq, {"seq": seq, "status": "pending"})
        if e["kind"] == "inference_submitted":
            obs = e.get("selected_observation", {}).get("observation_time_ns")
            row.update(start_ns=e.get("monotonic_ns", obs), exact_start="monotonic_ns" in e)
        elif e["kind"] == "decode_submitted":
            row["decode_expected_ns"] = e.get("expected_start_ns")
        elif e["kind"] == "plan_accepted":
            row.update(status="accepted", end_ns=e.get("rtc_retime", {}).get("handoff_time_ns"))
        elif e["kind"] in ("plan_rejected", "inference_rejected"):
            row.update(status="rejected", end_ns=e.get("monotonic_ns"), reason=e.get("reason", ""))
        elif e["kind"] == "plan_boundary":
            row.update(end_ns=e.get("monotonic_ns"))
    notes.append(
        "Feedback TCP is FK of recorded encoders, not an external measurement "
        "or eventual endpoint. IK curves follow source progress before the handoff bridge; "
        "execution reference includes the bridge."
    )
    if episode.name.endswith(".partial"):
        notes.append(
            "Interrupted episode: only saved samples are shown; this is not a completed trial."
        )
    result_path = episode / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        video = result.get("video_recording", {})
        if video.get("error") or video.get("dropped_bundles", 0):
            notes.append(f"Video recording: {video}")
    return TrackingData(
        episode,
        stamps,
        poses,
        errors,
        ids,
        coverage,
        segments(stamps, labels),
        list(requests.values()),
        plans,
        notes,
    )
