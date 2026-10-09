#!/usr/bin/env python3
"""Build a self-contained speed report from finalized dual 7-DoF arm rollouts.

The only robot interface used is offline RobotModel FK. See
docs/usage/chunk-speed.md for the evidence contract and optional Pi05 probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from manimux.embodiments.robot import RobotModel  # noqa: E402

ARMS = ("left_arm", "right_arm")
SOURCES = ("scheduled", "command", "state")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, (np.floating, float)):
        return round(float(value), 6) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def geometry(model, joints):
    if joints.ndim != 2 or joints.shape[1] != 8 or not len(joints):
        raise ValueError("Expected nonempty [samples, 7 arm joints + 1 gripper] data")
    if not np.isfinite(joints).all():
        raise ValueError("Non-finite joint samples")
    transforms = np.array([model.fk(row) for row in joints])
    if transforms.shape != (len(joints), 4, 4) or not np.isfinite(transforms).all():
        raise ValueError("Invalid FK transforms")
    return transforms[:, :3, 3] * 1000, transforms[:, :3, :3]


def speeds(joints, xyz, rotations, times):
    dt = np.diff(times)
    if not np.isfinite(times).all() or np.any(dt <= 0):
        raise ValueError("Non-increasing sample times")
    relative_trace = np.einsum("nij,nij->n", rotations[:-1], rotations[1:])
    return {
        "tcp": np.linalg.norm(np.diff(xyz, axis=0), axis=1) / dt,
        "joint": np.linalg.norm(np.diff(joints[:, :7], axis=0), axis=1) / dt * 180 / np.pi,
        "rotation": np.arccos(np.clip((relative_trace - 1) / 2, -1, 1)) / dt * 180 / np.pi,
    }


def weighted_mean(v, times, start, end):
    weights = np.maximum(0, np.minimum(times[1:], end) - np.maximum(times[:-1], start))
    total = weights.sum()
    return float(np.dot(v, weights) / total) if total > 0 else float("nan")


def summarize(v, times, trim=0.0):
    v, times = np.asarray(v), np.asarray(times)
    if len(times) != len(v) + 1:
        raise ValueError("Speed intervals must have one more time edge than values")
    if not np.isfinite(v).all() or not np.isfinite(times).all():
        raise ValueError("Non-finite speed data")
    if np.any(v < 0) or np.any(np.diff(times) <= 0) or trim < 0:
        raise ValueError("Expected nonnegative speeds and strictly increasing time edges")
    if len(times) < 2:
        return None
    start, end = times[0] + trim, times[-1]
    if end - start < 0.12:
        return None
    boundaries = np.linspace(start, end, 13)
    mean = weighted_mean(v, times, start, end)
    early = weighted_mean(v, times, start, start + (end - start) / 3)
    late = weighted_mean(v, times, end - (end - start) / 3, end)
    profile = np.array(
        [
            weighted_mean(v, times, a, b)
            for a, b in zip(boundaries[:-1], boundaries[1:], strict=True)
        ]
    )
    return {
        "mean": mean,
        "early": early,
        "late": late,
        "ratio": late / early if early > 1 else None,
        "profile": profile,
        "duration": end - start,
    }


def aggregate(rows, threshold=5):
    eligible = [r for r in rows if r and r["mean"] >= threshold and r["ratio"] is not None]
    if not eligible:
        return {"n": 0, "total": len(rows)}
    ratio = np.array([r["ratio"] for r in eligible])
    profiles = np.array([r["profile"] for r in eligible])
    normalized = profiles / np.array([r["mean"] for r in eligible])[:, None]
    return {
        "n": len(eligible),
        "total": len(rows),
        "median_ratio": np.median(ratio),
        "slower_count": int((ratio < 1).sum()),
        "slow20_count": int((ratio <= 0.8).sum()),
        "early_median": np.median([r["early"] for r in eligible]),
        "late_median": np.median([r["late"] for r in eligible]),
        "profile_median": np.median(profiles, axis=0),
        "normalized_median": np.median(normalized, axis=0),
        "normalized_q25": np.quantile(normalized, 0.25, axis=0),
        "normalized_q75": np.quantile(normalized, 0.75, axis=0),
    }


def check_fk_sources(manifest):
    provenance = (
        manifest.get("source_provenance", {})
        .get("repositories", {})
        .get("manimux", {})
        .get("files", {})
    )
    paths = [
        p
        for p in provenance
        if p.startswith("manimux/embodiments/") or p.startswith("manimux/configs/embodiment/")
    ]
    if not paths:
        raise ValueError("Session has no recorded embodiment source hashes to verify FK")
    mismatches = [
        p for p in paths if not (ROOT / p).is_file() or digest(ROOT / p) != provenance[p]["sha256"]
    ]
    if mismatches:
        raise ValueError(f"FK source/config differs from recorded session: {mismatches}")
    return {"checked_files": len(paths), "mismatches": mismatches}


def analyze_session(session):
    """Read recorded evidence and compute speeds without loading a hardware driver."""
    session = session.resolve()
    manifest = json.loads((session / "session-manifest.json").read_text())
    fk_source_check = check_fk_sources(manifest)
    config = manifest["config"]
    robot_config = Path(config["robot"]["config"])
    if "manimux/configs/embodiment/" not in robot_config.as_posix():
        raise ValueError("Expected a recorded repository embodiment config")
    robot_config = ROOT / (
        "manimux/configs/embodiment/"
        + robot_config.as_posix().split("manimux/configs/embodiment/", 1)[1]
    )
    robot = RobotModel.from_config(robot_config)
    models = robot.kinematics.models
    if not set(ARMS).issubset(models):
        raise ValueError("Report requires left_arm and right_arm FK models")
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "session": str(session),
        "session_id": session.name,
        "recorded_config": {
            "algorithm": config["inference"]["algorithm"],
            "rtc": config["inference"]["rtc"],
            "skip": config["inference"]["handoff_skip_steps"],
            "handoff": config["inference"]["handoff"],
            "action_dt_s": config["policy"]["action_dt_s"],
            "horizon": config["policy"]["horizon_policy_steps"],
            "control_hz": config["robot"]["control_hz"],
            "adapter": config["policy"]["adapter"],
            "executor": config["executor"],
        },
        "episodes": [],
        "exclusions": [],
        "fingerprints": {},
        "fk_source_check": fk_source_check,
        "method": {
            "version": "chunk-speed-v2",
            "early_late": "time-weighted first and last thirds",
            "min_mean_tcp_speed_mm_s": 5,
            "min_early_speed": 1,
            "slow20_ratio": 0.8,
            "phase_bins": 12,
            "execution": (
                "consecutive ticks with the same nonempty plan_id, excluding first and last "
                "plan runs and gaps > 50 ms"
            ),
            "full_saved": (
                "canonical_raw decoded joint rows, own dt, excludes startup; not original "
                "model tensors or necessarily all 32 steps"
            ),
            "speed": (
                "FK TCP distance / actual tick dt; no cross-plan differencing; gripper "
                "excluded from joint L2"
            ),
            "uncertainty": (
                "descriptive within-session analysis; correlated chunks, no population "
                "confidence claim"
            ),
        },
    }
    report["fingerprints"][str(session / "session-manifest.json")] = digest(
        session / "session-manifest.json"
    )
    identities = []
    for ep in sorted(session.glob("rollout-*")):
        if ep.name.endswith(".partial") or not (ep / "result.json").exists():
            report["exclusions"].append({"episode": str(ep), "reason": "unfinished"})
            continue
        meta = json.loads((ep / "meta.json").read_text())
        events = [json.loads(line) for line in (ep / "events.jsonl").read_text().splitlines()]
        accepts = {e["request_seq"]: e for e in events if e["kind"] == "plan_accepted"}
        submits = {e["request_seq"]: e for e in events if e["kind"] == "inference_submitted"}
        z = zarr.open_group(str(ep / "data.zarr"), mode="r")
        t_ns = z["ticks/monotonic_ns"][:]
        if len(t_ns) < 2:
            report["exclusions"].append({"episode": str(ep), "reason": "fewer than two ticks"})
            continue
        times = (t_ns - t_ns[0]) / 1e9
        ids = z["ticks/plan_id"][:]
        if len(ids) != len(times) or not np.any(ids != ""):
            report["exclusions"].append(
                {"episode": str(ep), "reason": "no usable tick plan ownership"}
            )
            continue
        identities.append(meta["policy_backend"])
        startup_id = str(next(plan_id for plan_id in ids if plan_id))
        boundaries = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
        episode = {
            "id": ep.name,
            "path": str(ep),
            "duration": times[-1],
            "tick_count": len(times),
            "tick_dt_ms": np.quantile(np.diff(times) * 1000, [0.5, 0.95, 1]),
            "event_counts": dict(Counter(e["kind"] for e in events)),
            "result": json.loads((ep / "result.json").read_text()),
            "plans": [],
            "ticks": {"t": times, "arms": {}},
        }
        for arm in ARMS:
            episode["ticks"]["arms"][arm] = {}
            for source in SOURCES:
                joints = z[f"ticks/{source}/{arm}"][:]
                xyz, rot = geometry(models[arm], joints)
                episode["ticks"]["arms"][arm][source] = {
                    "xyz": xyz,
                    "speed": speeds(joints, xyz, rot, times),
                }
        plan_map = {}
        for name in sorted(z["plans"].group_keys(), key=int):
            node = z["plans"][name]
            raw, committed = node["canonical_raw"], node["committed"]
            seq = int(node.attrs["request_seq"])
            metadata = dict(raw.attrs).get("metadata", {})
            plan = {
                "key": ep.name + "/" + name,
                "index": int(name),
                "seq": seq,
                "id": str(committed.attrs["plan_id"]),
                "conditioned": submits.get(seq, {}).get("conditioned"),
                "attrs": {"canonical": dict(raw.attrs), "committed": dict(committed.attrs)},
                "accept": accepts.get(seq, {}),
                "arms": {},
                "execution": None,
                "start": (int(committed.attrs["start_time_ns"]) - int(t_ns[0])) / 1e9,
                "saved_rows": len(raw[ARMS[0]]),
                "source_offset": raw.attrs["source_offset_steps"],
                "lead_in_steps": metadata.get("handoff_lead_in_steps", 0),
            }
            plan["startup"] = plan["id"] == startup_id
            for arm in ARMS:
                plan["arms"][arm] = {}
                for label, stage in (("saved", raw), ("committed", committed)):
                    joints = stage[arm][:]
                    dt = int(stage.attrs["dt_ns"]) / 1e9
                    local_t = np.arange(len(joints)) * dt
                    xyz, rot = geometry(models[arm], joints)
                    velocity = speeds(joints, xyz, rot, local_t)
                    plan["arms"][arm][label] = {
                        "t": local_t,
                        "xyz": xyz,
                        "speed": velocity,
                        "stats": {metric: summarize(v, local_t) for metric, v in velocity.items()},
                    }
            if plan["id"] in plan_map:
                raise ValueError(f"Duplicate saved plan id: {plan['id']}")
            plan_map[plan["id"]] = plan
            episode["plans"].append(plan)
        for a, b in zip(boundaries[:-1], boundaries[1:], strict=True):
            if not ids[a]:
                continue
            plan = plan_map.get(str(ids[a]))
            if plan is None:
                raise ValueError(f"Tick plan missing from saved plans: {ids[a]}")
            if plan["execution"] is not None:
                raise ValueError(
                    "Plan reappeared in multiple tick segments; explicit segmentation required"
                )
            local = times[a:b]
            reasons = []
            if plan["startup"]:
                reasons.append("startup")
            if a > 0 and not ids[a - 1]:
                reasons.append("no preceding plan ownership")
            if b == len(ids):
                reasons.append("last plan interrupted by finish; right-censored")
            elif not ids[b]:
                reasons.append("no following plan ownership; right-censored")
            if b - a < 15:
                reasons.append("fewer than 15 ticks")
            if np.any(np.diff(local) > 0.05):
                reasons.append("tick gap > 50 ms")
            if a and times[a] - times[a - 1] > 0.05:
                reasons.append("gap at takeover")
            if b < len(ids) and times[b] - times[b - 1] > 0.05:
                reasons.append("gap at next takeover")
            plan["execution"] = {
                "eligible": not reasons,
                "reasons": reasons,
                "tick_start": int(a),
                "tick_end": int(b),
                "start": local[0],
                "end": local[-1],
                "duration": local[-1] - local[0],
                "next_tick": times[b] if b < len(times) else None,
            }
            if reasons:
                report["exclusions"].append({"plan": plan["key"], "reason": reasons})
            for arm in ARMS:
                for source in SOURCES:
                    velocity = episode["ticks"]["arms"][arm][source]["speed"]
                    plan["arms"][arm][source] = {
                        "stats": {
                            metric: summarize(v[a : b - 1], local) for metric, v in velocity.items()
                        },
                        "sensitivity": {
                            str(trim): summarize(velocity["tcp"][a : b - 1], local, trim)
                            for trim in (0.05, 0.10)
                        },
                    }
                    # Resampling checks whether tick-level feedback quantization drives the result.
                    if len(local) < 2:
                        plan["arms"][arm][source]["sensitivity"]["resampled"] = None
                        continue
                    xyz = episode["ticks"]["arms"][arm][source]["xyz"][a:b]
                    sample_t = np.linspace(local[0], local[-1], 13)
                    sample_xyz = np.column_stack(
                        [np.interp(sample_t, local, xyz[:, j]) for j in range(3)]
                    )
                    resampled_v = np.linalg.norm(np.diff(sample_xyz, axis=0), axis=1) / np.diff(
                        sample_t
                    )
                    smoothed_v = np.convolve(
                        np.pad(resampled_v, (1, 1), mode="edge"), np.ones(3) / 3, mode="valid"
                    )
                    plan["arms"][arm][source]["sensitivity"]["resampled"] = summarize(
                        smoothed_v, sample_t
                    )
        for p in [ep / "meta.json", ep / "result.json", ep / "events.jsonl"]:
            report["fingerprints"][str(p)] = digest(p)
        tree = hashlib.sha256()
        for p in sorted((ep / "data.zarr").rglob("*")):
            if p.is_file():
                tree.update(
                    str(p.relative_to(ep / "data.zarr")).encode() + b"\0" + bytes.fromhex(digest(p))
                )
        report["fingerprints"][str(ep / "data.zarr")] = {
            "sha256": tree.hexdigest(),
            "convention": "sorted relative UTF-8 path + NUL + sha256(file) bytes",
        }
        report["episodes"].append(episode)
        print(f"Loaded {ep.name}: {len(episode['plans'])} plans, {len(times)} ticks", flush=True)
    if not report["episodes"]:
        raise ValueError("No finalized episodes")
    identity_json = [json.dumps(i, sort_keys=True) for i in identities]
    if len(set(identity_json)) != 1:
        raise ValueError("Mixed backend identities within selected session")
    report["identity"] = identities[0]
    all_plans = [p for e in report["episodes"] for p in e["plans"]]
    eligible_plans = [p for p in all_plans if p["execution"] and p["execution"]["eligible"]]
    summary = {
        "all_plans": len(all_plans),
        "eligible_execution": len(eligible_plans),
        "saved_rows_counts": dict(Counter(p["saved_rows"] for p in all_plans)),
        "duration_median": np.median([p["execution"]["duration"] for p in eligible_plans])
        if eligible_plans
        else None,
        "arms": {},
    }
    for arm in ARMS:
        summary["arms"][arm] = {}
        for source in (*SOURCES, "saved"):
            chosen = (
                [p for p in all_plans if not p["startup"]] if source == "saved" else eligible_plans
            )
            rows = [p["arms"][arm][source]["stats"]["tcp"] for p in chosen]
            value = aggregate(rows)
            value["by_episode"] = {
                e["id"]: aggregate(
                    [
                        p["arms"][arm][source]["stats"]["tcp"]
                        for p in chosen
                        if p["key"].startswith(e["id"] + "/")
                    ]
                )
                for e in report["episodes"]
            }
            if source in SOURCES:
                value["sensitivity"] = {
                    str(trim): aggregate(
                        [p["arms"][arm][source]["sensitivity"][str(trim)] for p in chosen]
                    )
                    for trim in (0.05, 0.10, "resampled")
                }
                value["thresholds"] = {
                    str(threshold): aggregate(rows, threshold) for threshold in (1, 5, 10, 30)
                }
            summary["arms"][arm][source] = value
    summary["ik_lag"] = {
        arm: {
            "plans_over_5mm": sum(
                p["attrs"]["canonical"]
                .get("metadata", {})
                .get("diff_ik_lag", {})
                .get(arm, {})
                .get("worst_lag_mm", 0)
                > 5
                for p in all_plans
            ),
            "max_mm": max(
                p["attrs"]["canonical"]
                .get("metadata", {})
                .get("diff_ik_lag", {})
                .get(arm, {})
                .get("worst_lag_mm", 0)
                for p in all_plans
            ),
        }
        for arm in ARMS
    }
    report["summary"] = summary
    return report


def attach_offline(report, directory):
    """Validate checkpoint pairing and add native prediction/GT diagnostics."""
    offline_provenance = json.loads((directory / "provenance.json").read_text())
    if "finished_utc" not in offline_provenance:
        raise ValueError("Offline inference has not finished")
    model_identity = report["identity"]["model"]
    if offline_provenance["checkpoint_source"] != model_identity["checkpoint_source"]:
        raise ValueError("Offline/live checkpoint mismatch")
    if "sha256_" + offline_provenance["norm_stats_sha256"] != model_identity["norm_stats_source"]:
        raise ValueError("Offline/live normalization mismatch")
    if (
        offline_provenance["horizon"] != 32
        or not np.isfinite(offline_provenance["fps"])
        or offline_provenance["fps"] <= 0
        or offline_provenance["stride_frames"] <= 0
    ):
        raise ValueError("Offline report requires a 32-action horizon and positive fps")
    offline_episodes = json.loads((directory / "episodes.json").read_text())
    if not offline_episodes or any(not e["chunks"] for e in offline_episodes):
        raise ValueError("Offline episodes must contain predictions")
    if sum(len(e["chunks"]) for e in offline_episodes) != offline_provenance["chunks"]:
        raise ValueError("Offline chunk count disagrees with completion metadata")
    for episode in offline_episodes:
        if not isinstance(episode["id"], int) or episode["duration"] <= 0:
            raise ValueError("Expected integer offline episode id and positive duration")
        for chunk in episode["chunks"]:
            for side in ("left", "right"):
                if not chunk["images"][side].startswith("data:image/jpeg;base64,"):
                    raise ValueError("Offline preview must be an embedded JPEG")
            for arm in ARMS:
                chunk["arms"][arm]["stats"] = {}
                for source in ("pred", "gt"):
                    speed = np.asarray(chunk["arms"][arm][source])
                    if speed.shape != (31,) or not np.isfinite(speed).all() or np.any(speed < 0):
                        raise ValueError("Invalid offline speed series")
                    chunk["arms"][arm]["stats"][source] = {}
                    for mode, length in (("full", 31), ("prefix", 12)):
                        time_edges = np.arange(length + 1) / offline_provenance["fps"]
                        chunk["arms"][arm]["stats"][source][mode] = summarize(
                            speed[:length], time_edges
                        )
                    chunk["arms"][arm]["stats"][source]["ratio8"] = (
                        float(speed[-8:].mean() / speed[:8].mean())
                        if speed[:8].mean() > 1
                        else None
                    )
    offline_summary = {
        arm: {
            source: {
                mode: aggregate(
                    [
                        c["arms"][arm]["stats"][source][mode]
                        for e in offline_episodes
                        for c in e["chunks"]
                    ]
                )
                for mode in ("full", "prefix")
            }
            for source in ("pred", "gt")
        }
        for arm in ARMS
    }
    report["offline"] = {
        "provenance": offline_provenance,
        "episodes": offline_episodes,
        "summary": offline_summary,
    }
    report["fingerprints"][str(directory.resolve() / "episodes.json")] = digest(
        directory / "episodes.json"
    )
    report["fingerprints"][str(directory.resolve() / "provenance.json")] = digest(
        directory / "provenance.json"
    )


def write_report(report, out):
    """Embed all data and frontend assets; the HTML needs no network or server."""
    out.mkdir(parents=True, exist_ok=True)
    assets = Path(__file__).with_name("chunk_speed_assets")
    report["fingerprints"]["analysis_generator"] = digest(__file__)
    report["fingerprints"]["frontend_assets"] = {
        path.name: digest(path) for path in sorted(assets.iterdir()) if path.is_file()
    }
    report = plain(report)
    (out / "analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, separators=(",", ":"))
    )
    (out / "summary.json").write_text(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if "offline" in report:
        (out / "offline-summary.json").write_text(
            json.dumps(report["offline"]["summary"], ensure_ascii=False, indent=2)
        )
    template = (assets / "template.html").read_text()
    payload = json.dumps(report, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    template = template.replace("__STYLE__", (assets / "style.css").read_text())
    template = template.replace("__REPORT_SCRIPT__", (assets / "report.js").read_text())
    template = template.replace("__OFFLINE_SCRIPT__", (assets / "offline.js").read_text())
    (out / "chunk-speed.html").write_text(template.replace("__REPORT_DATA__", payload))
    print(
        json.dumps(
            {
                arm: {
                    s: {
                        k: v for k, v in stats.items() if k in ("n", "median_ratio", "slow20_count")
                    }
                    for s, stats in values.items()
                }
                for arm, values in report["summary"]["arms"].items()
            },
            indent=2,
        )
    )
    print(out / "chunk-speed.html")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--offline", type=Path, help="Optional completed offline probe directory")
    args = parser.parse_args()
    session, out = args.session.expanduser().resolve(), args.out.expanduser().resolve()
    if out == session or session in out.parents:
        parser.error("Derived outputs must be outside recordings")
    if args.offline is not None:
        args.offline = args.offline.expanduser().resolve()
        if out == args.offline or args.offline in out.parents:
            parser.error("Report output must be separate from offline inputs")
    report = analyze_session(session)
    if args.offline is not None:
        attach_offline(report, args.offline)
    write_report(report, out)


if __name__ == "__main__":
    main()
