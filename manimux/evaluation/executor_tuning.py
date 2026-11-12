"""Offline, frozen-reference screening of legacy SmoothExecutor arm settings.

This is an executor intervention, not a policy/IK/hardware simulation. The batched
kernel is checked against the public executor and each recording must reproduce
before its candidate scores can participate in selection.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import zarr

ARMS = ("left_arm", "right_arm")
TASKS = {"pack_plate": "pack_plate", "pack the plate": "pack_plate", "pack_bottle": "pack_bottle"}
LOCKED = [
    "handoff_skip_steps",
    "all inference settings",
    "action/control timing",
    "IK",
    "arm max_velocity",
    "max_step_dt_s",
    "position limits",
    "gripper",
]


def sha(value):
    return hashlib.sha256(value).hexdigest()


def json_sha(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def candidates():
    """Caps never relax a recorded acceleration limit; None retains its value."""
    result = [dict(id="recorded", cutoff_hz=None, acceleration_cap=None, mode=None)]
    for cutoff in (4, 8, 12, 20, 40):
        for cap in (None, 8, 16, 32):
            for mode in ("isotropic", "per_joint"):
                result.append(
                    dict(
                        id=f"f{cutoff}_a{cap or 'recorded'}_{mode}",
                        cutoff_hz=cutoff,
                        acceleration_cap=cap,
                        mode=mode,
                    )
                )
    return result


def parameters(config, candidate):
    p = dict(config["executor"]["smooth"])
    if candidate["cutoff_hz"] is not None:
        p["cutoff_hz"] = candidate["cutoff_hz"]
    if candidate["mode"] is not None:
        p["mode"] = candidate["mode"]
    cap = candidate["acceleration_cap"]
    if cap is not None:
        p["max_acceleration"] = min(p["max_acceleration"] or cap, cap)
    return p


def load_episode(path):
    path = Path(path).resolve()
    manifest_bytes = (path.parent / "session-manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    config = manifest["config"]
    meta_bytes = (path / "meta.json").read_bytes()
    meta = json.loads(meta_bytes)
    task = TASKS.get(meta.get("task"))
    if task is None:
        raise ValueError("unsupported_or_missing_task")
    if "umi_dp" not in config.get("policy", {}).get("adapter", {}).get("type", ""):
        raise ValueError("unsupported_policy_adapter")
    executor = config.get("executor", {})
    if config.get("execution", {}).get("ik", {}).get("mode", "chunk") != "chunk":
        raise ValueError("step_ik_reference_depends_on_command")
    if executor.get("type") != "smooth":
        raise ValueError("unsupported_executor")
    smooth = executor["smooth"]
    if smooth.get("tracking_mode", "legacy") != "legacy":
        raise ValueError("braking_requires_unrecorded_lookahead")
    if smooth.get("release_guard") or smooth.get("grasp_guard"):
        raise ValueError("feedback_guard_not_replayable")
    indices = (smooth.get("gripper") or {}).get("group_indices", {})
    if any(indices.get(g) != 7 for g in ARMS):
        raise ValueError("requires_independent_gripper_at_index_7")
    root = zarr.open_group(str(path / "data.zarr"), mode="r")
    ticks = root["ticks"]
    times = ticks["monotonic_ns"][:]
    if len(times) < 100:
        raise ValueError("fewer_than_100_ticks")
    if np.any(np.diff(times) <= 0):
        raise ValueError("nonmonotonic_ticks")
    arrays = {
        stage: np.stack([ticks[stage][g][:] for g in ARMS], axis=1)
        for stage in ("state", "scheduled", "command")
    }
    if any(x.shape != (len(times), 2, 8) or not np.isfinite(x).all() for x in arrays.values()):
        raise ValueError("invalid_dual_arm_arrays")
    ids = ticks["plan_id"][:].astype(str)
    known = set()
    for plan in root["plans"].values():
        known.add(plan.attrs["plan_id"])
        for stage in ("canonical_raw", "committed"):
            if dict(plan[stage].attrs).get("hold_from_step"):
                raise ValueError("per_group_hold_not_replayable")
    if any(pid and pid not in known for pid in ids):
        raise ValueError("unknown_tick_plan")
    # Edge records measured-state placeholders in inference gaps. A baseline
    # replay gate below rejects episodes where this classification is ambiguous.
    active = (ids != "") & np.any(
        abs(arrays["scheduled"][..., :7] - arrays["state"][..., :7]) > 1e-10, axis=(1, 2)
    )
    if active.sum() < 100:
        raise ValueError("fewer_than_100_active_ticks")
    event_bytes = (path / "events.jsonl").read_bytes()
    events = [json.loads(line) for line in event_bytes.splitlines() if line.strip()]
    dt = 1 / config["robot"]["control_hz"]
    resets = np.r_[True, np.diff(times) > max(0.25, 10 * dt) * 1e9]
    for event in events:
        if (event.get("kind") == "robogui_control_state" and event.get("paused")) or event.get(
            "kind"
        ) in {"robogui_home_requested", "robogui_reset_requested"}:
            i = (
                np.searchsorted(times, event["monotonic_ns"])
                if "monotonic_ns" in event
                else event.get("step", -1)
            )
            if 0 <= i < len(times):
                resets[i] = True
    source_files = (
        manifest.get("source_provenance", {})
        .get("repositories", {})
        .get("manimux", {})
        .get("files", {})
    )
    source_check = {}
    for filename in ("manimux/runtime/executors/smooth.py", "manimux/runtime/executors/limits.py"):
        source_check[filename] = source_files.get(filename, {}).get("sha256") == sha(
            Path(filename).read_bytes()
        )
    digest = hashlib.sha256(
        manifest_bytes
        + meta_bytes
        + event_bytes
        + times.tobytes()
        + json.dumps(ids.tolist()).encode()
    )
    for x in arrays.values():
        digest.update(x.tobytes())
    result = json.loads((path / "result.json").read_text())
    info = dict(
        path=str(path),
        session=str(path.parent),
        task=task,
        status="partial" if path.name.endswith(".partial") else "finalized",
        run_completed=result.get("run_completed"),
        terminal_reason=result.get("terminal_reason"),
        ticks=len(times),
        active_ticks=int(active.sum()),
        reset_anchors=int(resets.sum()),
        config_sha256_canonical_json=json_sha(config),
        manifest_sha256=sha(manifest_bytes),
        replay_input_sha256=digest.hexdigest(),
        executor_source_matches=source_check,
        recorded_smooth=smooth,
        inference=config["inference"],
        policy_identity=meta.get("policy_backend", {}),
        expected_backend=config.get("policy", {}).get("expected_backend", {}),
    )
    return dict(
        info=info,
        config=config,
        manifest=manifest,
        arrays=arrays,
        times=times,
        ids=ids,
        active=active,
        resets=resets,
        dt=dt,
        root=root,
    )


def replay(data, choices):
    """Carry hypothetical q/v across ALL chunk changes; never reanchor per tick.

    Explicit discontinuities anchor once to a recorded command and are excluded
    from derivative metrics. Grippers are kept outside this arm-only kernel.
    """
    settings = [parameters(data["config"], c) for c in choices]
    dt = data["dt"]

    def expand(x):
        return np.asarray(x)[:, None, None]

    alpha = expand([dt / (dt + 1 / (2 * np.pi * p["cutoff_hz"])) for p in settings])
    vmax = expand(
        [
            np.inf
            if p["max_velocity"] is None
            else p["max_velocity"]
            * (min(dt, p["max_step_dt_s"]) / dt if p["max_step_dt_s"] is not None else 1)
            for p in settings
        ]
    )
    amax = expand(
        [np.inf if p["max_acceleration"] is None else p["max_acceleration"] for p in settings]
    )
    pos = expand(
        [np.inf if p["position_limit_abs"] is None else p["position_limit_abs"] for p in settings]
    )
    isotropic = expand([p["mode"] == "isotropic" for p in settings])
    recorded = data["arrays"]["command"][..., :7]
    ref = data["arrays"]["scheduled"][..., :7]
    output = np.empty((len(choices), len(ref), 2, 7))
    speed_flags = np.zeros((len(choices), len(ref), 2), bool)
    accel_flags = np.zeros_like(speed_flags)
    q = np.broadcast_to(recorded[0], (len(choices), 2, 7)).copy()
    v = np.zeros_like(q)

    def bound(values, limit):
        scale = np.minimum(
            1.0, limit / np.maximum(np.max(abs(values), axis=-1, keepdims=True), 1e-300)
        )
        return np.where(isotropic, values * scale, np.clip(values, -limit, limit))

    for i in range(len(ref)):
        if data["resets"][i]:
            q[:] = recorded[i]
            v[:] = 0
        elif not data["active"][i]:
            v[:] = 0
        else:
            target = q + alpha * (ref[i] - q)
            desired = (target - q) / dt
            speed_flags[:, i] = np.any(abs(desired) > vmax + 1e-10, axis=-1)
            limited = bound(desired, vmax)
            new_v = v + bound(limited - v, amax * dt)
            accel_flags[:, i] = np.any(abs(new_v - limited) > 1e-10, axis=-1)
            q = np.clip(q + new_v * dt, -pos, pos)
            v = new_v
        output[:, i] = q
    return output, speed_flags, accel_flags


def masks(data):
    valid = data["active"] & ~data["resets"]
    # Derivatives retain chunk switches and holds, but never cross a reset.
    velocity = ~data["resets"].copy()
    velocity[0] = False
    acceleration = velocity & np.r_[False, velocity[:-1]]
    jerk = acceleration & np.r_[False, acceleration[:-1]]
    seam = valid & np.r_[False, data["ids"][1:] != data["ids"][:-1]] & acceleration
    return valid, velocity, acceleration, jerk, seam


def metrics(data, output, speed_flags, accel_flags):
    def rms(x):
        return float(np.sqrt(np.mean(np.square(x)))) if x.size else None

    def p95(x):
        return float(np.percentile(x, 95)) if x.size else None

    valid, mv, ma, mj, seam = masks(data)
    ref = data["arrays"]["scheduled"][..., :7]
    v = np.diff(output, axis=1, prepend=output[:, :1]) / data["dt"]
    a = np.diff(v, axis=1, prepend=v[:, :1]) / data["dt"]
    j = np.diff(a, axis=1, prepend=a[:, :1]) / data["dt"]
    since = np.maximum.accumulate(
        np.where(np.r_[True, data["ids"][1:] != data["ids"][:-1]], np.arange(len(valid)), 0)
    )
    early = (data["times"] - data["times"][since]) < 100_000_000
    reference_velocity = np.diff(ref, axis=0, prepend=ref[:1]) / data["dt"]
    reference_pairs = valid & np.r_[False, valid[:-1]]
    smooth = data["config"]["executor"]["smooth"]
    speed_limit = smooth["max_velocity"]
    if speed_limit is not None and smooth["max_step_dt_s"] is not None:
        speed_limit *= min(data["dt"], smooth["max_step_dt_s"]) / data["dt"]
    result = []
    for k, q in enumerate(output):
        groups = {}
        for g, name in enumerate(ARMS):
            error = q[valid, g] - ref[valid, g]
            lower = data["config"]["executor"]["command_safety"]["position_lower"][name][:7]
            upper = data["config"]["executor"]["command_safety"]["position_upper"][name][:7]
            groups[name] = dict(
                error_rms_deg=float(np.degrees(rms(error))),
                error_peak_joint_p95_deg=p95(np.degrees(np.max(abs(error), axis=-1))),
                acceleration_rms_rad_s2=rms(a[k, ma, g]),
                jerk_rms_rad_s3=rms(j[k, mj, g]),
                seam_delta_v_p95_rad_s=p95(np.max(abs(a[k, seam, g]), axis=-1) * data["dt"]),
                speed_clip_fraction=float(speed_flags[k, valid, g].mean()),
                acceleration_clip_fraction=float(accel_flags[k, valid, g].mean()),
                max_joint_speed_rad_s=float(np.max(abs(v[k, mv, g]))),
                max_joint_acceleration_rad_s2=float(np.max(abs(a[k, ma, g]))),
                position_violation_ticks=int(
                    np.any(
                        (q[valid, g] < np.asarray(lower) - 1e-7)
                        | (q[valid, g] > np.asarray(upper) + 1e-7),
                        axis=-1,
                    ).sum()
                ),
            )
            for label, window in (("first_100ms", early), ("later", ~early)):
                mask = valid & window
                groups[name][f"{label}_speed_clip_fraction"] = (
                    float(speed_flags[k, mask, g].mean()) if mask.any() else None
                )
            peak = np.max(abs(reference_velocity[:, g]), axis=-1)
            mask = reference_pairs & ~early
            groups[name]["later_reference_at_90pct_speed_limit_fraction"] = (
                float(np.mean(peak[mask] >= 0.9 * speed_limit))
                if mask.any() and speed_limit is not None
                else None
            )
            groups[name]["reference_required_speed_p95_rad_s"] = p95(peak[reference_pairs])
        result.append(groups)
    return result


def split_sessions(infos):
    sessions = sorted({r["session"] for r in infos}, key=lambda p: Path(p).name)
    if len(sessions) < 2:
        return {s: "development" for s in sessions}
    n = max(1, int(np.ceil(len(sessions) / 3)))
    return {s: "holdout" if s in sessions[-n:] else "development" for s in sessions}


def aggregate(rows, choices, split):
    selected = [r for r in rows if r["split"] == split]
    if not selected:
        return []
    result = []
    for k, candidate in enumerate(choices):
        values = {}
        violations = []
        for name in ARMS:
            keys = selected[0]["metrics"][k][name]
            values[name] = {
                key: float(
                    np.mean(
                        [
                            r["metrics"][k][name][key]
                            for r in selected
                            if r["metrics"][k][name][key] is not None
                        ]
                    )
                )
                if any(r["metrics"][k][name][key] is not None for r in selected)
                else None
                for key in keys
            }
        # Require smoothness guardrails per episode AND per arm, not only pooled.
        for row in selected:
            p = parameters(row["config"], candidate)
            dt = 1 / row["config"]["robot"]["control_hz"]
            speed = p["max_velocity"]
            if speed is not None and p["max_step_dt_s"] is not None:
                speed *= min(dt, p["max_step_dt_s"]) / dt
            for name in ARMS:
                base, trial = row["metrics"][0][name], row["metrics"][k][name]
                for key in ("acceleration_rms_rad_s2", "jerk_rms_rad_s3", "seam_delta_v_p95_rad_s"):
                    if base[key] is not None and trial[key] > base[key] * 1.10 + 1e-8:
                        violations.append(
                            dict(
                                path=row["info"]["path"],
                                arm=name,
                                metric=key,
                                baseline=base[key],
                                candidate=trial[key],
                                limit_ratio=1.10,
                            )
                        )
                if trial["error_rms_deg"] > base["error_rms_deg"] * 1.05 + 1e-8:
                    violations.append(
                        dict(
                            path=row["info"]["path"],
                            arm=name,
                            metric="error_rms_deg",
                            baseline=base["error_rms_deg"],
                            candidate=trial["error_rms_deg"],
                            limit_ratio=1.05,
                        )
                    )
                if trial["position_violation_ticks"]:
                    violations.append(
                        dict(path=row["info"]["path"], arm=name, metric="position_violation_ticks")
                    )
                if speed is not None and trial["max_joint_speed_rad_s"] > speed + 1e-7:
                    violations.append(
                        dict(path=row["info"]["path"], arm=name, metric="velocity_limit")
                    )
        score = float(np.mean([values[name]["error_rms_deg"] for name in ARMS]))
        result.append(
            dict(
                candidate=candidate["id"],
                error_score_deg=score,
                passes_joint_screen=not violations,
                violations=violations,
                episodes=len(selected),
                groups=values,
            )
        )
    return result


def exploratory_choices(development):
    """Development-only tradeoffs for inspection, NOT approved recommendations.

    Average constraints may hide individual regressions; callers must retain the
    stricter per-episode violations and label these choices experimental.
    """
    base = development[0]

    def ratio(item, key):
        return np.mean([item["groups"][g][key] / max(base["groups"][g][key], 1e-12) for g in ARMS])

    safe_average = [
        x
        for x in development
        if all(
            x["groups"][g][key] <= base["groups"][g][key] * 1.10 + 1e-8
            for g in ARMS
            for key in ("jerk_rms_rad_s3", "acceleration_rms_rad_s2", "seam_delta_v_p95_rad_s")
        )
    ]
    near_tracking = [
        x for x in development if x["error_score_deg"] <= base["error_score_deg"] * 1.10
    ]
    return dict(
        tracking_only=min(development, key=lambda x: x["error_score_deg"])["candidate"],
        average_balance=min(safe_average, key=lambda x: x["error_score_deg"])["candidate"],
        smoothness_priority=min(near_tracking, key=lambda x: ratio(x, "jerk_rms_rad_s3"))[
            "candidate"
        ],
    )


def tcp_metrics(data, output):
    """Offline FK only. Never construct a RobotBase or open a device.

    All recorded embodiment source hashes must match. Full control-rate FK
    retains seam impulses. Derivatives use the executor's nominal control dt.
    """
    from manimux.embodiments.robot import RobotModel

    files = (
        data["manifest"]
        .get("source_provenance", {})
        .get("repositories", {})
        .get("manimux", {})
        .get("files", {})
    )
    paths = [
        p for p in files if p.startswith(("manimux/embodiments/", "manimux/configs/embodiment/"))
    ]
    if not paths or any(
        not Path(p).is_file() or sha(Path(p).read_bytes()) != files[p]["sha256"] for p in paths
    ):
        raise ValueError("FK embodiment sources absent or different from recording")
    robot = RobotModel.from_config(data["config"]["robot"]["config"])
    valid, mv, ma, mj, _ = masks(data)
    result = [{} for _ in output]
    measured = {}
    for g, name in enumerate(ARMS):
        model = robot.kinematics.models[name]

        def fk(rows, model=model):
            return np.asarray([model.fk(q) for q in rows])

        reference = fk(data["arrays"]["scheduled"][:, g])
        state = fk(data["arrays"]["state"][:, g])
        for k, arm in enumerate(output):
            joints = data["arrays"]["command"][:, g].copy()
            joints[:, :7] = arm[:, g]
            pose = fk(joints)
            xyz = pose[:, :3, 3] * 1000
            v = np.diff(xyz, axis=0, prepend=xyz[:1]) / data["dt"]
            a = np.diff(v, axis=0, prepend=v[:1]) / data["dt"]
            j = np.diff(a, axis=0, prepend=a[:1]) / data["dt"]
            error = np.linalg.norm(pose[valid, :3, 3] - reference[valid, :3, 3], axis=-1) * 1000
            trace = np.einsum("nij,nij->n", pose[valid, :3, :3], reference[valid, :3, :3])
            angle = np.degrees(np.arccos(np.clip((trace - 1) / 2, -1, 1)))
            result[k][name] = dict(
                reference_error_rms_mm=float(np.sqrt(np.mean(error**2))),
                reference_error_p95_mm=float(np.percentile(error, 95)),
                reference_rotation_p95_deg=float(np.percentile(angle, 95)),
                speed_p95_mm_s=float(np.percentile(np.linalg.norm(v[mv], axis=-1), 95)),
                acceleration_rms_mm_s2=float(np.sqrt(np.mean(np.sum(a[ma] ** 2, axis=-1)))),
                jerk_rms_mm_s3=float(np.sqrt(np.mean(np.sum(j[mj] ** 2, axis=-1)))),
            )
            if k == 0:
                measured[name] = dict(
                    recorded_command_state_p95_mm=float(
                        np.percentile(
                            np.linalg.norm(pose[valid, :3, 3] - state[valid, :3, 3], axis=-1)
                            * 1000,
                            95,
                        )
                    )
                )
    return result, dict(checked_embodiment_files=len(paths), recorded_tracking_context=measured)


def tcp_screen(base, trial):
    for name in ARMS:
        for key in (
            "reference_error_rms_mm",
            "reference_error_p95_mm",
            "reference_rotation_p95_deg",
        ):
            if trial[name][key] > base[name][key] * 1.05 + 1e-6:
                return False
        for key in ("acceleration_rms_mm_s2", "jerk_rms_mm_s3", "speed_p95_mm_s"):
            if trial[name][key] > base[name][key] * 1.10 + 1e-6:
                return False
    return True
