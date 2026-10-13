"""Offline EEF smoothness of recorded scheduled references, never measured motion.

Profile v1 uses speed-normalized dimensionless jerk (T**3 / peak_speed**2),
and the positive spectral arc length of speed. Derivative stencils crossing
plan changes are excluded from jerk; SPARC retains the full speed sequence.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def velocity_metrics(velocity, dt, *, valid_jerk=None, cutoff_hz=10.0, threshold=0.05):
    """Vector velocity in m/s or rad/s; return negative LDLJ/SPARC (lower better)."""
    v = np.asarray(velocity, dtype=float)
    if v.ndim != 2 or v.shape[1] != 3 or len(v) < 8 or not np.isfinite(v).all():
        raise ValueError("velocity requires at least eight finite XYZ samples")
    if not np.isfinite(dt) or dt <= 0 or not 0 < cutoff_hz <= 0.5 / dt:
        raise ValueError("invalid dt or spectral cutoff above Nyquist")
    if not 0 < threshold < 1:
        raise ValueError("spectral threshold must be between zero and one")
    speed = np.linalg.norm(v, axis=1)
    peak = float(speed.max())
    result = {"nldlj": None, "nsparc": None, "peak_speed": peak, "reason": None}
    if peak < 1e-8:
        return result | {"reason": "stationary"}
    jerk = np.diff(v, n=2, axis=0) / dt**2
    mask = np.ones(len(jerk), dtype=bool) if valid_jerk is None else np.asarray(valid_jerk, bool)
    if mask.shape != (len(jerk),):
        raise ValueError("jerk mask length mismatch")
    energy = float(np.sum(jerk[mask] ** 2) * dt)
    result["jerk_samples"] = int(mask.sum())
    result["excluded_jerk_samples"] = int((~mask).sum())
    if mask.sum() < 3:
        result["reason"] = "insufficient_within_plan_jerk_samples"
    elif energy <= 1e-24:
        result["reason"] = "zero_jerk_log_undefined"
    else:
        result["nldlj"] = float(np.log((len(v) * dt) ** 3 * energy / peak**2))
    nfft = 2 ** (int(np.ceil(np.log2(len(speed)))) + 4)
    spectrum = np.abs(np.fft.rfft(speed, n=nfft))
    spectrum /= spectrum.max()
    frequency = np.fft.rfftfreq(nfft, dt)
    selected = np.flatnonzero((frequency <= cutoff_hz) & (spectrum >= threshold))
    end = int(selected[-1])
    if end:
        f, amplitude = frequency[: end + 1], spectrum[: end + 1]
        result["nsparc"] = float(np.sqrt((np.diff(f) / f[-1]) ** 2 + np.diff(amplitude) ** 2).sum())
    else:
        result["reason"] = result["reason"] or "insufficient_spectral_band"
    return result


def pose_metrics(poses, dt, plan_ids, **options):
    """Differentiate translation and spatial SO(3) increments, not Euler angles."""
    poses = np.asarray(poses, dtype=float)
    ids = np.asarray(plan_ids)
    if poses.shape != (len(ids), 4, 4) or len(ids) < 9 or not np.isfinite(poses).all():
        raise ValueError("poses require at least nine finite SE(3) samples")
    r = poses[:, :3, :3]
    if not np.allclose(r @ r.transpose(0, 2, 1), np.eye(3), atol=1e-5) or not np.allclose(
        np.linalg.det(r), 1, atol=1e-5
    ):
        raise ValueError("invalid FK rotation")
    linear = np.diff(poses[:, :3, 3], axis=0) / dt
    angular = Rotation.from_matrix(r[1:] @ r[:-1].transpose(0, 2, 1)).as_rotvec() / dt
    # One third difference spans four poses. Invalid interpolation samples use
    # unique IDs so that they cannot accidentally form a within-plan stencil.
    mask = (ids[:-3] == ids[1:-2]) & (ids[:-3] == ids[2:-1]) & (ids[:-3] == ids[3:])
    return {
        name: velocity_metrics(v, dt, valid_jerk=mask, **options)
        for name, v in (("linear", linear), ("angular", angular))
    }


def make_profile(config, base):
    """Resolve an explicit task-wide profile and build offline FK only."""
    from manimux.embodiments.robot import RobotModel

    required = {"robot_config", "groups", "sample_hz", "max_gap_s"}
    if (
        not isinstance(config, dict)
        or not required <= config.keys()
        or config.keys() - required - {"cutoff_hz", "threshold"}
    ):
        raise ValueError("smoothness requires robot_config, groups, sample_hz, max_gap_s")
    path = (base / Path(config["robot_config"]).expanduser()).resolve()
    hz, gap = float(config["sample_hz"]), float(config["max_gap_s"])
    cutoff, threshold = float(config.get("cutoff_hz", 10)), float(config.get("threshold", 0.05))
    if (
        not np.isfinite([hz, gap, cutoff, threshold]).all()
        or hz <= 0
        or gap <= 0
        or not 0 < cutoff <= hz / 2
        or not 0 < threshold < 1
    ):
        raise ValueError("invalid smoothness sampling/spectral parameters")
    groups = config["groups"]
    if (
        not isinstance(groups, list)
        or not groups
        or any(not isinstance(g, str) for g in groups)
        or len(set(groups)) != len(groups)
    ):
        raise ValueError("smoothness groups must be a nonempty unique list")
    robot = RobotModel.from_config(path)
    models = {g: robot.kinematics.models[g] for g in groups}
    sources = {Path(c["source"]) for c in robot.components.values()}
    sources.update(Path(inspect.getfile(type(model))) for model in models.values())
    profile = {
        "schema": "scheduled-eef-smoothness-v1",
        "source": "ticks/scheduled",
        "window": "first_to_last_recorded_running_tick",
        "sample_hz": hz,
        "max_gap_s": gap,
        "cutoff_hz": cutoff,
        "threshold": threshold,
        "groups": groups,
        "robot_config": str(path),
        "robot_config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "component_source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)
        },
        "metric_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "nldlj": "ln(T^3 * integral(norm(vector_jerk)^2 dt) / peak_speed^2)",
        "jerk_boundary": "exclude four-pose stencils crossing plan changes",
        "nsparc": "positive speed spectral arc length; FFT padding level 4",
        "aggregation": "equal active EEF linear/angular channels; equal episodes",
    }
    return profile, models


def analyze_episode(episode: Path, profile, models):
    """Read frozen records. Reject gaps rather than inventing motion across pauses."""
    import zarr

    ticks = zarr.open_group(str(episode / "data.zarr"), mode="r")["ticks"]
    times = np.asarray(ticks["monotonic_ns"][:], dtype=np.int64)
    ids = np.asarray(ticks["plan_id"][:]).astype(str)
    if len(times) < 9 or ids.shape != times.shape or np.any(np.diff(times) <= 0):
        raise ValueError("missing_or_nonmonotonic_ticks")
    if np.max(np.diff(times)) / 1e9 > profile["max_gap_s"]:
        raise ValueError("running_tick_gap_or_pause")
    if np.any(ids == ""):
        raise ValueError("missing_tick_plan_identity")
    events_path = episode / "events.jsonl"
    if events_path.is_file():
        for line in events_path.read_text().splitlines():
            event = json.loads(line)
            if (
                event.get("kind") == "robogui_control_state"
                and event.get("paused")
                and times[0] < event.get("monotonic_ns", -1) < times[-1]
            ):
                raise ValueError("pause_inside_recorded_window")
            if event.get("kind") == "robogui_home_requested" and 0 < event.get("step", 0) < len(
                times
            ):
                raise ValueError("home_inside_recorded_window")
    dt = 1 / profile["sample_hz"]
    t = (times - times[0]) / 1e9
    grid = np.arange(int(np.floor(t[-1] / dt)) + 1) * dt
    if len(grid) < 9:
        raise ValueError("too_short_for_smoothness")
    lower = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 1)
    upper = np.minimum(lower + 1, len(t) - 1)
    grid_ids = ids[lower].astype(object)
    for i in np.flatnonzero((ids[lower] != ids[upper]) & (grid != t[lower])):
        grid_ids[i] = ("boundary", int(i))
    digest = hashlib.sha256(times.tobytes() + json.dumps(ids.tolist()).encode())
    components = {}
    for group, model in models.items():
        q = np.asarray(ticks["scheduled"][group][:], dtype=float)
        if q.ndim != 2 or len(q) != len(t) or not np.isfinite(q).all():
            raise ValueError(f"invalid_scheduled_reference:{group}")
        digest.update(group.encode() + str(q.shape).encode() + q.tobytes())
        sampled = np.column_stack([np.interp(grid, t, col) for col in q.T])
        poses = np.asarray([model.fk(row) for row in sampled])
        digest.update(poses.tobytes())
        components[group] = pose_metrics(
            poses, dt, grid_ids, cutoff_hz=profile["cutoff_hz"], threshold=profile["threshold"]
        )
    values = {}
    for metric in ("nldlj", "nsparc"):
        active = {
            f"{g}/{c}": value[metric]
            for g, cs in components.items()
            for c, value in cs.items()
            if value["reason"] != "stationary"
        }
        values[metric] = {
            "value": float(np.mean(list(active.values())))
            if active and all(v is not None for v in active.values())
            else None,
            "channels": sorted(active),
        }
    return {
        "status": "available",
        "reason": None,
        "profile": profile,
        "profile_signature": hashlib.sha256(
            json.dumps(profile, sort_keys=True).encode()
        ).hexdigest(),
        "input_sha256": digest.hexdigest(),
        "duration_s": float(grid[-1]),
        "sample_count": len(grid),
        "components": components,
        **values,
    }
