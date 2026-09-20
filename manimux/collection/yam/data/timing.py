"""Independent collection timelines and explicit alignment for schema v2."""

from __future__ import annotations

import numpy as np

from .schema import (
    action_gripper_key,
    action_joint_key,
    cam_image_key,
    cam_timestamp_key,
    controller_timestamp_key,
    feedback_timestamp_key,
    gripper_pos_key,
    joint_pos_key,
)


def previous_indices(source_ns, query_ns) -> np.ndarray:
    """Latest available sample; -1 means no source existed at that query time."""
    source = np.asarray(source_ns, dtype=np.int64)
    query = np.asarray(query_ns, dtype=np.int64)
    if source.ndim != 1 or np.any(np.diff(source) < 0):
        raise ValueError("source timestamps must be one-dimensional and sorted")
    return np.searchsorted(source, query, side="right") - 1


def camera_times_ns(buffers: dict, role: str) -> np.ndarray:
    return np.rint(np.asarray(buffers[cam_timestamp_key(role)]) * 1e6).astype(np.int64)


def rate_stats(timestamp_ns, nominal_hz: float) -> dict:
    t = np.asarray(timestamp_ns, dtype=np.int64)
    dt = np.diff(t) / 1e6
    return {
        "count": len(t),
        "nominal_hz": nominal_hz,
        "actual_hz": float((len(t) - 1) * 1e9 / (t[-1] - t[0]))
        if len(t) > 1 and t[-1] > t[0]
        else None,
        "period_p50_ms": float(np.median(dt)) if len(dt) else None,
        "period_p95_ms": float(np.percentile(dt, 95)) if len(dt) else None,
        "period_max_ms": float(dt.max()) if len(dt) else None,
        "intervals_over_1_5_periods": int(np.count_nonzero(dt > 1500 / nominal_hz)),
    }


def describe_timing(buffers: dict, arms, cameras, control_hz: float) -> dict:
    ticks = np.asarray(buffers["tick-timestamp-ns"], dtype=np.int64)
    camera_stats = {}
    for cam in cameras:
        times = camera_times_ns(buffers, cam.role)
        # Persist the map so one image can be referenced by several control rows
        # without storing duplicates. Before the first capture the map is -1.
        buffers[f"{cam.role}-frame-index"] = previous_indices(times, ticks)
        camera_stats[cam.role] = {
            "count": len(times),
            "actual_hz": float((len(times) - 1) * 1e9 / (times[-1] - times[0]))
            if len(times) > 1 and times[-1] > times[0]
            else None,
        }
    return {
        "layout": "independent_streams",
        "num_frames_scope": "control samples",
        "tick_timestamp": "tick-timestamp-ns",
        "tick_clock": "Unix nanoseconds",
        "tick_scope": "after observation reads, before publishing the leader target",
        "monotonic_timestamp": "tick-monotonic-ns",
        "feedback_timestamp_scope": "driver snapshot read time, not per-motor CAN receive time",
        "camera_timestamp_scope": (
            "host camera timestamp in milliseconds; not exposure synchronization"
        ),
        "camera_alignment": "latest capture at or before tick; -1 before first capture",
        "control": rate_stats(buffers["tick-monotonic-ns"], control_hz),
        "commands": {
            arm: rate_stats(buffers[controller_timestamp_key(arm)], control_hz) for arm in arms
        },
        "cameras": camera_stats,
    }


def uniform_training_buffers(meta, buffers: dict) -> dict:
    """Project v2 onto LeRobot's single FPS grid, retaining original files.

    Commands/states are held, not interpolated. Each image is the last available
    capture and can repeat in the exported dataset. Explicitly reject long gaps
    rather than disguising missing data as hundreds of fresh samples.
    """
    hz = float(meta.control_hz)
    if not hz.is_integer():
        raise ValueError("LeRobot export needs an integer collection_hz")
    sources = {}
    for arm in meta.arm_names:
        sources[arm + "/action"] = np.asarray(
            buffers[controller_timestamp_key(arm)], dtype=np.int64
        )
        sources[arm + "/state"] = np.asarray(buffers[feedback_timestamp_key(arm)], dtype=np.int64)
    for cam in meta.cameras:
        sources[cam.role + "/image"] = camera_times_ns(buffers, cam.role)
    if not sources or any(len(t) < 2 for t in sources.values()):
        raise ValueError("LeRobot export requires at least two samples in every stream")
    origin = min(int(t[0]) for t in sources.values())
    start = max(int(t[0]) for t in sources.values())
    end = min(int(t[-1]) for t in sources.values())
    first = int(np.ceil((start - origin) * hz / 1e9))
    last = int(np.floor((end - origin) * hz / 1e9))
    query = origin + np.rint(np.arange(first, last + 1) * (1e9 / hz)).astype(np.int64)
    if len(query) < 2:
        raise ValueError("No shared time interval for LeRobot export")
    out = {}
    for name, times in sources.items():
        indices = previous_indices(times, query)
        stream, kind = name.rsplit("/", 1)
        if kind == "image":
            cam = next(c for c in meta.cameras if c.role == stream)
            gap_s = 3 / cam.fps
            for key in cam.image_keys:
                image_key = cam_image_key(stream, key)
                out[image_key] = [buffers[image_key][i] for i in indices]
            out[cam_timestamp_key(stream)] = query / 1e6
        else:
            gap_s = 3 / hz
            keys = (
                (action_joint_key(stream), action_gripper_key(stream))
                if kind == "action"
                else (joint_pos_key(stream), gripper_pos_key(stream))
            )
            for key in keys:
                out[key] = np.asarray(buffers[key])[indices]
        if np.any(indices < 0) or np.any(query - times[indices] > round(gap_s * 1e9)):
            raise ValueError(f"Stream {name} has gaps longer than three nominal periods")
    return out
