"""Read saved YAM trajectories for synchronized, hardware-free Viewer replay."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .joint_rate_analysis import compare_joint


@dataclass(frozen=True)
class ReplayCamera:
    path: Path
    timestamps_ns: np.ndarray

    def frame_at(self, timestamp_ns: int) -> int:
        """Original video ordinal of the latest image, including duplicate captures."""
        return int(np.searchsorted(self.timestamps_ns, timestamp_ns, side="right") - 1)


@dataclass(frozen=True)
class JointReplay:
    episode: Path
    source: str
    origin_ns: int
    time_s: np.ndarray
    held: dict[str, np.ndarray]
    linear: dict[str, np.ndarray]
    cameras: dict[str, ReplayCamera]
    source_rates_hz: dict[str, float]
    target_hz: float = 100.0
    low_hz: float = 30.0
    native: dict[str, np.ndarray] = field(default_factory=dict)
    reference_label: str = "原始实测"
    source_note: str = ""
    resampled: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    resample_rates_hz: dict[str, float] = field(default_factory=dict)

    source_samples: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    @property
    def timestamps_ns(self) -> np.ndarray:
        return self.origin_ns + np.rint(self.time_s * 1e9).astype(np.int64)

    @property
    def trajectories(self) -> dict[str, dict[str, np.ndarray]]:
        if self.resampled:
            return {"native": self.native, **self.resampled}
        result = {"native": self.native} if self.native else {}
        return {**result, "linear": self.linear, "held": self.held}


def load_replay_cameras(episode: Path, meta: dict) -> dict[str, ReplayCamera]:
    cameras = {}
    for camera in meta["cameras"]:
        name = camera["name"]
        role = camera.get("role", name)
        path = _child(episode, f"{role}-images-rgb.mp4")
        ts_ms = np.load(_child(episode, f"{role}-timestamp.npy"), allow_pickle=False)
        expected = (
            meta.get("extra", {}).get("timing", {}).get("cameras", {}).get(role, {}).get("count")
        )
        if expected is None:
            expected = len(ts_ms) if meta.get("schema_version") == 2 else meta["num_frames"]
        if (
            not path.is_file()
            or ts_ms.shape != (expected,)
            or not len(ts_ms)
            or not np.all(np.isfinite(ts_ms))
            or np.any(np.diff(ts_ms) < 0)
        ):
            raise ValueError(f"Missing video or invalid camera timestamps: {name}")
        # Recorded camera unit is Unix milliseconds, not seconds or MP4 PTS.
        cameras[name] = ReplayCamera(path, np.rint(ts_ms * 1e6).astype(np.int64))
    if not cameras:
        raise ValueError("Episode contains no recorded cameras")
    return cameras


def _child(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if path.parent != root:
        raise ValueError(f"Expected a file directly inside {root}: {name!r}")
    return path


def _series(episode: Path, value_name: str, time_name: str, width: int):
    values = np.load(_child(episode, value_name), allow_pickle=False)
    ts = np.load(_child(episode, time_name), allow_pickle=False)
    if ts.ndim != 1 or ts.dtype.kind not in "iu" or len(ts) < 2:
        raise ValueError(f"Expected at least two integer Unix-ns timestamps: {time_name}")
    if values.shape != (len(ts), width) or not np.all(np.isfinite(values)):
        raise ValueError(f"Invalid shape or nonfinite values: {value_name}")
    if np.any(np.diff(ts) < 0):
        raise ValueError(f"Timestamps went backwards: {time_name}")
    return ts.astype(np.int64), values.astype(float)


def _previous(ts: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    index = np.searchsorted(ts, query, side="right") - 1
    result = values[np.maximum(index, 0)].copy()
    result[(index < 0) | (query > ts[-1])] = np.nan
    return result


def load_joint_replay(episode: Path | str, source: str = "feedback") -> JointReplay:
    """Compare native->100 Hz, native->30->100 Hz linear, and 30 Hz hold.

    Feedback mode reuses the analysis tool's phase-zero causal downsampling and
    50 ms gap mask. Command mode retains the *actual*, irregular command times;
    it does not invent a native 100 Hz command reference. Both grippers use the
    same saved low-rate values in all views. Internal data gaps remain NaN.
    Only feedback mode has a native reference; command mode leaves it empty.
    Feedback replay starts/ends on shared 30/100 Hz sample knots, so matching
    endpoints arise from identical input samples, without changing joint values.
    """
    episode = Path(episode).expanduser().resolve()
    if source not in {"feedback", "command"}:
        raise ValueError("Replay source must be feedback or command")
    if not (episode / "write_complete.flag").is_file():
        raise ValueError("Episode is not finalized (missing write_complete.flag)")
    meta = json.loads((episode / "metadata.json").read_text())
    arms = meta["arm_names"]
    if (
        meta.get("schema_version") not in {1, 2}
        or meta.get("num_arm_joints") != 6
        or not arms
        or not set(arms) <= {"left", "right"}
        or len(set(arms)) != len(arms)
    ):
        raise ValueError("Replay currently supports canonical YAM joint episodes only")
    # This comparison has an explicit 30 Hz lower-rate contract.
    multirate = meta.get("schema_version") == 2
    if not multirate and not np.isclose(float(meta["control_hz"]), 30.0):
        raise ValueError("This replay compares 30 Hz collection with 100 Hz reconstruction")
    native_root = episode / "native_joints"
    native = None
    if source == "feedback" and (not multirate or (native_root / "metadata.json").is_file()):
        native = json.loads((native_root / "metadata.json").read_text())
        if native.get("schema_version") != 1 or not native.get("complete"):
            raise ValueError("Feedback replay requires a complete native joint recording")
        origin = int(native["start_timestamp_ns"])
    else:
        prefix = "action-{arm}-joint.npy" if source == "command" else "{arm}-joint_pos.npy"
        time_file = (
            "controller-{arm}-timestamp-ns.npy"
            if source == "command"
            else "{arm}-feedback-timestamp-ns.npy"
        )
        origin = min(
            int(_series(episode, prefix.format(arm=arm), time_file.format(arm=arm), 6)[0][0])
            for arm in arms
        )

    curves = {}
    grippers = {}
    source_rates = {}
    for arm in arms:
        if native is not None:
            info = native["streams"][f"follower_{arm}"]
            if len(info["joint_names"]) != 6 or info.get("error") or info.get("dropped_frames"):
                raise ValueError(f"Incomplete native feedback: follower_{arm}")
            samples: list[list[tuple[int, float]]] = [[] for _ in range(6)]
            with _child(native_root, info["file"]).open() as handle:
                for line in handle:
                    row = json.loads(line)
                    joint = row["joint_index"]
                    if not isinstance(joint, int) or not 0 <= joint < 6:
                        raise ValueError(f"Invalid native joint index: {joint}")
                    q = row["position_rad"] if row["motor_error"] == "0x1" else np.nan
                    samples[joint].append((row["timestamp_ns"], q))
            results = []
            for joint, rows in enumerate(samples):
                if len(rows) != info["samples"][joint]:
                    raise ValueError(f"Native sample count mismatch: {arm}/{joint}")
                result = compare_joint(
                    [r[0] for r in rows],
                    [r[1] for r in rows],
                    origin_ns=origin,
                    cutoff_hz=0,
                    max_gap_s=0.05,
                )
                held = _previous(
                    result["low_time_s"],
                    result["low_positions_rad"],
                    result["time_s"],
                )
                held[~np.isfinite(result["curves"]["native_to_target"])] = np.nan
                results.append(
                    (
                        result["time_s"],
                        held,
                        result["curves"]["low_to_target_linear"],
                        result["curves"]["native_to_target"],
                    )
                )
                source_rates[f"{arm}/J{joint + 1}"] = result["native_hz_mean"]
            curves[arm] = results
            grippers[arm] = _series(
                episode,
                f"{arm}-gripper_pos.npy",
                f"{arm}-feedback-timestamp-ns.npy",
                1,
            )
        elif multirate:
            value_file = (
                f"action-{arm}-joint.npy" if source == "command" else f"{arm}-joint_pos.npy"
            )
            time_file = (
                f"controller-{arm}-timestamp-ns.npy"
                if source == "command"
                else f"{arm}-feedback-timestamp-ns.npy"
            )
            ts, q = _series(episode, value_file, time_file, 6)
            # Retain the last value if a driver snapshot carries a repeated time.
            keep = np.r_[np.diff(ts) > 0, True]
            ts, q = ts[keep], q[keep]
            results = []
            for joint in range(6):
                result = compare_joint(
                    ts,
                    q[:, joint],
                    origin_ns=origin,
                    cutoff_hz=0,
                    max_gap_s=max(0.05, 3 / float(meta["control_hz"])),
                )
                held = _previous(
                    result["low_time_s"], result["low_positions_rad"], result["time_s"]
                )
                held[~np.isfinite(result["curves"]["native_to_target"])] = np.nan
                results.append(
                    (
                        result["time_s"],
                        held,
                        result["curves"]["low_to_target_linear"],
                        result["curves"]["native_to_target"],
                    )
                )
                source_rates[f"{arm}/J{joint + 1}"] = result["native_hz_mean"]
            curves[arm] = results
            grip_file = (
                f"action-{arm}-gripper.npy" if source == "command" else f"{arm}-gripper_pos.npy"
            )
            grippers[arm] = _series(episode, grip_file, time_file, 1)
        else:
            ts, q = _series(
                episode,
                f"action-{arm}-joint.npy",
                f"controller-{arm}-timestamp-ns.npy",
                6,
            )
            if np.any(np.diff(ts) <= 0):
                raise ValueError(f"Command timestamps must increase strictly: {arm}")
            t = (ts - origin) / 1e9
            grid = np.arange(np.ceil(t[0] * 100), np.floor(t[-1] * 100) + 1) / 100
            held = _previous(t, q, grid)
            linear = np.column_stack([np.interp(grid, t, col) for col in q.T])
            # Three missed nominal updates: show a gap instead of bridging it.
            index = np.maximum(np.searchsorted(t, grid, side="right") - 1, 0)
            gap = t[np.minimum(index + 1, len(t) - 1)] - t[index] > 0.1
            gap &= ~np.isclose(grid, t[index], rtol=0, atol=1e-10)
            held[gap] = np.nan
            linear[gap] = np.nan
            curves[arm] = [(grid, held[:, j], linear[:, j]) for j in range(6)]
            source_rates[arm] = (len(t) - 1) / (t[-1] - t[0])
            grippers[arm] = _series(
                episode,
                f"action-{arm}-gripper.npy",
                f"controller-{arm}-timestamp-ns.npy",
                1,
            )

    first = max(int(round(c[0][0] * 100)) for results in curves.values() for c in results)
    last = min(int(round(c[0][-1] * 100)) for results in curves.values() for c in results)
    ticks = np.arange(first, last + 1)
    time_s = ticks / 100.0
    query_ns = origin + ticks * 10_000_000
    held_groups, linear_groups, native_groups = {}, {}, {}
    for arm, results in curves.items():
        held = np.column_stack([c[1][ticks - int(round(c[0][0] * 100))] for c in results])
        linear = np.column_stack([c[2][ticks - int(round(c[0][0] * 100))] for c in results])
        grip_ts, grip_q = grippers[arm]
        grip = _previous(grip_ts, grip_q, query_ns)
        held_groups[arm] = np.column_stack([held, grip])
        linear_groups[arm] = np.column_stack([linear, grip])
        if native is not None or multirate:
            reference = np.column_stack([c[3][ticks - int(round(c[0][0] * 100))] for c in results])
            native_groups[arm] = np.column_stack([reference, grip])
    valid = np.ones(len(ticks), dtype=bool)
    for values in (*held_groups.values(), *linear_groups.values(), *native_groups.values()):
        valid &= np.isfinite(values).all(axis=1)
    if native is not None or multirate:
        # A 100 Hz tick is not necessarily a retained 30 Hz sample. Bound the
        # replay by common knots (every 0.1 s at these rates) instead of altering
        # either trajectory to force equal endpoint poses. Keep interior gaps.
        valid &= np.isclose(time_s * 30, np.rint(time_s * 30), rtol=0, atol=1e-9)
    indices = np.flatnonzero(valid)
    if len(indices) < 2:
        raise ValueError("No shared valid sample endpoints for the replay")
    # Trim ends only. Retain every 100 Hz tick, including interior data gaps.
    interval = slice(indices[0], indices[-1] + 1)

    cameras = load_replay_cameras(episode, meta)
    return JointReplay(
        episode,
        source,
        origin,
        time_s[interval],
        {k: v[interval] for k, v in held_groups.items()},
        {k: v[interval] for k, v in linear_groups.items()},
        cameras,
        source_rates,
        native={k: v[interval] for k, v in native_groups.items()},
        reference_label=("已录 command" if source == "command" else "已录 joint")
        if multirate and native is None
        else "原始实测",
        source_note=(
            f"采集设置 {meta['control_hz']:g} Hz；相机按自身时间戳独立记录。"
            "蓝色为保存的数据直接取到 100 Hz 网格；另两组先降到 30 Hz，再保持或线性插值。"
            "100 Hz 是比较网格，不保证每格都有新测量；夹爪各组使用同一份已录数据。"
        )
        if multirate
        else "",
    )
