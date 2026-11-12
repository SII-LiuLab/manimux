"""Reconstruct saved achieved joints; no commands or controller simulation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .rate_resampling import resample_joint_streams, sampling_strides
from .replay import JointReplay, _series, load_replay_cameras


def load_feedback_rate_replay(
    episode: Path | str, *, target_hz: float = 60.0,
    low_rates_hz: tuple[float, ...] = (10.0, 30.0), max_gap_s: float = 0.1,
) -> JointReplay:
    """Use each follower's own saved snapshot timestamps, independent of config Hz.

    Preserve raw observations for plots. The reference on the comparison grid is
    latest-at-or-before sampling, not interpolation or a fresh measured stream.
    """
    sampling_strides(target_hz, low_rates_hz)
    episode = Path(episode).expanduser().resolve()
    if not (episode / "write_complete.flag").is_file():
        raise ValueError("Episode is not finalized (missing write_complete.flag)")
    meta = json.loads((episode / "metadata.json").read_text())
    arms = meta.get("arm_names", [])
    if (meta.get("schema_version") != 2 or meta.get("num_arm_joints") != 6
            or not arms or not set(arms) <= {"left", "right"} or len(set(arms)) != len(arms)):
        raise ValueError("Achieved rate comparison requires a schema 2 YAM joint episode")
    raw, original, rates = {}, {}, {}
    for arm in arms:
        time_file = f"{arm}-feedback-timestamp-ns.npy"
        ts, q = _series(episode, f"{arm}-joint_pos.npy", time_file, 6)
        _, grip = _series(episode, f"{arm}-gripper_pos.npy", time_file, 1)
        if len(ts) != meta["num_frames"] or np.any((grip < 0) | (grip > 1)):
            raise ValueError(f"Invalid achieved count or gripper range: {arm}")
        original[arm] = (ts, np.column_stack([q, grip]))
        # Duplicate snapshot times do not represent an extra temporal sample.
        keep = np.r_[np.diff(ts) > 0, True]
        ts, values = ts[keep], original[arm][1][keep]
        if len(ts) < 2:
            raise ValueError(f"Need at least two distinct achieved timestamps: {arm}")
        raw[arm] = (ts, values)
        rates[arm] = (len(ts) - 1) * 1e9 / int(ts[-1] - ts[0])
    origin, time_s, reference, variants = resample_joint_streams(
        raw, target_hz=target_hz, low_rates_hz=low_rates_hz, max_gap_s=max_gap_s,
    )
    actual = " / ".join(f"{arm} {rate:.2f}" for arm, rate in rates.items())
    low_label = ", ".join(f"{r:g}" for r in low_rates_hz)
    return JointReplay(
        episode=episode, source="feedback", origin_ns=origin, time_s=time_s,
        held={}, linear={}, native=reference, resampled=variants,
        target_hz=target_hz, low_hz=min(low_rates_hz),
        cameras=load_replay_cameras(episode, meta), source_rates_hz=rates,
        reference_label="Original achieved", source_samples=original,
        resample_rates_hz=dict(zip(variants, low_rates_hz, strict=True)),
        source_note=(
            f"Capture setting {meta['control_hz']:g} Hz; recorded follower achieved states actually average {actual} Hz.\n\n"
            f"Blue: use each arm's feedback timestamps to select the most recent measurement on the {target_hz:g} Hz comparison grid; "
            f"Orange/green: sample the same achieved data by time at {low_label} Hz, "
            f"then linearly interpolate the six joints to {target_hz:g} Hz. This is time sampling, not selecting every nth source row. "
            "The joint report also retains original sample points; repeated grid values are not new measurements.\n\n"
            "All three gripper series share recorded measured openings without downsampling, interpolation or rewriting records. "
            f"Crop endpoints to shared sample points; source-time gaps longer than {max_gap_s * 1000:g} ms are not filled. "
            "Use offline future points without additional low-pass or antialias filtering; error measures reconstruction loss from low-rate recording. "
            "These are measured trajectories and reconstructions, not three hardware command experiments or MIT control accuracy."
        ),
    )
