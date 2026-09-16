"""Compare native->100 Hz with native->30 Hz->linear 100 Hz, per joint.

Run offline: python -m manimux.collection.yam.data.joint_rate_analysis EPISODE
This measures reconstruction error on the same recording, not robot tracking
error or the task performance of a controller running at another frequency.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def _sample_previous(t, q, query, max_gap_s):
    """Causal sample: latest feedback at/before each tick; mask unobserved gaps."""
    index = np.searchsorted(t, query, side="right") - 1
    left = np.clip(index, 0, len(t) - 1)
    right = np.minimum(left + 1, len(t) - 1)
    valid = (index >= 0) & (query <= t[-1])
    valid &= (t[right] - t[left] <= max_gap_s) | np.isclose(
        query, t[left], rtol=0, atol=1e-10
    )
    return np.where(valid, q[left], np.nan)


def _grid(start, end, hz, phase=0.0):
    first = math.ceil((start - phase) * hz - 1e-9)
    last = math.floor((end - phase) * hz + 1e-9)
    return np.arange(first, last + 1, dtype=float) / hz + phase


def compare_joint(
    timestamps_ns, positions_rad, *, origin_ns=None, target_hz=100.0,
    low_hz=30.0, phase_ms=0.0, max_gap_s=0.05, cutoff_hz=8.0,
):
    """Return time-aligned curves; no anti-alias filter before the 30 Hz sampling.

    Positions may contain NaN for invalid motor feedback. They stay invalid in
    reconstruction. Never extrapolate past the lower-rate stream's last sample.
    """
    ts = np.asarray(timestamps_ns, dtype=np.int64)
    q = np.asarray(positions_rad, dtype=float)
    if ts.ndim != 1 or q.shape != ts.shape or len(ts) < 2:
        raise ValueError("need at least two timestamps and matching joint positions")
    if np.any(np.diff(ts) <= 0) or np.any(np.isinf(q)):
        raise ValueError("timestamps must increase strictly; positions cannot be infinite")
    if not all(math.isfinite(v) and v > 0 for v in (target_hz, low_hz, max_gap_s)):
        raise ValueError("rates and maximum gap must be finite and positive")
    if low_hz >= target_hz:
        raise ValueError("low_hz must be below target_hz")
    if not math.isfinite(phase_ms) or not 0 <= phase_ms < 1000 / low_hz:
        raise ValueError("phase_ms must lie within one lower-rate sample period")
    if not math.isfinite(cutoff_hz) or cutoff_hz < 0:
        raise ValueError("cutoff_hz must be finite and nonnegative (0 disables smoothing)")
    origin = int(ts[0]) if origin_ns is None else int(origin_ns)
    t = (ts - origin) / 1e9
    grid = _grid(t[0], t[-1], target_hz)
    low_t = _grid(t[0], t[-1], low_hz, phase_ms / 1000)
    if len(grid) < 2 or len(low_t) < 2:
        raise ValueError("recording is too short for the requested rates and phase")
    baseline = _sample_previous(t, q, grid, max_gap_s)
    low = _sample_previous(t, q, low_t, max_gap_s)
    linear = np.interp(grid, low_t, low, left=np.nan, right=np.nan)
    # Also mask gaps seen at target-rate sampling, even if both 30 Hz endpoints
    # happened to be valid; interpolation cannot invent missing feedback.
    linear[~np.isfinite(baseline)] = np.nan
    curves = {"native_to_target": baseline, "low_to_target_linear": linear}
    if cutoff_hz:
        alpha = (1 / target_hz) / (1 / (2 * np.pi * cutoff_hz) + 1 / target_hz)
        smooth = np.full_like(linear, np.nan)
        previous = None
        for i, value in enumerate(linear):
            if not np.isfinite(value):
                previous = None
                continue
            if previous is None:
                previous = baseline[i]
            previous += alpha * (value - previous)
            smooth[i] = previous
        curves["low_to_target_linear_smooth"] = smooth
    return {
        "time_s": grid, "low_time_s": low_t, "low_positions_rad": low,
        "curves": curves,
        "native_hz_mean": (len(t) - 1) / (t[-1] - t[0]),
        "native_max_gap_ms": float(np.max(np.diff(t)) * 1000),
    }


def error_metrics(reference, reconstructed):
    valid = np.isfinite(reference) & np.isfinite(reconstructed)
    error = np.asarray(reconstructed)[valid] - np.asarray(reference)[valid]
    if not len(error):
        return {"valid_samples": 0, "excluded_samples": len(reference),
                "rmse_deg": None, "mae_deg": None, "p95_abs_deg": None, "max_abs_deg": None}
    error = np.rad2deg(error)
    return {
        "valid_samples": len(error), "excluded_samples": int(len(reference) - len(error)),
        "rmse_deg": float(np.sqrt(np.mean(error ** 2))),
        "mae_deg": float(np.mean(np.abs(error))),
        "p95_abs_deg": float(np.percentile(np.abs(error), 95)),
        "max_abs_deg": float(np.max(np.abs(error))),
    }


def analyze_episode(episode, output=None, **options):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episode = Path(episode).resolve()
    native = episode / "native_joints"
    meta = json.loads((native / "metadata.json").read_text())
    if meta.get("schema_version") != 1 or not meta.get("end_timestamp_ns"):
        raise ValueError("native joint recording is not finalized or has an unsupported schema")
    output = Path(output) if output is not None else episode / "joint_rate_analysis"
    output.mkdir(parents=True, exist_ok=True)
    params = {"target_hz": 100.0, "low_hz": 30.0, "phase_ms": 0.0,
              "max_gap_s": 0.05, "cutoff_hz": 8.0, **options}
    metrics = []
    failures = []
    warnings = []
    colors = ("#222222", "#d55e00", "#0072b2")
    labels = (
        f"native -> {params['target_hz']:g} Hz (reference)",
        f"native -> {params['low_hz']:g} -> {params['target_hz']:g} Hz (linear)",
        f"linear + {params['cutoff_hz']:g} Hz low-pass",
    )
    for stream, info in meta["streams"].items():
        source = (native / info["file"]).resolve()
        if source.parent != native or not stream.replace("_", "").isalnum():
            raise ValueError("invalid native stream file or name")
        joints = [[] for _ in info["joint_names"]]
        with source.open() as handle:
            for line in handle:
                sample = json.loads(line)
                q = sample["position_rad"] if sample["motor_error"] == "0x1" else np.nan
                joints[sample["joint_index"]].append((sample["timestamp_ns"], q))
        fig, axes = plt.subplots(len(joints), 2, figsize=(15, 2.5 * len(joints)), squeeze=False)
        for index, samples in enumerate(joints):
            name = info["joint_names"][index]
            try:
                result = compare_joint(
                    [s[0] for s in samples], [s[1] for s in samples],
                    origin_ns=meta["start_timestamp_ns"], **params,
                )
            except ValueError as exc:
                failures.append({"stream": stream, "joint": name, "error": str(exc)})
                axes[index, 0].set_title(f"{name}: unavailable ({exc})")
                continue
            curves = result["curves"]
            if result["native_hz_mean"] < params["target_hz"]:
                warnings.append(
                    f"{stream}/{name}: native mean rate is below target_hz; "
                    "the reference repeats feedback and is not a native 100 Hz measurement"
                )
            reference = curves["native_to_target"]
            t = result["time_s"]
            for (method, q), color, label in zip(curves.items(), colors, labels, strict=False):
                axes[index, 0].plot(t, np.rad2deg(q), color=color, linewidth=0.8, label=label)
                if method == "native_to_target":
                    continue
                metric = {"stream": stream, "joint": name, "method": method,
                          "native_hz_mean": result["native_hz_mean"],
                          "native_max_gap_ms": result["native_max_gap_ms"],
                          **error_metrics(reference, q)}
                metrics.append(metric)
                rmse = metric["rmse_deg"]
                short = "linear" if method == "low_to_target_linear" else "linear + low-pass"
                error_label = f"{short}: RMSE={rmse:.3f} deg" if rmse is not None else short
                axes[index, 1].plot(t, np.rad2deg(q - reference), color=color, linewidth=0.8,
                                    label=error_label)
            axes[index, 1].legend(fontsize=7)
            for column, ylabel in enumerate(("position (deg)", "error (deg)")):
                axes[index, column].set_title(name)
                axes[index, column].set_ylabel(ylabel)
                axes[index, column].grid(alpha=0.2)
            with (output / f"{stream}-{name}.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["time_s", *[f"{key}_rad" for key in curves]])
                writer.writerows(zip(t, *curves.values(), strict=True))
        axes[0, 0].legend(fontsize=7)
        for axis in axes[-1]:
            axis.set_xlabel("episode time (s)")
        quality = "" if meta["complete"] else " | INCOMPLETE CAPTURE: see metadata"
        fig.suptitle(f"{stream} | phase={params['phase_ms']:g} ms{quality}")
        fig.tight_layout()
        fig.savefig(output / f"{stream}.png", dpi=150)
        fig.savefig(output / f"{stream}.svg")
        plt.close(fig)
    if metrics:
        with (output / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
            writer.writeheader()
            writer.writerows(metrics)
    summary = {
        "source_episode": str(episode), "native_recording_complete": meta["complete"],
        "parameters": params, "metrics": metrics, "unavailable_joints": failures,
        "warnings": warnings,
        "sampling": "latest feedback at/before each tick; no anti-alias prefilter",
        "scope": "offline reconstruction of one trajectory, not control/task performance",
        "smoothing": "legacy first-order low-pass only; no velocity/acceleration limits",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target-hz", type=float, default=100.0)
    parser.add_argument("--low-hz", type=float, default=30.0)
    parser.add_argument("--phase-ms", type=float, default=0.0)
    parser.add_argument("--max-gap-s", type=float, default=0.05)
    parser.add_argument("--cutoff-hz", type=float, default=8.0)
    args = vars(parser.parse_args(argv))
    summary = analyze_episode(**args)
    valid = [row for row in summary["metrics"] if row["valid_samples"]]
    print(f"Wrote {len(valid)} joint/method comparisons; "
          f"native capture complete={summary['native_recording_complete']}")
    if not valid:
        raise SystemExit("No valid comparisons; inspect summary.json")


if __name__ == "__main__":
    main()
