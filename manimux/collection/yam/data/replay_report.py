"""Export the exact achieved replay arrays as an offline joint comparison report."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
from pathlib import Path

import numpy as np

from .feedback_resampling import load_feedback_rate_replay
from .joint_rate_analysis import error_metrics
from .replay import JointReplay


def joint_metrics(data: JointReplay) -> list[dict]:
    """Use identical valid frames for every reconstruction of a given joint."""
    rows = []
    for arm, reference in data.native.items():
        for joint in range(6):
            valid = np.isfinite(reference[:, joint])
            for groups in data.resampled.values():
                valid &= np.isfinite(groups[arm][:, joint])
            for method, groups in data.resampled.items():
                stats = error_metrics(
                    np.where(valid, reference[:, joint], np.nan), groups[arm][:, joint],
                )
                rows.append({"arm": arm, "joint": joint + 1, "method": method,
                             "low_hz": data.resample_rates_hz[method], **stats})
    return rows


def _json_values(values):
    values = np.asarray(values)
    return np.where(np.isfinite(values), values, None).tolist()


def report_payload(data: JointReplay, camera: str) -> dict:
    """Frame IDs and source indices are shared with Viser, not video ordinals."""
    cameras = data.cameras[camera]
    arms = {}
    for arm, (ts, values) in data.source_samples.items():
        index = np.searchsorted(ts, data.timestamps_ns, side="right") - 1
        arms[arm] = {
            "raw_time_s": ((ts - data.origin_ns) / 1e9).tolist(),
            "raw_deg": np.rad2deg(values[:, :6]).tolist(),
            "source_index": index.tolist(),
            "source_age_ms": ((data.timestamps_ns - ts[index]) / 1e6).tolist(),
            "curves": {name: _json_values(np.rad2deg(groups[arm][:, :6]))
                       for name, groups in data.trajectories.items()},
        }
    return {
        "episode": str(data.episode), "source": data.source,
        "target_hz": data.target_hz, "time_s": data.time_s.tolist(),
        "source_rates_hz": data.source_rates_hz, "arms": arms,
        "methods": [{"key": "native", "label": "原始 achieved（此前最近值）"}, *(
            {"key": key, "label": f"{rate:g} → {data.target_hz:g} Hz", "low_hz": rate}
            for key, rate in data.resample_rates_hz.items()
        )],
        "camera": camera, "video_path": str(cameras.path),
        "camera_time_s": ((cameras.timestamps_ns - data.origin_ns) / 1e9).tolist(),
        "video_index": [cameras.frame_at(int(ts)) for ts in data.timestamps_ns],
        "metrics": joint_metrics(data), "note": data.source_note,
    }


def _write_plots(data: JointReplay, output: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#2069ac", "#ea6a00", "#15966e", "#984cc6")
    ticks = np.arange(len(data.time_s))
    for arm in data.native:
        fig, axes = plt.subplots(6, 1, figsize=(15, 17), sharex=True, layout="constrained")
        for joint, ax in enumerate(axes):
            for i, (name, groups) in enumerate(data.trajectories.items()):
                label = ("Original achieved (previous sample)" if name == "native" else
                         f"{data.resample_rates_hz[name]:g} -> {data.target_hz:g} Hz linear")
                ax.plot(ticks, np.rad2deg(groups[arm][:, joint]), color=colors[i % len(colors)],
                        label=label, lw=1.1, linestyle="-" if i == 0 else "--",
                        drawstyle="steps-post" if i == 0 else "default")
            ax.set_ylabel(f"{arm} J{joint + 1} (deg)")
            ax.grid(alpha=0.2)
        axes[0].legend(loc="best", ncol=3, fontsize=9)
        axes[-1].set_xlabel(f"Comparison frame (0-based, {1000 / data.target_hz:.3f} ms/frame)")
        fig.suptitle(f"{arm.title()} achieved: original vs reconstruction | {data.episode.name}")
        fig.savefig(output / f"{arm}-all-joints.png", dpi=160)
        fig.savefig(output / f"{arm}-all-joints.svg")
        plt.close(fig)


def write_report(data: JointReplay, output: Path, camera: str = "top") -> dict:
    import av

    if not data.source_samples or data.source != "feedback" or not data.resampled:
        raise ValueError("This report needs the achieved rate replay with original samples")
    output = Path(output).expanduser().resolve()
    if output == data.episode or data.episode in output.parents:
        raise ValueError("Write derived reports outside the original episode")
    payload = report_payload(data, camera)
    video_path = data.cameras[camera].path
    with av.open(str(video_path)) as container:
        pts = [float(frame.pts * frame.time_base) for frame in container.decode(video=0)]
    if len(pts) != len(payload["camera_time_s"]) or np.any(np.diff(pts) <= 0):
        raise ValueError("Video frame count or PTS does not match saved camera data")
    payload["video_pts_s"] = pts
    output.mkdir(parents=True, exist_ok=True)
    _write_plots(data, output)
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["metrics"][0]))
        writer.writeheader()
        writer.writerows(payload["metrics"])
    with (output / "curves.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        names = list(data.trajectories)
        writer.writerow(["frame", "episode_time_s", "unix_ns", "video_frame", "arm", "joint",
                         "original_sample_index", "original_sample_age_ms",
                         *[f"{n}_deg" for n in names]])
        for frame, timestamp in enumerate(data.timestamps_ns):
            for arm, info in payload["arms"].items():
                for joint in range(6):
                    writer.writerow([frame, data.time_s[frame], int(timestamp),
                                     payload["video_index"][frame], arm, joint + 1,
                                     info["source_index"][frame], info["source_age_ms"][frame], *(
                                         info["curves"][name][frame][joint] for name in names
                                     )])
    (output / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False))
    serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    from .replay_report_html import PAGE

    page = PAGE.replace("@@DATA@@", serialized)
    page = page.replace("@@VIDEO@@", base64.b64encode(video_path.read_bytes()).decode())
    page = page.replace("@@NOTE@@", html.escape(data.source_note).replace("\n\n", "<br><br>"))
    (output / "index.html").write_text(page)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-hz", type=float, default=60)
    parser.add_argument("--low-hz", type=float, nargs="+", default=[10, 30])
    parser.add_argument("--camera", default="top")
    args = parser.parse_args()
    data = load_feedback_rate_replay(args.episode, target_hz=args.target_hz,
                                     low_rates_hz=tuple(args.low_hz))
    payload = write_report(data, args.output, args.camera)
    print(f"Wrote {len(payload['metrics'])} joint/method comparisons "
          f"to {args.output / 'index.html'}")


if __name__ == "__main__":
    main()
