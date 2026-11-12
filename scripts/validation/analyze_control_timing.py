#!/usr/bin/env python3
"""Summarize recorded Edge host timing without loading models, devices, or ticks.

Usage: python -m scripts.validation.analyze_control_timing --episode EPISODE
       python -m scripts.validation.analyze_control_timing EPISODE --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from manimux.timing import summarize_control_timing


def analyze_episode(episode: Path | str) -> dict:
    episode = Path(episode)
    rows_path = episode / "control_timing.jsonl"
    metadata_path = episode / "control_timing_summary.json"
    if not rows_path.is_file() or not metadata_path.is_file():
        raise ValueError(
            f"Missing control timing artifacts in {episode}; historical ticks cannot "
            "reconstruct loop stages. Record an episode with control timing enabled."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))["metadata"]
    if metadata.get("schema") != "manimux-control-timing-v1":
        raise ValueError("Unsupported control timing schema")
    with rows_path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != metadata["rows_retained"]:
        raise ValueError("Control timing row count does not match its metadata")
    if any(
        right["cycle_index"] <= left["cycle_index"]
        or right["cycle_monotonic_ns"] <= left["cycle_monotonic_ns"]
        for left, right in zip(rows, rows[1:], strict=False)
    ):
        raise ValueError("Control timing rows must have increasing cycle indices and timestamps")
    report = summarize_control_timing(rows, metadata)
    report["episode"] = str(episode.resolve())
    return report


def _ms(stats: dict, key: str) -> str:
    value = stats.get(key)
    return "-" if value is None else f"{value:.3f}"


def format_report(report: dict) -> str:
    metadata = report["metadata"]
    lines = [
        "Control timing: host calls only; this does not establish motor/control hardware Hz.",
        f"Clock: {metadata['measurement_clock']}; scheduler: {metadata['clock_source']}; "
        f"target {metadata['period_ns'] / 1e6:g} ms",
        f"Rows retained {metadata['rows_retained']}/{metadata['cycles_observed']}; "
        f"dropped {metadata['rows_dropped']}; truncated={metadata['truncated']}",
        "Each continuous phase is separate. Period excludes incomplete/nonadjacent cycles.",
    ]
    for segment in report["segments"]:
        hz = "-" if segment["actual_hz"] is None else f"{segment['actual_hz']:.2f}"
        send_hz = (
            "-" if segment["send_actual_hz"] is None else f"{segment['send_actual_hz']:.2f}"
        )
        period = segment["period"]
        lines.extend([
            "",
            f"#{segment['segment_id']} {segment['phase']}: "
            f"{segment['completed_rows']} complete / {segment['incomplete_rows']} incomplete; "
            f"host Hz={hz}; send-call Hz={send_hz}",
            "  Period ms mean/P50/P95/P99/max: " + "/".join(
                _ms(period, key) for key in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms")
            ) + f"; >10ms={period.get('over_10ms_count', 0)}/{period['count']}",
            "  Work mean/P95, CPU mean, sleep overshoot P95, send interval P95 (ms): "
            f"{_ms(segment['work'], 'mean_ms')}/{_ms(segment['work'], 'p95_ms')}, "
            f"{_ms(segment['work_cpu'], 'mean_ms')}, "
            f"{_ms(segment['sleep_overshoot'], 'p95_ms')}, "
            f"{_ms(segment['send_interval'], 'p95_ms')}",
            f"  Work >target: {segment['work']['over_target_count']}/"
            f"{segment['completed_rows']}; deadline lag P95 "
            f"{_ms(segment['deadline_lag'], 'p95_ms')} ms",
        ])
        for name, stats in segment["stages"].items():
            lines.append(
                f"    {name}: calls={stats['wall']['count']}; wall mean/P95/max "
                f"{_ms(stats['wall'], 'mean_ms')}/{_ms(stats['wall'], 'p95_ms')}/"
                f"{_ms(stats['wall'], 'max_ms')} ms; CPU mean "
                f"{_ms(stats['cpu'], 'mean_ms')} ms"
            )
    lines.append("Nested stage times overlap: do not add them. Full distributions are in JSON.")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, help="episode directory")
    parser.add_argument("--episode", type=Path, help="episode directory")
    parser.add_argument("--output", type=Path, help="write the recomputed JSON report")
    parser.add_argument(
        "--json", action="store_true", help="print JSON instead of the compact report",
    )
    args = parser.parse_args(argv)
    if (args.path is None) == (args.episode is None):
        parser.error("provide exactly one episode path, either positional or --episode")
    try:
        report = analyze_episode(args.episode or args.path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Control timing analysis failed: {exc}\n")
    if args.output is not None:
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
    print(json.dumps(report, indent=2, allow_nan=False) if args.json else format_report(report))


if __name__ == "__main__":
    main()
