#!/usr/bin/env python3
"""Screen executor parameters on immutable historical UMI rollout inputs.

Run from the repository root with its analysis environment. Outputs are analysis
artifacts, never deployment configs. No network, policy calls or hardware I/O.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from manimux.evaluation.executor_tuning import (  # noqa: E402
    LOCKED,
    TASKS,
    aggregate,
    candidates,
    exploratory_choices,
    json_sha,
    load_episode,
    metrics,
    parameters,
    replay,
    sha,
    split_sessions,
    tcp_metrics,
    tcp_screen,
)


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def discover(root):
    # Bounded metadata glob; never recursively walk Zarr chunks or videos.
    paths = set()
    for pattern in (
        "session-*/rollout-*/result.json",
        "*/session-*/rollout-*/result.json",
        "*/*/session-*/rollout-*/result.json",
    ):
        paths.update(p.parent for p in root.glob(pattern))
    return sorted(paths, key=lambda p: (p.parent.name, p.name), reverse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True, help="Recorded UMI campaign or parent directory"
    )
    parser.add_argument(
        "--output", type=Path, help="New immutable output directory; must not already exist"
    )
    parser.add_argument("--latest-per-task", type=int, default=20)
    parser.add_argument("--replay-tolerance-rad", type=float, default=1e-6)
    args = parser.parse_args()
    if args.latest_per_task < 2 or args.replay_tolerance_rad <= 0:
        parser.error("sample size must be >= 2 and replay tolerance positive")
    dest = args.output or Path("data/analysis/umi-executor-tuning") / datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%S.%fZ"
    )
    dest.mkdir(parents=True, exist_ok=False)
    choices = candidates()
    save(
        dest / "search-space.json",
        dict(
            candidates=choices,
            locked=LOCKED,
            joint_screen=(
                "Every development episode/arm: tracking RMS <= 105%; acceleration "
                "RMS, jerk RMS, seam delta-v p95 <= 110% of baseline; "
                "unchanged velocity and position bounds."
            ),
            tcp_screen=(
                "Every development episode/arm: TCP RMS/p95 and rotation p95 <= 105%; "
                "speed p95, acceleration RMS, jerk RMS <= 110% of baseline."
            ),
            selection=(
                "Lowest equal-episode, equal-arm joint RMS among passing development "
                "candidates; TCP screening of up to five best joint-screen survivors; "
                "holdout never selects."
            ),
            limits=(
                "Frozen scheduled references and grippers. Command changes would alter "
                "future observations, model output, IK seeds and possibly contacts in a "
                "real rollout. No prediction of success or measured tracking. Derivatives "
                "use nominal executor dt, not physical-time reconstruction. Gaps hold "
                "hypothetical command. Recorded discontinuities reanchor explicitly. "
                "Finalized is not success; partials are separate stress evidence."
            ),
        ),
    )
    rows, cohort, counts = [], [], Counter()
    for path in discover(args.root):
        entry = dict(path=str(path.resolve()), included=False)
        try:
            meta = json.loads((path / "meta.json").read_text())
            entry.update(
                task=TASKS.get(meta.get("task")),
                recorded_task=meta.get("task"),
                status="partial" if path.name.endswith(".partial") else "finalized",
            )
            if entry["task"] is None:
                raise ValueError("unsupported_or_missing_task")
            data = load_episode(path)
            if counts[data["info"]["task"]] >= args.latest_per_task:
                raise ValueError("outside_latest_eligible_cohort")
            # Freeze cohort before replay/score inspection. Rejected replay cases
            # consume a sample slot; do not replace failures with easier traces.
            counts[data["info"]["task"]] += 1
            output, speed, accel = replay(data, choices)
            error = float(np.max(abs(output[0] - data["arrays"]["command"][..., :7])))
            entry.update(data["info"], baseline_replay_max_error_rad=error)
            if error > args.replay_tolerance_rad:
                raise ValueError("baseline_free_run_replay_failed")
            row = dict(
                info=data["info"],
                config=data["config"],
                metrics=metrics(data, output, speed, accel),
                baseline_replay_max_error_rad=error,
            )
            rows.append(row)
            entry.update(included=True, reason="baseline_reproduced")
            print(
                f"{data['info']['task']} {path.parent.name}/{path.name}: "
                f"{len(output[0])} ticks, replay {error:.2g} rad",
                flush=True,
            )
        except (ValueError, KeyError, FileNotFoundError) as error:
            entry["reason"] = str(error)
        cohort.append(entry)
    summary = {}
    for task in sorted({r["info"]["task"] for r in rows}):
        task_rows = [r for r in rows if r["info"]["task"] == task]
        splits = split_sessions([r["info"] for r in task_rows])
        for row in task_rows:
            row["split"] = splits[row["info"]["session"]]
        development = aggregate(task_rows, choices, "development")
        holdout = aggregate(task_rows, choices, "holdout")
        shortlist = sorted(
            [a for a in development if a["passes_joint_screen"] and a["candidate"] != "recorded"],
            key=lambda a: a["error_score_deg"],
        )[:5]
        exploratory = exploratory_choices(development)
        ids = list(
            dict.fromkeys(
                ["recorded"] + [s["candidate"] for s in shortlist] + list(exploratory.values())
            )
        )
        indices = [next(i for i, c in enumerate(choices) if c["id"] == cid) for cid in ids]
        # FK only short-listed settings; no holdout input is used for selection.
        tcp_pass = {cid: True for cid in ids}
        for row in task_rows:
            if row["split"] != "development":
                continue
            data = load_episode(row["info"]["path"])
            outputs, _, _ = replay(data, [choices[i] for i in indices])
            try:
                values, context = tcp_metrics(data, outputs)
                row["tcp"] = dict(zip(ids, values, strict=True))
                row["tcp_context"] = context
                for cid, value in zip(ids, values, strict=True):
                    tcp_pass[cid] &= tcp_screen(values[0], value)
            except ValueError as error:
                row["tcp_exclusion"] = str(error)
                tcp_pass = {cid: False for cid in ids}
        winner = next(
            (
                s["candidate"]
                for s in shortlist
                if tcp_pass[s["candidate"]]
                and s["error_score_deg"] < development[0]["error_score_deg"] * 0.99
            ),
            "recorded",
        )
        winner_idx = next(i for i, c in enumerate(choices) if c["id"] == winner)
        holdout_tcp_pass = True
        for row in task_rows:
            if row["split"] != "holdout":
                continue
            data = load_episode(row["info"]["path"])
            review_ids = list(dict.fromkeys(["recorded", winner] + list(exploratory.values())))
            review_indices = [
                next(i for i, c in enumerate(choices) if c["id"] == cid) for cid in review_ids
            ]
            outputs, _, _ = replay(data, [choices[i] for i in review_indices])
            try:
                values, context = tcp_metrics(data, outputs)
                row["tcp"] = dict(zip(review_ids, values, strict=True))
                row["tcp_context"] = context
                holdout_tcp_pass &= tcp_screen(values[0], values[review_ids.index(winner)])
            except ValueError as error:
                row["tcp_exclusion"] = str(error)
                holdout_tcp_pass = False
        holdout_selected = next((a for a in holdout if a["candidate"] == winner), None)
        summary[task] = dict(
            selected_on_development=winner,
            development=development,
            holdout=holdout,
            tcp_development_pass=tcp_pass,
            exploratory_candidates=exploratory,
            holdout_joint_pass=holdout_selected["passes_joint_screen"]
            if holdout_selected
            else None,
            holdout_tcp_pass=holdout_tcp_pass if holdout_selected else None,
            counts=dict(Counter(r["info"]["status"] + "/" + r["split"] for r in task_rows)),
            sessions=splits,
            baseline_replay_max_error_rad=max(
                r["baseline_replay_max_error_rad"] for r in task_rows
            ),
        )
        strata = {}
        for row in task_rows:
            smooth = row["info"]["recorded_smooth"]
            identity = dict(
                expected_backend=row["info"]["expected_backend"],
                control_hz=row["config"]["robot"]["control_hz"],
                velocity=smooth["max_velocity"],
                acceleration=smooth["max_acceleration"],
                cutoff=smooth["cutoff_hz"],
                mode=smooth["mode"],
                inference=row["config"]["inference"],
            )
            key = json_sha(identity)
            strata.setdefault(key, dict(identity=identity, rows=[]))["rows"].append(row)
        summary[task]["strata"] = {
            key: dict(
                identity=entry["identity"],
                counts=dict(Counter(r["info"]["status"] + "/" + r["split"] for r in entry["rows"])),
                development=aggregate(entry["rows"], choices, "development"),
                holdout=aggregate(entry["rows"], choices, "holdout"),
            )
            for key, entry in strata.items()
        }
        # A review artifact with resolved per-recording settings, deliberately NOT
        # an executable experiment YAML or a patch to a mature deployment.
        spec = dict(
            artifact_type="offline_candidate_review_only",
            task=task,
            candidate=choices[winner_idx],
            locked=LOCKED,
            exploratory_candidates={
                role: choices[next(i for i, c in enumerate(choices) if c["id"] == cid)]
                for role, cid in exploratory.items()
            },
            exploratory_warning=(
                "Development-average tradeoffs only. Individual episodes may "
                "regress; see per-episode joint/TCP violations. "
                "Not approved for deployment."
            ),
            evidence_status="no_improving_candidate_passed"
            if winner == "recorded"
            else "requires_hardware_validation",
            holdout_joint_pass=summary[task]["holdout_joint_pass"],
            holdout_tcp_pass=summary[task]["holdout_tcp_pass"],
            episodes=[
                dict(
                    path=r["info"]["path"],
                    split=r["split"],
                    config_sha256_canonical_json=r["info"]["config_sha256_canonical_json"],
                    handoff_skip_steps=r["config"]["inference"]["handoff_skip_steps"],
                    tested_smooth=parameters(r["config"], choices[winner_idx]),
                    exploratory_smooth={
                        role: parameters(
                            r["config"],
                            choices[next(i for i, c in enumerate(choices) if c["id"] == cid)],
                        )
                        for role, cid in exploratory.items()
                    },
                )
                for r in task_rows
            ],
        )
        (dest / f"{task}-candidate-review.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
        print(
            f"{task}: selected={winner}, holdout joint={summary[task]['holdout_joint_pass']}, "
            f"TCP={summary[task]['holdout_tcp_pass']}",
            flush=True,
        )
    save(dest / "summary.json", summary)
    save(dest / "episode-metrics.json", rows)
    (dest / "cohort.jsonl").write_text("".join(json.dumps(e) + "\n" for e in cohort))
    save(
        dest / "provenance.json",
        dict(
            created_at=datetime.now(UTC).isoformat(),
            argv=sys.argv,
            root=str(args.root.resolve()),
            candidates_sha256=json_sha(choices),
            code_sha256={
                str(p.relative_to(ROOT)): sha(p.read_bytes())
                for p in [Path(__file__).resolve(), ROOT / "manimux/evaluation/executor_tuning.py"]
            },
            exclusions=dict(Counter(e["reason"] for e in cohort if not e["included"])),
            weighting=(
                "Equal episode, then equal arm; session chronological holdout; "
                "partial/finalized labels retained."
            ),
        ),
    )
    print(dest.resolve(), flush=True)


if __name__ == "__main__":
    main()
