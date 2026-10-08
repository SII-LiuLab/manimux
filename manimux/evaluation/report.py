"""Task-scoped offline reports from recorded attempts and existing metric artifacts.

No hardware, policy inference, PRM inference, or human-label mutation occurs here.
A report is an evidence inventory, not automatic certification of a formal cohort.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import socket
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import yaml

from manimux.evaluation.rubric import count_result, evaluation_parameters
from manimux.evaluation.trajectory import METRICS


def _json(path):
    def reject(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path(base, value):
    p = Path(value).expanduser()
    return (p if p.is_absolute() else base / p).resolve()


def _lookup(document, dotted):
    for part in dotted.split("."):
        if not isinstance(document, dict) or part not in document:
            return None
        document = document[part]
    return document


def _mean(values):
    return sum(values) / len(values) if values else None


def _wilson(success, total):
    if not total:
        return None
    z = 1.959963984540054
    p = success / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def _human(path, eligible, recorded_evaluation=None):
    if not eligible:
        return {"status": "excluded", "reason": "attempt_not_eligible", "result": None}
    if not path.is_file():
        return {"status": "missing", "reason": "unreviewed", "result": None}
    try:
        d = _json(path)
        if (
            d.get("label_schema") not in {"human-label-v1", "human-label-v2"}
            or d.get("task_result") not in {"success", "failure", "invalid"}
            or d.get("review_mode") not in {"live", "video"}
        ):
            raise ValueError("unsupported human label schema, result or review mode")
        count_fields = {}
        if d.get("evaluation", {}).get("kind") == "count":
            profile = evaluation_parameters(d["evaluation"])
            if profile != evaluation_parameters(recorded_evaluation):
                raise ValueError("human count rubric differs from recorded evaluation")
            if d["task_result"] != "invalid":
                result = count_result(profile, d.get("completed_count"))
                if result != d["task_result"] or d.get("target_count") != profile["target_count"]:
                    raise ValueError("inconsistent human count/result")
                count_fields = {
                    "score": d["completed_count"] / profile["target_count"],
                    "completed_count": d["completed_count"],
                    "target_count": profile["target_count"],
                    "rubric_id": profile["rubric_id"],
                    "rubric_signature": json.dumps(profile, sort_keys=True),
                }
        return {
            **count_fields,
            "status": "available",
            "reason": None,
            "result": d["task_result"],
            "review_mode": d["review_mode"],
            "label_schema": d["label_schema"],
            "source": str(path),
            "source_sha256": _hash(path),
        }
    except (ValueError, TypeError, AttributeError, OSError) as exc:
        return {"status": "error", "reason": str(exc), "result": None}


def _overlap(path, episode, eligible):
    if not eligible:
        return {"status": "excluded", "reason": "attempt_not_eligible"}, None
    if path is None:
        return {"status": "missing", "reason": "overlap_artifact_not_supplied"}, None
    try:
        d = _json(path)
        if d["schema"] != "committed-reference-overlap-v1" or d["formal_evaluation"] is not False:
            raise ValueError("unsupported overlap schema/scope")
        if Path(d["provenance"]["episode"]).resolve() != episode:
            raise ValueError("overlap artifact belongs to a different episode")
        if d.get("selection", {}).get("truncated"):
            raise ValueError("truncated overlap artifact is not used in task summaries")
        for name, digest in d["provenance"].get("input_metadata_sha256", {}).items():
            if name not in {"meta.json", "events.jsonl", "result.json"}:
                raise ValueError("unexpected provenance filename")
            if not (episode / name).is_file() or _hash(episode / name) != digest:
                raise ValueError(f"stale overlap metadata: {name}")
        if not d.get("rows") or not d.get("summary"):
            raise ValueError("expected overlap-metrics.json with nonempty rows and summary")
        windows = d["windows_policy_steps"]
        if not windows or any(type(n) is not int or n <= 0 for n in windows):
            raise ValueError("invalid overlap windows")
        for r in d["rows"]:
            for metric in METRICS:
                value = r[metric]
                if value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError("invalid overlap metric value")
        if type(d["candidate_pair_count"]) is not int:
            raise ValueError("invalid candidate count")
        for arm in d["summary"].values():
            for n in windows:
                summary = arm["common_boundary_by_window"][str(n)]
                if type(summary["numeric_valid_count"]) is not int:
                    raise ValueError("invalid numeric boundary count")
                for key in METRICS:
                    stat = summary["metrics"][key]
                    if type(stat["count"]) is not int or stat["count"] < 0:
                        raise ValueError("invalid summary count")
                    if stat["count"] and (
                        type(stat["mean"]) not in (float, int) or not math.isfinite(stat["mean"])
                    ):
                        raise ValueError("invalid summary mean")
        signature = json.dumps(
            {
                "schema": d["schema"],
                "definition": d["definition"],
                "windows": windows,
                "sample_dt_ns": sorted({r["sample_dt_ns"] for r in d["rows"]}),
                "fk_sha256": d["provenance"]["robot_config_sha256"],
                "metric_code_sha256": d["provenance"]["metric_code_sha256"],
            },
            sort_keys=True,
        )
        return {
            "status": "available",
            "reason": None,
            "source": str(path),
            "source_sha256": _hash(path),
            "profile_signature": signature,
            "artifact_provenance": d["provenance"],
            "selected_reference_sha256": d.get("selected_reference_sha256"),
            "numeric_scope": "candidate references; execution eligibility not checked",
        }, d
    except (ValueError, TypeError, KeyError, AttributeError, OSError) as exc:
        return {"status": "error", "reason": str(exc), "source": str(path)}, None


def _handoff_figures(entry, base, eligible, attempt_id):
    """Import explicitly selected plotter images; never certify formal seam scores."""
    images = entry.get("handoff_images", [])
    if not images or not eligible:
        return [], []
    try:
        if not isinstance(images, list) or any(not isinstance(n, str) for n in images):
            raise ValueError("handoff_images must be a list of PNG filenames")
        summary_path = _path(base, entry["handoff_summary"])
        summary = _json(summary_path)
        if not isinstance(summary, list):
            raise ValueError("handoff_summary must contain plotter rows")
        imported = []
        for name in dict.fromkeys(images):
            if Path(name).name != name or Path(name).suffix.lower() != ".png":
                raise ValueError("handoff image must be a local PNG basename")
            image_path = (summary_path.parent / name).resolve()
            if not image_path.is_relative_to(summary_path.parent):
                raise ValueError("handoff image escapes summary directory")
            matches = [r for r in summary if r.get("image") == name]
            if len(matches) != 1:
                raise ValueError("handoff image requires one matching summary row")
            row = matches[0]
            for key in ("old_chunk", "new_chunk"):
                if type(row[key]) is not int or row[key] < 0:
                    raise ValueError("invalid handoff boundary identity")
            for key in ("left_position_seam_mm", "right_position_seam_mm"):
                if (
                    type(row[key]) not in (float, int)
                    or not math.isfinite(row[key])
                    or row[key] < 0
                ):
                    raise ValueError("invalid seam value")
            if not image_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("handoff image is not a PNG")
            imported.append(
                {
                    "source": str(image_path),
                    "source_sha256": _hash(image_path),
                    "summary_source": str(summary_path),
                    "summary_sha256": _hash(summary_path),
                    "filename": f"handoff-{attempt_id}-{name}",
                    "binding": "explicit task.yaml episode attachment",
                    "formal_evaluation": False,
                    **{
                        key: row[key]
                        for key in (
                            "old_chunk",
                            "new_chunk",
                            "left_position_seam_mm",
                            "right_position_seam_mm",
                        )
                    },
                }
            )
        return imported, []
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return [], [str(exc)]


def _experiment_table(config, base):
    """Read the selected register table without inferring any cohort bindings."""
    spec = config.get("experiment_table")
    if spec is None:
        return None
    path = _path(base, spec["path"])
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    heading = spec["heading"]
    sections = re.split(r"(?m)^### ", text)
    matches = [s for s in sections if s.splitlines() and s.splitlines()[0] == heading]
    if len(matches) != 1:
        raise ValueError("experiment_table heading must match exactly one section")
    lines = matches[0].splitlines()[1:]
    headers, rows, model = None, [], None
    bindings = spec.get("bindings", {})
    group_ids = {g["setting_id"] for g in config["groups"]}
    used = set()
    for line in lines:
        if not line.startswith("|"):
            if headers is not None:
                break
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if headers is None:
            if cells[0] == "Model":
                headers = [c.split()[0] for c in cells]
            continue
        if all(re.fullmatch(r"[-: ]+", c) for c in cells):
            continue
        if len(cells) != len(headers):
            raise ValueError("experiment table row width differs from header")
        label = cells[0].replace("**", "")
        if label.startswith("+ ") and model:
            method = label[2:]
        elif " · " in label:
            model, method = label.split(" · ", 1)
        else:
            raise ValueError("experiment table requires model and method rows")
        key = f"{model} / {method}"
        binding = bindings.get(key)
        if binding is not None:
            if binding not in group_ids or binding in used:
                raise ValueError("experiment table binding must select a unique known setting")
            used.add(binding)
        rows.append(
            {
                "model": model,
                "method": method,
                "setting_id": binding,
                "metrics": dict(zip(headers[1:], cells[1:], strict=True)),
            }
        )
    if not rows or set(bindings) - {f"{r['model']} / {r['method']}" for r in rows}:
        raise ValueError("experiment table has no rows or unknown row bindings")
    return {
        "source": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "heading": heading,
        "rows": rows,
    }


def load_task(task_dir: Path):
    config_path = task_dir.expanduser().resolve() / "task.yaml"
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != "task-report-v1":
        raise ValueError("task.yaml requires schema: task-report-v1")
    if not isinstance(data.get("task_id"), str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", data["task_id"]
    ):
        raise ValueError("task_id must be a safe stable identifier")
    if not isinstance(data.get("groups"), list):
        raise ValueError("groups must be an explicit list")
    if not isinstance(data.get("output_root"), str):
        raise ValueError("output_root must be explicitly set in task.yaml")
    seen = set()
    for g in data["groups"]:
        if not isinstance(g, dict):
            raise ValueError("each group must be a mapping")
        for k in ("setting_id", "model", "method", "split"):
            if not isinstance(g.get(k), str) or not g[k]:
                raise ValueError(f"each group requires {k}")
        if g["setting_id"] in seen:
            raise ValueError("duplicate setting_id; use separate IDs for versions/splits")
        seen.add(g["setting_id"])
        target = g.get("target_attempts")
        if target is not None and (type(target) is not int or target <= 0):
            raise ValueError("target_attempts must be null or positive integer")
        if not isinstance(g.get("expected", {}), dict):
            raise ValueError("expected must map dotted metadata fields to exact values")
        if (g.get("episodes") or g.get("raw_roots")) and "meta.task" not in g.get("expected", {}):
            raise ValueError(
                "groups with inputs require expected.meta.task (explicit prompt mapping)"
            )
    return config_path, data


def collect_task(config_path: Path, config: dict):
    base = config_path.parent
    groups, episodes, seen_episodes, raw_roots = [], [], set(), []
    for group in config["groups"]:
        discoveries, notes = {}, []
        for value in group.get("raw_roots", []):
            root = _path(base, value)
            raw_roots.append(root)
            if not root.is_dir():
                notes.append(f"raw_root_missing: {root}")
                continue
            for pattern in ("session-*/rollout-*", "rollout-*"):
                for path in sorted(root.glob(pattern)):
                    if path.is_dir():
                        discoveries[path.resolve()] = {}
        for entry in group.get("episodes", []):
            if not isinstance(entry, dict) or "episode_dir" not in entry:
                raise ValueError("episodes entries require episode_dir")
            path = _path(base, entry["episode_dir"])
            discoveries[path] = entry
        group_episodes, metric_payloads = [], []
        for episode, entry in sorted(discoveries.items()):
            if episode in seen_episodes:
                raise ValueError(f"episode is mapped to multiple settings: {episode}")
            seen_episodes.add(episode)
            attempt_id = hashlib.sha256(str(episode).encode()).hexdigest()[:16]
            row = {
                "attempt_id": attempt_id,
                "task": config["task_id"],
                "setting_id": group["setting_id"],
                "model": group["model"],
                "method": group["method"],
                "split": group["split"],
                "episode_dir": str(episode),
                "host": socket.gethostname(),
                "errors": [],
            }
            documents = {}
            for key, path in {
                "meta": episode / "meta.json",
                "result": episode / "result.json",
                "manifest": episode.parent / "session-manifest.json",
            }.items():
                row[key + "_path"] = str(path) if path.is_file() else None
                try:
                    documents[key] = _json(path)
                    if not isinstance(documents[key], dict):
                        raise ValueError("expected a JSON object")
                except (OSError, ValueError) as exc:
                    documents[key] = {}
                    row["errors"].append(f"{key}: {exc}")
            meta = documents["meta"]
            row["finalized"] = bool(row["result_path"] and not episode.name.endswith(".partial"))
            row["partial"] = episode.name.endswith(".partial") or not row["result_path"]
            mismatch = [
                k for k, v in group.get("expected", {}).items() if _lookup(documents, k) != v
            ]
            row["binding"] = {
                "status": "mismatch" if mismatch else "matched_fields",
                "mismatched_fields": mismatch,
                "expected": group.get("expected", {}),
            }
            row["layout_id"], row["repeat_id"] = meta.get("layout_id"), meta.get("repeat_id")
            row["reference_layout"] = meta.get("reference_layout")
            row["recorded_model"] = _lookup(documents, "meta.policy_backend.model")
            row["recorded_inference"] = _lookup(documents, "manifest.config.inference")
            row["source_hashes"] = {
                k: _hash(Path(row[k + "_path"])) for k in documents if row[k + "_path"]
            }
            row["videos_index"] = (
                str(episode / "videos/index.json")
                if (episode / "videos/index.json").is_file()
                else None
            )
            row["video_paths"] = [str(p) for p in sorted((episode / "videos").glob("*.mp4"))]
            row["data_zarr"] = (
                str(episode / "data.zarr") if (episode / "data.zarr").is_dir() else None
            )
            row["events"] = (
                str(episode / "events.jsonl") if (episode / "events.jsonl").is_file() else None
            )
            eligible = row["finalized"] and not row["errors"] and not mismatch
            row["human"] = _human(
                episode / "evaluation/human-label.json", eligible, meta.get("evaluation")
            )
            overlap_path = (
                _path(base, entry["overlap_metrics"]) if entry.get("overlap_metrics") else None
            )
            row["overlap"], payload = _overlap(overlap_path, episode, eligible)
            row["handoff_figures"], row["handoff_figure_errors"] = _handoff_figures(
                entry, base, eligible, attempt_id
            )
            row["formal_seam"] = {
                "status": "missing",
                "reason": "execution_boundary_review_not_integrated",
            }
            row["prm"] = {
                "status": "missing",
                "reason": "matched_PRM_profile_and_case_import_not_integrated",
            }
            if payload is not None:
                row["overlap_summary"] = payload["summary"]
                row["overlap_windows"] = payload["windows_policy_steps"]
                row["overlap_candidate_count"] = payload["candidate_pair_count"]
                metric_payloads.append((row, payload))
            group_episodes.append(row)
        signatures = {r["overlap"]["profile_signature"] for r, p in metric_payloads}
        if len(signatures) > 1:
            for row, _ in metric_payloads:
                row["overlap"]["status"] = "excluded"
                row["overlap"]["reason"] = "incompatible_overlap_profiles_in_setting"
            notes.append("Overlap profiles differ: split this setting before comparing metrics.")
        groups.append(_group_summary(group, group_episodes, notes))
        episodes.extend(group_episodes)
    return {
        "task_id": config["task_id"],
        "title": config.get("title", config["task_id"]),
        "scope": "Task evidence report; candidate overlap diagnostics are not formal seam scores.",
        "formal_evaluation": False,
        "groups": groups,
        "experiment_table": _experiment_table(config, base),
        "episodes": episodes,
    }, raw_roots


def _group_summary(group, episodes, notes):
    human = {}
    for mode in ("live", "video"):
        labels = [r["human"]["result"] for r in episodes if r["human"].get("review_mode") == mode]
        count = Counter(labels)
        total = count["success"] + count["failure"]
        scored = [
            r["human"]
            for r in episodes
            if r["human"].get("review_mode") == mode and "score" in r["human"]
        ]
        score_groups = []
        for signature in sorted({h["rubric_signature"] for h in scored}):
            selected = [h for h in scored if h["rubric_signature"] == signature]
            score_groups.append(
                {
                    "rubric_id": selected[0]["rubric_id"],
                    "target_count": selected[0]["target_count"],
                    "n": len(selected),
                    "mean": _mean([h["score"] for h in selected]),
                }
            )
        human[mode] = {
            "count_scores_by_rubric": score_groups,
            "success": count["success"],
            "failure": count["failure"],
            "invalid": count["invalid"],
            "n": total,
            "sr": count["success"] / total if total else None,
            "wilson95": _wilson(count["success"], total),
        }
    slots = Counter(
        (r["layout_id"], r["repeat_id"])
        for r in episodes
        if isinstance(r["layout_id"], str)
        and r["layout_id"] in {f"{i:02d}" for i in range(1, 11)}
        and type(r["repeat_id"]) is int
        and 1 <= r["repeat_id"] <= 3
    )
    diagnostic = [r for r in episodes if r["overlap"]["status"] == "available"]
    windows = sorted({n for r in diagnostic for n in r["overlap_windows"]})
    overlap = {}
    for arm in ("left_arm", "right_arm"):
        overlap[arm] = {}
        for n in windows:
            per = [
                r["overlap_summary"][arm]["common_boundary_by_window"][str(n)]
                for r in diagnostic
                if arm in r["overlap_summary"]
            ]
            overlap[arm][str(n)] = {
                "numeric_boundary_count": sum(s["numeric_valid_count"] for s in per),
                "metrics": {
                    key: {
                        "episode_count": sum(s["metrics"][key]["count"] > 0 for s in per),
                        "mean_of_episode_means": _mean(
                            [
                                s["metrics"][key]["mean"]
                                for s in per
                                if s["metrics"][key]["count"] > 0
                            ]
                        ),
                    }
                    for key in METRICS
                },
            }
    return {
        k: group.get(k) for k in ("setting_id", "model", "method", "split", "target_attempts")
    } | {
        "attempts": len(episodes),
        "finalized": sum(r["finalized"] for r in episodes),
        "partial": sum(r["partial"] for r in episodes),
        "binding_mismatch": sum(r["binding"]["status"] == "mismatch" for r in episodes),
        "processing_errors": sum(bool(r["errors"]) for r in episodes),
        "human": human,
        "human_status": dict(Counter(r["human"]["status"] for r in episodes)),
        "known_slots": len(slots),
        "unknown_slots": sum(slots.values()) != len(episodes),
        "slots": [
            {"layout_id": k[0], "repeat_id": k[1], "attempts": v} for k, v in sorted(slots.items())
        ],
        "duplicate_slots": sum(v > 1 for v in slots.values()),
        "overlap_episodes": len(diagnostic),
        "overlap": overlap,
        "notes": notes,
        "overlap_missing_reasons": dict(
            Counter(r["overlap"]["reason"] for r in episodes if r["overlap"].get("reason"))
        ),
        "formal_seam_reason": "execution_boundary_review_not_integrated",
        "prm_reason": "matched_PRM_profile_and_case_import_not_integrated",
    }


def generate_task_report(task_dir: Path) -> Path:
    from manimux.evaluation.report_html import render_report

    config_path, config = load_task(task_dir)
    report, raw_roots = collect_task(config_path, config)
    parent = (
        _path(config_path.parent, config["output_root"]) / config["task_id"] / "reports"
    ).resolve()
    for raw in raw_roots + [Path(r["episode_dir"]) for r in report["episodes"]]:
        if parent == raw or raw in parent.parents:
            raise ValueError("report output must be outside all raw inputs")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_id = f"{stamp}-{uuid.uuid4().hex[:8]}"
    output = parent / report_id
    staging = parent / ("." + report_id + ".building")
    staging.mkdir(parents=True)
    report.update(
        report_id=report_id,
        generated_at=datetime.now(UTC).isoformat(),
        output_dir=str(output),
        html_path=str(output / "index.html"),
        task_config=str(config_path),
    )
    try:

        def dump(name, value):
            (staging / name).write_text(
                json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )

        provenance = {
            "task_config": str(config_path),
            "task_config_sha256": _hash(config_path),
            "source_code_sha256": {
                name: _hash(Path(__file__).with_name(name))
                for name in ("report.py", "report_html.py", "report_template.html")
            },
            "experiment_table": report["experiment_table"],
            "inputs": [
                {
                    "episode_dir": r["episode_dir"],
                    "hashes": r["source_hashes"],
                    "overlap": r["overlap"],
                    "human": r["human"],
                    "handoff_figures": r["handoff_figures"],
                }
                for r in report["episodes"]
            ],
            "limitations": [
                "raw reference array hashes are retained from overlap artifacts, not recomputed",
                "human live/video results are separate attempt-level scores",
                "formal seam qualification and PRM case import remain unavailable",
            ],
        }
        dump("summary.json", {k: v for k, v in report.items() if k != "episodes"})
        dump("episode-metrics.json", report["episodes"])
        dump("provenance.json", provenance)
        (staging / "task.yaml").write_bytes(config_path.read_bytes())
        with (staging / "cohort.jsonl").open("w", encoding="utf-8") as f:
            for row in report["episodes"]:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        fields = [
            "setting_id",
            "model",
            "method",
            "split",
            "attempt_id",
            "episode_dir",
            "finalized",
            "partial",
            "human_result",
            "human_review_mode",
            "overlap_status",
            "overlap_reason",
        ]
        with (staging / "episodes.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in report["episodes"]:
                cells = {k: r.get(k) for k in fields}
                cells.update(
                    human_result=r["human"].get("result"),
                    human_review_mode=r["human"].get("review_mode"),
                    overlap_status=r["overlap"]["status"],
                    overlap_reason=r["overlap"]["reason"],
                )
                # Prevent spreadsheet formula execution from free-form recorded text.
                writer.writerow(
                    {
                        k: (
                            "'" + v
                            if isinstance(v, str) and v.startswith(("=", "+", "-", "@", "\t", "\r"))
                            else v
                        )
                        for k, v in cells.items()
                    }
                )
        figures = staging / "figures"
        figures.mkdir()
        for row in report["episodes"]:
            for figure in row["handoff_figures"]:
                data = Path(figure["source"]).read_bytes()
                if hashlib.sha256(data).hexdigest() != figure["source_sha256"]:
                    raise ValueError("handoff image changed while building report")
                (figures / figure["filename"]).write_bytes(data)
        page = render_report(report, figures)
        (staging / "index.html").write_text(page, encoding="utf-8")
        staging.rename(output)
        latest = parent.parent / "latest.json"
        temporary = latest.with_name(".latest-" + uuid.uuid4().hex + ".json")
        temporary.write_text(
            json.dumps(
                {
                    "report_id": report_id,
                    "html": str(output / "index.html"),
                    "summary": str(output / "summary.json"),
                    "generated_at": report["generated_at"],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(latest)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output / "index.html"
