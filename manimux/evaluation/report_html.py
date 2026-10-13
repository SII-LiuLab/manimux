"""Self-contained task report rendering, with native SVG figures and safe text."""

from __future__ import annotations

import base64
import html
import json
import re
from pathlib import Path


def esc(value):
    return html.escape(str(value), quote=True)


def number(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def method_label(value):
    """Format the current ordinary asynchronous algorithm label."""
    return "Async" if value == "async" else value


def line_chart(series, *, title, ylabel, cosine=False):
    """No interpolation beyond the reported per-window summary points."""
    series = [s for s in series if s["points"]]
    if not series:
        return ""
    palette = ["#067d74", "#4262a1", "#af7824", "#864e82", "#526851", "#8d5349"]
    xs = sorted({x for s in series for x, y in s["points"]})
    ys = [y for s in series for x, y in s["points"]]
    ymin, ymax = (-1, 1) if cosine else (0, max(ys) * 1.15 if max(ys) > 0 else 1)
    width, height = 820, 370 + len(series) * 22

    def xp(x):
        return 90 + (x - min(xs)) / (max(xs) - min(xs) or 1) * 660

    def yp(y):
        return 275 - (y - ymin) / (ymax - ymin) * 215

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{esc(title)}">',
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        f'<text x="30" y="27" font-size="16" fill="#172e36">{esc(title)}</text>',
        f'<text x="30" y="48" font-size="11" fill="#667780">{esc(ylabel)}</text>',
    ]
    for i in range(5):
        y = ymin + (ymax - ymin) * i / 4
        py = yp(y)
        parts.extend(
            [
                f'<line x1="90" y1="{py}" x2="750" y2="{py}" stroke="#e1e8e5"/>',
                f'<text x="78" y="{py + 4}" text-anchor="end" '
                f'font-size="11" fill="#667780">{y:.2f}</text>',
            ]
        )
    for x in xs:
        parts.append(
            f'<text x="{xp(x)}" y="295" text-anchor="middle" '
            f'font-size="11" fill="#667780">{x}</text>'
        )
    parts.append(
        '<text x="420" y="318" text-anchor="middle" '
        'font-size="12" fill="#667780">Window N / '
        "model action steps</text>"
    )
    for i, s in enumerate(series):
        color = palette[(i // 2) % len(palette)]
        dash = 'stroke-dasharray="5 4"' if i % 2 else ""
        points = " ".join(f"{xp(x):.3f},{yp(y):.3f}" for x, y in s["points"])
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2" {dash}/>'
        )
        for x, y in s["points"]:
            parts.append(
                f'<circle cx="{xp(x)}" cy="{yp(y)}" r="4" fill="{color}">'
                f"<title>{esc(s['label'])} / N={x}: {y:.4f}</title>"
                f"</circle>"
            )
        ly = 346 + i * 22
        parts.append(
            f'<line x1="90" y1="{ly}" x2="120" y2="{ly}" '
            f'stroke="{color}" stroke-width="2" {dash}/>'
            f'<text x="130" y="{ly + 4}" font-size="11" '
            f'fill="#172e36">{esc(s["label"])}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


METRIC_COLUMNS = [
    "H-Score",
    "H-SR",
    "Time(s)",
    "NLDLJ",
    "NSPARC",
    "L-Mean",
    "L-P95",
    "L-Max",
    "R-Mean",
    "R-P95",
    "R-Max",
    "M25",
    "M50",
    "M75",
    "P-SR",
    "MP",
    "PPL",
    "CRA",
    "STR",
    "DRR",
    "FNS",
    "SQS",
]


def experiment_rows(report):
    """Keep register order and explicit setting bindings; isolate extra cohorts."""
    groups = {g["setting_id"]: g for g in report["groups"]}
    register = report.get("experiment_table")
    roster = list(register["rows"]) if register else []
    bound = {r["setting_id"] for r in roster}
    roster += [
        {
            "model": g["model"],
            "method": g["method"],
            "setting_id": key,
            "metrics": {},
            "extra": bool(register),
        }
        for key, g in groups.items()
        if key not in bound
    ]
    output = []
    for row in roster or [{"model": "Not registered", "method": "—", "metrics": {}}]:
        g = groups.get(row.get("setting_id"))
        values = {k: esc(row["metrics"].get(k, "—")) for k in METRIC_COLUMNS}
        if g:
            times = []
            for mode, name in (("live", "Live"), ("video", "Video")):
                s = g["human"][mode].get("completion_time", {})
                if s.get("n"):
                    times.append(f"{number(s['mean'])}<small>{name} · n={s['n']}</small>")
            values["Time(s)"] = "<br>".join(times) or "—"
            for metric in ("nldlj", "nsparc"):
                s = g.get("smoothness", {}).get(metric, {})
                values[metric.upper()] = (
                    f"{number(s['mean'])}<small>n={s['n']}</small>" if s.get("n") else "—"
                )
            for key in ("H-Score", "H-SR"):
                scores = []
                for mode, name in (("live", "Live"), ("video", "Video")):
                    h = g["human"][mode]
                    if key == "H-SR" and h["n"]:
                        scores.append(
                            f"{h['sr'] * 100:.1f}<small>{name} {h['success']}/{h['n']}</small>"
                        )
                    if key == "H-Score":
                        scores += [
                            f"{v['mean'] * 100:.1f}<small>{name} · "
                            f"{esc(v['rubric_id'])} · n={v['n']}</small>"
                            for v in h["scores_by_rubric"]
                        ]
                if scores:
                    values[key] = "<br>".join(scores)
        suffix = ""
        if row.get("extra") and g:
            suffix = f"<small>{esc(g['split'])} · {esc(g['setting_id'])}</small>"
        output.append(
            "<tr>"
            + f"<td>{esc(row['model'])} · {esc(method_label(row['method']))}{suffix}</td>"
            + "".join(f"<td>{values[k]}</td>" for k in METRIC_COLUMNS)
            + "</tr>"
        )
    return "".join(output)


def experiment_table(rows):
    lower = {"CRA", "STR", "Time(s)", "NLDLJ", "NSPARC"}
    headers = "".join(
        f"<th>{k} {'↓' if k.startswith(('L-', 'R-')) or k in lower else '↑'}</th>"
        for k in METRIC_COLUMNS
    )
    return (
        '<div class="table-scroll experiment-matrix"><table><thead><tr>'
        "<th>Method</th>"
        + headers
        + "</tr></thead><tbody>"
        + rows
        + "</tbody></table></div>"
    )


def outcome_plots(report, figures):
    """Per-attempt dots with a mean line; no zero-filled missing measurements."""
    output = []
    for metric, title in (
        ("time", "Successful completion time / s"),
        ("nldlj", "NLDLJ ↓"),
        ("nsparc", "NSPARC ↓"),
    ):
        series = []
        for group in report["groups"]:
            rows = [r for r in report["episodes"] if r["setting_id"] == group["setting_id"]]
            label = f"{group['model']} / {method_label(group['method'])} · {group['setting_id']}"
            if metric == "time":
                for mode in ("live", "video"):
                    values = [
                        r["human"]["completion_time"]["seconds"]
                        for r in rows
                        if r["human"].get("review_mode") == mode
                        and r["human"].get("completion_time", {}).get("status") == "available"
                    ]
                    if values:
                        series.append((f"{label} / {mode}", values))
            elif group.get("smoothness", {}).get(metric, {}).get("n"):
                values = [
                    r["smoothness"][metric]["value"]
                    for r in rows
                    if r["smoothness"].get(metric, {}).get("value") is not None
                ]
                series.append((label, values))
        if not series:
            continue
        values = [v for _, vs in series for v in vs]
        lo, hi = min(values), max(values)
        pad = max((hi - lo) * 0.1, 0.1)
        lo, hi = lo - pad, hi + pad

        def xp(v, lo=lo, hi=hi):
            return 400 + (v - lo) / (hi - lo) * 320

        height = 85 + len(series) * 55
        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 {height}" '
            f'role="img" aria-label="{esc(title)}">'
            f'<rect width="820" height="{height}" fill="white"/>'
            f'<text x="20" y="25" font-size="17">{esc(title)}</text>'
        ]
        for i in range(5):
            v = lo + (hi - lo) * i / 4
            svg.append(
                f'<text x="{xp(v):.1f}" y="48" font-size="11" text-anchor="middle">{v:.2f}</text>'
            )
        for i, (label, vs) in enumerate(series):
            y = 78 + 55 * i
            svg.append(
                f'<text x="15" y="{y}" font-size="10">{esc(label[:66])}'
                f"<title>{esc(label)}</title></text>"
            )
            for j, v in enumerate(vs):
                svg.append(
                    f'<circle cx="{xp(v):.2f}" cy="{y - 4 + (j % 5 - 2) * 3}" '
                    f'r="3" fill="#067d74" opacity=".6"><title>{v:.4f}</title></circle>'
                )
            mean = sum(vs) / len(vs)
            svg.append(
                f'<path d="M {xp(mean):.2f} {y - 17} v 26" stroke="#172e36" stroke-width="2"/>'
                f'<text x="740" y="{y}" font-size="11">n={len(vs)}</text>'
            )
        svg.append("</svg>")
        content = "".join(svg)
        filename = f"outcome-{metric}.svg"
        (figures / filename).write_text(content, encoding="utf-8")
        output.append(
            f'<figure>{content}<figcaption><a href="figures/{filename}">{esc(title)}</a>'
            " · Points are rollouts; vertical lines are means.</figcaption></figure>"
        )
    return "".join(output) or '<p class="empty-state">No completion-time or smoothness values available.</p>'


def scope_figure(row, figures):
    scope = row.get("manipulation_scope", {})
    if scope.get("status") != "available":
        return ""
    encoded = base64.b64encode((figures / scope["filename"]).read_bytes()).decode("ascii")
    return (
        '<figure class="seam-figure">'
        f'<a href="figures/{esc(scope["filename"])}" target="_blank" rel="noopener">'
        f'<img src="data:image/jpeg;base64,{encoded}" alt="manipulation scope"></a>'
        "<figcaption>First front frame with the training data's manipulation scope."
        "</figcaption></figure>"
    )


def render_report(report: dict, figures: Path) -> str:
    groups = report["groups"]
    episodes = report["episodes"]
    metric_rows = []
    progress = []
    details = []
    plots = []
    seam_figures = []
    for g in groups:
        label = f"{g['model']} / {method_label(g['method'])}"
        sub = f"{g['setting_id']} · {g['split']}"
        for arm, arm_label in (("left_arm", "Left arm"), ("right_arm", "Right arm")):
            for n, s in g["overlap"][arm].items():
                m = s["metrics"]
                metric_rows.append(
                    "<tr>"
                    + f"<td>{esc(label)}<small>{esc(sub)}</small></td>"
                    + f"<td>{arm_label}</td><td>{esc(n)}</td>"
                    + f"<td>{number(m['position_rmse_mm']['mean_of_episode_means'])}</td>"
                    + f"<td>{number(m['shape_rmse_mm']['mean_of_episode_means'])}</td>"
                    + f"<td>{number(m['endpoint_mm']['mean_of_episode_means'])}</td>"
                    + f"<td>{number(m['velocity_cosine']['mean_of_episode_means'], 3)}</td>"
                    + f"<td>{m['position_rmse_mm']['episode_count']} "
                    f"/ {m['velocity_cosine']['episode_count']}</td>"
                    + f"<td>{s['numeric_boundary_count']}</td></tr>"
                )
        slots = {(s["layout_id"], s["repeat_id"]): s["attempts"] for s in g["slots"]}
        cells = []
        for i in range(1, 11):
            for r in range(1, 4):
                count = slots.get((f"{i:02d}", r), 0)
                state = "duplicate" if count > 1 else "recorded" if count else "empty"
                cells.append(
                    f'<span class="slot {state}" '
                    f'title="Layout {i:02d} / repeat {r} / attempts {count}">{i:02d}.{r}</span>'
                )
        grid = "".join(cells)
        notes = "; ".join(g["notes"])
        missing = "; ".join(f"{k}: {v}" for k, v in g["overlap_missing_reasons"].items())
        target = g["target_attempts"] or "Unspecified"
        slot_note = (
            "Legacy records with unknown layout/repeat IDs are not assigned to slots."
            if g["unknown_slots"]
            else "Slots indicate recorded attempts, not success or acceptance."
        )
        progress.append(
            f'<article class="coverage"><h3>{esc(label)}</h3><small>{esc(sub)}</small>'
            + f"<p>Recorded {g['attempts']} attempts; target {target}. "
            f"Known layout slots {g['known_slots']}/30; Duplicate slots {g['duplicate_slots']}."
            f"</p>"
            + f'<div class="slots">{grid}</div>'
            + f'<p class="caption">{slot_note}</p>'
            + f'<p class="caption">Identity mismatches {g["binding_mismatch"]}; '
            f"Metadata processing errors {g['processing_errors']}; Available trajectory artifacts "
            f"{g['overlap_episodes']} episodes.</p>"
            + (f'<p class="warning">{esc(notes)}</p>' if notes else "")
            + (f'<p class="caption mono">{esc(missing)}</p>' if missing else "")
            + "</article>"
        )
    for metric, title, ylabel, filename, cosine in (
        (
            "position_rmse_mm",
            "Position deviation between old and new plans",
            "Position RMSE / mm",
            "overlap-position-rmse.svg",
            False,
        ),
        (
            "velocity_cosine",
            "Motion-direction agreement between old and new plans",
            "Velocity cosine / dimensionless",
            "overlap-velocity-cosine.svg",
            True,
        ),
    ):
        series = []
        for g in groups:
            for arm, arm_label in (("left_arm", "Left arm"), ("right_arm", "Right arm")):
                points = [
                    (int(n), s["metrics"][metric]["mean_of_episode_means"])
                    for n, s in g["overlap"][arm].items()
                    if s["metrics"][metric]["mean_of_episode_means"] is not None
                ]
                series.append({"label": f"{g['setting_id']} · {arm_label}", "points": points})
        svg = line_chart(series, title=title, ylabel=ylabel, cosine=cosine)
        if svg:
            (figures / filename).write_text(svg, encoding="utf-8")
            plots.append(
                f'<figure>{svg}<figcaption><a download href="figures/{filename}">'
                f"Download SVG</a>"
                f"</figcaption>"
                f"</figure>"
            )
    for row in episodes:
        for figure in row.get("handoff_figures", []):
            image_path = figures / figure["filename"]
            encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
            label = (
                f"{row['setting_id']} · {Path(row['episode_dir']).name} · "
                f"chunk {figure['old_chunk']} → {figure['new_chunk']}"
            )
            seam_figures.append(
                '<figure class="seam-figure">'
                f'<a href="figures/{esc(figure["filename"])}" target="_blank" rel="noopener">'
                f'<img src="data:image/png;base64,{encoded}" alt="{esc(label)}"></a>'
                f"<figcaption>{esc(label)}<br>"
                f"Left arm {figure['left_position_seam_mm']:.2f} mm · "
                f"Right arm {figure['right_position_seam_mm']:.2f} mm. "
                "Candidate reference seam: blue is the old plan, red is the new plan, and black marks the switch time."
                "</figcaption></figure>"
            )
        h = row["human"]
        o = row["overlap"]
        notes = row["errors"] + row.get("handoff_figure_errors", [])
        scope = row.get("manipulation_scope", {})
        if scope.get("status") == "error":
            notes.append("manipulation scope: " + scope["reason"])
        notes += [str(v) for v in (h.get("reason"), o.get("reason")) if v]
        notes += [
            str(v)
            for v in (
                h.get("completion_time", {}).get("reason"),
                row.get("smoothness", {}).get("reason"),
            )
            if v
        ]
        if row["binding"]["status"] == "mismatch":
            notes.append("identity mismatch: " + ", ".join(row["binding"]["mismatched_fields"]))
        binding_text = esc(json.dumps(row["binding"]["expected"], ensure_ascii=False))
        details.append(
            f'<details class="attempt">'
            f"<summary>"
            f"<b>{esc(row['setting_id'])}</b> / {esc(Path(row['episode_dir']).name)} "
            f"<span>{esc(row['split'])} · Human: {esc(h.get('result') or h['status'])} "
            f"· overlap: {esc(o['status'])}</span>"
            f"</summary>"
            + f'<p class="mono path">{esc(row["episode_dir"])}</p>'
            + f"<p>Layout: {esc(row['layout_id'])} / Repeat: {esc(row['repeat_id'])} "
            f"/ attempt ID: {esc(row['attempt_id'])}</p>"
            + (f'<p class="warning">{esc("; ".join(notes))}</p>' if notes else "")
            + scope_figure(row, figures)
            + f'<p class="caption">Matching fields: {binding_text}</p>'
            + f'<p class="caption path">Trajectory source: {esc(o.get("source") or "Not provided")}</p>'
            + "<details><summary>Completion-time / smoothness details</summary><pre>"
            + esc(
                json.dumps(
                    {
                        "completion_time": h.get("completion_time"),
                        "smoothness": row.get("smoothness"),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            + "</pre></details>"
            + '<p class="caption">Formal seams: actual execution-boundary filtering is not integrated; '
            "PRM: matching profile/case import is not integrated.</p></details>"
        )
    total = sum(g["attempts"] for g in groups)
    tokens = {
        "MAIN_TABLE": experiment_table(experiment_rows(report)),
        "OUTCOME_FIGURES": outcome_plots(report, figures),
        "TITLE": esc(report["title"]),
        "TASK": esc(report["task_id"]),
        "GENERATED": esc(report["generated_at"]),
        "OUTPUT": esc(report["output_dir"]),
        "REPORT_ID": esc(report["report_id"]),
        "SEAM_FIGURES": "".join(seam_figures) or '<p class="empty-state">No matching seam plots available.</p>',
        "METRIC_ROWS": "".join(metric_rows) or '<tr><td colspan="9">No trajectory metrics available.</td></tr>',
        "PROGRESS": "".join(progress)
        or '<p class="empty-state">Task registered; model/method settings still need to be bound.</p>',
        "FIGURES": "".join(plots) or '<p class="empty-state">No similarity plots available.</p>',
        "ATTEMPTS": "".join(details) or '<p class="empty-state">No evaluation records available.</p>',
        "STATUS": "Partial results" if total else "Pending evaluation",
        "DATA": json.dumps(report, ensure_ascii=False, allow_nan=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026"),
    }
    template = Path(__file__).with_name("report_template.html").read_text(encoding="utf-8")
    return re.sub(r"@@([A-Z_]+)@@", lambda match: tokens[match[1]], template)
