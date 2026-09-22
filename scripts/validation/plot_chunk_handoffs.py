#!/usr/bin/env python3
"""Plot adjacent Tianji canonical-chunk handoffs in a recorded rollout.

The full canonical chunks are aligned on their recorded source clocks.  Seam
markers use the actual committed Timeline values at the incoming plan's
start_time_ns, so the two black markers represent the executed handoff rather
than an approximation from source-row timestamps. Generated PNG, PDF, CSV and
JSON artifacts are written outside the episode so recordings remain unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DEFAULT_ROBOT_CONFIG = ROOT / "manimux/configs/embodiment/robot/tianji_taccap.yaml"

WIDTH, HEIGHT = 1800, 2040
# Every subplot uses exactly the same physical vertical scale.  Eight equal
# intervals at 0.02 m per grid cell provide a seam-centred 0.16 m window.
Y_GRID_M = 0.02
Y_GRID_INTERVALS = 8
Y_SPAN_M = Y_GRID_M * Y_GRID_INTERVALS
BG = "#f7f8fa"
PANEL_BG = "#fbfcfd"
FRAME = "#aab4c0"
GRID = "#dce2e8"
TEXT = "#20252b"
MUTED = "#687381"
OLD = "#2f6fbd"
NEW = "#e34d59"
EXPIRED = "#fff0cf"
POST_SWITCH = "#e7f1fd"
BLACK = "#15191d"
WHITE = "#ffffff"


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


F_TITLE = font(34, bold=True)
F_SUBTITLE = font(20)
F_PANEL = font(25, bold=True)
F_AXIS = font(18)
F_TICK = font(16)
F_NOTE = font(16)
F_NOTE_BOLD = font(16, bold=True)


@dataclass(frozen=True)
class Plan:
    index: int
    request_seq: int
    source_start_ns: int
    committed_start_ns: int
    dt_ns: int
    canonical: dict[str, np.ndarray]
    committed: dict[str, np.ndarray]

    @property
    def trim_steps(self) -> int:
        lag = max(0, self.committed_start_ns - self.source_start_ns)
        return math.ceil(lag / self.dt_ns) if lag else 0


def load_plans(episode: Path) -> list[Plan]:
    root = zarr.open_group(str(episode / "data.zarr"), mode="r")
    plans = root["plans"]
    loaded: list[Plan] = []
    for name in sorted(key for key in plans.group_keys() if key.isdigit()):
        group = plans[name]
        canonical = group["canonical_raw"]
        committed = group["committed"]
        loaded.append(
            Plan(
                index=int(name),
                request_seq=int(group.attrs["request_seq"]),
                source_start_ns=int(canonical.attrs["observation_time_ns"]),
                committed_start_ns=int(committed.attrs["start_time_ns"]),
                dt_ns=int(canonical.attrs["dt_ns"]),
                canonical={
                    arm: np.asarray(canonical[arm][:], dtype=np.float64)
                    for arm in ("left_arm", "right_arm")
                },
                committed={
                    arm: np.asarray(committed[arm][:], dtype=np.float64)
                    for arm in ("left_arm", "right_arm")
                },
            )
        )
    return loaded


def fk_xyz(model, rows: np.ndarray) -> np.ndarray:
    return np.asarray([model.fk(row)[:3, 3] for row in rows], dtype=np.float64)


def sample_committed(plan: Plan, arm: str, time_ns: int) -> np.ndarray:
    values = plan.committed[arm]
    position = (time_ns - plan.committed_start_ns) / plan.dt_ns
    position = float(np.clip(position, 0.0, len(values) - 1))
    lower = min(int(math.floor(position)), len(values) - 1)
    upper = min(lower + 1, len(values) - 1)
    alpha = min(position - lower, 1.0)
    return (1.0 - alpha) * values[lower] + alpha * values[upper]


def text_center(
    draw: ImageDraw.ImageDraw, xy: tuple[float, float], value: str, used_font, fill: str
) -> None:
    box = draw.textbbox((0, 0), value, font=used_font)
    draw.text(
        (xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2),
        value,
        font=used_font,
        fill=fill,
    )


def draw_dashed(
    draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], fill: str, width: int = 4
) -> None:
    dash, gap = 12.0, 7.0
    phase = 0.0
    drawing = True
    for p0, p1 in zip(points, points[1:], strict=False):
        x0, y0 = p0
        x1, y1 = p1
        length = math.hypot(x1 - x0, y1 - y0)
        if length == 0:
            continue
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        cursor = 0.0
        while cursor < length:
            remaining = (dash if drawing else gap) - phase
            step = min(remaining, length - cursor)
            if drawing:
                draw.line(
                    [
                        (x0 + ux * cursor, y0 + uy * cursor),
                        (x0 + ux * (cursor + step), y0 + uy * (cursor + step)),
                    ],
                    fill=fill,
                    width=width,
                )
            cursor += step
            phase += step
            limit = dash if drawing else gap
            if phase >= limit - 1e-9:
                drawing = not drawing
                phase = 0.0


def plot_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    title: str,
    old_t: np.ndarray,
    old_y: np.ndarray,
    new_t: np.ndarray,
    new_y: np.ndarray,
    switch_t: float,
    overlap_end: float,
    outgoing_y: float,
    incoming_y: float,
) -> None:
    x0, y0, x1, y1 = rect
    ml, mr, mt, mb = 88, 24, 54, 58
    px0, py0, px1, py1 = x0 + ml, y0 + mt, x1 - mr, y1 - mb
    draw.rectangle((px0, py0, px1, py1), fill=PANEL_BG, outline=FRAME, width=2)
    text_center(draw, ((x0 + x1) / 2, y0 + 18), title, F_PANEL, TEXT)

    xmin = float(min(old_t.min(), new_t.min()))
    xmax = float(max(old_t.max(), new_t.max()))
    # Keep tick values on exact 2 cm boundaries and center the 16 cm window on
    # the actual Timeline seam.  Full source trajectories remain on the x axis;
    # portions outside the seam window are clipped to the plot frame.
    seam_min, seam_max = sorted((outgoing_y, incoming_y))
    centered_min = (seam_min + seam_max - Y_SPAN_M) / 2
    ymin = round(centered_min / Y_GRID_M) * Y_GRID_M
    ymin = min(ymin, math.floor(seam_min / Y_GRID_M) * Y_GRID_M)
    if ymin + Y_SPAN_M < seam_max:
        ymin = math.ceil(seam_max / Y_GRID_M) * Y_GRID_M - Y_SPAN_M
    ymax = ymin + Y_SPAN_M

    def sx(value: float) -> float:
        return px0 + (value - xmin) / (xmax - xmin) * (px1 - px0)

    def sy(value: float) -> float:
        return py1 - (value - ymin) / (ymax - ymin) * (py1 - py0)

    overlap_start = max(0.0, xmin)
    expired_end = min(switch_t, overlap_end)
    if expired_end > overlap_start:
        draw.rectangle((sx(overlap_start), py0, sx(expired_end), py1), fill=EXPIRED)
    if overlap_end > max(switch_t, overlap_start):
        draw.rectangle(
            (sx(max(switch_t, overlap_start)), py0, sx(overlap_end), py1), fill=POST_SWITCH
        )

    for value in ymin + np.arange(Y_GRID_INTERVALS + 1) * Y_GRID_M:
        yy = sy(float(value))
        draw.line((px0, yy, px1, yy), fill=GRID, width=1)
        label = f"{value:.3f}"
        box = draw.textbbox((0, 0), label, font=F_TICK)
        draw.text((px0 - 10 - (box[2] - box[0]), yy - 9), label, font=F_TICK, fill=MUTED)
    for value in np.linspace(xmin, xmax, 7):
        xx = sx(float(value))
        draw.line((xx, py0, xx, py1), fill=GRID, width=1)
        label = f"{value:+.2f}"
        box = draw.textbbox((0, 0), label, font=F_TICK)
        draw.text((xx - (box[2] - box[0]) / 2, py1 + 8), label, font=F_TICK, fill=MUTED)

    old_points = [(sx(float(t)), sy(float(v))) for t, v in zip(old_t, old_y, strict=True)]
    new_points = [(sx(float(t)), sy(float(v))) for t, v in zip(new_t, new_y, strict=True)]
    # Draw trajectories on a transparent layer, then paste only the plot-area
    # crop so seam-focused y windows cannot leak into labels or adjacent panels.
    layer = Image.new("RGBA", draw._image.size, (0, 0, 0, 0))
    layer_draw = ImageDraw.Draw(layer)
    layer_draw.line(old_points, fill=OLD, width=4, joint="curve")
    draw_dashed(layer_draw, new_points, NEW, width=4)
    for xx, yy in old_points:
        layer_draw.ellipse((xx - 2.2, yy - 2.2, xx + 2.2, yy + 2.2), fill=OLD)
    for xx, yy in new_points:
        layer_draw.ellipse((xx - 2.2, yy - 2.2, xx + 2.2, yy + 2.2), fill=NEW)
    clipped = layer.crop((px0, py0, px1 + 1, py1 + 1))
    draw._image.paste(clipped, (px0, py0), clipped)

    seam_x = sx(switch_t)
    draw.line((seam_x, py0, seam_x, py1), fill=BLACK, width=3)
    draw.ellipse(
        (seam_x - 6, sy(outgoing_y) - 6, seam_x + 6, sy(outgoing_y) + 6),
        fill=WHITE,
        outline=BLACK,
        width=2,
    )
    draw.ellipse(
        (seam_x - 6, sy(incoming_y) - 6, seam_x + 6, sy(incoming_y) + 6),
        fill=BLACK,
        outline=BLACK,
        width=2,
    )
    draw.text((seam_x + 7, py0 + 7), f"switch {switch_t:.3f}s", font=F_NOTE_BOLD, fill=TEXT)

    draw.text((x0 + 7, (py0 + py1) / 2 - 8), "m", font=F_AXIS, fill=MUTED)
    text_center(
        draw, ((px0 + px1) / 2, y1 - 13), "T relative to incoming source row 0 (s)", F_AXIS, MUTED
    )


def plot_handoff(
    old: Plan,
    new: Plan,
    models: dict[str, object],
    rollout_label: str,
) -> tuple[Image.Image, dict[str, object]]:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    old_steps = len(next(iter(old.canonical.values())))
    new_steps = len(next(iter(new.canonical.values())))
    old_t = (old.source_start_ns + np.arange(old_steps) * old.dt_ns - new.source_start_ns) / 1e9
    new_t = np.arange(new_steps) * new.dt_ns / 1e9
    switch_t = (new.committed_start_ns - new.source_start_ns) / 1e9
    overlap_end = float(min(old_t.max(), new_t.max()))

    canonical_xyz: dict[tuple[str, str], np.ndarray] = {}
    seam_xyz: dict[tuple[str, str], np.ndarray] = {}
    jumps_mm: dict[str, float] = {}
    for arm in ("left_arm", "right_arm"):
        model = models[arm]
        canonical_xyz[("old", arm)] = fk_xyz(model, old.canonical[arm])
        canonical_xyz[("new", arm)] = fk_xyz(model, new.canonical[arm])
        outgoing = model.fk(sample_committed(old, arm, new.committed_start_ns))[:3, 3]
        incoming = model.fk(new.committed[arm][0])[:3, 3]
        seam_xyz[("old", arm)] = outgoing
        seam_xyz[("new", arm)] = incoming
        jumps_mm[arm] = float(np.linalg.norm(incoming - outgoing) * 1000.0)

    title = f"Rollout {rollout_label} — full chunk {old.index:02d} → {new.index:02d} EEF handoff"
    text_center(draw, (WIDTH / 2, 34), title, F_TITLE, TEXT)
    subtitle = (
        f"requests {old.request_seq}→{new.request_seq} · "
        f"{old_steps}/{new_steps} source points · "
        f"incoming trim={new.trim_steps} · actual position seam: "
        f"L {jumps_mm['left_arm']:.1f} mm, R {jumps_mm['right_arm']:.1f} mm · "
        f"Y grid={Y_GRID_M * 100:.0f} cm × {Y_GRID_INTERVALS} (seam-centred)"
    )
    text_center(draw, (WIDTH / 2, 73), subtitle, F_SUBTITLE, MUTED)

    legend_y = 110
    draw.line((205, legend_y, 265, legend_y), fill=OLD, width=5)
    draw.text((278, legend_y - 11), f"old chunk {old.index:02d} (solid)", font=F_NOTE, fill=TEXT)
    draw_dashed(draw, [(525, legend_y), (585, legend_y)], NEW, width=5)
    draw.text((598, legend_y - 11), f"new chunk {new.index:02d} (dashed)", font=F_NOTE, fill=TEXT)
    draw.rectangle((895, legend_y - 10, 930, legend_y + 10), fill=EXPIRED)
    draw.text((942, legend_y - 11), "overlap, expired prefix", font=F_NOTE, fill=TEXT)
    draw.rectangle((1240, legend_y - 10, 1275, legend_y + 10), fill=POST_SWITCH)
    draw.text((1287, legend_y - 11), "overlap after switch", font=F_NOTE, fill=TEXT)

    left, right = 52, WIDTH - 38
    top, bottom = 145, HEIGHT - 92
    col_gap, row_gap = 42, 35
    panel_w = (right - left - col_gap) // 2
    panel_h = (bottom - top - 2 * row_gap) // 3
    arm_labels = {"left_arm": "Left arm", "right_arm": "Right arm"}
    for row, axis in enumerate("XYZ"):
        for col, arm in enumerate(("left_arm", "right_arm")):
            x0 = left + col * (panel_w + col_gap)
            y0 = top + row * (panel_h + row_gap)
            plot_panel(
                draw,
                (x0, y0, x0 + panel_w, y0 + panel_h),
                f"{arm_labels[arm]} {axis}",
                old_t,
                canonical_xyz[("old", arm)][:, row],
                new_t,
                canonical_xyz[("new", arm)][:, row],
                switch_t,
                overlap_end,
                float(seam_xyz[("old", arm)][row]),
                float(seam_xyz[("new", arm)][row]),
            )

    footer = (
        f"Overlap: 0.000 to {overlap_end:.3f}s · incoming rows 0–{new.trim_steps - 1} expired · "
        f"row {new.trim_steps} first retained · ○ outgoing Timeline reference · "
        "● incoming committed first row"
    )
    text_center(draw, (WIDTH / 2, HEIGHT - 58), footer, F_NOTE, MUTED)
    text_center(
        draw,
        (WIDTH / 2, HEIGHT - 30),
        "Y: 2 cm/grid × 8, seam-centred · source tails outside the Y window are clipped",
        F_NOTE,
        MUTED,
    )
    summary = {
        "old_chunk": old.index,
        "new_chunk": new.index,
        "old_request_seq": old.request_seq,
        "new_request_seq": new.request_seq,
        "switch_s_from_new_row0": switch_t,
        "incoming_trim_steps": new.trim_steps,
        "overlap_end_s": overlap_end,
        "left_position_seam_mm": jumps_mm["left_arm"],
        "right_position_seam_mm": jumps_mm["right_arm"],
    }
    return image, summary


def make_contact_sheet(
    images: list[Image.Image],
    summaries: list[dict[str, object]],
    rollout_label: str,
    *,
    page: int,
    pages: int,
) -> Image.Image:
    columns = 3
    thumb_w, thumb_h = 720, 816
    header = 100
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (columns * thumb_w, header + rows * thumb_h), BG)
    draw = ImageDraw.Draw(sheet)
    heading = f"Rollout {rollout_label} — every adjacent EEF chunk handoff"
    if pages > 1:
        heading += f" · overview {page}/{pages}"
    text_center(draw, (sheet.width / 2, 32), heading, F_TITLE, TEXT)
    text_center(
        draw,
        (sheet.width / 2, 70),
        "Solid = outgoing chunk · dashed = incoming chunk · black line = actual Timeline switch",
        F_SUBTITLE,
        MUTED,
    )
    for index, (image, summary) in enumerate(zip(images, summaries, strict=True)):
        row, col = divmod(index, columns)
        thumb = image.resize((thumb_w, int(HEIGHT * thumb_w / WIDTH)), Image.Resampling.LANCZOS)
        y = header + row * thumb_h
        sheet.paste(thumb, (col * thumb_w, y))
        label = (
            f"{summary['old_chunk']:02d}→{summary['new_chunk']:02d}   "
            f"L {summary['left_position_seam_mm']:.1f} mm   "
            f"R {summary['right_position_seam_mm']:.1f} mm"
        )
        text_center(
            draw, (col * thumb_w + thumb_w / 2, y + thumb.height + 17), label, F_NOTE_BOLD, TEXT
        )
    return sheet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        help="output directory (default: data/analysis/<rollout>-all-chunk-handoffs)",
    )
    parser.add_argument("--label", help="plot label (default: inferred from episode name)")
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=DEFAULT_ROBOT_CONFIG,
        help="offline Tianji assembly used for FK",
    )
    parser.add_argument(
        "--max-handoffs",
        type=int,
        help="render only the first N adjacent handoffs (useful for a quick check)",
    )
    return parser.parse_args()


def main() -> None:
    from manimux.embodiments.robot import RobotModel

    args = parse_args()
    episode = args.episode.expanduser().resolve()
    if not (episode / "data.zarr/plans").is_dir():
        raise SystemExit(f"missing recorded plans: {episode / 'data.zarr/plans'}")
    episode_name = episode.name.removesuffix(".partial")
    label = args.label or episode_name.removeprefix("rollout-")
    out = (
        args.out.expanduser().resolve()
        if args.out is not None
        else ROOT / "data/analysis" / f"{episode_name}-all-chunk-handoffs"
    )
    out.mkdir(parents=True, exist_ok=True)
    plans = load_plans(episode)
    if len(plans) < 2:
        raise SystemExit("need at least two recorded plans")
    pairs = list(zip(plans, plans[1:], strict=False))
    if args.max_handoffs is not None:
        if args.max_handoffs < 1:
            raise SystemExit("--max-handoffs must be positive")
        pairs = pairs[: args.max_handoffs]
    robot = RobotModel.from_config(args.robot_config.expanduser().resolve())
    models = dict(robot.kinematics.models)
    images: list[Image.Image] = []
    summaries: list[dict[str, object]] = []
    for old, new in pairs:
        image, summary = plot_handoff(old, new, models, label)
        path = out / f"chunk-{old.index:02d}-to-{new.index:02d}-eef-xyz.png"
        image.save(path, optimize=True)
        images.append(image)
        summaries.append(summary)

    per_page = 12
    pages = math.ceil(len(images) / per_page)
    for page_index in range(pages):
        start = page_index * per_page
        stop = min(start + per_page, len(images))
        contact = make_contact_sheet(
            images[start:stop],
            summaries[start:stop],
            label,
            page=page_index + 1,
            pages=pages,
        )
        suffix = "" if pages == 1 else f"-{page_index + 1:02d}"
        contact.save(
            out / f"rollout-{label}-all-handoffs-overview{suffix}.png",
            optimize=True,
        )
    images[0].save(
        out / f"rollout-{label}-all-handoffs.pdf",
        save_all=True,
        append_images=images[1:],
        resolution=120.0,
    )
    with (out / "handoff-summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (out / "handoff-summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(images)} handoff plots and {pages} overview page(s) from {episode} to {out}")


if __name__ == "__main__":
    main()
