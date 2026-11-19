"""Optional RoboGUI replay plugin: synchronized video, tracking and chunk evidence."""

from __future__ import annotations

import html
import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .replay_data.tracking import load_tracking
from .replay_layout import tracking_layout

CURVES = (
    ("policy_ik", "Policy → IK", "#b567df"),
    ("ik_feedback", "IK → feedback", "#ef9645"),
    ("reference_feedback", "Execution reference → feedback", "#e45f69"),
    ("command_feedback", "Command → feedback", "#4d9fd8"),
)


def camera_index(stamps, time_ns):
    """Last captured frame at the replay clock; never display a future frame."""
    return int(np.searchsorted(stamps, time_ns, side="right") - 1)


class RecordedVideo:
    """OpenCV file reader with bounded cache; no camera/device APIs."""

    def __init__(self, path, stamps):
        try:
            import cv2
        except ImportError as exc:
            raise ValueError("Install the 'video' extra to decode recorded MP4 files") from exc

        self.cv2 = cv2
        self.stamps = np.asarray(stamps, dtype=np.int64)
        # Bundled recording may repeat a slower camera while another advances.
        # Equal capture times are legitimate; searchsorted chooses the last copy.
        if not len(self.stamps) or np.any(np.diff(self.stamps) < 0):
            raise ValueError("Video timestamps must be nonempty and nondecreasing")
        self.cap = cv2.VideoCapture(str(path))
        self.cache = OrderedDict()
        if not self.cap.isOpened() or int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) != len(
            self.stamps
        ):
            self.close()
            raise ValueError("Video frame count does not match its timestamp index")

    def frame(self, index):
        if index not in self.cache:
            if int(self.cap.get(self.cv2.CAP_PROP_POS_FRAMES)) != index:
                self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = self.cap.read()
            if not ok:
                raise ValueError(f"Cannot decode video frame {index}")
            self.cache[index] = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
            while len(self.cache) > 8:
                self.cache.popitem(last=False)
        return self.cache[index]

    def close(self):
        self.cap.release()
        self.cache.clear()


def _svg(body, height, width=520):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        'preserveAspectRatio="none" '
        f'style="width:100%;display:block;background:#101824;border-radius:8px" '
        f'role="img">{body}</svg>'
    )


def _text(x, y, value, color="#bbc8d8", size=11):
    return (
        f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" '
        f'font-family="sans-serif">{html.escape(str(value))}</text>'
    )


def tracking_chart(times, series, index, unit, breaks=()):
    width, left, top, height = 460, 48, 22, 120
    duration = max(float(times[-1]), 1e-9)
    finite = np.concatenate([v[np.isfinite(v)] for v in series.values()])
    ymax = max(1.0, float(finite.max()) * 1.05) if len(finite) else 1.0
    body = _text(8, 15, f"TCP error ({unit})")
    for frac in (0, 0.5, 1):
        y = top + (1 - frac) * height
        body += f'<path d="M{left},{y}H508" stroke="#2c394b"/>'
        body += _text(3, y + 4, f"{frac * ymax:.1f}", size=10)
    for key, _label, color in CURVES:
        values = series[key]
        good = np.isfinite(values)
        starts = np.r_[True, ~good[:-1]] | np.isin(np.arange(len(times)), breaks)
        # Keep both extrema in each bin, retaining short error spikes and gaps.
        keep = {index, 0, len(times) - 1}
        keep.update(np.flatnonzero(starts).tolist())
        keep.update(np.flatnonzero(np.r_[good[:-1] != good[1:], False]).tolist())
        for chunk in np.array_split(np.flatnonzero(good), min(240, max(1, int(good.sum())))):
            if len(chunk):
                keep.update(
                    (int(chunk[np.argmin(values[chunk])]), int(chunk[np.argmax(values[chunk])]))
                )
        parts, pen = [], False
        previous = -1
        for i in sorted(keep):
            if not good[i]:
                pen = False
                previous = i
                continue
            if previous >= 0 and (
                not good[previous : i + 1].all() or starts[previous + 1 : i + 1].any()
            ):
                pen = False
            x, y = left + width * times[i] / duration, top + height * (1 - values[i] / ymax)
            parts.append(f"{'L' if pen else 'M'}{x:.2f},{y:.2f}")
            pen, previous = True, i
        body += f'<path d="{" ".join(parts)}" fill="none" stroke="{color}" stroke-width="1.5"/>'
    x = left + width * times[index] / duration
    body += f'<path d="M{x:.2f},18V148" stroke="white" stroke-width="1"/>'
    body += _text(left, 160, "0 s") + _text(450, 160, f"{duration:.2f} s")
    return _svg(body, 170)


def chunk_chart(data, index, *, width=1440):
    start, end = int(data.stamps[0]), int(data.stamps[-1])

    def x(t):
        return 80 + (width - 96) * np.clip((int(t) - start) / max(1, end - start), 0, 1)

    body = _text(8, 16, "Chunk execution and inference")
    body += _text(5, 46, "Execute", size=10) + _text(5, 83, "Requests", size=10)
    for a, b, label in data.timeline:
        color = (
            "#467d9a"
            if label.startswith("#")
            else "#b9654e"
            if label.startswith("Hold")
            else "#647080"
        )
        xa, xb = x(a), x(b)
        body += (
            f'<rect x="{xa:.2f}" y="28" width="{max(0.6, xb - xa):.2f}" '
            f'height="27" fill="{color}"><title>{html.escape(label)}</title></rect>'
        )
        if xb - xa > 19:
            body += _text(
                xa + 2,
                46,
                label if label.startswith("#") else "Hold" if label.startswith("Hold") else "Pause",
                "#fff",
                9,
            )
    for r in data.requests:
        a, b = r.get("start_ns"), r.get("end_ns")
        if a is None:
            continue
        color = {"accepted": "#55c39f", "rejected": "#f07772"}.get(r["status"], "#b6c0d0")
        # Historical rejections have no decision timestamp: anchor their marker
        # to the request/observation, never fabricate a decision time.
        title = f"#{r['seq']} {r['status']}; " + str(r.get("reason", ""))
        if b is None:
            title += "; decision time not recorded; marker at request observation"
        if not r.get("exact_start"):
            title += "; start uses observation timestamp"
        body += f"<g><title>{html.escape(title)}</title>"
        if b is not None:
            body += f'<path d="M{x(a):.2f},78H{x(b):.2f}" stroke="{color}" stroke-width="3"/>'
        at = x(b if b is not None else a)
        if r["status"] == "rejected":
            body += f'<path d="M{at - 3:.2f},73l6,10m-6,0l6,-10" stroke="{color}" fill="none"/>'
        else:
            body += f'<circle cx="{at:.2f}" cy="78" r="3" fill="{color}"/>'
        body += "</g>"
    at = x(data.stamps[index])
    body += f'<path d="M{at:.2f},23V90" stroke="white"/>'
    body += _text(68, 105, "Green: accepted · red ×: rejected · brown: hold", size=10)
    return _svg(body, 115, width)


class TrackingReplayPlugin:
    """Factory contract: (player, episode), then update(tick_index) / close()."""

    def __init__(self, player, episode: Path):
        self.player = player
        self.data = load_tracking(episode, player.robot.kinematics.models)
        if len(self.data.stamps) != player.count:
            raise ValueError("Tracking and player tick counts differ")
        self.times = (self.data.stamps - self.data.stamps[0]) / 1e9
        self.videos = {}
        self._image_keys = {}
        self._last_update = 0.0
        self._index = 0
        self.markers = {}
        try:
            index_path = episode / "videos/index.json"
            if index_path.exists():
                video_index = json.loads(index_path.read_text())
                for name, info in video_index.get("cameras", {}).items():
                    try:
                        path = (episode / "videos" / info["file"]).resolve()
                        if not path.is_relative_to((episode / "videos").resolve()):
                            raise ValueError(
                                "Video index points outside the episode video directory"
                            )
                        self.videos[name] = RecordedVideo(path, info["capture_monotonic_ns"])
                    except (ValueError, OSError) as exc:
                        self.data.notes.append(f"{name}: video unavailable ({exc})")
            self._build_gui()
            self._build_markers()
        except BaseException:
            self.close()
            raise

    def _build_gui(self):
        gui = self.player.server.gui
        gui.configure_theme(control_layout="fixed", control_width="large", dark_mode=True)
        self.layout = gui.add_html(tracking_layout(False))
        self.metric = gui.add_dropdown("Error", ("Position (mm)", "Rotation (deg)"))
        options = {"Start": 0}
        for p in self.data.plans.values():
            ix = int(
                np.clip(np.searchsorted(self.data.stamps, p["start_ns"]), 0, len(self.times) - 1)
            )
            options[f"Chunk #{p['seq']} · {self.times[ix]:.2f} s"] = ix
        jump = gui.add_dropdown("Jump to chunk", tuple(options), initial_value="Start")
        self.scene_enabled = gui.add_checkbox("Show 3D reference", False)

        @jump.on_update
        def _jump(event):
            if event.client is not None:
                self.player.play.value = False
                self.player.seek(options[jump.value])

        @self.scene_enabled.on_update
        def _scene(event):
            if event.client is not None:
                self.layout.content = tracking_layout(self.scene_enabled.value)

        self.cameras, self.images, self.video_status, self.charts, self.values = {}, {}, {}, {}, {}
        for group in self.data.poses:
            side = group.removesuffix("_arm")
            with gui.add_folder(f"{side.title()} wrist · {group}"):
                gui.add_html('<div class="mm-replay-arm"></div>')
                camera_names = tuple(self.videos) or ("Unavailable",)
                preferred = next((name for name in camera_names if side in name), camera_names[0])
                self.cameras[group] = gui.add_dropdown(
                    f"{side.title()} video", camera_names, initial_value=preferred
                )
                self.images[group] = gui.add_image(
                    np.zeros((240, 320, 3), dtype=np.uint8), format="jpeg", jpeg_quality=80
                )
                self.video_status[group] = gui.add_markdown("")
                self.charts[group] = gui.add_html("")
                self.values[group] = gui.add_html("")
        with gui.add_folder("Chunk execution and inference"):
            gui.add_html('<div class="mm-replay-wide"></div>')
            self.timeline = gui.add_html("")
        for handle in (*self.cameras.values(), self.metric):

            @handle.on_update
            def _change(event):
                if event.client is not None:
                    with self.player._lock:
                        self._last_update = 0
                        self.update(self._index)

        with gui.add_folder("Replay evidence and limitations", expand_by_default=False):
            gui.add_html('<div class="mm-replay-wide"></div>')
            gui.add_markdown(
                "\n\n".join(self.data.notes)
                + "\n\nPolicy/IK comparisons use the same source phase, before bridge shaping. "
                "Execution reference includes retiming and the bridge. "
                "Gaps are missing data, not zero error. Historical rejection markers without "
                "decision timestamps are placed at the request observation. These charts use "
                "simultaneous feedback, without time-shifting it to hide tracking lag."
            )
            gui.add_markdown(f"Episode: `{self.data.episode}`")

    def _build_markers(self):
        colors = {
            "policy": (181, 103, 223),
            "ik": (239, 150, 69),
            "reference": (228, 95, 105),
            "feedback": (77, 159, 216),
        }
        for group in self.player.robot.groups:
            if group.name not in self.data.poses:
                continue
            root = f"{self.player.scene_root}/tracking/{group.name}"
            self.player.server.scene.add_frame(
                root, show_axes=False, position=group.base_position, wxyz=group.base_orientation
            )
            for key, color in colors.items():
                self.markers[group.name, key] = self.player.server.scene.add_icosphere(
                    f"{root}/{key}", radius=0.008, color=color, visible=False
                )

    def update(self, index):
        if self.player._closed:
            return
        now = time.monotonic()
        if self.player.play.value and now - self._last_update < 1 / 30:
            return
        self._last_update = now
        self._index = index
        unit = "mm" if self.metric.value.startswith("Position") else "deg"
        breaks = np.flatnonzero(self.data.plan_ids[1:] != self.data.plan_ids[:-1]) + 1
        self.timeline.content = (
            '<div class="mm-replay-chunks">' + chunk_chart(self.data, index) + "</div>"
        )
        for group in self.data.poses:
            series = {k: v[unit] for k, v in self.data.errors[group].items()}
            self.charts[group].content = (
                '<div class="mm-replay-chart">'
                + tracking_chart(self.times, series, index, unit, breaks)
                + "</div>"
            )
            rows = []
            for key, label, color in CURVES:
                value = series[key][index]
                text = (
                    f"{value:.2f} {unit}"
                    if np.isfinite(value)
                    else "Not recorded / no aligned path"
                )
                rows.append(
                    f'<div><span style="color:{color}">{label}</span><strong>{text}</strong></div>'
                )
            self.values[group].content = '<div class="mm-replay-values">' + "".join(rows) + "</div>"
            self._update_video(group, index)
        for (name, key), handle in self.markers.items():
            pos = self.data.poses[name][key][index, :3, 3]
            handle.visible = bool(self.data.active[index] and np.isfinite(pos).all())
            if handle.visible:
                handle.position = pos

    def _update_video(self, group, index):
        camera = self.cameras[group].value
        image, status = self.images[group], self.video_status[group]
        if camera in self.videos:
            video = self.videos[camera]
            frame = camera_index(video.stamps, self.data.stamps[index])
            if frame < 0:
                image.visible = False
                status.content = "Video has not started at this timestamp."
            else:
                try:
                    if self._image_keys.get(group) != (camera, frame):
                        image.image = video.frame(frame)
                        self._image_keys[group] = (camera, frame)
                    image.visible = True
                    age = (int(self.data.stamps[index]) - int(video.stamps[frame])) / 1e6
                    status.content = (
                        f"Frame {frame + 1}/{len(video.stamps)} · capture age {age:.1f} ms"
                        + (" · **stale / video gap**" if age > 250 else "")
                    )
                except ValueError as exc:
                    image.visible = False
                    status.content = f"Video unavailable: {exc}"
        else:
            image.visible = False
            status.content = "No indexed video recorded for this episode."

    def close(self):
        for video in self.videos.values():
            video.close()
