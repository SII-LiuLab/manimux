"""Read-only collection replay using the existing Viser robot adapters."""

from __future__ import annotations

import signal
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import viser
from viser.extras import ViserUrdf

from manimux.collection.yam.data.replay import JointReplay, load_joint_replay

from .robots.base import RobotAdapter


class ReplayVideo:
    """Index decoded ordinals once; seek by PTS with a bounded RGB frame cache.

    PTS is used only for locating a frame in the MP4. The recording's separate
    host timestamps determine *when* that frame belongs on the robot timeline.
    """

    def __init__(self, path: Path, expected_frames: int, cache_size: int = 32):
        import av

        self.container = av.open(str(path))
        try:
            self.stream = self.container.streams.video[0]
            self.pts = [frame.pts for frame in self.container.decode(self.stream)]
            if len(self.pts) != expected_frames:
                raise ValueError(
                    f"{path.name}: {len(self.pts)} video frames != {expected_frames} timestamps"
                )
            if not self.pts or any(p is None for p in self.pts):
                raise ValueError(f"Video has missing presentation timestamps: {path}")
            if any(a >= b for a, b in zip(self.pts, self.pts[1:], strict=False)):
                raise ValueError(f"Video presentation timestamps must increase: {path}")
            self.ordinal = {pts: index for index, pts in enumerate(self.pts)}
            self.cache: OrderedDict[int, np.ndarray] = OrderedDict()
            self.cache_size = max(1, cache_size)
            self.next_index = len(self.pts)
            self.decoder = iter(())
        except BaseException:
            self.container.close()
            raise

    def frame(self, index: int) -> np.ndarray:
        if not 0 <= index < len(self.pts):
            raise IndexError(index)
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        if index < self.next_index or index > self.next_index + 48:
            self.container.seek(self.pts[index], stream=self.stream, backward=True)
            self.decoder = iter(self.container.decode(self.stream))
        for frame in self.decoder:
            ordinal = self.ordinal[frame.pts]
            self.next_index = ordinal + 1
            if ordinal < index:
                continue
            if ordinal != index:
                raise ValueError(f"Video seek skipped requested frame {index}")
            rgb = frame.to_ndarray(format="rgb24")
            self.cache[index] = rgb
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
            return rgb
        raise ValueError(f"Unable to decode video frame {index}")

    def close(self) -> None:
        self.container.close()
        self.cache.clear()


class CollectionReplayViewer:
    """One playback clock for original video and the available joint trajectories.

    This class owns only Viser and recorded file readers. It intentionally does
    not instantiate PolicyViewer, its command server, or a hardware runtime.
    """

    def __init__(
        self, data: JointReplay, robot: RobotAdapter, *, host: str = "127.0.0.1",
        port: int = 8087, camera: str | None = None, server=None,
    ):
        if robot.name != "yam":
            raise ValueError("Collection joint replay currently supports the YAM adapter")
        self.data = data
        self.robot = robot
        self.trajectories = data.trajectories
        self.methods = []
        if data.native:
            self.methods.append((
                "native", f"{data.reference_label} → {data.target_hz:g} Hz · reference",
            ))
        if data.resampled:
            self.methods.extend(
                (name, f"{rate:g} → {data.target_hz:g} Hz · linear")
                for name, rate in data.resample_rates_hz.items()
            )
        else:
            self.methods.extend((
                ("linear", f"{data.low_hz:g} → {data.target_hz:g} Hz · linear"),
                ("held", f"{data.low_hz:g} Hz · hold"),
            ))
        self.reference_method = "native" if data.native else "held"
        self._style_selection = None
        camera = camera or ("top" if "top" in data.cameras else next(iter(data.cameras)))
        if camera not in data.cameras:
            raise ValueError(f"Unknown recorded camera {camera!r}; choose {list(data.cameras)}")
        self._video = None
        self._video_name = None
        self._last_image = None
        self._last_render = None
        self._timestamps_ns = data.timestamps_ns
        self._lock = threading.RLock()
        self._frame = 0
        self._anchor_frame = 0
        self._anchor_time = time.monotonic()
        self._playing = False
        self._speed = 1.0
        self.server = server or viser.ViserServer(host=host, port=port, label="ManiMux Replay")
        try:
            self.server.gui.configure_theme(
                control_layout="fixed", control_width="large", dark_mode=False,
                show_logo=False, show_share_button=False, brand_color=(70, 103, 190),
            )
            self._build_scene()
            self._build_gui(camera)
            self.tick()
        except BaseException:
            self.close()
            raise

    def _build_scene(self):
        scene = self.server.scene
        scene.set_up_direction("+z")
        self.server.initial_camera.position = (1.8, -2.0, 1.45)
        self.server.initial_camera.look_at = (0.3, 0.0, 0.3)
        self.server.initial_camera.up = (0.0, 0.0, 1.0)
        self.server.initial_camera.fov = np.deg2rad(50.0)
        scene.add_grid(
            "/floor", width=3.0, height=4.2, position=(0.3, 0.0, -0.01),
        )
        self.roots = {}
        self.robot_handles = {}
        self.method_roots = {}
        self.environment_roots = {}
        self.labels = {}
        colors = {"native": (0.12, 0.38, 0.78), "linear": (1.0, 0.42, 0.05),
                  "held": (0.1, 0.6, 0.4)}
        palette = ((1.0, 0.42, 0.05), (0.1, 0.6, 0.4), (0.65, 0.3, 0.85))
        colors.update({
            name: palette[i % len(palette)] for i, name in enumerate(self.data.resampled)
        })
        for method, label in self.methods:
            root = f"/replay/{method}"
            self.method_roots[method] = scene.add_frame(root, show_axes=False)
            self.labels[method] = scene.add_label(
                f"{root}/label", label, position=(0.0, 0.0, 0.85), visible=False,
            )
            self.environment_roots[method] = scene.add_frame(
                f"{root}/environment", show_axes=False,
            )
            for box in self.robot.scene_boxes:
                scene.add_box(
                    f"{root}/environment/{box.name}", dimensions=box.dimensions,
                    position=box.position, color=box.color,
                )
            for group in self.robot.groups:
                if group.name not in self.trajectories[self.reference_method]:
                    continue
                if group.urdf_path is None:
                    raise ValueError(f"Replay requires URDF geometry for {group.name}")
                name = f"{root}/{group.name}"
                key = (method, group.name)
                self.roots[key] = scene.add_frame(
                    name, show_axes=False, position=group.base_position,
                    wxyz=group.base_orientation,
                )
                self.robot_handles[key] = ViserUrdf(
                    self.server, group.urdf_path, root_node_name=name,
                    mesh_color_override=(*colors[method], 1.0),
                )

    def _build_gui(self, camera):
        gui = self.server.gui
        gui.add_markdown("## 轨迹叠加对比\n实体为基准，半透明线框虚影为对照；原始视频共用时间轴。")
        source_note = (
            "从臂实测 joint：基准每 10 ms 取此前最近的原始反馈；"
            "另两组先取同一份 30 Hz 数据，再分别线性插值到 100 Hz 或直接保持。"
            if self.data.source == "feedback" else
            "Command：沿用已保存的实际发送时间（标称 30 Hz），分别保持或线性插值。"
            "这份数据没有原生 100 Hz command。"
        )
        self.source_note = gui.add_markdown(self.data.source_note or source_note)
        achieved_comparison = self.data.source == "feedback" and bool(self.data.resampled)
        self.layout = gui.add_dropdown(
            "对比布局", options=("叠加", "三组叠加", "并排"),
            initial_value="三组叠加" if achieved_comparison else "叠加",
        )
        if self.data.resampled:
            candidates = {
                f"{rate:g} → {self.data.target_hz:g} Hz 插值": name
                for name, rate in self.data.resample_rates_hz.items()
            }
        else:
            candidates = {f"{self.data.low_hz:g} → {self.data.target_hz:g} Hz 插值": "linear"}
            if self.data.native:
                candidates[f"{self.data.low_hz:g} Hz 保持"] = "held"
        self.candidates = candidates
        self.ghost = gui.add_dropdown(
            "对照虚影", options=tuple(candidates), initial_value=next(iter(candidates)),
        )
        self.opacity = gui.add_slider(
            "虚影不透明度", min=0.05, max=0.8, step=0.05, initial_value=0.55,
        )
        self.show_reference = gui.add_checkbox("显示原始实体", initial_value=True)
        self.show_candidate = gui.add_checkbox("显示对照虚影", initial_value=True)
        self.comparison = gui.add_markdown("")
        self.play = gui.add_button("播放")
        self.frame_slider = gui.add_slider(
            f"{self.data.target_hz:g} Hz 帧索引", min=0, max=len(self.data.time_s) - 1,
            step=1, initial_value=0,
        )
        period_ms = f"{1000 / self.data.target_hz:.4g}"
        self.previous = gui.add_button(f"上一帧 · −{period_ms} ms")
        self.next = gui.add_button(f"下一帧 · +{period_ms} ms")
        self.first = gui.add_button("回到起点")
        self.last = gui.add_button("跳到终点")
        self.speed = gui.add_dropdown(
            "播放速度", options=("0.1×", "0.25×", "0.5×", "1×", "2×"), initial_value="1×",
        )
        self.loop = gui.add_checkbox("循环", initial_value=False)
        self.status = gui.add_markdown("")
        self.joint_values = None
        if achieved_comparison:
            with gui.add_folder("逐关节对比 · 当前帧（度）", expand_by_default=False):
                self.joint_values = gui.add_markdown("")
        with gui.add_folder("原始相机视频"):
            self.camera = gui.add_dropdown(
                "已录制视角", options=tuple(self.data.cameras), initial_value=camera,
            )
            self.image = gui.add_image(
                np.zeros((1, 1, 3), dtype=np.uint8), label="原始视频", visible=False,
                format="jpeg", jpeg_quality=90,
            )
            self.image_status = gui.add_markdown("")
        with gui.add_folder("数据说明", expand_by_default=False):
            rates = tuple(self.data.source_rates_hz.values())
            gui.add_markdown(
                f"Episode：`{self.data.episode.name}`\n\n"
                f"有效区间：{self.data.time_s[0]:.3f}–{self.data.time_s[-1]:.3f} s；"
                f"源数据平均频率：{min(rates):.2f}–{max(rates):.2f} Hz。\n\n"
                "各组夹爪均保持相同的已保存值，不对夹爪插值，只比较六个 arm joint。"
                "图像按主机采集时间戳匹配，非曝光硬同步；视频显示原帧索引和帧龄。"
                "场景底座为 Viewer 展示布局。\n\n"
                f"{self.data.target_hz:g} Hz 是轨迹采样网格，浏览器显示帧率取决于设备；"
                "用慢放或逐帧检查细节。"
                "这是轨迹回放，不模拟控制器或物体动力学。"
                + (
                    "\n\n首尾取所有比较频率共有的采样点，"
                    "因此实体和虚影的端点使用同一源数据值。只裁剪回放区间，"
                    "不添加端点约束，不修改角度或空间变换。"
                    if self.data.native else ""
                )
            )

        @self.play.on_click
        def _play(_):
            with self._lock:
                if not self._playing and self._frame == len(self.data.time_s) - 1:
                    self._frame = 0
                self._playing = not self._playing
                self._anchor()

        @self.frame_slider.on_update
        def _seek(event):
            if event.client is not None:
                self.seek(int(event.target.value))

        @self.previous.on_click
        def _previous(_):
            self.seek(self._frame - 1)

        @self.next.on_click
        def _next(_):
            self.seek(self._frame + 1)

        @self.first.on_click
        def _first(_):
            self.seek(0)

        @self.last.on_click
        def _last(_):
            self.seek(len(self.data.time_s) - 1)

        @self.speed.on_update
        def _speed(_):
            with self._lock:
                self._speed = float(self.speed.value.removesuffix("×"))
                self._anchor()

    def _anchor(self):
        self._anchor_frame = self._frame
        self._anchor_time = time.monotonic()
        self.play.label = "暂停" if self._playing else "播放"

    def seek(self, frame: int):
        with self._lock:
            self._frame = max(0, min(int(frame), len(self.data.time_s) - 1))
            self._playing = False
            self._anchor()

    def tick(self, now: float | None = None):
        with self._lock:
            now = time.monotonic() if now is None else now
            if self._playing:
                elapsed_frames = int((now - self._anchor_time) * self.data.target_hz * self._speed)
                frame = self._anchor_frame + max(0, elapsed_frames)
                if self.loop.value:
                    self._frame = frame % len(self.data.time_s)
                else:
                    self._frame = min(frame, len(self.data.time_s) - 1)
                    if frame >= len(self.data.time_s) - 1:
                        self._playing = False
                        self.play.label = "播放"
            selection = (
                self._frame, self.camera.value, self.layout.value,
                self.candidates[self.ghost.value], self.opacity.value,
                self.show_reference.value, self.show_candidate.value,
            )
            if selection == self._last_render:
                return
            self._render(*selection)
            self._last_render = selection

    def _apply_style(self, layout: str, ghost: str, opacity: float):
        selection = (layout, ghost, opacity)
        if selection == self._style_selection:
            return
        overlay = layout != "并排"
        previous_layout = self._style_selection[0] if self._style_selection else None
        if layout != previous_layout:
            position = (1.8, -2.0, 1.45) if overlay else (3.9, -4.6, 2.9)
            self.server.initial_camera.position = position
            # Update connected clients only for an explicit layout change; do
            # not reset an operator's camera while playing or changing opacity.
            if previous_layout is not None:
                for client in self.server.get_clients().values():
                    client.camera.position = position
                    client.camera.look_at = (0.3, 0.0, 0.3)
        for index, (method, _) in enumerate(self.methods):
            offset = 0.0 if overlay else ((len(self.methods) - 1) / 2 - index) * 1.4
            self.method_roots[method].position = (0.0, offset, 0.0)
            self.method_roots[method].visible = (
                not overlay or layout == "三组叠加" or method in {self.reference_method, ghost}
            )
            self.labels[method].visible = not overlay
            # Only one table in overlay mode; coincident copies cause flicker.
            self.environment_roots[method].visible = not overlay or method == self.reference_method
        for (method, _), handle in self.robot_handles.items():
            # ViserUrdf owns the MeshHandles returned by add_mesh_simple. The
            # RGBA override above ensures these expose a mutable opacity.
            for mesh in handle._meshes:
                is_ghost = overlay and method != self.reference_method and (
                    method == ghost or layout == "三组叠加"
                )
                # In Viser, None selects the opaque render pass; 1.0 still
                # enables transparent rendering and its depth-sorting rules.
                mesh.opacity = opacity if is_ghost else None
                # A back-face shell disappears inside the solid at matching
                # poses. A front-face wireframe keeps both colors visible and
                # avoids two filled, coplanar surfaces fighting for depth.
                # Geometry, base transforms, and joint values stay unchanged.
                mesh.side = "front"
                mesh.wireframe = is_ghost
                mesh.cast_shadow = not is_ghost
                mesh.receive_shadow = not is_ghost
        self._style_selection = selection

    def _render(
        self, index: int, camera: str, layout: str, ghost: str, opacity: float,
        show_reference: bool, show_candidate: bool,
    ):
        recorded = self.data.cameras[camera]
        if self._video_name != camera:
            # Open the replacement before closing the current decoder.
            replacement = ReplayVideo(recorded.path, len(recorded.timestamps_ns))
            if self._video is not None:
                self._video.close()
            self._video = replacement
            self._video_name = camera
        timestamp = int(self._timestamps_ns[index])
        image_index = recorded.frame_at(timestamp)
        gaps = []
        with self.server.atomic():
            self._apply_style(layout, ghost, opacity)
            for (method, group), handle in self.robot_handles.items():
                q = self.trajectories[method][group][index]
                valid = bool(np.all(np.isfinite(q)))
                layer_visible = (
                    show_reference if method == self.reference_method else show_candidate
                )
                self.roots[(method, group)].visible = valid and layer_visible
                if valid:
                    handle.update_cfg(self.robot.visual_configuration(group, q))
                else:
                    gaps.append(f"{method}/{group}")
            self.frame_slider.value = index
            self.status.content = (
                f"**帧 {index} / {len(self.data.time_s) - 1}** · "
                f"episode **{self.data.time_s[index]:.3f} s**"
                + (f"\n\n缺失反馈，暂时隐藏：{', '.join(gaps)}" if gaps else "")
            )
            reference_name = (
                f"蓝色实体：{self.data.reference_label}→{self.data.target_hz:g} Hz"
                if self.data.native else "绿色实体：command 保持"
            )
            ghost_label = next(
                label for label, method in self.candidates.items() if method == ghost
            )
            ghost_name = f"对照线框虚影：{ghost_label}"
            differences = []
            for group in self.trajectories[self.reference_method]:
                reference = self.trajectories[self.reference_method][group][index, :6]
                candidate = self.trajectories[ghost][group][index, :6]
                error = np.rad2deg(candidate - reference)
                value = f"{np.max(np.abs(error)):.4f}°" if np.all(np.isfinite(error)) else "缺失"
                differences.append(f"{group} 最大关节差：{value}")
            self.comparison.content = (
                f"{reference_name} · {ghost_name}\n\n" + " · ".join(differences)
            )
            if layout == "三组叠加" and self.data.resampled:
                labels = list(self.candidates)
                self.comparison.content = (
                    f"{reference_name}；橙色：{labels[0]}；绿色：{labels[1]}"
                    if len(labels) == 2 else f"{reference_name}；所有重建同时显示"
                )
            if self.joint_values is not None:
                header = " | ".join(f"{r:g}→{self.data.target_hz:g}" for r in
                                    self.data.resample_rates_hz.values())
                columns = 2 + len(self.data.resampled)
                rows = [f"| 关节 | 原始 | {header} |", "|" + "---|" * columns]
                for arm, reference in self.data.native.items():
                    for joint in range(6):
                        values = [reference[index, joint], *(
                            group[arm][index, joint] for group in self.data.resampled.values()
                        )]
                        cells = [f"{np.rad2deg(v):.3f}" if np.isfinite(v) else "缺失"
                                 for v in values]
                        rows.append(f"| {arm} J{joint + 1} | " + " | ".join(cells) + " |")
                self.joint_values.content = "\n".join(rows)
            self.image.visible = image_index >= 0
            if image_index >= 0:
                if (camera, image_index) != self._last_image:
                    self.image.image = self._video.frame(image_index)
                    self._last_image = (camera, image_index)
                age_ms = (timestamp - int(recorded.timestamps_ns[image_index])) / 1e6
                self.image_status.content = (
                    f"{camera} · 原视频帧 **{image_index} / {len(recorded.timestamps_ns) - 1}**"
                    f" · 帧龄 {age_ms:.1f} ms"
                    + (" · **图像已过期，仍显示最后一帧**" if age_ms > 100 else "")
                )
            else:
                self.image_status.content = f"{camera} · 此时尚无已录制图像"

    def close(self):
        with self._lock:
            if self._video is not None:
                self._video.close()
                self._video = None
            self.server.stop()


def serve_collection_replay(
    episode: Path, robot: RobotAdapter, *, source: str, camera: str | None,
    host: str, port: int, target_hz: float | None = None,
    low_rates_hz: tuple[float, ...] | None = None,
) -> None:
    if target_hz is not None or low_rates_hz is not None:
        if source not in {"command", "feedback"} or target_hz is None or low_rates_hz is None:
            raise ValueError("Custom rates require a replay source and both target and lower rates")
        from manimux.collection.yam.data.command_resampling import load_command_rate_replay
        from manimux.collection.yam.data.feedback_resampling import load_feedback_rate_replay

        loader = load_feedback_rate_replay if source == "feedback" else load_command_rate_replay
        data = loader(episode, target_hz=target_hz, low_rates_hz=low_rates_hz)
    else:
        data = load_joint_replay(episode, source)
    viewer = CollectionReplayViewer(data, robot, host=host, port=port, camera=camera)
    stop = threading.Event()
    previous_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.signal(sig, lambda *_: stop.set())
    address = "127.0.0.1" if host == "0.0.0.0" else host
    print(f"Read-only {source} replay: {data.episode}")
    print(f"Open http://{address}:{port} — paused; press Play in Viser")
    try:
        while not stop.wait(0.005):
            viewer.tick()
    finally:
        viewer.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
