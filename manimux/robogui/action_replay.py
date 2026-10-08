"""Offline playback of absolute, named robot configurations without runtime sockets."""

from __future__ import annotations

import signal
import threading
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import viser
from viser.extras import ViserUrdf

from .robot_view import RobotView


def load_actions(path: Path, robot: RobotView) -> dict[str, np.ndarray]:
    """Read an NPZ with one (steps, coordinates) array per robot group."""
    with np.load(path, allow_pickle=False) as archive:
        return robot.validate_groups(dict(archive), sequence=True)


class ActionReplayRoboGUI:
    """Display joint targets only; never construct a robot driver or policy client."""

    def __init__(
        self,
        actions: Mapping[str, np.ndarray],
        robot: RobotView,
        *,
        action_dt_s: float,
        host: str = "127.0.0.1",
        port: int = 8087,
        server=None,
        timestamps_ns: np.ndarray | None = None,
        source_label: str = "joint targets",
    ):
        if not np.isfinite(action_dt_s) or action_dt_s <= 0:
            raise ValueError("action_dt_s must be finite and positive")
        self.actions = {
            name: values.copy()
            for name, values in robot.validate_groups(actions, sequence=True).items()
        }
        self.robot = robot
        self.action_dt_s = float(action_dt_s)
        self.count = len(next(iter(self.actions.values())))
        self.source_label = source_label
        self.times = np.arange(self.count, dtype=np.float64) * self.action_dt_s
        if timestamps_ns is not None:
            stamps = np.asarray(timestamps_ns, dtype=np.int64)
            if stamps.shape != (self.count,) or np.any(np.diff(stamps) <= 0):
                raise ValueError("Replay timestamps must match samples and strictly increase")
            self.times = (stamps - stamps[0]).astype(np.float64) / 1e9
        self._lock = threading.RLock()
        self._position = 0.0
        self._last_time = time.monotonic()
        self._rendered = None
        self.server = server or viser.ViserServer(
            host=host, port=port, label="ManiMux Action Replay"
        )
        try:
            self._build_scene()
            self.server.gui.add_markdown(
                f"## Offline replay\n{source_label}; no hardware connection or execution."
            )
            self.play = self.server.gui.add_checkbox("Play", False, disabled=self.count == 1)
            self.speed = self.server.gui.add_dropdown(
                "Speed",
                options=("0.25", "0.5", "1", "2", "4"),
                initial_value="1",
            )
            self.frame = self.server.gui.add_slider(
                "Frame",
                min=0,
                max=max(1, self.count - 1),
                step=1,
                initial_value=0,
                disabled=self.count == 1,
            )
            self.status = self.server.gui.add_markdown("")

            @self.play.on_update
            def _play(_event):
                with self._lock:
                    if self.play.value and self._position >= self.times[-1]:
                        self._position = 0.0
                    self._last_time = time.monotonic()

            @self.speed.on_update
            def _speed(_event):
                with self._lock:
                    self._last_time = time.monotonic()

            @self.frame.on_update
            def _seek(event):
                # Server-driven slider updates must not reset the playback clock.
                if event.client is not None:
                    self.seek(int(self.frame.value))

            self.tick()
        except BaseException:
            self.server.stop()
            raise

    def _build_scene(self):
        scene = self.server.scene
        view = self.robot.scene_view
        scene.set_up_direction("+z")
        self.server.initial_camera.position = view.camera_position
        self.server.initial_camera.look_at = view.camera_look_at
        self.server.initial_camera.up = (0.0, 0.0, 1.0)
        scene.add_grid(
            "/floor",
            width=view.grid_size[0],
            height=view.grid_size[1],
            position=view.grid_position,
        )
        for box in self.robot.scene_boxes:
            scene.add_box(
                f"/environment/{box.name}",
                dimensions=box.dimensions,
                position=box.position,
                color=box.color,
            )
        for mesh in self.robot.static_meshes:
            root = f"/environment/{mesh.name}"
            scene.add_frame(root, show_axes=False, position=mesh.position, wxyz=mesh.orientation)
            ViserUrdf(self.server, mesh.urdf_path, root_node_name=root)
        self.robot_handles = {}
        for group in self.robot.groups:
            if group.urdf_path is None:
                raise ValueError(f"Action replay requires URDF geometry for {group.name}")
            root = f"/actions/{group.name}"
            scene.add_frame(
                root,
                show_axes=False,
                position=group.base_position,
                wxyz=group.base_orientation,
            )
            handle = ViserUrdf(self.server, group.urdf_path, root_node_name=root)
            handle.update_cfg(self.robot.initial_configuration(group.name))
            self.robot_handles[group.name] = handle

    def seek(self, index: int) -> None:
        with self._lock:
            self._position = float(self.times[int(np.clip(index, 0, self.count - 1))])
            self._last_time = time.monotonic()
            self.tick(now=self._last_time)

    def tick(self, *, now: float | None = None) -> None:
        with self._lock:
            now = time.monotonic() if now is None else now
            if self.play.value:
                self._position = min(
                    self.times[-1],
                    self._position + max(0.0, now - self._last_time) * float(self.speed.value),
                )
                if self._position >= self.times[-1]:
                    self.play.value = False
            self._last_time = now
            index = int(np.searchsorted(self.times, self._position, side="right") - 1)
            if index == self._rendered:
                return
            with self.server.atomic():
                for name, actions in self.actions.items():
                    self.robot_handles[name].update_cfg(
                        self.robot.visual_configuration(name, actions[index])
                    )
                self.frame.value = index
                self.status.content = (
                    f"Frame **{index + 1} / {self.count}** · "
                    f"time **{self.times[index]:.3f} s** · {self.source_label}"
                )
            self._rendered = index

    def close(self) -> None:
        self.server.stop()


def serve_action_replay(path: Path, robot: RobotView, *, action_dt_s: float, host: str, port: int):
    robogui = ActionReplayRoboGUI(
        load_actions(path, robot),
        robot,
        action_dt_s=action_dt_s,
        host=host,
        port=port,
    )
    stop = threading.Event()
    handlers = {}
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, lambda *_: stop.set())
        address = "127.0.0.1" if host == "0.0.0.0" else host
        print(f"Offline action replay: http://{address}:{robogui.server.get_port()} (paused)")
        while not stop.wait(0.01):
            robogui.tick()
    finally:
        robogui.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
