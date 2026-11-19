"""Hierarchical recording selection on one hardware-free replay server."""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path

import viser

from .replay import create_recorded_replay


def discover_episodes(root: Path, *, allow_partial: bool = False) -> list[Path]:
    """Index closed recordings without walking Zarr chunks or decoding video."""
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Recording root is not a directory: {root}")
    episodes = []
    for directory, children, files in os.walk(root, followlinks=False):
        path = Path(directory)
        if "data.zarr" in children:
            children[:] = []
            if "result.json" in files and (allow_partial or not path.name.endswith(".partial")):
                episodes.append(path)
            continue
        children[:] = [
            name
            for name in children
            if not name.startswith(".") and name != "videos" and not name.endswith(".zarr")
        ]
    return sorted(episodes)


def episode_description(episode: Path, root: Path) -> str:
    """Display saved configuration, not a frequency inferred from folder names."""
    description = f"**{episode.name}**"
    description += (
        " · **Interrupted / partial**" if episode.name.endswith(".partial") else " · Finalized"
    )
    try:
        config = json.loads((episode.parent / "session-manifest.json").read_text())["config"]
        inference = config.get("inference", {})
        mode = inference.get("algorithm")
        if inference.get("strategy") == "manimux.runtime.rtc.retimed:RetimedRtcStrategy":
            mode = "retimed RTC"
        if mode:
            description += f" · algorithm: `{mode}`"
        dt = config.get("policy", {}).get("action_dt_s")
        if isinstance(dt, (float, int)) and dt > 0:
            description += f" · action playback: **{1 / dt:g} Hz**"
    except (OSError, ValueError, KeyError, TypeError):
        description += " · saved configuration unavailable"
    return description


class RecordedReplayBrowser:
    """Stage a replacement before retiring the current player and its readers."""

    def __init__(
        self,
        root,
        episode,
        robot,
        *,
        source="state",
        plugin=None,
        allow_partial=False,
        host="127.0.0.1",
        port=8087,
    ):
        self.root = Path(root).expanduser().resolve()
        episode = Path(episode).expanduser().resolve()
        if not episode.is_relative_to(self.root):
            raise ValueError("Initial episode must be inside --replay-root")
        self.episodes = discover_episodes(self.root, allow_partial=allow_partial)
        if episode not in self.episodes:
            raise ValueError("Initial episode is not a closed recording in the selected root")
        self.robot, self.source, self.plugin = robot, source, plugin
        self._lock = threading.RLock()
        self._closed = False
        self.player = None
        self._folder = self._scene = None
        self.episode = episode
        self.server = viser.ViserServer(host=host, port=port, label="RoboGUI Recorded Replay")
        try:
            with self.server.gui.add_folder("Choose rollout", order=-100):
                self.server.gui.add_markdown(
                    f"Browse `{self.root.name}` · choose folders, then load a rollout."
                )
                self.partial = self.server.gui.add_checkbox(
                    "Include interrupted rollouts", allow_partial
                )
                self.refresh = self.server.gui.add_button("Refresh recordings")
                self.folder_controls = []
                self.folder_container = self.server.gui.add_folder("Folders")
                self.rollout = self.server.gui.add_dropdown("Rollout", (episode.name,))
                self.selection = self.server.gui.add_markdown("")
                self.load = self.server.gui.add_button("Load selected rollout")
                self.status = self.server.gui.add_markdown("")
            self._rebuild_folders(episode)

            @self.partial.on_update
            def _partial(event):
                if event.client is not None:
                    self.rescan()

            @self.refresh.on_click
            def _refresh(_):
                self.rescan()

            @self.rollout.on_update
            def _selection(event):
                if event.client is not None:
                    with self._lock:
                        self._describe_selection()

            @self.load.on_click
            def _load(_):
                with self._lock:
                    selected = self.selected_episode()
                    if selected is not None and not self._closed:
                        self._set_disabled(True)
                        self.status.content = f"Loading `{selected.relative_to(self.root)}`…"
                        try:
                            self.switch_episode(selected)
                        except Exception as exc:
                            self.status.content = (
                                f"**Could not load selection:** {exc}\n\n"
                                f"Still loaded: `{self.episode.relative_to(self.root)}`"
                            )
                        finally:
                            self._set_disabled(False)

            self.switch_episode(episode)
        except BaseException:
            self.close()
            raise

    def _set_disabled(self, value):
        for handle in (self.partial, self.refresh, self.rollout, self.load, *self.folder_controls):
            handle.disabled = value
        if not value:
            self.load.disabled = self.selected_episode() is None

    def _rebuild_folders(self, preferred):
        for handle in self.folder_controls:
            handle.remove()
        self.folder_controls = []
        self.depth = max(
            (len(p.relative_to(self.root).parts) - 1 for p in self.episodes), default=0
        )
        with self.folder_container:
            for i in range(self.depth):
                labels = (
                    ("Experiment", "Mode / folder", "Session")
                    if self.root.name == "experiments"
                    else ()
                )
                label = labels[i] if i < len(labels) else f"Folder {i + 1}"
                handle = self.server.gui.add_dropdown(label, ("(this folder)",))
                self.folder_controls.append(handle)

                @handle.on_update
                def _folder_changed(event, level=i):
                    if event.client is not None:
                        with self._lock:
                            self._cascade(level + 1)

        self._cascade(0, preferred)

    def _parts(self, episode):
        parts = episode.relative_to(self.root).parent.parts
        return parts + ("(this folder)",) * (self.depth - len(parts))

    def _cascade(self, start, preferred=None):
        candidates = self.episodes[:]
        desired = self._parts(preferred) if preferred in self.episodes else ()
        for i, handle in enumerate(self.folder_controls):
            options = sorted({self._parts(p)[i] for p in candidates})
            if i >= start:
                previous = desired[i] if desired else handle.value
                handle.options = tuple(options) or ("(no recordings)",)
                handle.value = previous if previous in options else handle.options[0]
            handle.visible = bool(options) and options != ["(this folder)"]
            candidates = [p for p in candidates if self._parts(p)[i] == handle.value]
        self._rollouts = {p.name: p for p in reversed(candidates)}
        previous = preferred.name if preferred in candidates else self.rollout.value
        self.rollout.options = tuple(self._rollouts) or ("(no recordings)",)
        self.rollout.value = previous if previous in self._rollouts else self.rollout.options[0]
        self.load.disabled = not bool(self._rollouts)
        self._describe_selection()

    def selected_episode(self):
        return self._rollouts.get(self.rollout.value)

    def _describe_selection(self):
        episode = self.selected_episode()
        self.selection.content = (
            episode_description(episode, self.root)
            if episode
            else "No closed recordings in this selection. Enable interrupted rollouts or refresh."
        )

    def rescan(self):
        with self._lock:
            if self._closed:
                return
            selected = self.selected_episode()
            self.episodes = discover_episodes(self.root, allow_partial=self.partial.value)
            self._rebuild_folders(selected)

    def switch_episode(self, episode):
        """Keep the old paused replay intact if loading or plugin initialization fails."""
        with self._lock:
            if self._closed:
                return
            episode = Path(episode).resolve()
            if episode not in self.episodes:
                raise ValueError("Choose a recording from the current catalog")
            if self.player is not None:
                with self.player._lock:
                    self.player.play.value = False
            folder = self.server.gui.add_folder("Playback", visible=False)
            root = f"/replay/{uuid.uuid4().hex}"
            scene = self.server.scene.add_frame(root, show_axes=False, visible=False)
            candidate = None
            try:
                with folder:
                    candidate = create_recorded_replay(
                        episode,
                        self.robot,
                        source=self.source,
                        plugin=self.plugin,
                        allow_partial=self.partial.value,
                        server=self.server,
                        scene_root=root,
                    )
            except BaseException:
                if candidate is not None:
                    candidate.close()
                folder.remove()
                scene.remove()
                raise
            old, old_folder, old_scene = self.player, self._folder, self._scene
            with self.server.atomic():
                if old is not None:
                    old.close()
                    old_folder.remove()
                    old_scene.remove()
                self.player, self._folder, self._scene = candidate, folder, scene
                self.episode = episode
                folder.visible = scene.visible = True
                self.status.content = (
                    f"**Loaded · paused at start**\n\n`{episode.relative_to(self.root)}`"
                )

    def tick(self):
        with self._lock:
            if self.player is not None and not self._closed:
                self.player.tick()

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                if self.player is not None:
                    self.player.close()
            finally:
                self.server.stop()
