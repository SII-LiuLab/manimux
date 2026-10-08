"""Read finalized ManiMux records and launch isolated, hardware-free playback."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import zarr

from .action_replay import ActionReplayRoboGUI

SOURCES = {
    "Measured state": "state",
    "Scheduled reference": "scheduled",
    "Executor command": "command",
}


def load_episode_trajectory(episode: Path, source: str, robot):
    """Read tick samples with their original host times; never resample feedback."""
    if episode.name.endswith(".partial") or not (episode / "result.json").is_file():
        raise ValueError("Choose a finalized episode with result.json")
    if source not in SOURCES.values():
        raise ValueError("Choose state, scheduled or command")
    ticks = zarr.open_group(str(episode / "data.zarr"), mode="r")["ticks"]
    actions = robot.validate_groups(
        {name: np.asarray(values) for name, values in ticks[source].arrays()}, sequence=True
    )
    stamps = np.asarray(ticks["monotonic_ns"], dtype=np.int64)
    count = len(next(iter(actions.values())))
    if stamps.shape != (count,) or np.any(np.diff(stamps) <= 0):
        raise ValueError("Recorded timestamps must match samples and strictly increase")
    return actions, stamps


class RecordsPanel:
    def __init__(self, gui, robot, *, host: str, port: int):
        self.robot, self.host, self.port = robot, host, port
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._robogui = None
        with gui.add_folder("Recorded rollouts", expand_by_default=False):
            self.root = gui.add_text("Session directory", "")
            self.refresh = gui.add_button("Refresh episodes")
            self.episode = gui.add_dropdown("Episode", ("—",))
            self.source = gui.add_dropdown("Replay source", tuple(SOURCES))
            self.open = gui.add_button("Open offline replay")
            self.summary = gui.add_markdown("Select a session directory, then refresh.")
            self.link = gui.add_markdown("")

        @self.refresh.on_click
        def _refresh(_event):
            with self._lock:
                try:
                    folder = Path(self.root.value).expanduser()
                    episodes = tuple(
                        sorted(
                            path.name
                            for path in folder.iterdir()
                            if path.is_dir()
                            and not path.name.endswith(".partial")
                            and (path / "result.json").is_file()
                        )
                    )
                    selected = self.episode.value
                    self.episode.options = episodes or ("—",)
                    self.episode.value = (
                        selected if selected in episodes else self.episode.options[-1]
                    )
                    self._describe()
                except (OSError, ValueError) as exc:
                    self.summary.content = f"Could not read session: {exc}"

        @self.episode.on_update
        def _select(_event):
            with self._lock:
                self._describe()

        @self.open.on_click
        def _open(_event):
            with self._lock:
                try:
                    episode = Path(self.root.value).expanduser() / self.episode.value
                    actions, stamps = load_episode_trajectory(
                        episode, SOURCES[self.source.value], self.robot
                    )
                    self._close_replay()
                    self._robogui = ActionReplayRoboGUI(
                        actions,
                        self.robot,
                        action_dt_s=1.0,
                        timestamps_ns=stamps,
                        source_label=self.source.value,
                        host=self.host,
                        port=self.port,
                    )
                    self._stop.clear()
                    self._thread = threading.Thread(target=self._play, daemon=True)
                    self._thread.start()
                    host = "127.0.0.1" if self.host == "0.0.0.0" else self.host
                    if ":" in host:
                        host = f"[{host}]"
                    self.link.content = (
                        f"[Open paused replay](http://{host}:{self._robogui.server.get_port()})"
                        " · separate view; live controls are unchanged."
                    )
                except (OSError, ValueError, KeyError) as exc:
                    self.summary.content = f"Could not open replay: {exc}"

    def _describe(self):
        if self.episode.value == "—":
            self.summary.content = "No finalized episodes in this session."
            return
        episode = Path(self.root.value).expanduser() / self.episode.value
        try:
            meta = json.loads((episode / "meta.json").read_text())
            result = json.loads((episode / "result.json").read_text())
            human = episode / "evaluation" / "human-label.json"
            details = {
                key: meta.get(key)
                for key in ("experiment_name", "condition", "notes", "layout_id", "repeat_id")
                if meta.get(key)
            }
            details.update(
                terminal_reason=result.get("terminal_reason"),
                steps=result.get("steps"),
                human_label="saved" if human.is_file() else "not saved",
            )
            self.summary.content = (
                "```json\n" + json.dumps(details, indent=2, ensure_ascii=False) + "\n```"
            )
        except (OSError, ValueError) as exc:
            self.summary.content = f"Could not read episode: {exc}"

    def _play(self):
        while not self._stop.wait(0.01):
            self._robogui.tick()

    def _close_replay(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._robogui is not None:
            self._robogui.close()
            self._robogui = None

    def close(self):
        with self._lock:
            self._close_replay()
