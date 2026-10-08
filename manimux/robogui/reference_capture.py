"""Standalone camera-only tool for saving named Top references per task.

Run: python -m manimux.robogui.reference_capture --task put_bottles_into_the_bin
"""

from __future__ import annotations

import argparse
import signal
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import viser

from manimux.embodiments.sensor.camera_server.client import CameraSubscriber

from .reference_layouts import DEFAULT_LAYOUT_ROOT, ReferenceLayouts

EMPTY_TASK = "(Create a task)"


class ReferenceCapture:
    def __init__(self, gui: Any, layouts: ReferenceLayouts, camera: str) -> None:
        self.layouts = layouts
        self.camera = camera
        self._lock = threading.RLock()
        self._live: np.ndarray | None = None
        self._timestamp = 0.0
        tasks = layouts.tasks()
        gui.add_markdown(
            "## Top reference capture\n"
            "Camera only. Choose a task and a reference ID, then save."
        )
        self.task = gui.add_dropdown("Task", tasks or (EMPTY_TASK,))
        self.new_task = gui.add_text("New task name", "")
        self.create = gui.add_button("Create task")
        self.slot = gui.add_text("Reference ID", "01")
        self.preview = gui.add_image(np.zeros((480, 640, 3), np.uint8), label="Live Top view")
        self.save = gui.add_button("Save to slot (replace existing image)", disabled=True)
        self.status = gui.add_markdown("Waiting for camera service.")
        self.inventory = gui.add_markdown("")
        self.saved_preview = gui.add_image(
            np.zeros((480, 640, 3), np.uint8), label="Saved reference for selected slot"
        )

        @self.create.on_click
        def _create(_event: Any) -> None:
            with self._lock:
                task = self.new_task.value.strip()
                try:
                    layouts.create_task(task)
                    self.task.options = layouts.tasks()
                    self.task.value = task
                    self._selection()
                except (OSError, ValueError) as exc:
                    self.status.content = str(exc)

        @self.task.on_update
        @self.slot.on_update
        def _select(_event: Any) -> None:
            with self._lock:
                self._selection()

        @self.save.on_click
        def _save(_event: Any) -> None:
            with self._lock:
                if not self._fresh():
                    self.status.content = "No fresh camera frame. Image not saved."
                    return
                if self.task.value == EMPTY_TASK:
                    self.status.content = "Create a task first."
                    return
                try:
                    assert self._live is not None
                    layouts.save(self.task.value, self.slot.value, self._live)
                    self.status.content = f"Saved {self.task.value}/{self.slot.value}.png"
                    self._selection()
                except (OSError, ValueError) as exc:
                    self.status.content = f"Save failed: {exc}"

        self._selection()

    def _fresh(self) -> bool:
        return self._live is not None and 0 <= time.time() - self._timestamp < 0.5

    def _selection(self) -> None:
        slots = self.layouts.slots(self.task.value) if self.task.value != EMPTY_TASK else ()
        self.inventory.content = f"Captured **{len(slots)}**: {', '.join(slots) or 'None'}"
        self.saved_preview.visible = self.slot.value in slots
        if self.saved_preview.visible:
            try:
                self.saved_preview.image = self.layouts.load(self.task.value, self.slot.value)
            except (OSError, ValueError) as exc:
                self.saved_preview.visible = False
                self.status.content = f"Could not load reference: {exc}"
        self.save.disabled = not self._fresh() or self.task.value == EMPTY_TASK

    def update(self, bundle: dict[str, Any] | None) -> None:
        with self._lock:
            if bundle is not None:
                image = (bundle.get("frames") or {}).get(self.camera)
                timestamp = (bundle.get("timestamps") or {}).get(self.camera, 0)
                if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3:
                    self._live = image.copy()
                    self._timestamp = float(timestamp or 0)
                    self.preview.image = self._live
            fresh = self._fresh()
            self.save.disabled = not fresh or self.task.value == EMPTY_TASK
            if not fresh:
                self.status.content = (
                    f"Waiting for a fresh {self.camera} frame. Check camera service."
                )
            elif self.status.content.startswith("Waiting"):
                self.status.content = "Arrange the scene, enter a reference ID, then save."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", help="create/select this task on opening")
    parser.add_argument("--root", type=Path, default=DEFAULT_LAYOUT_ROOT)
    parser.add_argument("--camera", default="d405_front")
    parser.add_argument("--camera-endpoint", default="tcp://127.0.0.1:5556")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8087)
    args = parser.parse_args()
    layouts = ReferenceLayouts(args.root)
    if args.task:
        layouts.create_task(args.task)
    server = viser.ViserServer(host=args.host, port=args.port, label="Top reference capture")
    server.gui.configure_theme(control_layout="fixed", control_width="large", show_logo=False)
    capture = ReferenceCapture(server.gui, layouts, args.camera)
    if args.task:
        capture.task.value = args.task
        capture._selection()
    subscriber = CameraSubscriber(args.camera_endpoint)
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    print(f"Reference capture: http://{args.host}:{args.port}")
    print(f"Image library: {args.root.resolve()}")
    try:
        while not stop.wait(0.1):
            capture.update(subscriber.try_recv_bundle())
    finally:
        subscriber.close()
        server.stop()


if __name__ == "__main__":
    main()
