"""Standalone camera-only tool for collecting ten Top references per task.

Run: python -m manimux.viewer.reference_capture --task put_bottles_into_the_bin
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

from manimux.sensors.camera_server.client import CameraSubscriber

from .reference_layouts import DEFAULT_LAYOUT_ROOT, REFERENCE_SLOTS, ReferenceLayouts

EMPTY_TASK = "（请创建 Task）"


class ReferenceCapture:
    def __init__(self, gui: Any, layouts: ReferenceLayouts, camera: str) -> None:
        self.layouts = layouts
        self.camera = camera
        self._lock = threading.RLock()
        self._live: np.ndarray | None = None
        self._timestamp = 0.0
        tasks = layouts.tasks()
        gui.add_markdown("## Top 参考图采集\n只读取相机；每个 Task 保存 01–10 共十个布局。")
        self.task = gui.add_dropdown("Task", tasks or (EMPTY_TASK,))
        self.new_task = gui.add_text("新 Task 名称", "")
        self.create = gui.add_button("创建 Task")
        self.slot = gui.add_dropdown("参考图编号", REFERENCE_SLOTS)
        self.preview = gui.add_image(np.zeros((480, 640, 3), np.uint8), label="Top 实时画面")
        self.save = gui.add_button("保存到选中编号（已有图片会替换）", disabled=True)
        self.status = gui.add_markdown("等待相机服务。")
        self.inventory = gui.add_markdown("")
        self.saved_preview = gui.add_image(
            np.zeros((480, 640, 3), np.uint8), label="选中编号的已保存参考"
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
                    self.status.content = "没有新鲜的相机画面，未保存。"
                    return
                if self.task.value == EMPTY_TASK:
                    self.status.content = "请先创建 Task。"
                    return
                try:
                    assert self._live is not None
                    layouts.save(self.task.value, self.slot.value, self._live)
                    self.status.content = f"已保存 {self.task.value}/{self.slot.value}.png"
                    self._selection()
                except (OSError, ValueError) as exc:
                    self.status.content = f"保存失败：{exc}"

        self._selection()

    def _fresh(self) -> bool:
        return self._live is not None and 0 <= time.time() - self._timestamp < 0.5

    def _selection(self) -> None:
        slots = self.layouts.slots(self.task.value) if self.task.value != EMPTY_TASK else ()
        self.inventory.content = f"已采集 **{len(slots)}/10**：{', '.join(slots) or '无'}"
        self.saved_preview.visible = self.slot.value in slots
        if self.saved_preview.visible:
            try:
                self.saved_preview.image = self.layouts.load(self.task.value, self.slot.value)
            except (OSError, ValueError) as exc:
                self.saved_preview.visible = False
                self.status.content = f"参考图读取失败：{exc}"
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
                self.status.content = f"等待 {self.camera} 新鲜画面；请检查相机服务。"
            elif self.status.content.startswith("等待"):
                self.status.content = "摆好场景，选择 01–10 中的编号后保存。"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", help="create/select this task on opening")
    parser.add_argument("--root", type=Path, default=DEFAULT_LAYOUT_ROOT)
    parser.add_argument("--camera", default="front_camera")
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
