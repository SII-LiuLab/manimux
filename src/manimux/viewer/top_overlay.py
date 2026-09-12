"""Read-only task/reference selection and a lower-left Top ghost view."""

from __future__ import annotations

import html
import threading
from pathlib import Path
from typing import Any

import numpy as np

from .reference_layouts import DEFAULT_LAYOUT_ROOT, ReferenceLayouts

EMPTY_TASK = "（暂无 Task）"
EMPTY_SLOT = "（尚未采集）"

_PANEL_STYLE = """
<style>
  :root {
    --reference-top: calc(294px + var(--manimux-camera-height, 397px));
    --reference-image-height: clamp(100px,
      calc(100dvh - var(--reference-top) - 92px),
      calc((var(--manimux-camera-width, 460px) - 22px) * .75));
  }
  div:has(> .manimux-chunk-anchor) { anchor-name: --manimux-chunks; }
  div:has(> .manimux-reference-anchor) {
    position: fixed; left: 16px; top: var(--reference-top);
    width: var(--manimux-camera-width, 460px); anchor-name: --manimux-reference;
    z-index: 4; margin: 0; padding: 0 !important;
  }
  .manimux-reference-panel {
    padding: 10px; border: 1px solid rgba(128,138,156,.35); border-radius: 12px;
    background: rgba(18,23,32,.94); color: #eaf0fb;
    box-shadow: 0 8px 28px rgba(0,0,0,.16); font: 12px/1.35 system-ui,sans-serif;
  }
  .manimux-reference-panel header { height: 34px; margin-bottom: 6px; }
  .manimux-reference-panel header span {
    display: block; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    color: #a6b5c7;
  }
  .manimux-reference-image-space { height: var(--reference-image-height); }
  .manimux-reference-panel footer { margin-top: 7px; color: #a6b5c7; }
  div:has(> .manimux-reference-anchor) + div {
    position: fixed; left: 27px; top: calc(var(--reference-top) + 51px);
    width: calc(var(--manimux-camera-width, 460px) - 22px);
    height: var(--reference-image-height); z-index: 5;
    margin: 0; padding: 0 !important; border-radius: 8px;
    background: #0e131c; overflow: hidden;
  }
  div:has(> .manimux-reference-anchor) + div > div { width: 100%; height: 100%; }
  div:has(> .manimux-reference-anchor) + div img {
    width: 100%; height: 100% !important; max-width: none !important;
    object-fit: contain; display: block;
  }
  @supports (top: anchor(--manimux-chunks bottom)) {
    div:has(> .manimux-reference-anchor) {
      top: calc(anchor(--manimux-chunks bottom) + 12px);
    }
    div:has(> .manimux-reference-anchor) + div {
      top: calc(anchor(--manimux-reference top) + 51px);
    }
  }
  @media (max-width: 900px), (max-height: 720px) {
    div:has(> .manimux-reference-anchor),
    div:has(> .manimux-reference-anchor) + div {
      position: static; width: auto; height: auto; margin: 8px 0;
    }
    .manimux-reference-image-space { display: none; }
    div:has(> .manimux-reference-anchor) + div img {
      height: auto !important; max-height: 280px;
    }
  }
</style>
"""


class TopViewOverlay:
    def __init__(self, gui: Any, root: Path = DEFAULT_LAYOUT_ROOT) -> None:
        self.layouts = ReferenceLayouts(root)
        self._lock = threading.RLock()
        self._live: np.ndarray | None = None
        self._reference: np.ndarray | None = None
        self._error = ""
        # Keep the native image as the anchor's direct sibling, like the camera panel.
        # Only its binary image data changes on a frame; never replace the HTML subtree.
        with gui.add_folder(None):
            self.panel = gui.add_html("")
            self.image = gui.add_image(
                np.zeros((480, 640, 3), dtype=np.uint8),
                label=None,
                format="jpeg",
                jpeg_quality=90,
            )
        with gui.add_folder("参考布局 · Top", expand_by_default=True):
            tasks = self.layouts.tasks()
            self.task = gui.add_dropdown("Task", tasks or (EMPTY_TASK,))
            self.slot = gui.add_dropdown("参考图", (EMPTY_SLOT,))
            self.refresh = gui.add_button("刷新参考图库")
            self.enabled = gui.add_checkbox("显示参考蒙版", True)
            self.opacity = gui.add_slider(
                "参考透明度", min=0.0, max=1.0, step=0.05, initial_value=0.5
            )
            self.white = gui.add_slider(
                "参考白底强度", min=0.0, max=1.0, step=0.05, initial_value=0.2
            )
            self.status = gui.add_markdown("")

        @self.task.on_update
        def _task(_event: Any) -> None:
            with self._lock:
                self._load_slots()

        @self.slot.on_update
        def _slot(_event: Any) -> None:
            with self._lock:
                self._load_reference()

        @self.refresh.on_click
        def _refresh(_event: Any) -> None:
            with self._lock:
                tasks = self.layouts.tasks() or (EMPTY_TASK,)
                selected = self.task.value
                self.task.options = tasks
                self.task.value = selected if selected in tasks else tasks[0]
                self._load_slots()

        @self.enabled.on_update
        @self.opacity.on_update
        @self.white.on_update
        def _redraw(_event: Any) -> None:
            with self._lock:
                self._render()

        self._load_slots()

    def _load_slots(self) -> None:
        slots = self.layouts.slots(self.task.value) if self.task.value != EMPTY_TASK else ()
        selected = self.slot.value
        self.slot.options = slots or (EMPTY_SLOT,)
        self.slot.value = selected if selected in slots else self.slot.options[0]
        self._load_reference()

    def _load_reference(self) -> None:
        self._reference = None
        self._error = ""
        if self.task.value != EMPTY_TASK and self.slot.value != EMPTY_SLOT:
            try:
                self._reference = self.layouts.load(self.task.value, self.slot.value)
            except (OSError, ValueError) as exc:
                self._error = f"参考图读取失败：{exc}"
        self._render()

    def update(self, image: np.ndarray) -> None:
        with self._lock:
            self._live = image.copy()
            self._render()

    def _render(self) -> None:
        label = f"{self.task.value} / {self.slot.value}"
        display = self._live
        note = (
            f"参考 {float(self.opacity.value):.0%} · 白底 {float(self.white.value):.0%}"
            " · 摆齐后 Start。"
            if self.enabled.value
            else "仅显示实时 Top · 参考蒙版已关闭。"
        )
        if self._error:
            note = self._error
        elif self._reference is None:
            note = "先用独立采集工具准备参考图，再刷新图库。"
        elif self._live is None:
            display = self._reference
            note = "当前仅显示参考图；等待 Prepare 后的 Top 实时画面。"
        elif self._reference.shape != self._live.shape:
            note = "参考与实时图尺寸不一致，当前仅显示实时画面。"
        elif self.enabled.value:
            alpha = float(self.opacity.value)
            white = float(self.white.value)
            reference = (1.0 - white) * self._reference.astype(np.float32) + white * 255.0
            display = np.rint(
                (1.0 - alpha) * self._live.astype(np.float32) + alpha * reference
            ).astype(np.uint8)
        if display is not None:
            self.image.image = display
        if self.status.content != note:
            self.status.content = note
        content = (
            _PANEL_STYLE + '<div class="manimux-reference-anchor"></div>'
            '<section class="manimux-reference-panel">'
            f"<header><strong>TOP · 布局复现</strong><span>{html.escape(label)}</span></header>"
            '<div class="manimux-reference-image-space"></div>'
            f"<footer>{html.escape(note)}</footer></section>"
        )
        if self.panel.content != content:
            self.panel.content = content
