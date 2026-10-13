"""Read-only task/reference selection and a lower-left Top ghost view."""

from __future__ import annotations

import html
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .reference_layouts import DEFAULT_LAYOUT_ROOT, ReferenceLayouts

EMPTY_TASK = "(No tasks)"
EMPTY_SLOT = "(No references)"

_PANEL_STYLE = """
<style>
  :root {
    --reference-top: calc(var(--manimux-camera-height, 397px) + 274px);
    --reference-image-height: clamp(100px,
      calc(100dvh - var(--reference-top) - 92px),
      calc((var(--manimux-camera-width, 460px) - 22px) * .75));
  }
  div:has(> .manimux-chunk-anchor) { anchor-name: --manimux-chunks; }
  div:has(> .manimux-reference-anchor) {
    position: relative; left: 0; top: 0;
    width: var(--manimux-camera-width, 460px); anchor-name: --manimux-reference;
    z-index: 4; margin: 12px 0 0; padding: 0 !important;
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
    position: absolute; left: 11px; top: calc(var(--reference-top) + 51px);
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
    div:has(> .manimux-reference-anchor) + div {
      top: calc(anchor(--manimux-reference top) + 51px);
    }
  }
  @media (max-width: 900px) {
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
    def __init__(
        self,
        gui: Any,
        root: Path = DEFAULT_LAYOUT_ROOT,
        *,
        display_container: Any | None = None,
    ) -> None:
        self.layouts = ReferenceLayouts(root)
        self._lock = threading.RLock()
        self._live: np.ndarray | None = None
        self._reference: np.ndarray | None = None
        self._reference_identity: dict[str, Any] | None = None
        self._selection_enabled = True
        self._error = ""
        self._task = EMPTY_TASK
        self._slot = EMPTY_SLOT
        # Keep the native image as the anchor's direct sibling, like the camera panel.
        # Only its binary image data changes on a frame; never replace the HTML subtree.
        container = display_container if display_container is not None else gui.add_folder(None)
        with container:
            self.panel = gui.add_html("")
            self.image = gui.add_image(
                np.zeros((480, 640, 3), dtype=np.uint8),
                label=None,
                format="jpeg",
                jpeg_quality=90,
            )
        with gui.add_folder("Reference appearance", expand_by_default=True):
            self.enabled = gui.add_checkbox("Show reference overlay", True)
            self.opacity = gui.add_slider(
                "Reference opacity", min=0.0, max=1.0, step=0.05, initial_value=0.5
            )
            self.white = gui.add_slider(
                "White tint", min=0.0, max=1.0, step=0.05, initial_value=0.2
            )
            self.status = gui.add_markdown("")

        @self.enabled.on_update
        @self.opacity.on_update
        @self.white.on_update
        def _redraw(_event: Any) -> None:
            with self._lock:
                self._render()

        self._render()

    def select(self, task: str, slot: str) -> None:
        """The dashboard's single Layout selector owns both preview and identity."""
        with self._lock:
            if not self._selection_enabled:
                return
            self._task, self._slot = task or EMPTY_TASK, slot or EMPTY_SLOT
            self._reference = None
            self._reference_identity = None
            self._error = ""
            if task and slot:
                try:
                    self._reference, self._reference_identity = self.layouts.snapshot(task, slot)
                except (OSError, ValueError) as exc:
                    self._error = f"Could not load reference image: {exc}"
            self._render()

    def set_selection_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._selection_enabled = enabled

    def freeze_selection(self) -> dict[str, Any]:
        """Snapshot and lock the selected reference; no work runs on control ticks."""
        with self._lock:
            if self._task == EMPTY_TASK or self._slot == EMPTY_SLOT:
                raise ValueError("Select a task and a saved reference image first.")
            pixels, identity = self.layouts.snapshot(self._task, self._slot)
            self._reference = pixels
            self._reference_identity = identity
            self._error = ""
            self.set_selection_enabled(False)
            self._render()
            return deepcopy(identity)

    def restore_selection(self, identity: dict[str, Any]) -> None:
        """Restore an active rollout after reconnect without showing a different image."""
        with self._lock:
            self.set_selection_enabled(False)
            reference = identity["reference_layout"]
            task, slot = reference["task"], identity["layout_id"]
            self._task, self._slot = task, slot
            if identity == self._reference_identity:
                return
            self._reference_identity = deepcopy(identity)
            self._reference = None
            try:
                pixels, current = self.layouts.snapshot(reference["task"], identity["layout_id"])
                if current["reference_layout"]["sha256"] != reference["sha256"]:
                    raise ValueError(
                        "The reference image has changed since this rollout was prepared."
                    )
                self._reference = pixels
                self._error = ""
            except (OSError, ValueError) as exc:
                self._error = f"Could not restore this rollout's reference image: {exc}"
            self._render()

    def update(self, image: np.ndarray | None) -> None:
        with self._lock:
            self._live = None if image is None else image.copy()
            self._render()

    def _render(self) -> None:
        identity = self._reference_identity
        label = (
            f"{identity['reference_layout']['task']} / {identity['layout_id']}"
            if identity is not None
            else f"{self._task} / {self._slot}"
        )
        display = self._live
        note = (
            f"Reference {float(self.opacity.value):.0%} · White tint {float(self.white.value):.0%}"
            " · Match the layout, then Start."
            if self.enabled.value
            else "Live Top only · Reference overlay is off."
        )
        if self._error:
            note = self._error
        elif self._reference is None:
            note = "Capture reference images with the capture tool, then refresh the library."
        elif self._live is None:
            display = self._reference
            note = "Reference only · Live Top appears after Prepare."
        elif self._reference.shape != self._live.shape:
            note = "Reference and live image sizes differ · Showing live Top only."
        elif self.enabled.value:
            alpha = float(self.opacity.value)
            white = float(self.white.value)
            reference = (1.0 - white) * self._reference.astype(np.float32) + white * 255.0
            display = np.rint(
                (1.0 - alpha) * self._live.astype(np.float32) + alpha * reference
            ).astype(np.uint8)
        self.image.image = (
            display if display is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        )
        if self.status.content != note:
            self.status.content = note
        content = (
            _PANEL_STYLE + '<div class="manimux-reference-anchor"></div>'
            '<section class="manimux-reference-panel">'
            "<header><strong>TOP · Layout reference</strong>"
            f"<span>{html.escape(label)}</span></header>"
            '<div class="manimux-reference-image-space"></div>'
            f"<footer>{html.escape(note)}</footer></section>"
        )
        if self.panel.content != content:
            self.panel.content = content
