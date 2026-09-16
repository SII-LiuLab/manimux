"""Camera previews selected from policy input metadata, with explicit manual overrides."""

from __future__ import annotations

import base64
import html
import io
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from PIL import Image

from .config import ViewerConfig

PANEL_SLOTS = ("top", "left", "right")


def _camera_panel_html(labels: Mapping[str, tuple[str, str]] | None = None) -> str:
    """Place stable Viser image handles in the scrollable left overlay."""

    labels = labels if labels is not None else {slot: (slot, "") for slot in PANEL_SLOTS}
    style = """
<style>
  :root {
    --manimux-camera-top: 16px;
    --manimux-camera-width: clamp(300px, 26vw, 460px);
    --manimux-camera-inner: calc(var(--manimux-camera-width) - 20px);
    --manimux-camera-top-height: clamp(157.5px, calc(14.625vw - 11.25px), 247.5px);
    --manimux-camera-small-width: clamp(136px, calc(13vw - 14px), 216px);
    --manimux-camera-height: clamp(262px, calc(21.9375vw + 8.875px), 397px);
  }
  .mantine-Paper-root:has(.manimux-left-overlay-root):not([style*="position: absolute"]) {
    position: fixed; left: 16px; top: var(--manimux-camera-top); bottom: 16px;
    width: calc(var(--manimux-camera-width) + 8px); z-index: 4;
    box-sizing: border-box; margin: 0; padding: 0 8px 0 0;
    border: 0; background: transparent; box-shadow: none;
    overflow-x: hidden; overflow-y: auto; scrollbar-width: thin;
    scrollbar-color: rgba(135,148,167,.72) rgba(18,23,32,.38);
  }
  .mantine-Paper-root:has(.manimux-left-overlay-root):not([style*="position: absolute"])
    > .mantine-Paper-root:first-child { display: none; }
  .mantine-Paper-root:has(.manimux-left-overlay-root):not([style*="position: absolute"])
    > div:not(:first-child) > div > div { padding-top: 0 !important; }
  .mantine-Paper-root:has(
    .manimux-left-overlay-root
  ):not([style*="position: absolute"])::-webkit-scrollbar {
    width: 6px;
  }
  .mantine-Paper-root:has(
    .manimux-left-overlay-root
  ):not([style*="position: absolute"])::-webkit-scrollbar-thumb {
    border-radius: 999px; background: rgba(135,148,167,.72);
  }
  .mantine-Paper-root:has(
    .manimux-left-overlay-root
  ):not([style*="position: absolute"])::-webkit-scrollbar-track {
    background: rgba(18,23,32,.38);
  }
  div:has(> .manimux-camera-anchor) {
    position: relative; width: var(--manimux-camera-width); margin: 0; padding: 0 !important;
  }
  .manimux-camera-panel {
    position: relative;
    width: var(--manimux-camera-width);
    height: var(--manimux-camera-height);
    z-index: 4;
    padding: 10px; box-sizing: border-box; pointer-events: none;
    border: 1px solid rgba(128, 138, 156, 0.35); border-radius: 12px;
    background: rgba(18, 23, 32, 0.92); box-shadow: 0 8px 28px rgba(0,0,0,0.18);
  }
  .manimux-camera-label { position: absolute; z-index: 7;
    padding: 3px 9px; border-radius: 999px; color: white; background: rgba(8,12,18,0.78);
    font: 600 12px/1.4 system-ui, sans-serif; box-sizing: border-box;
    max-width: var(--manimux-camera-small-width); overflow: hidden; }
  .manimux-camera-label strong {
    display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .manimux-camera-label-top {
    left: 10px; top: 10px;
    max-width: var(--manimux-camera-inner); }
  .manimux-camera-label-left { left: 10px;
    top: calc(18px + var(--manimux-camera-top-height)); }
  .manimux-camera-label-right {
    left: calc(18px + var(--manimux-camera-small-width));
    top: calc(18px + var(--manimux-camera-top-height)); }

  div:has(> .manimux-camera-anchor) + div,
  div:has(> .manimux-camera-anchor) + div + div,
  div:has(> .manimux-camera-anchor) + div + div + div {
    position: absolute; z-index: 5; margin: 0; padding: 0 !important;
    overflow: hidden; border-radius: 8px; background: #0e131c;
    pointer-events: auto;
  }
  div:has(> .manimux-camera-anchor) + div {
    left: 10px; top: 10px;
    width: var(--manimux-camera-inner);
    height: var(--manimux-camera-top-height);
  }
  div:has(> .manimux-camera-anchor) + div + div {
    left: 10px; top: calc(18px + var(--manimux-camera-top-height));
    width: var(--manimux-camera-small-width); aspect-ratio: 16 / 9;
  }
  div:has(> .manimux-camera-anchor) + div + div + div {
    left: calc(18px + var(--manimux-camera-small-width));
    top: calc(18px + var(--manimux-camera-top-height));
    width: var(--manimux-camera-small-width); aspect-ratio: 16 / 9;
  }
  div:has(> .manimux-camera-anchor) + div > div,
  div:has(> .manimux-camera-anchor) + div + div > div,
  div:has(> .manimux-camera-anchor) + div + div + div > div {
    width: 100%; height: 100%;
  }
  div:has(> .manimux-camera-anchor) + div img,
  div:has(> .manimux-camera-anchor) + div + div img,
  div:has(> .manimux-camera-anchor) + div + div + div img {
    width: 100%; height: 100% !important; max-width: none !important;
    object-fit: cover; display: block;
  }
  @media (max-width: 900px) {
    .mantine-Paper-root:has(.manimux-left-overlay-root):not([style*="position: absolute"]) {
      position: static; width: auto; height: auto; margin: 8px 0; padding: 0;
      overflow: visible;
    }
    .manimux-camera-panel, .manimux-camera-label,
    div:has(> .manimux-camera-anchor) + div,
    div:has(> .manimux-camera-anchor) + div + div,
    div:has(> .manimux-camera-anchor) + div + div + div { display: none; }
  }
</style>
<div class="manimux-left-overlay-root"></div>
<div class="manimux-camera-anchor"></div>
<section class="manimux-camera-panel"></section>
"""
    # Missing cameras retain their black image handles and fixed layout positions.
    labels = {slot: labels.get(slot, (slot, "")) for slot in PANEL_SLOTS}
    display_names = {"left": "left side", "right": "right side"}
    return style + "".join(
        f'<span class="manimux-camera-label manimux-camera-label-{slot}" '
        f'title="{html.escape(display_names.get(slot, name))}">'
        f'<strong>{html.escape(display_names.get(slot, name))}</strong></span>'
        for slot, (name, _source) in labels.items() if slot in PANEL_SLOTS
    )


class CameraPanel:
    """Keep model names as labels; GUI handle IDs never depend on a model namespace.

    Recognized camera sources use fixed top/left/right preview positions. Further
    inputs remain visible in a separate native GUI folder. Image handles persist
    across frames and model switches; only labels change when routing changes.
    """

    def __init__(
        self, gui: Any, config: ViewerConfig, normalize: Callable[[str], str],
        on_main_image: Callable[[np.ndarray | None], None],
    ) -> None:
        self._gui = gui
        self._config = config
        self._normalize = normalize
        self._on_main_image = on_main_image
        self._policy_map: dict[str, str] = {}
        self._invalid_map = False
        self._views: list[tuple[str, str] | None] = list(config.cameras.model_dump().items())
        self._received: set[int] = set()
        self._sources: dict[int, str] = {}
        self._placeholder = np.zeros((90, 160, 3), dtype=np.uint8)
        self.display_container = gui.add_folder("")
        with self.display_container:
            self.panel = gui.add_html(_camera_panel_html())
            self.images = [self._add_image() for _ in PANEL_SLOTS]
        self.details = gui.add_html("")
        self.extra_folder = gui.add_folder("更多输入相机", visible=False)
        self._render()

    def _add_image(self) -> Any:
        return self._gui.add_image(self._placeholder, label=None, format="jpeg", jpeg_quality=70)

    def _clear_image(self, index: int) -> None:
        self.images[index].image = self._placeholder
        self._received.discard(index)
        if index == 0:
            self._on_main_image(None)

    def clear_images(self) -> None:
        for index in range(len(self.images)):
            self._clear_image(index)
        self._render()

    def _policy_views(self, policy_map: Mapping[str, str]) -> list[tuple[str, str] | None]:
        views: list[tuple[str, str] | None] = [None] * len(PANEL_SLOTS)
        unassigned = []
        extra = []
        for name, source in policy_map.items():
            slot = self._normalize(source)
            if slot in PANEL_SLOTS:
                index = PANEL_SLOTS.index(slot)
                if views[index] is None:
                    views[index] = (name, source)
                else:
                    extra.append((name, source))
            elif source.endswith("_prev"):
                # Temporal model inputs are not additional physical viewpoints.
                extra.append((name, source))
            else:
                unassigned.append((name, source))
        # Preserve config order for sources without a known spatial role, after
        # reserving positions for all recognized cameras.
        for view in unassigned:
            if None in views:
                views[views.index(None)] = view
            else:
                extra.append(view)
        return views + extra

    def set_policy_map(self, raw: object, *, reset: bool = False) -> None:
        valid = isinstance(raw, Mapping) and all(
            isinstance(key, str) and bool(key.strip())
            and isinstance(value, str) and bool(value.strip())
            for key, value in raw.items()
        )
        policy_map = dict(raw) if valid else {}
        invalid = raw is not None and not valid
        if (
            not reset and list(policy_map.items()) == list(self._policy_map.items())
            and invalid == self._invalid_map
        ):
            return
        self._policy_map = policy_map
        self._invalid_map = invalid
        self._views = (
            self._policy_views(policy_map)
            if self._config.camera_mode == "policy" and policy_map
            else list(self._config.cameras.model_dump().items())
        )
        with self.extra_folder:
            while len(self.images) < len(self._views):
                self.images.append(self._add_image())
        self.extra_folder.visible = len(self._views) > len(PANEL_SLOTS)
        self._sources.clear()
        for index, handle in enumerate(self.images):
            handle.visible = index < max(len(PANEL_SLOTS), len(self._views))
            self._clear_image(index)
        self._render()

    def update_images(self, payloads: Mapping[str, str]) -> None:
        # Normal state heartbeats omit images between camera updates.
        if not payloads:
            return
        follow_policy = self._config.camera_mode == "policy" and bool(self._policy_map)
        resolved = {} if follow_policy else self._config.cameras.resolve(payloads, self._normalize)
        for index, view in enumerate(self._views):
            if view is None:
                continue
            name, source = view
            selected = source if follow_policy else resolved.get(name, source)
            self._sources[index] = selected
            if selected not in payloads:
                if index in self._received:
                    self._clear_image(index)
                continue
            image = np.array(
                Image.open(io.BytesIO(base64.b64decode(payloads[selected]))).convert("RGB"),
                copy=True,
            )
            self.images[index].image = image
            self._received.add(index)
            if index == 0:
                self._on_main_image(image)
        self._render()

    def _render(self) -> None:
        labels = {
            PANEL_SLOTS[index]: (view[0], self._sources.get(index, view[1]))
            for index, view in enumerate(self._views[:3]) if view is not None
        }
        panel_html = _camera_panel_html(labels)
        if self.panel.content != panel_html:
            self.panel.content = panel_html
        if self._config.camera_mode == "manual":
            title = "手动预览 · 不改变模型输入"
        elif self._policy_map:
            title = "模型输入相机 · 实时预览"
        elif self._invalid_map:
            title = "默认预览 · 模型输入映射无效"
        else:
            title = "默认预览 · 尚未获取模型输入配置"
        rows = []
        for index, view in enumerate(self._views):
            if view is None:
                rows.append(f"<tr><td>{PANEL_SLOTS[index]}</td><td>—</td><td>未配置</td></tr>")
                continue
            name, source = view
            source = self._sources.get(index, source)
            status = "预览中" if index in self._received else "等待图像"
            rows.append(
                f"<tr><td>{html.escape(name)}</td><td>{html.escape(source)}</td>"
                f"<td>{status}</td></tr>"
            )
            if index >= len(PANEL_SLOTS):
                self.images[index].label = f"{name} ← {source} · {status}"
        details = (
            f"<strong>{title}</strong><table><thead><tr>"
            "<th>输入 / 预览位</th><th>相机来源</th><th>状态</th>"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        )
        if self._config.camera_mode == "manual":
            inputs = "；".join(
                f"{html.escape(name)} ← {html.escape(source)}"
                for name, source in self._policy_map.items()
            )
            details += f"<p>模型输入：{inputs or '尚未获取配置'}</p>"
        if self.details.content != details:
            self.details.content = details
