"""Camera previews selected from policy input metadata, with explicit manual overrides."""

from __future__ import annotations

import base64
import html
import io
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from PIL import Image

PANEL_SLOTS = ("top", "left", "right")


def _camera_panel_html(
    labels: Mapping[str, tuple[str, str]] | None = None, slots: tuple[str, ...] = PANEL_SLOTS
) -> str:
    """Place stable Viser image handles in the scrollable left overlay."""

    labels = (
        labels
        if labels is not None
        else {
            slot: ({"left": "left side", "right": "right side"}.get(slot, slot), "")
            for slot in slots
        }
    )
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
  /* Select only the innermost Paper that owns the camera folder. Viser 1.1
     wraps it in a dock Paper; matching that ancestor moves the main panel left. */
  .mantine-Paper-root:has(.manimux-left-overlay-root):not(
    :has(.mantine-Paper-root .manimux-left-overlay-root)
  ):not([style*="position: absolute"]) {
    position: fixed; left: 16px; top: var(--manimux-camera-top); bottom: 16px;
    width: calc(var(--manimux-camera-width) + 8px); z-index: 4;
    box-sizing: border-box; margin: 0; padding: 0 8px 0 0;
    border: 0; background: transparent; box-shadow: none;
    overflow-x: hidden; overflow-y: auto; scrollbar-width: thin;
    scrollbar-color: rgba(135,148,167,.72) rgba(18,23,32,.38);
  }
  .mantine-Paper-root:has(.manimux-left-overlay-root):not(
    :has(.mantine-Paper-root .manimux-left-overlay-root)
  ):not([style*="position: absolute"])
    > .mantine-Paper-root:first-child { display: none; }
  .mantine-Paper-root:has(.manimux-left-overlay-root):not(
    :has(.mantine-Paper-root .manimux-left-overlay-root)
  ):not([style*="position: absolute"])
    > div:not(:first-child) > div > div { padding-top: 0 !important; }
  .mantine-Paper-root:has(
    .manimux-left-overlay-root
  ):not(:has(
    .mantine-Paper-root .manimux-left-overlay-root
  )):not([style*="position: absolute"])::-webkit-scrollbar {
    width: 6px;
  }
  .mantine-Paper-root:has(
    .manimux-left-overlay-root
  ):not(:has(
    .mantine-Paper-root .manimux-left-overlay-root
  )):not([style*="position: absolute"])::-webkit-scrollbar-thumb {
    border-radius: 999px; background: rgba(135,148,167,.72);
  }
  .mantine-Paper-root:has(
    .manimux-left-overlay-root
  ):not(:has(
    .mantine-Paper-root .manimux-left-overlay-root
  )):not([style*="position: absolute"])::-webkit-scrollbar-track {
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
    .mantine-Paper-root:has(.manimux-left-overlay-root):not(
      :has(.mantine-Paper-root .manimux-left-overlay-root)
    ):not([style*="position: absolute"]) {
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
    # Keep the original three-camera CSS exactly; override positions only for
    # explicitly smaller layouts. No empty agent/top preview is manufactured.
    labels = {slot: labels.get(slot, (slot, "")) for slot in slots}
    if slots != PANEL_SLOTS:
        # The original sibling selectors assume three image handles. Remove
        # unused selectors so they cannot position the following timeline as an image.
        for count in range(3, len(slots), -1):
            selector = "div:has(> .manimux-camera-anchor)" + " + div" * count
            style = style.replace(selector, f".manimux-unused-camera-{count}")
        rules = []
        if "top" not in slots:
            rules.append(
                ":root { --manimux-camera-top-height: 0px; "
                "--manimux-camera-height: clamp(104px, 7.3125vw + 9px, 149px); }"
            )
        if not slots:
            rules.append(".manimux-camera-panel { display: none; }")
        for index, slot in enumerate(slots):
            selector = "div:has(> .manimux-camera-anchor)" + " + div" * (index + 1)
            if slot == "top" or len(slots) == 1:
                geometry = (
                    "left:10px;top:10px;width:var(--manimux-camera-inner);"
                    "height:var(--manimux-camera-top-height);aspect-ratio:16/9;"
                )
                if len(slots) == 1:
                    rules.append(
                        ":root { --manimux-camera-top-height: "
                        "clamp(157.5px, calc(14.625vw - 11.25px), 247.5px); "
                        "--manimux-camera-height: "
                        "calc(var(--manimux-camera-top-height) + 20px); }"
                    )
            else:
                left = (
                    "10px" if slot == "left" else "calc(18px + var(--manimux-camera-small-width))"
                )
                top = "calc(18px + var(--manimux-camera-top-height))" if "top" in slots else "10px"
                geometry = (
                    f"left:{left};top:{top};width:var(--manimux-camera-small-width);"
                    "height:auto;aspect-ratio:16/9;"
                )
            rules.append(f"{selector} {{{geometry}}}")
        if "top" not in slots:
            rules.append(".manimux-camera-label-left,.manimux-camera-label-right {top:10px;}")
        style += "<style>" + "".join(rules) + "</style>"
    return style + "".join(
        f'<span class="manimux-camera-label manimux-camera-label-{slot}" '
        f'title="{html.escape(name)}"><strong>{html.escape(name)}</strong></span>'
        for slot, (name, _source) in labels.items()
    )


class CameraPanel:
    """Keep model names as labels; GUI handle IDs never depend on a model namespace.

    Recognized camera sources use fixed top/left/right preview positions. Further
    inputs remain visible in a separate native GUI folder. Image handles persist
    across frames and model switches; only labels change when routing changes.
    """

    def __init__(
        self,
        gui: Any,
        config: dict,
        normalize: Callable[[str], str],
        on_main_image: Callable[[np.ndarray | None], None],
        *,
        defer_diagnostics: bool = False,
    ) -> None:
        self._gui = gui
        self._config = config
        self._normalize = normalize
        self._on_main_image = on_main_image
        self._policy_map: dict[str, str] = {}
        self._temporal_views: list[tuple[str, str]] = []
        self._invalid_map = False
        cameras = config.get("cameras", [])
        self._slots = tuple(
            slot for slot in PANEL_SLOTS if any(c.get("slot", c["source"]) == slot for c in cameras)
        )
        self._configured = [
            next((c["label"], c["source"]) for c in cameras if c.get("slot", c["source"]) == slot)
            for slot in self._slots
        ]
        self._configured += [
            (c["label"], c["source"])
            for c in cameras
            if c.get("slot", c["source"]) not in PANEL_SLOTS
        ]
        self._views: list[tuple[str, str] | None] = list(self._configured)
        self._received: set[int] = set()
        self._sources: dict[int, str] = {}
        self._placeholder = np.zeros((90, 160, 3), dtype=np.uint8)
        self.display_container = gui.add_folder("")
        with self.display_container:
            self.panel = gui.add_html(_camera_panel_html(slots=self._slots))
            self.images = [self._add_image() for _ in self._slots]
        self.diagnostics_folder = None
        self.details = None
        self.extra_folder = None
        if not defer_diagnostics:
            self.add_diagnostics()
        self._render()

    def add_diagnostics(self, *, expand_by_default: bool = True) -> None:
        """Add routing diagnostics at the caller-selected point in the GUI."""

        if self.diagnostics_folder is not None:
            return
        self.diagnostics_folder = self._gui.add_folder(
            "Images", expand_by_default=expand_by_default
        )
        with self.diagnostics_folder:
            self.details = self._gui.add_html("")
            self.extra_folder = self._gui.add_folder("更多输入相机", visible=False)
            with self.extra_folder:
                while len(self.images) < len(self._views):
                    self.images.append(self._add_image())
            self.extra_folder.visible = len(self._views) > len(self._slots)
        self._render()

    def _add_image(self) -> Any:
        return self._gui.add_image(self._placeholder, label=None, format="jpeg", jpeg_quality=70)

    def _clear_image(self, index: int) -> None:
        self.images[index].image = self._placeholder
        self._received.discard(index)
        if self._slots and self._slots[0] == "top" and index == 0:
            self._on_main_image(None)

    def clear_images(self) -> None:
        for index in range(len(self.images)):
            self._clear_image(index)
        self._render()

    def _policy_views(self, policy_map: Mapping[str, str]) -> list[tuple[str, str] | None]:
        views: list[tuple[str, str] | None] = [None] * len(self._slots)
        unassigned = []
        extra = []
        for name, source in policy_map.items():
            slot = self._normalize(source)
            if slot in self._slots:
                index = self._slots.index(slot)
                if views[index] is None:
                    views[index] = (name, source)
                else:
                    extra.append((name, source))
            elif source.endswith("_prev"):
                # Temporal model inputs reuse a physical camera; report their
                # routing without manufacturing duplicate black previews.
                continue
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
            isinstance(key, str)
            and bool(key.strip())
            and isinstance(value, str)
            and bool(value.strip())
            for key, value in raw.items()
        )
        policy_map = dict(raw) if valid else {}
        invalid = raw is not None and not valid
        if (
            not reset
            and list(policy_map.items()) == list(self._policy_map.items())
            and invalid == self._invalid_map
        ):
            return
        self._policy_map = policy_map
        self._temporal_views = [
            (name, source) for name, source in policy_map.items() if source.endswith("_prev")
        ]
        self._invalid_map = invalid
        self._views = (
            self._policy_views(policy_map)
            if self._config.get("camera_mode", "policy") == "policy" and policy_map
            else list(self._configured)
        )
        if self.extra_folder is not None:
            with self.extra_folder:
                while len(self.images) < len(self._views):
                    self.images.append(self._add_image())
            self.extra_folder.visible = len(self._views) > len(self._slots)
        self._sources.clear()
        for index, handle in enumerate(self.images):
            handle.visible = index < len(self._views)
            self._clear_image(index)
        self._render()

    def update_images(self, payloads: Mapping[str, str]) -> None:
        # Normal state heartbeats omit images between camera updates.
        if not payloads:
            return
        follow_policy = self._config.get("camera_mode", "policy") == "policy" and bool(
            self._policy_map
        )
        resolved = {}
        if not follow_policy:
            for _name, source in self._configured:
                if source in payloads:
                    resolved[source] = source
                else:
                    role = self._normalize(source)
                    selected = next((key for key in payloads if self._normalize(key) == role), None)
                    if selected is not None:
                        resolved[source] = selected
        for index, view in enumerate(self._views):
            if view is None:
                continue
            if index >= len(self.images):
                # Deferred diagnostics allocate extra preview handles later.
                continue
            name, source = view
            selected = source if follow_policy else resolved.get(source, source)
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
            if self._slots and self._slots[0] == "top" and index == 0:
                self._on_main_image(image)
        self._render()

    def _render(self) -> None:
        labels = {
            self._slots[index]: (
                self._configured[index][0] if self._slots[index] != "top" else view[0],
                self._sources.get(index, view[1]),
            )
            for index, view in enumerate(self._views[: len(self._slots)])
            if view is not None
        }
        panel_html = _camera_panel_html(labels, self._slots)
        if self.panel.content != panel_html:
            self.panel.content = panel_html
        if self._config.get("camera_mode", "policy") == "manual":
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
                rows.append(f"<tr><td>{self._slots[index]}</td><td>—</td><td>未配置</td></tr>")
                continue
            name, source = view
            source = self._sources.get(index, source)
            status = "预览中" if index in self._received else "等待图像"
            rows.append(
                f"<tr><td>{html.escape(name)}</td><td>{html.escape(source)}</td>"
                f"<td>{status}</td></tr>"
            )
            if len(self._slots) <= index < len(self.images):
                self.images[index].label = f"{name} ← {source} · {status}"
        rows.extend(
            f"<tr><td>{html.escape(name)}</td><td>{html.escape(source)}</td>"
            "<td>时序输入 · 复用物理相机</td></tr>"
            for name, source in self._temporal_views
        )
        details = (
            f"<strong>{title}</strong><table><thead><tr>"
            "<th>输入 / 预览位</th><th>相机来源</th><th>状态</th>"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        )
        if self._config.get("camera_mode", "policy") == "manual":
            inputs = "；".join(
                f"{html.escape(name)} ← {html.escape(source)}"
                for name, source in self._policy_map.items()
            )
            details += f"<p>模型输入：{inputs or '尚未获取配置'}</p>"
        if self.details is not None and self.details.content != details:
            self.details.content = details
