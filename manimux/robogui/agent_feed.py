"""Bottom message bar showing the RPent planner's answers, thinking and tool lines."""

from __future__ import annotations

import html
import time
from typing import Any

# One glyph per line kind, as in a terminal transcript.
_GLYPHS = {"answer": "●", "thinking": "∴", "tool": "$", "error": "!", "user": "›", "system": "·"}
_PHASES = {"waiting": "waiting for Start", "running": "running", "paused": "paused",
           "completed": "finished", "failed": "failed", "stopped": "stopped"}

_STYLE = """
<style>
  /* Lift the innermost Paper owning this bar out of the control panel, as the camera
     overlay does, between the left camera column and the right-docked panel. */
  .mantine-Paper-root:has(.manimux-agent-feed-root):not(
    :has(.mantine-Paper-root .manimux-agent-feed-root)
  ):not([style*="position: absolute"]) {
    position: fixed; bottom: 16px; z-index: 4;
    left: calc(var(--manimux-camera-width, 0px) + 40px); right: calc(24em + 40px);
    margin: 0; padding: 0; border: 0; background: transparent; box-shadow: none;
  }
  .mantine-Paper-root:has(.manimux-agent-feed-root):not(
    :has(.mantine-Paper-root .manimux-agent-feed-root)
  ):not([style*="position: absolute"]) > .mantine-Paper-root:first-child { display: none; }
  .manimux-agent-feed-root {
    box-sizing: border-box; padding: 8px 12px 10px; border-radius: 10px;
    border: 1px solid rgba(128,138,156,.35); background: rgba(14,18,26,.9);
    box-shadow: 0 8px 28px rgba(0,0,0,.18); color: #d7deea;
    font: 14px/1.4 ui-sans-serif, system-ui, sans-serif;
  }
  .manimux-agent-feed-grip { height: 10px; margin: -6px -12px 2px; cursor: ns-resize;
    display: flex; align-items: center; justify-content: center; }
  .manimux-agent-feed-grip::before { content: ""; width: 42px; height: 4px; border-radius: 2px;
    background: rgba(143,157,176,.45); }
  .manimux-agent-feed-grip:hover::before { background: rgba(143,157,176,.85); }
  .manimux-agent-feed-head { color: #8f9db0; margin-bottom: 6px; font-size: 13px; }
  .manimux-agent-feed-head b { color: #e6ebf3; font-weight: 600; margin-right: 4px; }
  .manimux-agent-feed-dot { display: inline-block; width: 7px; height: 7px; margin-right: 6px;
    border-radius: 50%; background: #6b7686; vertical-align: 1px; }
  .manimux-agent-feed-dot.live { background: #f08a4b; }
  /* column-reverse keeps the scroll anchored at the newest (bottom) line on re-render.
     The grip stores a dragged height on :root, so re-rendered content keeps it. */
  .manimux-agent-feed-list { display: flex; flex-direction: column-reverse;
    height: var(--manimux-agent-feed-height, max(18vh, 140px));
    overflow-y: auto; scrollbar-width: thin; }
  .manimux-agent-feed-row { display: flex; gap: 8px; padding: 1px 0; }
  .manimux-agent-feed-glyph { flex: none; width: 12px; text-align: center; color: #8f9db0; }
  .manimux-agent-feed-text { white-space: pre-wrap; word-break: break-word; min-width: 0; }
  .kind-answer .manimux-agent-feed-text, .kind-answer .manimux-agent-feed-glyph { color: #eef2f8; }
  .kind-thinking .manimux-agent-feed-text { color: #8e9aac; font-style: italic;
    display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
  .kind-tool .manimux-agent-feed-text, .kind-tool .manimux-agent-feed-glyph {
    color: #7fb0ff; font-family: ui-monospace, monospace; font-size: 13px; }
  .kind-tool .manimux-agent-feed-text { display: -webkit-box; -webkit-line-clamp: 2;
    -webkit-box-orient: vertical; overflow: hidden; }
  .kind-error .manimux-agent-feed-text, .kind-error .manimux-agent-feed-glyph { color: #ff8f8f; }
  .kind-user .manimux-agent-feed-text, .kind-user .manimux-agent-feed-glyph { color: #9be3a8; }
  .kind-system .manimux-agent-feed-text, .kind-system .manimux-agent-feed-glyph {
    color: #6b7686; font-size: 12px; }
</style>
"""


# Inline handler: Viser renders this HTML with innerHTML, so <script> would not run.
# Inline handlers resolve bare names on the element first; use window explicitly.
# Dragging the top grip up makes the bottom-anchored bar taller.
_RESIZE = (
    "(function(e){e.preventDefault();"
    "var l=e.currentTarget.parentNode.querySelector('.manimux-agent-feed-list');"
    "var y0=e.clientY,h0=l.getBoundingClientRect().height;"
    "var w=window;function mv(v){var h=Math.max(80,Math.min(w.innerHeight*0.85,h0+y0-v.clientY));"
    "document.documentElement.style.setProperty('--manimux-agent-feed-height',h+'px');}"
    "function up(){w.removeEventListener('mousemove',mv);w.removeEventListener('mouseup',up);}"
    "w.addEventListener('mousemove',mv);w.addEventListener('mouseup',up);})(event)"
)


def feed_html(messages: list[dict[str, Any]], *, phase: str, elapsed_s: float | None) -> str:
    """Chronological, newest at the bottom; hover a line for its time."""
    status = _PHASES.get(phase, phase or "ready")
    if elapsed_s is not None:
        status += f" · {int(elapsed_s) // 60:02d}:{int(elapsed_s) % 60:02d} since Start"
    if phase == "running" and not any(m.get("kind") != "system" for m in messages):
        status += " · planner is thinking"
    # Emitted newest first; the list's column-reverse shows them oldest first.
    rows = "".join(
        f'<div class="manimux-agent-feed-row kind-{html.escape(m.get("kind", "system"))}" '
        f'title="{time.strftime("%H:%M:%S", time.localtime(m.get("wall_time", 0)))}">'
        f'<span class="manimux-agent-feed-glyph">{_GLYPHS.get(m.get("kind"), "·")}</span>'
        f'<span class="manimux-agent-feed-text">{html.escape(str(m.get("text", "")))}</span></div>'
        for m in reversed(messages)
    ) or '<div class="manimux-agent-feed-row kind-system"><span class="manimux-agent-feed-text">' \
         'No planner messages yet.</span></div>'
    live = " live" if phase == "running" else ""
    return (f'{_STYLE}<div class="manimux-agent-feed-root">'
            f'<div class="manimux-agent-feed-grip" title="Drag to resize" '
            f'onmousedown="{_RESIZE}"></div>'
            f'<div class="manimux-agent-feed-head"><span class="manimux-agent-feed-dot{live}">'
            f'</span><b>Agent</b> RPent · {html.escape(status)} · {len(messages)} messages</div>'
            f'<div class="manimux-agent-feed-list">{rows}</div></div>')


class AgentFeed:
    """Accumulate heartbeat tails by sequence and re-render only when the text changes."""

    def __init__(self, gui: Any) -> None:
        self.container = gui.add_folder("", visible=False)
        with self.container:
            self.panel = gui.add_html("")
        self.reset()

    def reset(self) -> None:
        self._messages: dict[int, dict[str, Any]] = {}
        self._phase = ""
        self._running_since: float | None = None
        self._elapsed_s: float | None = None
        self._content = None
        self.render()

    def set_visible(self, visible: bool) -> None:
        self.container.visible = visible

    def update(self, messages: list[dict[str, Any]], status: dict[str, Any]) -> None:
        for message in messages:
            self._messages[int(message.get("sequence", 0))] = message
        for sequence in sorted(self._messages)[:-200]:
            del self._messages[sequence]
        phase = str(status.get("phase", self._phase))
        if phase == "running" and self._running_since is None:
            self._running_since = time.monotonic()
        self._phase = phase
        if self._running_since is not None and phase in {"running", "paused"}:
            self._elapsed_s = time.monotonic() - self._running_since
        self.render()

    def render(self) -> None:
        content = feed_html([self._messages[k] for k in sorted(self._messages)],
                            phase=self._phase, elapsed_s=self._elapsed_s)
        if content != self._content:
            self._content = content
            self.panel.content = content
