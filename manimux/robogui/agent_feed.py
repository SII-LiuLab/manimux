"""Bottom message bar showing the RPent planner's answers, thinking and tool lines."""

from __future__ import annotations

import html
import time
from typing import Any

# One glyph per line kind, as in a terminal transcript.
_GLYPHS = {"answer": "●", "thinking": "∴", "tool": "$", "error": "!", "user": "›", "system": "·"}
_PHASES = {"waiting": "waiting for Start", "running": "running", "paused": "paused",
           "completed": "finished", "failed": "failed", "stopped": "stopped"}
_KEEP = 200

# The bar is one Viser folder: a header element, then _KEEP preallocated message slots.
# Adding a Viser element re-creates its siblings' DOM, but changing one element's content
# does not; filling the next slot therefore keeps scrolling and text selection in place
# while the header's timer updates. CSS lifts the folder's Paper out of the control panel, as the
# camera overlay does, and turns its children container into the scroll area.
_FOLDER = ".mantine-Paper-root:has(.manimux-agent-feed-root):not(" \
          ":has(.mantine-Paper-root .manimux-agent-feed-root))"
_STYLE = f"""
<style>
  {_FOLDER} {{
    position: fixed; bottom: 16px; z-index: 4;
    left: calc(var(--manimux-camera-width, 0px) + 40px); right: calc(24em + 40px);
    margin: 0; padding: 0; border: 1px solid rgba(128,138,156,.35); border-radius: 10px;
    background: rgba(14,18,26,.92); box-shadow: 0 8px 28px rgba(0,0,0,.18); overflow: hidden;
  }}
  {_FOLDER} > .mantine-Paper-root:first-child, {_FOLDER} > div[aria-hidden="true"] {{
    display: none; }}
  {_FOLDER} > div[aria-hidden="false"] > div {{ padding: 0 !important; }}
  {_FOLDER} > div[aria-hidden="false"] > div > div {{
    height: var(--manimux-agent-feed-height, max(18vh, 140px)); overflow-y: auto;
    padding: 0 12px 8px !important; scrollbar-width: thin; }}
  {_FOLDER} > div[aria-hidden="false"] > div > div > div {{ padding: 0 !important; margin: 0; }}
  {_FOLDER} > div[aria-hidden="false"] > div > div > div:first-child {{
    position: sticky; top: 0; z-index: 1; background: rgb(14,18,26); }}
  .manimux-agent-feed-root {{ color: #8f9db0; font: 13px/1.4 ui-sans-serif, system-ui, sans-serif;
    padding-bottom: 6px; }}
  .manimux-agent-feed-root b {{ color: #e6ebf3; font-weight: 600; margin-right: 4px; }}
  .manimux-agent-feed-grip {{ height: 12px; margin: 0 -12px; cursor: ns-resize;
    display: flex; align-items: center; justify-content: center; }}
  .manimux-agent-feed-grip::before {{ content: ""; width: 42px; height: 4px; border-radius: 2px;
    background: rgba(143,157,176,.45); }}
  .manimux-agent-feed-grip:hover::before {{ background: rgba(143,157,176,.85); }}
  .manimux-agent-feed-dot {{ display: inline-block; width: 7px; height: 7px; margin-right: 6px;
    border-radius: 50%; background: #6b7686; vertical-align: 1px; }}
  .manimux-agent-feed-dot.live {{ background: #f08a4b; }}
  .manimux-agent-msg {{ display: flex; gap: 8px; padding: 1px 0; color: #d7deea;
    font: 14px/1.4 ui-sans-serif, system-ui, sans-serif; user-select: text; cursor: text; }}
  .manimux-agent-msg .g {{ flex: none; width: 12px; text-align: center; color: #8f9db0;
    user-select: none; }}
  .manimux-agent-msg .t {{ white-space: pre-wrap; word-break: break-word; min-width: 0; }}
  .manimux-agent-msg.kind-answer, .manimux-agent-msg.kind-answer .g {{ color: #eef2f8; }}
  .manimux-agent-msg.kind-thinking .t {{ color: #8e9aac; font-style: italic; }}
  .manimux-agent-msg.kind-tool, .manimux-agent-msg.kind-tool .g {{ color: #7fb0ff;
    font-family: ui-monospace, monospace; font-size: 13px; }}
  .manimux-agent-msg.kind-error, .manimux-agent-msg.kind-error .g {{ color: #ff8f8f; }}
  .manimux-agent-msg.kind-user, .manimux-agent-msg.kind-user .g {{ color: #9be3a8; }}
  .manimux-agent-msg.kind-system, .manimux-agent-msg.kind-system .g {{ color: #6b7686;
    font-size: 12px; }}
</style>
"""

# Viser inserts this HTML with innerHTML: <script> does not run, but inline handlers do,
# and an <img> with an empty data URL fires onerror once inserted. Inline handlers
# resolve bare names on the element first, so window is named explicitly.
_SCROLLER = "var c=this;while(c&&getComputedStyle(c).overflowY!='auto')c=c.parentElement;"
# Header: follow new lines only while the view is at the bottom.
_BIND = _SCROLLER + (
    "if(c&&!c.__mxBound){c.__mxBound=1;c.__mxPinned=true;c.addEventListener('scroll',"
    "function(){c.__mxPinned=c.scrollTop+c.clientHeight>=c.scrollHeight-8;});}"
)
_FOLLOW = _SCROLLER + "if(c&&c.__mxPinned!==false)c.scrollTop=c.scrollHeight;"
# Dragging the top grip up makes the bottom-anchored bar taller; :root keeps the height.
_RESIZE = (
    "(function(e){e.preventDefault();var w=window,c=e.currentTarget;"
    "while(c&&getComputedStyle(c).overflowY!='auto')c=c.parentElement;if(!c)return;"
    "var y0=e.clientY,h0=c.getBoundingClientRect().height;"
    "function mv(v){var h=Math.max(80,Math.min(w.innerHeight*0.85,h0+y0-v.clientY));"
    "document.documentElement.style.setProperty('--manimux-agent-feed-height',h+'px');}"
    "function up(){w.removeEventListener('mousemove',mv);w.removeEventListener('mouseup',up);}"
    "w.addEventListener('mousemove',mv);w.addEventListener('mouseup',up);})(event)"
)
_RUN = '<img alt="" src="data:," style="display:none" onerror="{}">'


def header_html(*, phase: str, elapsed_s: float | None, count: int, replied: bool) -> str:
    status = _PHASES.get(phase, phase or "ready")
    if elapsed_s is not None:
        status += f" · {int(elapsed_s) // 60:02d}:{int(elapsed_s) % 60:02d} since Start"
    if phase == "running" and not replied:
        status += " · planner is thinking"
    live = " live" if phase == "running" else ""
    empty = "" if count else " · no planner messages yet"
    return (f'{_STYLE}<div class="manimux-agent-feed-root">'
            f'<div class="manimux-agent-feed-grip" title="Drag to resize" '
            f'onmousedown="{_RESIZE}"></div>'
            f'<span class="manimux-agent-feed-dot{live}"></span><b>Agent</b>RPent · '
            f'{html.escape(status)} · {count} messages{empty}</div>{_RUN.format(_BIND)}')


def message_html(message: dict[str, Any]) -> str:
    kind = str(message.get("kind", "system"))
    stamp = time.strftime("%H:%M:%S", time.localtime(message.get("wall_time", 0)))
    return (f'<div class="manimux-agent-msg kind-{html.escape(kind)}" title="{stamp}">'
            f'<span class="g">{_GLYPHS.get(kind, "·")}</span>'
            f'<span class="t">{html.escape(str(message.get("text", "")))}</span></div>'
            f'{_RUN.format(_FOLLOW)}')


class AgentFeed:
    """Fill preallocated slots with heartbeat messages by sequence."""

    def __init__(self, gui: Any) -> None:
        self.container = gui.add_folder("", visible=False)
        with self.container:
            self.header = gui.add_html("")
            self._slots = [gui.add_html("") for _ in range(_KEEP)]
        self._messages: list[dict[str, Any]] = []
        self.reset()

    def reset(self) -> None:
        for slot in self._slots[:len(self._messages)]:
            slot.content = ""
        self._messages = []
        self._phase = ""
        self._replied = False
        self._running_since: float | None = None
        self._elapsed_s: float | None = None
        self._header = None
        self.render_header()

    def set_visible(self, visible: bool) -> None:
        self.container.visible = visible

    def update(self, messages: list[dict[str, Any]], status: dict[str, Any]) -> None:
        last = int(self._messages[-1]["sequence"]) if self._messages else 0
        new = sorted((m for m in messages if int(m.get("sequence", 0)) > last),
                     key=lambda m: int(m["sequence"]))
        for message in new:
            if len(self._messages) == _KEEP:
                # Full: drop the oldest and re-render every slot once.
                self._messages = self._messages[1:] + [message]
                for slot, kept in zip(self._slots, self._messages, strict=True):
                    slot.content = message_html(kept)
            else:
                self._slots[len(self._messages)].content = message_html(message)
                self._messages.append(message)
            self._replied |= message.get("kind") != "system"
        phase = str(status.get("phase", self._phase))
        if phase == "running" and self._running_since is None:
            self._running_since = time.monotonic()
        self._phase = phase
        if self._running_since is not None and phase in {"running", "paused"}:
            self._elapsed_s = time.monotonic() - self._running_since
        self.render_header()

    def render_header(self) -> None:
        content = header_html(phase=self._phase, elapsed_s=self._elapsed_s,
                              count=len(self._messages), replied=self._replied)
        if content != self._header:
            self._header = content
            self.header.content = content
