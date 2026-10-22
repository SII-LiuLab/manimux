"""Bottom message bar showing the RPent planner's answers, thinking and tool lines."""

from __future__ import annotations

import html
import time
from typing import Any

_LABELS = {"answer": "agent", "thinking": "thinking", "tool": "tool",
           "error": "error", "user": "user", "system": "system"}
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
    box-sizing: border-box; padding: 8px 12px; border-radius: 12px;
    border: 1px solid rgba(128,138,156,.35); background: rgba(18,23,32,.92);
    box-shadow: 0 8px 28px rgba(0,0,0,.18); color: #d7deea;
    font: 12px/1.45 ui-sans-serif, system-ui, sans-serif;
  }
  .manimux-agent-feed-head { color: #8f9db0; margin-bottom: 6px; }
  .manimux-agent-feed-head b { color: #e6ebf3; }
  .manimux-agent-feed-list { max-height: 26vh; overflow-y: auto; scrollbar-width: thin; }
  .manimux-agent-feed-row { display: flex; gap: 8px; padding: 3px 0;
    border-top: 1px solid rgba(128,138,156,.18); }
  .manimux-agent-feed-row:first-child { border-top: 0; }
  .manimux-agent-feed-time { flex: none; color: #7d8a9d; font-variant-numeric: tabular-nums; }
  .manimux-agent-feed-kind { flex: none; width: 58px; color: #8f9db0; }
  .manimux-agent-feed-text { white-space: pre-wrap; word-break: break-word; min-width: 0; }
  .kind-answer .manimux-agent-feed-text { color: #f1f4f9; }
  .kind-answer .manimux-agent-feed-kind { color: #8fb4ff; }
  .kind-thinking .manimux-agent-feed-text { color: #9aa6b8; font-style: italic; }
  .kind-tool .manimux-agent-feed-text { color: #b8d3c0;
    font-family: ui-monospace, monospace; font-size: 11px; }
  .kind-error .manimux-agent-feed-text, .kind-error .manimux-agent-feed-kind { color: #ff8f8f; }
  .kind-user .manimux-agent-feed-kind { color: #9be3a8; }
  .kind-system .manimux-agent-feed-text { color: #7d8a9d; }
</style>
"""


def feed_html(messages: list[dict[str, Any]], *, phase: str, elapsed_s: float | None) -> str:
    """Newest first, so the latest answer stays visible without scrolling."""
    status = _PHASES.get(phase, phase or "ready")
    if elapsed_s is not None:
        status += f" · {int(elapsed_s) // 60:02d}:{int(elapsed_s) % 60:02d} since Start"
    if phase == "running" and not any(m.get("kind") != "system" for m in messages):
        status += " · planner is thinking"
    rows = "".join(
        f'<div class="manimux-agent-feed-row kind-{html.escape(m.get("kind", "system"))}">'
        f'<span class="manimux-agent-feed-time">'
        f'{time.strftime("%H:%M:%S", time.localtime(m.get("wall_time", 0)))}</span>'
        f'<span class="manimux-agent-feed-kind">{_LABELS.get(m.get("kind"), "system")}</span>'
        f'<span class="manimux-agent-feed-text">{html.escape(str(m.get("text", "")))}</span></div>'
        for m in reversed(messages)
    ) or '<div class="manimux-agent-feed-row"><span class="manimux-agent-feed-text">' \
         'No planner messages yet.</span></div>'
    return (f'{_STYLE}<div class="manimux-agent-feed-root">'
            f'<div class="manimux-agent-feed-head"><b>RPent</b> · {html.escape(status)} · '
            f'{len(messages)} messages</div>'
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
