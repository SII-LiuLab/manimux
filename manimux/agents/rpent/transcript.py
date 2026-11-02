"""Follow the native planner's live transcript for RoboGUI and recordings."""

from __future__ import annotations

import re
from pathlib import Path

# Codex and Claude Code write `[tag] text` blocks to <backend>_planner.txt while they run.
# Codex echoes operator steering as `[codex][user] text`.
_TAG = re.compile(r"^\[([^\]\n]+)\](\[user\])?(?: |$)")
_ANSWER = {"codex", "claude"}
_THINKING = {"codex-reasoning", "claude-thinking"}
_TOOL = {"tool->", "tool<-"}
_USER = {"user"}


def message_kind(tag: str) -> str:
    if tag in _ANSWER:
        return "answer"
    if tag in _THINKING:
        return "thinking"
    if tag in _TOOL:
        return "tool"
    if tag in _USER:
        return "user"
    if tag.endswith("error"):
        return "error"
    return "system"


def parse_transcript(text: str) -> list[dict[str, str]]:
    """Split a transcript into tagged blocks; untagged lines continue the previous block."""
    blocks: list[list[str]] = []
    for line in text.splitlines():
        match = _TAG.match(line)
        if match:
            tag = "user" if match.group(2) else match.group(1)
            blocks.append([tag, line[match.end():]])
        elif blocks:
            blocks[-1][1] += "\n" + line
    messages = []
    for tag, body in blocks:
        body = body.strip()
        if body:
            messages.append({"kind": message_kind(tag), "tag": tag, "text": body})
    return messages


class TranscriptFollower:
    """Return new complete blocks; the last block may still grow until the planner exits."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self._sent = 0

    def poll(self, *, final: bool = False) -> list[dict[str, str]]:
        paths = sorted(self.directory.glob("*_planner.txt"))
        if not paths:
            return []
        messages = parse_transcript(paths[0].read_text(errors="replace"))
        end = len(messages) if final else max(0, len(messages) - 1)
        new, self._sent = messages[self._sent:end], max(self._sent, end)
        return new
