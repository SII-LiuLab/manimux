"""Expanded, mutually exclusive score choices using supported Viser controls."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import viser


class ScoreChoices:
    """Radio-style choices with no preselected outcome.

    Viser has no native radio input. Circular button icons expose the selection
    while ordinary button events retain its existing client/server transport.
    """

    def __init__(
        self, gui: Any, container: Any, label: str, options: Sequence[str], *,
        disabled: bool = True, visible: bool = True,
        lock: Any,
    ) -> None:
        self._gui, self._container, self._lock = gui, container, lock
        self._value = "unlabeled"
        self._disabled, self._visible = disabled, visible
        self._callbacks: list[Callable] = []
        self._buttons: dict[str, Any] = {}
        self.display_labels: dict[str, str] = {}
        self._heading = gui.add_markdown(label, visible=visible)
        self._generation = 0
        self._replace_options(options)

    def on_update(self, callback: Callable) -> Callable:
        self._callbacks.append(callback)
        return callback

    @property
    def label(self) -> str:
        return self._heading.content

    @label.setter
    def label(self, value: str) -> None:
        self._heading.content = value

    @property
    def options(self) -> tuple[str, ...]:
        return self._options

    @options.setter
    def options(self, values: Sequence[str]) -> None:
        with self._container:
            self._replace_options(values)

    def _replace_options(self, values: Sequence[str]) -> None:
        self._generation += 1
        for button in self._buttons.values():
            button.remove()
        self._buttons.clear()
        self._options = tuple(values)
        if self._value not in self._options:
            self._value = "unlabeled"
        for index, value in enumerate(v for v in values if v != "unlabeled"):
            button = self._gui.add_button(
                self.display_labels.get(value, value),
                icon=viser.Icon.CIRCLE, color="gray",
                disabled=self._disabled, visible=self._visible,
                order=self._heading.order + (index + 1) / (len(values) + 1),
            )
            self._buttons[value] = button

            @button.on_click
            def choose(event: Any, selected: str = value, generation: int = self._generation) -> None:
                with self._lock:
                    # Reject stale events after a rubric change or Finish.
                    if (self._disabled or not self._visible
                            or generation != self._generation):
                        return
                    self.value = selected
                    for callback in self._callbacks:
                        callback(event)
        self._paint()

    def _paint(self) -> None:
        for value, button in self._buttons.items():
            selected = value == self._value
            button.icon = viser.Icon.CIRCLE_DOT if selected else viser.Icon.CIRCLE
            button.color = "blue" if selected else "gray"

    @property
    def value(self) -> str:
        return self._value

    @value.setter
    def value(self, value: str) -> None:
        if value != "unlabeled" and value not in self._options:
            raise ValueError(f"Unknown scoring choice: {value}")
        self._value = value
        self._paint()

    @property
    def disabled(self) -> bool:
        return self._disabled

    @disabled.setter
    def disabled(self, value: bool) -> None:
        self._disabled = value
        for button in self._buttons.values():
            button.disabled = value

    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        self._visible = value
        self._heading.visible = value
        for button in self._buttons.values():
            button.visible = value
