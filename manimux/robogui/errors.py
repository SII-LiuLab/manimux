"""Operator-visible errors, independent of live status and robot recovery."""

from datetime import datetime
from html import escape


class OperatorErrors:
    def __init__(self, gui):
        self.message = gui.add_html("", visible=False)
        self.show_detail = gui.add_checkbox("Show original error", initial_value=False, visible=False)
        self.detail = gui.add_html("", visible=False)
        self.show_detail.on_update(lambda _: self._update_detail_visibility())
        self.dismiss = gui.add_button("Dismiss error message", color="gray", visible=False)
        self.operation = None
        self._sources = {}
        self.dismiss.on_click(lambda _: self.clear())

    def _update_detail_visibility(self):
        self.detail.visible = self.operation is not None and bool(self.show_detail.value)

    def report(self, operation: str, detail: str, *, failure: dict | None = None) -> None:
        self.operation = operation
        timestamp = datetime.now().astimezone().strftime("%H:%M:%S")
        if failure is None and operation == "Warmup / Start":
            from manimux.failures import describe_failure
            failure = describe_failure(detail)
            failure["stage"] = "warmup / start"
        title = f"{operation} failed"
        body = escape(str(detail))
        if failure:
            title = str(failure.get("title") or title)
            body = "".join(
                f"<div><strong>{label}:</strong> {escape(str(failure[key]))}</div>"
                for label, key in (("Source", "source"), ("Stage", "stage"),
                                   ("Device / endpoint", "target"), ("Reason", "reason"),
                                   ("Next step", "action")) if failure.get(key)
            )
        raw_detail = str((failure or {}).get("detail") or detail)
        self.detail.content = (
            '<div style="white-space:pre-wrap;overflow-wrap:anywhere;user-select:text">'
            + escape(raw_detail) + '</div>'
        )
        self.message.content = (
            '<div role="alert" style="border:1px solid #e9a6a6;border-left:4px solid #c92a2a;'
            "border-radius:6px;padding:12px;background:#fff5f5;color:#781c1c;"
            'overflow-wrap:anywhere;font:14px/1.5 system-ui">'
            f"<strong>{escape(title)}</strong>"
            f'<span style="float:right;font-size:12px">{timestamp}</span>'
            '<div style="white-space:pre-wrap;user-select:text;margin-top:8px">'
            f"{body}</div></div>"
        )
        self.message.visible = self.dismiss.visible = True
        self.show_detail.visible = True
        self._update_detail_visibility()

    def sync(self, operation: str, detail: str, *, identity: str = "") -> None:
        """Deduplicate durable heartbeat errors, including after dismissal."""
        token = (identity, detail)
        if detail and self._sources.get(operation) != token:
            self.report(operation, detail)
        elif not detail:
            self.clear(operation)
        self._sources[operation] = token

    def clear(self, operation: str | None = None) -> None:
        # Dismissal only affects presentation. Never touch control state here.
        if operation is not None and operation != self.operation:
            return
        self.operation = None
        self.message.visible = self.dismiss.visible = False
        self.show_detail.visible = self.detail.visible = False

    def reset(self) -> None:
        """Forget errors and deduplication tokens when the runtime service changes."""
        self.clear()
        self._sources.clear()
