"""Operator-controlled recording selection, reversible deletion and repeat refresh."""

from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path

from manimux.recording.management import list_recordings, restore_recordings, trash_recordings

RECORDS_STYLE = """
<style>
.mm-records-session { --mm-records-muted: #59636f; box-sizing: border-box;
  min-width: 0; padding: 8px 12px 12px; font: 12px/1.55 system-ui, sans-serif;
  color: var(--mantine-color-text, #263445); overflow-wrap: anywhere; }
[data-mantine-color-scheme="dark"] .mm-records-session { --mm-records-muted: #b1bac6; }
.mm-records-session header { margin-bottom: 4px; font-size: 11px; font-weight: 500;
  font-variant-numeric: tabular-nums; color: var(--mm-records-muted); }
.mm-records-entry + .mm-records-entry {
  border-top: 1px solid var(--mantine-color-default-border, #e8edf2); }
.mm-records-entry summary { display: grid; grid-template-columns: minmax(0, 1fr) 10px;
  column-gap: 8px; padding: 10px 0; list-style: none; cursor: pointer; }
.mm-records-entry summary::-webkit-details-marker { display: none; }
.mm-records-entry summary::after { content: ''; width: 5px; height: 5px;
  border-right: 1px solid currentColor; border-bottom: 1px solid currentColor;
  transform: rotate(-45deg); grid-column: 2; grid-row: 1 / 3; align-self: center;
  color: var(--mm-records-muted); }
.mm-records-entry[open] summary::after { transform: rotate(45deg); }
.mm-records-entry summary:focus-visible {
  outline: 2px solid var(--mantine-primary-color-filled, #228be6); outline-offset: 2px; }
.mm-records-entry summary:hover .mm-records-name { text-decoration: underline;
  text-underline-offset: 3px; }
.mm-records-name { font-weight: 500; }
.mm-records-meta { grid-column: 1; margin-top: 3px; font-size: 11px;
  color: var(--mm-records-muted); }
.mm-records-files { padding: 0 0 12px; font-size: 11px;
  color: var(--mm-records-muted); }
.mm-records-files pre { margin: 8px 0 0; white-space: pre-wrap; overflow-wrap: anywhere; }
</style>
"""


def _session_labels(sessions):
    """Readable UTC timestamps, retaining a suffix only when labels would collide."""
    labels = {}
    for session in sessions:
        stamp = session.removeprefix("session-").partition("-")[0]
        try:
            labels[session] = datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").strftime(
                "%Y-%m-%d · %H:%M:%S UTC"
            )
        except ValueError:
            labels[session] = session
    duplicates = {label for label in labels.values() if list(labels.values()).count(label) > 1}
    return {
        session: f"{label} · {session.rsplit('-', 1)[-1]}" if label in duplicates else label
        for session, label in labels.items()
    }


def _record_label(row):
    parts = [row["episode"], row["status"]]
    if row["score"] is not None:
        parts.append(f"score {row['score']}")
    if row["layout"]:
        parts.append(f"Layout {row['layout']}")
    if row["repeat"] is not None:
        parts.append(f"Repeat {row['repeat']}")
    return " · ".join(parts)


class RecordingsPanel:
    def __init__(self, dashboard):
        self.host = dashboard
        self.root = None
        self.rows = []
        self.checks = {}
        self.handles = []
        self.epoch = 0
        self.busy = False
        self.loading = False
        self.last_receipt = None
        self.session_ids = {}
        gui = dashboard.server.gui
        self.folder = gui.add_folder("Experiment records", expand_by_default=False, order=10000)
        with self.folder:
            self.info = gui.add_markdown("No recordings loaded.", order=0)
            self.status = gui.add_markdown("", order=1)
            self.refresh_button = gui.add_button("Refresh records", color="gray", order=2)
            self.filters = gui.add_folder(
                "Filter records", expand_by_default=False, visible=False, order=3
            )
            with self.filters:
                self.layout = gui.add_dropdown("Recorded layout", ("All layouts",))
                self.session = gui.add_dropdown("Session", ("All sessions",))
            self.management = gui.add_folder(
                "Manage records", expand_by_default=False, visible=False, order=10000
            )
            with self.management:
                gui.add_markdown(
                    "Select completed rollouts to delete and release their repeat slots.", order=0
                )
                self.select_all = gui.add_button("Select all shown", color="gray", order=10000)
                self.clear = gui.add_button("Clear selection", color="gray", order=10001)
                self.delete = gui.add_button(
                    "Delete selected…", color="red", disabled=True, order=10002
                )
                self.undo = gui.add_button("Undo last deletion", color="gray", disabled=True,
                                           visible=False, order=10003)
            self.details = gui.add_folder("Storage details", expand_by_default=False, order=10001)
            with self.details:
                self.output = gui.add_html("")
                gui.add_markdown(
                    "Deleted rollouts move to `_archive/deleted`. Undo restores the last deletion. "
                    "Imported history is read-only."
                )

        @self.refresh_button.on_click
        def refresh(_event):
            with self.host.lock:
                self.refresh()

        @self.layout.on_update
        @self.session.on_update
        def filter_records(event):
            if event.client is not None:
                with self.host.lock:
                    self.render()

        @self.select_all.on_click
        def select(_event):
            with self.host.lock:
                for handle in self.checks.values():
                    if not handle.disabled:
                        handle.value = True
                self.update_enabled()

        @self.clear.on_click
        def clear(_event):
            with self.host.lock:
                for handle in self.checks.values():
                    handle.value = False
                self.update_enabled()

        @self.delete.on_click
        def delete(event):
            with self.host.lock:
                self.confirm_delete(event.client)

        @self.undo.on_click
        def undo(_event):
            with self.host.lock:
                if self.allowed() and self.last_receipt is not None:
                    self.mutate({}, receipt=self.last_receipt)

    def allowed(self):
        host = self.host
        return (
            self.root is not None and host.service_ready and not self.busy and not self.loading
            and not host.episode_active and not host.preparing_rollout
            and not host.new_rollout_requested and host.evaluation_complete
            and not host._saving_evaluation and not host._recovery_pending()
            and not host._pending_experiment_id and not host._pending_initial_pose_id
            and not host.shutdown_phase and not host.home_requested and not host.finish_requested
        )

    def update_enabled(self):
        allowed = self.allowed()
        self.refresh_button.disabled = self.busy or self.loading or self.root is None
        self.layout.disabled = self.session.disabled = self.busy or self.loading
        self.select_all.disabled = self.clear.disabled = not allowed
        self.delete.disabled = not allowed or not any(h.value for h in self.checks.values())
        self.undo.visible = self.last_receipt is not None
        self.undo.disabled = not allowed or self.last_receipt is None
        revisions = {r["relative"]: r["revision"] for r in self.rows}
        for relative, handle in self.checks.items():
            handle.disabled = not allowed or not revisions.get(relative)

    def configure(self, root):
        root = Path(root).resolve() if root else None
        if root != self.root:
            self.last_receipt = None
        self.root = root
        self.refresh()

    def refresh(self):
        if self.busy:
            return
        self.epoch += 1
        epoch, root = self.epoch, self.root
        self.loading = True
        self.status.content = "Reading recordings…"
        self.update_enabled()

        def apply(future):
            with self.host.lock:
                if epoch != self.epoch or not self.host.running:
                    return
                self.loading = False
                try:
                    self.rows = future.result()
                    self.output.content = (
                        '<div style="padding:8px 12px;font-size:12px;line-height:1.55;'
                        'overflow-wrap:anywhere">'
                        'Current Task / Method output:<br>'
                        '<code style="white-space:normal;overflow-wrap:anywhere">'
                        f'{escape(str(root))}</code>'
                        '</div>'
                    ) if root else "Waiting for a Task / Method output directory."
                    labels = _session_labels(
                        sorted({r["session"] for r in self.rows}, reverse=True)
                    )
                    self.session_ids = {label: session for session, label in labels.items()}
                    for widget, options in (
                        (self.layout, ("All layouts", *sorted({r["layout"] for r in self.rows
                                                              if r["layout"]}))),
                        (self.session, ("All sessions", *self.session_ids)),
                    ):
                        old = widget.value
                        widget.options = options
                        widget.value = old if old in options else options[0]
                        widget.visible = len(options) > 2
                    self.filters.visible = self.layout.visible or self.session.visible
                    self.render()
                    self.status.content = ""
                except (OSError, ValueError) as exc:
                    self.rows = []
                    self.render()
                    self.status.content = f"Could not read recordings: {exc}"
                    self.host._report_error("Read recordings", exc)
                self.update_enabled()

        future = self.host._jobs.submit(lambda: list_recordings(root) if root else [])
        future.add_done_callback(apply)

    def render(self):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        self.checks.clear()
        gui = self.host.server.gui
        session = self.session_ids.get(self.session.value)
        rows = [
            r for r in self.rows
            if (self.layout.value == "All layouts" or r["layout"] == self.layout.value)
            and (session is None or r["session"] == session)
        ]
        count = len(self.rows)
        self.info.content = (
            f"{count} recording{'s' if count != 1 else ''}"
            + (f" · {len(rows)} shown" if len(rows) != count else "")
            if count else "No recordings yet. Completed runs will appear here."
        )
        self.management.visible = bool(rows) or self.last_receipt is not None
        labels = _session_labels(list(dict.fromkeys(r["session"] for r in rows)))
        with self.folder:
            for session in dict.fromkeys(r["session"] for r in rows):
                content = (
                    RECORDS_STYLE + '<section class="mm-records-session">'
                    f'<header>{escape(labels[session])}</header>'
                )
                for row in (r for r in rows if r["session"] == session):
                    label = _record_label(row)
                    with self.management:
                        checkbox = gui.add_checkbox(
                            f"{labels[session]} · {label}", False, disabled=True,
                            order=10 + len(self.checks),
                        )
                        self.handles.append(checkbox)
                        self.checks[row["relative"]] = checkbox

                        @checkbox.on_update
                        def changed(_event):
                            with self.host.lock:
                                self.update_enabled()

                    path = self.root / row["relative"]
                    try:
                        names = [
                            p.name + ("/" if p.is_dir() else "")
                            for p in sorted(path.iterdir()) if not p.is_symlink()
                        ]
                    except OSError:
                        names = ["Recording no longer available; refresh."]
                    tree = "\n".join(escape(name) for name in names)
                    metadata = label.removeprefix(row["episode"] + " · ")
                    content += (
                        '<details class="mm-records-entry">'
                        f'<summary aria-label="{escape(label)} · Files and location">'
                        f'<span class="mm-records-name">{escape(row["episode"])}</span>'
                        f'<span class="mm-records-meta">{escape(metadata)}</span></summary>'
                        '<div class="mm-records-files">Files and location'
                        f'<div>{escape(str(path))}</div><pre>{tree}</pre>'
                        '</div></details>'
                    )
                self.handles.append(
                    gui.add_html(content + '</section>', order=10 + len(self.handles))
                )
        self.update_enabled()

    def confirm_delete(self, client):
        if client is None or not self.allowed():
            return
        selected = {
            row["relative"]: row["revision"] for row in self.rows
            if row["relative"] in self.checks and self.checks[row["relative"]].value
            and row["revision"] is not None
        }
        if not selected:
            return
        root = self.root
        with client.gui.add_modal("Delete recordings and allow retest?") as modal:
            client.gui.add_markdown(
                f"**{len(selected)} complete rollout(s)**, "
                "including scores, videos and trajectories, "
                "will move to the recoverable archive. Their scored repeat slots will be released."
                "\n\n```text\n" + str(root) + "\n" + "\n".join(selected) + "\n```"
            )
            reason = client.gui.add_text("Reason", initial_value="Operator requested retest")
            confirm = client.gui.add_button("Delete these records", color="red")
            cancel = client.gui.add_button("Cancel")
            cancel.on_click(lambda _event: modal.close())

            @confirm.on_click
            def confirmed(_event):
                with self.host.lock:
                    if self.root != root or not self.allowed():
                        self.status.content = (
                            "State changed. Return to idle, refresh and select again."
                        )
                        modal.close()
                        return
                    if not reason.value.strip():
                        return
                    modal.close()
                    self.mutate(selected, reason=reason.value)

    def mutate(self, selected, *, reason="", receipt=None):
        if not self.allowed():
            return
        root = self.root
        self.busy = True
        self.status.content = (
            "Restoring recordings…" if receipt else "Moving recordings to archive…"
        )
        self.host._update_prepare_enabled()
        self.update_enabled()

        def work():
            if receipt:
                restore_recordings(root, receipt)
                result = None
            else:
                result = trash_recordings(root, selected, reason=reason)
            # The index is derived; never report a successful move as failed
            # just because its inventory snapshot could not be refreshed.
            from manimux.collections import write_inventory
            warning = ""
            for parent in (root, *list(root.parents)[:3]):
                if (parent / "recordings-index-v2.json").is_file():
                    try:
                        write_inventory(parent)
                    except (OSError, ValueError, TypeError) as exc:
                        warning = f"Records changed, but inventory refresh failed: {exc}"
                    break
            return result, warning

        def apply(future):
            with self.host.lock:
                self.busy = False
                try:
                    self.last_receipt, warning = future.result()
                    removed = {root / p for p in selected}
                    if self.host._last_evaluated_episode in removed:
                        self.host._last_evaluated_episode = None
                        self.host.last_evaluation_btn.visible = False
                    if self.host.current_episode_dir in removed:
                        self.host.current_episode_dir = None
                        self.host.episode_finalized = False
                        self.host.episode_path.value = "Deleted for retest"
                    self.host._refresh_repeat_history()
                    if warning:
                        self.host._report_error("Refresh recording index", ValueError(warning))
                except (OSError, ValueError, TypeError) as exc:
                    self.status.content = f"Recording operation failed: {exc}"
                    self.host._report_error("Manage recordings", exc)
                self.host._update_prepare_enabled()
                self.update_enabled()

        self.host._jobs.submit(work).add_done_callback(apply)
