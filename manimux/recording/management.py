"""Reversible removal of finalized rollouts from a method's study history."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

_EPISODE = re.compile(r"rollout-\d+")


def _checked_episode(root: Path, relative: str) -> Path:
    part = Path(relative)
    if (
        part.is_absolute() or len(part.parts) != 2
        or not part.parts[0].startswith("session-")
        or not _EPISODE.fullmatch(part.parts[1])
    ):
        raise ValueError("Select a finalized session/rollout entry, not an arbitrary path")
    path = root / part
    if path.parent.is_symlink() or path.is_symlink():
        raise ValueError("Linked recording directories cannot be deleted")
    if path.resolve().parent.parent != root.resolve():
        raise ValueError("Recording is outside the selected method")
    if not path.is_dir() or not (path / "result.json").is_file():
        raise ValueError(f"Recording is missing or still being saved: {relative}")
    for name in ("meta.json", "result.json", "evaluation/human-label.json"):
        candidate = path / name
        if candidate.is_symlink() or candidate.parent.is_symlink():
            raise ValueError("Linked recording metadata cannot be deleted")
    return path


def record_revision(path: Path) -> str:
    """Detect a changed evaluation/result between selection and confirmation."""
    digest = hashlib.sha256()
    for name in ("meta.json", "result.json", "evaluation/human-label.json"):
        file = path / name
        digest.update(name.encode())
        digest.update(file.read_bytes() if file.exists() else b"<missing>")
    return digest.hexdigest()


def list_recordings(root: Path) -> list[dict]:
    """Read one method, without traversing videos, arrays, archives or aliases."""
    rows = []
    if not root.is_dir():
        return rows
    for session in sorted(root.glob("session-*"), reverse=True):
        if session.is_symlink() or not session.is_dir():
            continue
        for path in sorted(session.glob("rollout-*"), reverse=True):
            if path.is_symlink() or not path.is_dir():
                continue
            row = {
                "relative": str(path.relative_to(root)),
                "session": session.name, "episode": path.name,
                "layout": "", "repeat": None, "score": None,
                "status": "Saving / incomplete", "revision": None,
            }
            try:
                meta = json.loads((path / "meta.json").read_text())
                row.update(layout=meta.get("layout_id") or "", repeat=meta.get("repeat_id"),
                           task=meta.get("task", ""), method=meta.get("runtime", ""))
                if not _EPISODE.fullmatch(path.name) or not (path / "result.json").is_file():
                    rows.append(row)
                    continue
                _checked_episode(root, row["relative"])
                result = json.loads((path / "result.json").read_text())
                label_path = path / "evaluation/human-label.json"
                label = json.loads(label_path.read_text()) if label_path.exists() else {}
                row.update(
                    status=label.get("task_result") or (
                        "Preparation cancelled" if meta.get("preparation_cancelled") else
                        "Unscored" if meta.get("experiment_mode") else "Unscored run"
                    ),
                    score=label.get("score"),
                    steps=result.get("steps"),
                    revision=record_revision(path),
                )
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                row["status"] = f"Unreadable: {exc}"
            rows.append(row)
    return rows


def trash_recordings(root: Path, selected: dict[str, str], *, reason: str) -> Path:
    """Move a confirmed batch atomically per rollout, rolling back on an error.

    Callers serialize this with Prepare/evaluation. Partial directories are never
    eligible. A small .deleted marker reserves each rollout ID for later restore.
    The existing inventory and repeat readers exclude _archive.
    """
    root = root.resolve(strict=True)
    if not selected or not reason.strip():
        raise ValueError("Select at least one recording and provide a deletion reason")
    paths = []
    for relative, expected in selected.items():
        path = _checked_episode(root, relative)
        if record_revision(path) != expected:
            raise ValueError(f"Recording changed; refresh and select it again: {relative}")
        if path.with_suffix(".deleted").exists():
            raise ValueError(f"Deletion marker already exists: {relative}")
        paths.append(path)
    archive = root / "_archive" / "deleted"
    if any(p.is_symlink() for p in (root / "_archive", archive)):
        raise ValueError("Linked archive directories are not supported")
    batch = archive / uuid.uuid4().hex
    batch.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema": "deleted-recordings-v1", "root": str(root),
        "deleted_at": datetime.now(UTC).isoformat(), "reason": reason.strip(),
        "records": list(selected), "revisions": selected,
    }
    (batch / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    moved, markers = [], []
    try:
        for path in paths:
            marker = path.with_suffix(".deleted")
            with marker.open("x") as stream:
                json.dump({"receipt": str(batch / "receipt.json")}, stream)
            markers.append(marker)
            target = batch / path.relative_to(root)
            target.parent.mkdir(exist_ok=True)
            path.rename(target)
            moved.append((path, target))
    except Exception:
        for source, target in reversed(moved):
            target.rename(source)
        for marker in markers:
            marker.unlink(missing_ok=True)
        raise
    return batch / "receipt.json"


def restore_recordings(root: Path, receipt_path: Path) -> None:
    root = root.resolve(strict=True)
    receipt_path = receipt_path.resolve(strict=True)
    archive = root / "_archive" / "deleted"
    if receipt_path.parent.parent != archive or receipt_path.name != "receipt.json":
        raise ValueError("Receipt is outside this method's deleted recordings")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("root") != str(root) or receipt.get("restored_at"):
        raise ValueError("Receipt belongs to another method or was already restored")
    paths = []
    for relative in receipt["records"]:
        source = _checked_episode(receipt_path.parent, relative)
        target = root / relative
        marker = target.with_suffix(".deleted")
        if target.exists() or target.with_suffix(".partial").exists() or target.parent.is_symlink():
            raise ValueError(f"Restore would overwrite a recording: {relative}")
        if (
            not marker.is_file()
            or json.loads(marker.read_text()).get("receipt") != str(receipt_path)
        ):
            raise ValueError(f"Deletion marker no longer matches: {relative}")
        paths.append((source, target, marker))
    moved = []
    try:
        for source, target, _ in paths:
            source.rename(target)
            moved.append((source, target))
    except Exception:
        for source, target in reversed(moved):
            target.rename(source)
        raise
    for _, _, marker in paths:
        marker.unlink()
    receipt["restored_at"] = datetime.now(UTC).isoformat()
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
