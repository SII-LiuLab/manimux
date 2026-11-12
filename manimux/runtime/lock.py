"""Process lock preventing multiple runtimes from controlling one robot."""

from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import TextIO


class RuntimeLockError(RuntimeError):
    """Raised when another ManiMux process already owns the robot runtime."""


class RuntimeInstanceLock:
    """Hold an OS-backed exclusive lock for one local robot identity."""

    def __init__(
        self,
        identity: str,
        *,
        mode: str,
        config_path: Path,
        lock_dir: Path | None = None,
    ) -> None:
        safe_identity = re.sub(r"[^A-Za-z0-9_.-]+", "-", identity.strip()) or "robot"
        root = lock_dir or Path(tempfile.gettempdir())
        self.path = root / f"manimux-runtime-{safe_identity}.lock"
        self._identity = identity
        self._mode = mode
        self._config_path = config_path.expanduser().resolve()
        self._handle: TextIO | None = None

    def __enter__(self) -> RuntimeInstanceLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            raw_owner = handle.read().strip()
            handle.close()
            try:
                owner = json.loads(raw_owner)
                owner_summary = (
                    f"pid={owner['pid']}, mode={owner['mode']}, config={owner['config']}"
                )
            except (json.JSONDecodeError, KeyError, TypeError):
                owner_summary = raw_owner or "owner details unavailable"
            raise RuntimeLockError(
                f"another ManiMux runtime already owns robot {self._identity!r} ({owner_summary})"
            ) from exc
        handle.seek(0)
        handle.truncate()
        json.dump(
            {
                "pid": os.getpid(),
                "mode": self._mode,
                "config": str(self._config_path),
                "acquired_at": datetime.now(UTC).isoformat(),
            },
            handle,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        self._handle = handle
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        if self._handle is None:
            return
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None
