"""Local FCI ownership, shared by executing and receive-only SDK sessions."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


class FCIConnectionLease:
    """Refuse another local ManiMux connection before constructing either SDK.

    A receive-only SDK connection can disconnect the active FCI controller. This
    lease covers the configured address on this host; raw SDK clients and clients
    on other hosts must still coordinate with the runtime owner.
    """

    def __init__(self, address: str):
        key = hashlib.sha256(address.strip().lower().encode()).hexdigest()[:24]
        self.path = Path(tempfile.gettempdir()) / f"manimux-franka-fci-{key}.lock"
        self._handle = None

    def acquire(self, *, execute: bool):
        import fcntl

        if self._handle is not None:
            return
        flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW
        handle = os.fdopen(os.open(self.path, flags, 0o600), "r+", encoding="utf-8")
        try:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                owner = handle.read(1024).strip() or "owner details unavailable"
                raise RuntimeError(
                    f"Franka FCI connection already owned ({owner}). "
                    "Use the active runtime's feedback; finish its session before "
                    "opening another SDK connection, including read-only diagnostics."
                ) from exc
            handle.seek(0)
            handle.truncate()
            json.dump({"pid": os.getpid(), "mode": "execute" if execute else "read_only"}, handle)
            handle.flush()
        except BaseException:
            handle.close()
            raise
        self._handle = handle

    def release(self):
        if self._handle is not None:
            # Never unlink: concurrent waiters must continue to lock the same inode.
            self._handle.close()
            self._handle = None
