"""Shared Ctrl-C handling for blocking hardware moves."""

from __future__ import annotations

import contextlib
import logging
import signal
import threading
from collections.abc import Iterator
from typing import Any


@contextlib.contextmanager
def finish_move_before_interrupt(what: str, log: logging.Logger) -> Iterator[list[int]]:
    """Let a blocking arm move finish before Ctrl-C takes effect.

    A SIGINT otherwise lands as ``KeyboardInterrupt`` inside the move's sleep:
    the move stops with the arm mid-air and the shutdown that follows may drop
    it. Defer the first interrupt until the move returns. A second Ctrl-C
    restores the default handler and aborts immediately, so a genuinely stuck
    move is still escapable. The yielded list is non-empty once an interrupt was
    deferred.
    """
    if threading.current_thread() is not threading.main_thread():
        yield []
        return

    pending: list[int] = []

    def _handler(signum: int, frame: Any) -> None:
        if pending:
            signal.signal(signal.SIGINT, previous)
            raise KeyboardInterrupt
        pending.append(signum)
        log.warning(
            "Ctrl-C received; finishing %s first (press Ctrl-C again to abort now).",
            what,
        )

    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _handler)
    try:
        yield pending
    finally:
        with contextlib.suppress(ValueError, TypeError):
            signal.signal(signal.SIGINT, previous)
