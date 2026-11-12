from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    def now_ns(self) -> int: ...

    def sleep_until_ns(self, target_ns: int) -> None: ...


class SystemClock:
    def now_ns(self) -> int:
        return time.monotonic_ns()

    def sleep_until_ns(self, target_ns: int) -> None:
        remaining_ns = target_ns - time.monotonic_ns()
        if remaining_ns > 0:
            time.sleep(remaining_ns / 1_000_000_000)
