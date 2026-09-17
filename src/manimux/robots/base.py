"""Robot control contracts, independent of vendor SDKs.

New implementations inherit :class:`RobotBase`. Runtime consumers can keep
using :class:`RobotDriver`, which also accepts existing structural implementations
without requiring them to inherit the base class.
"""

from __future__ import annotations

from typing import Protocol

from manimux.embodiments.robot.base import RobotBase as RobotBase
from manimux.types import RobotCommand, RobotState


class RobotDriver(Protocol):
    """Structural interface shared by existing drivers and ``RobotBase``."""

    def connect(self) -> None: ...

    def get_state(self) -> RobotState: ...

    def send_command(self, command: RobotCommand) -> None: ...

    def home(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...
