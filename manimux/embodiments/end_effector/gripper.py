"""Shared position-control contract for one-coordinate grippers."""

import math
from dataclasses import dataclass

from manimux.embodiments.end_effector.base import EndEffectorBase


def _opening(value: float) -> None:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("opening must be finite and in [0, 1]")


@dataclass(frozen=True, slots=True)
class GripperState:
    """Measured normalized travel, 0 closed and 1 open (not pad distance).

    timestamp is seconds in the station Clock's monotonic domain. Drivers must
    state whether it marks acquisition or host receipt. Cached states retain
    their original timestamp; unavailable/invalid feedback raises an exception.
    Optional target_opening is the device's echoed target, not measured travel.
    contact identifies a device-reported stop during opening/closing; None means
    no confirmed contact. Neither field establishes object identity or retention.
    """

    opening: float
    timestamp: float
    target_opening: float | None = None
    contact: str | None = None

    def __post_init__(self) -> None:
        _opening(self.opening)
        if not math.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")
        if self.target_opening is not None:
            _opening(self.target_opening)
        if self.contact not in (None, "opening", "closing"):
            raise ValueError("contact must be opening, closing or unavailable")


@dataclass(frozen=True, slots=True)
class GripperCommand:
    """Absolute normalized travel target; submission is not motion completion."""

    opening: float

    def __post_init__(self) -> None:
        _opening(self.opening)


class GripperBase(EndEffectorBase[GripperState, GripperCommand]):
    """Position-controlled gripper with a common state/command representation.

    connect must not initiate homing, calibration or opening/closing. Drivers
    document when motors become enabled and their stop/resume semantics. Never
    replace measured state with a command target or silently clear faults.
    Robot assembly maps state.opening to the matching geometric coordinate;
    sensors and tool geometry remain separate components.
    """
