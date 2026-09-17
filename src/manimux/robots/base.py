"""Robot control contracts, independent of vendor SDKs.

New implementations inherit :class:`RobotBase`. Runtime consumers can keep
using :class:`RobotDriver`, which also accepts existing structural implementations
without requiring them to inherit the base class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from manimux.types import RobotCommand, RobotState


class RobotDriver(Protocol):
    """Structural interface shared by existing drivers and ``RobotBase``."""

    def connect(self) -> None: ...

    def get_state(self) -> RobotState: ...

    def send_command(self, command: RobotCommand) -> None: ...

    def home(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


class RobotBase(ABC):
    """Control a configured robot, including its configured end effectors.

    Group names, dimensions, ordering, units and gripper conventions follow the
    embodiment configuration and must match between states and commands. No
    particular number of arms or presence of a gripper is assumed.

    Subclasses own hardware communication and must implement every operation.
    Camera acquisition, FK/IK and trajectory scheduling have separate interfaces;
    this base provides no SDK loading, implicit connection or motion behavior.
    """

    @abstractmethod
    def connect(self) -> None:
        """Acquire the configured hardware connections and prepare control.

        Document any enabling or configured startup motion, and whether repeated
        calls are supported. Connecting must not silently clear controller faults.
        """
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> RobotState:
        """Read feedback in the configured group layout.

        Return measured feedback rather than commanded targets. Document caching,
        timestamp/sequence semantics and how stale or unavailable feedback is
        reported; this contract does not imply simultaneous sampling of groups.
        """
        raise NotImplementedError

    @abstractmethod
    def send_command(self, command: RobotCommand) -> None:
        """Submit grouped position targets in the configured representation.

        Validate supported groups, dimensions and values before issuing motion.
        Document partial-command handling, blocking behavior and any limitations
        on coordinated dispatch. Returning does not guarantee target arrival.
        """
        raise NotImplementedError

    @abstractmethod
    def home(self) -> None:
        """Execute the embodiment's configured return-to-home operation.

        Document the destination, gripper behavior and completion semantics. This
        is not implicitly encoder calibration or fault recovery. An unsupported
        operation must raise ``NotImplementedError`` rather than silently succeed.
        """
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """Request the embodiment's stop behavior for its controlled components.

        Document whether this holds position, decelerates or disables actuation,
        and what is required before commands can resume. This interface does not
        imply a hardware emergency stop.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release owned connections, threads and other hardware resources.

        Support cleanup after a partially completed connection and repeated calls.
        Document effects on actuation and report incomplete cleanup. Resources
        borrowed from another owner must not be closed by this driver.
        """
        raise NotImplementedError
