"""End-effector control contract, independent of robot and vendor SDKs.

Tool geometry and flange/TCP transforms remain in ``manimux.kinematics``.
Sensors mounted on a tool use the separate sensor interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

StateT = TypeVar("StateT")
CommandT = TypeVar("CommandT")


class EndEffectorBase(ABC, Generic[StateT, CommandT]):
    """Control an end effector such as a gripper, suction tool or hand.

    Capability-specific interfaces bind the state and command types, for example
    ``EndEffectorBase[GripperState, GripperCommand]``. Implementations of the same
    capability should share these types and their units and semantics; inheriting
    this base alone does not make different tool capabilities interchangeable.

    The base assumes no joint count, normalized aperture or open/close behavior.
    Calibration and homing are optional capabilities rather than universal tool
    operations. Subclasses may own a connection or borrow a shared SDK session;
    constructing this base never connects hardware or starts control.
    """

    @abstractmethod
    def connect(self) -> None:
        """Acquire or attach to the configured connection and prepare the device.

        Document enabling, calibration or motion side effects and whether repeated
        calls are supported. Do not silently clear device faults.
        """
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> StateT:
        """Read feedback using the capability's state type and units.

        Distinguish measured feedback from requested targets. Document timestamps,
        caching and how unavailable, stale or invalid feedback is reported.
        """
        raise NotImplementedError

    @abstractmethod
    def send_command(self, command: CommandT) -> None:
        """Submit a command supported by the device's declared capability.

        Validate modes, values and limits before actuation. Unsupported commands
        must be rejected explicitly. Document blocking behavior; returning does
        not by itself guarantee that a target or grasp has been achieved.
        """
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """Request the device's documented stop behavior.

        Specify whether this holds a target, disables actuation or changes suction,
        and how control can resume. Stopping does not universally mean opening a
        gripper or releasing a grasp, nor does it imply a hardware emergency stop.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release owned resources, including after a partial connection failure.

        Support repeated calls and report incomplete cleanup. Document effects on
        actuation. Do not close an SDK session owned by the robot, a sensor or the
        station that assembled the devices.
        """
        raise NotImplementedError
