"""Mechanical-arm control interfaces with explicit shared-session ownership."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from manimux.kinematics.base import (
    FlangeKinematicsBase,
    FloatArray,
    IKResult,
    KinematicCoordinate,
    ManipulatorKinematicsBase,
)


@dataclass(frozen=True, slots=True)
class ArmModel:
    """Component kinematics and visual assets, for a bare arm or an arm with a built-in gripper.

    Complete TCP models implement ManipulatorKinematicsBase; bare arms with separate tools
    use FlangeKinematicsBase. visual_mapping converts control coordinates to URDF coordinates,
    for example expanding one normalized gripper value into two finger joints; it does not change control commands.
    """

    kinematics: FlangeKinematicsBase | ManipulatorKinematicsBase
    coordinates: tuple[KinematicCoordinate, ...]
    urdf_path: Path
    flange_link: str | None
    base_frame: str = "arm_base"
    visual_mapping: Callable[[FloatArray], FloatArray] | None = None

    def visual_configuration(self, configuration: FloatArray) -> FloatArray:
        if self.visual_mapping is not None:
            return self.visual_mapping(configuration)
        return np.array(configuration, dtype=np.float64, copy=True)


@dataclass(frozen=True, slots=True)
class ArmState:
    """Measured arm joints and the original host-receipt timestamp."""

    joints: FloatArray
    monotonic_ns: int
    sequence: int

    def __post_init__(self) -> None:
        joints = np.array(self.joints, dtype=np.float64, copy=True)
        if joints.ndim != 1 or not joints.size or not np.isfinite(joints).all():
            raise ValueError("arm feedback must contain a finite joint vector")
        joints.setflags(write=False)
        object.__setattr__(self, "joints", joints)


class ArmController(ABC):
    """Own one hardware session, possibly shared by several arm components.

    Robot assembly calls lifecycle methods once per controller and batches all
    targets sharing it. An implementation declares its own dispatch guarantees.
    Controllers know no end effectors, policies or whole-robot TCP models.
    """

    @abstractmethod
    def connect(self) -> None:
        """Connect without implicit homing or fault recovery."""
        raise NotImplementedError

    @abstractmethod
    def get_states(self) -> Mapping[str, ArmState]:
        """Return measured states indexed by controller channel."""
        raise NotImplementedError

    def validate_commands(self, targets: Mapping[str, FloatArray]) -> None:
        """Optional device constraint hook; defaults to the coordinate vector resolved by the assembled robot.

        Controllers with device-specific constraints may override this method. The base interface does not revalidate
        configuration, types or dimensions, and does not access hardware.
        """
        return None

    @abstractmethod
    def send_commands(self, targets: Mapping[str, FloatArray]) -> None:
        """Submit a validated batch; returning does not imply arrival."""
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """Stop owned channels, attempting every channel even on failure."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release owned resources; failed cleanup must remain retryable."""
        raise NotImplementedError


class ArmBase(ABC):
    """An independently connectable arm component that may include a built-in gripper.

    num_joints is the complete coordinate count read/submitted together: seven for YAM, including the gripper.
    The robot assembles separately mounted tools; end_effector: null preserves all built-in component capabilities.

    The controller owns connection lifecycle. Calling connect/stop/close on a
    component affects that shared session, so an assembled robot owns these
    operations and deduplicates them. FK/IK never reads or commands hardware.
    """

    @classmethod
    def load_model(cls, **options) -> ArmModel:
        """Load component geometry without constructing a hardware connection."""
        raise NotImplementedError(f"{cls.__name__} does not provide an offline model")

    @property
    @abstractmethod
    def controller(self) -> ArmController:
        raise NotImplementedError

    @property
    @abstractmethod
    def channel(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_joints(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def kinematics(self) -> FlangeKinematicsBase | ManipulatorKinematicsBase:
        """The component flange or complete TCP solver; the robot manages scene placement."""
        raise NotImplementedError

    def connect(self) -> None:
        self.controller.connect()

    def get_state(self) -> ArmState:
        return self.controller.get_states()[self.channel]

    def send_command(self, joints: FloatArray) -> None:
        self.controller.send_commands({self.channel: joints})

    def stop(self) -> None:
        self.controller.stop()

    def close(self) -> None:
        self.controller.close()

    def fk(self, joints: FloatArray) -> FloatArray:
        if isinstance(self.kinematics, ManipulatorKinematicsBase):
            return self.kinematics.fk(joints)
        return self.kinematics.fk_flange(joints)

    def ik(self, target: FloatArray, seed: FloatArray, *, fixed_coordinates=None) -> IKResult:
        if isinstance(self.kinematics, ManipulatorKinematicsBase):
            return self.kinematics.ik(target, seed, fixed_coordinates=fixed_coordinates or {})
        return self.kinematics.ik_flange(target, seed)
