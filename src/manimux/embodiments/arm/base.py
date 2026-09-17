"""Mechanical-arm control interfaces with explicit shared-session ownership."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from manimux.kinematics.base import FlangeKinematicsBase, FloatArray, IKResult, KinematicCoordinate


@dataclass(frozen=True, slots=True)
class ArmModel:
    """One component's official flange model and corresponding CAD resources."""

    kinematics: FlangeKinematicsBase
    coordinates: tuple[KinematicCoordinate, ...]
    urdf_path: Path
    flange_link: str
    base_frame: str = "arm_base"


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

    @abstractmethod
    def validate_commands(self, targets: Mapping[str, FloatArray]) -> None:
        """Validate targets without contacting hardware."""
        raise NotImplementedError

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
    """One mechanical arm, excluding its mounted end effector.

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
    def kinematics(self) -> FlangeKinematicsBase:
        """The component's official flange solver, excluding installation offsets."""
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
        return self.kinematics.fk_flange(joints)

    def ik(self, target_flange: FloatArray, seed_joints: FloatArray) -> IKResult:
        return self.kinematics.ik_flange(target_flange, seed_joints)
