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
    """组件自己的运动学和显示资源，可为裸臂，也可包含内置夹爪。

    完整 TCP 模型直接实现 ManipulatorKinematicsBase；可挂独立工具的裸臂
    使用 FlangeKinematicsBase。visual_mapping 将控制坐标转换为 URDF 坐标，
    例如将一个归一化夹爪值展开成两个指尖关节；它不改变控制命令。
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
        """可选的设备约束钩子；默认直接使用整机已解析的坐标向量。

        有设备专用约束的 controller 可覆盖此方法。基础接口不再重复校验
        配置、类型和维度，也不接触硬件。
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
    """一个可独立接入的臂组件，允许包含设备自带的夹爪。

    num_joints 表示该组件一次读取/提交的完整坐标数：YAM 为 7，包含夹爪。
    独立挂载的末端由整机额外装配；end_effector: null 保留组件自身的全部能力。

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
        """组件的法兰求解器或完整 TCP 求解器；场景安装位置由整机管理。"""
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
