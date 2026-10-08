"""Shared offline model loading for arms with an integrated scalar gripper."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

from numpy.typing import ArrayLike

from manimux.embodiments.arm.base import ArmBase, ArmController, ArmModel
from manimux.kinematics.urdf_manipulator import URDFManipulatorKinematics


class _ChannelController(Protocol):
    channel: str


class IntegratedArm(ArmBase):
    """Six joint angles and normalized opening with an independently owned controller."""

    num_joints = 7

    def __init__(self, controller: ArmController, *, kinematics: URDFManipulatorKinematics) -> None:
        self._controller = controller
        self._kinematics = kinematics

    @property
    def controller(self) -> ArmController:
        return self._controller

    @property
    def channel(self) -> str:
        return cast(_ChannelController, self._controller).channel

    @property
    def kinematics(self) -> URDFManipulatorKinematics:
        return self._kinematics


class IntegratedURDFArm(IntegratedArm):
    """Load the subclass's model with joint1..6 and its nominal visual fingers."""

    model_path: Path
    visual_gripper_joints: tuple[str, ...] = ()
    visual_gripper_closed: tuple[float, ...] = ()
    visual_gripper_open: tuple[float, ...] = ()

    @classmethod
    def load_model(
        cls, *, tcp_frame: str = "link6", joint_limits: ArrayLike | None = None
    ) -> ArmModel:
        model = URDFManipulatorKinematics(
            cls.model_path,
            base_frame="base_link",
            tcp_frame=tcp_frame,
            joint_names=[f"joint{i}" for i in range(1, 7)],
            gripper_joints=cls.visual_gripper_joints,
            gripper_closed=cls.visual_gripper_closed,
            gripper_open=cls.visual_gripper_open,
            joint_limits=joint_limits,
        )
        return ArmModel(
            model,
            model.coordinates,
            cls.model_path,
            None,
            base_frame=model.base_frame,
            visual_mapping=model.visual_configuration,
        )
