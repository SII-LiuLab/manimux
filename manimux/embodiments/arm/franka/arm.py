"""Bare Franka arm component, independent of its gripper and policy source."""

from manimux.embodiments.arm.base import ArmBase, ArmModel
from manimux.kinematics.base import KinematicCoordinate

from .kinematics import FrankaKinematics


class FrankaArm(ArmBase):
    num_joints = 7

    def __init__(self, controller, *, kinematics):
        self._controller = controller
        self._kinematics = kinematics

    @classmethod
    def load_model(cls, **options):
        model = FrankaKinematics(**options)
        return ArmModel(
            model,
            tuple(KinematicCoordinate(f"joint{i}", "rad") for i in range(1, 8)),
            model.path,
            "link8",
            base_frame="link0",
        )

    @property
    def controller(self):
        return self._controller

    @property
    def channel(self):
        return self._controller.channel

    @property
    def kinematics(self):
        return self._kinematics
