"""Seven arm joints, without an implicit gripper coordinate."""

from pathlib import Path

from manimux.embodiments.arm.base import ArmBase, ArmModel
from manimux.kinematics.base import KinematicCoordinate

from .kinematics import FlexivKinematics


class FlexivArm(ArmBase):
    num_joints = 7
    model_path = Path(__file__).parent / "model.urdf"

    def __init__(self, controller, *, kinematics):
        self._controller = controller
        self._kinematics = kinematics

    @property
    def controller(self):
        return self._controller

    @property
    def channel(self):
        return self._controller.channel

    @property
    def kinematics(self):
        return self._kinematics

    @classmethod
    def load_model(cls, *, urdf_path=None, ik_max_joint_step_rad=None, ik_joint_motion_cost=0.0):
        path = Path(urdf_path).expanduser().resolve() if urdf_path else cls.model_path
        kin = FlexivKinematics(
            path,
            ik_max_joint_step_rad=ik_max_joint_step_rad,
            ik_joint_motion_cost=ik_joint_motion_cost,
        )
        return ArmModel(
            kin,
            tuple(KinematicCoordinate(f"joint{i}", "rad") for i in range(1, 8)),
            path,
            "flange",
            base_frame="base_link",
        )
