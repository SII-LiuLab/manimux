"""RoboTwin ALOHA follower-arm model for offline viewing and replay."""

from pathlib import Path

from manimux.embodiments.arm._urdf import IntegratedArm
from manimux.embodiments.arm.base import ArmModel

from .kinematics import AlohaAgilexKinematics


class AlohaAgilexArm(IntegratedArm):
    """Integrated seven-coordinate model; no hardware controller is provided.

    A future driver must supply ArmController with this exact coordinate contract.
    Offline model loading neither constructs nor imports an ARX/PiPER SDK session.
    """

    @classmethod
    def load_model(cls, *, side: str) -> ArmModel:
        if side not in {"left", "right"}:
            raise ValueError("ALOHA side must be left or right")
        prefix = "fl" if side == "left" else "fr"
        path = Path(__file__).parent / "assets/robotwin" / f"{side}_arm.urdf"
        kinematics = AlohaAgilexKinematics(path, prefix=prefix)
        return ArmModel(
            kinematics,
            kinematics.coordinates,
            path,
            None,
            base_frame=kinematics.base_frame,
            visual_mapping=kinematics.visual_configuration,
        )
