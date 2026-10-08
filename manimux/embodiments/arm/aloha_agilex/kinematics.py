"""RoboTwin follower-arm geometry and its normalized visual gripper mapping."""

from pathlib import Path

from manimux.kinematics.urdf_manipulator import URDFManipulatorKinematics


class AlohaAgilexKinematics(URDFManipulatorKinematics):
    """Simulation bounds and recipe-specific fingers; not hardware calibration."""

    def __init__(self, path: Path, *, prefix: str) -> None:
        super().__init__(
            path,
            base_frame=f"{prefix}_base_link",
            tcp_frame=f"{prefix}_tcp",
            joint_names=[f"{prefix}_joint{i}" for i in range(1, 7)],
            gripper_joints=[f"{prefix}_joint7", f"{prefix}_joint8"],
            gripper_closed=[-0.01, -0.01],
            gripper_open=[0.045, 0.045],
        )
