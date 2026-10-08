"""Standard PiPER component; pyAgxArm is imported only when connecting."""

from pathlib import Path

from manimux.embodiments.arm._urdf import IntegratedURDFArm


class PiperArm(IntegratedURDFArm):
    model_path = Path(__file__).parent / "model.urdf"
    visual_gripper_joints = ("joint7", "joint8")
    visual_gripper_closed = (0.0, 0.0)
    visual_gripper_open = (0.035, -0.035)
