"""ARX X5 (2023) component; SDK imports belong to its controller process."""

from pathlib import Path

from manimux.embodiments.arm._urdf import IntegratedURDFArm


class ArxX5Arm(IntegratedURDFArm):
    model_path = Path(__file__).parent / "model.urdf"
