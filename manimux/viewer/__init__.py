from pathlib import Path

from manimux.viewer.communication import PolicyPlan, RobotSnapshot, RuntimeEvent
from manimux.viewer.publisher import ViewerBridge, ViewerClient, ViewerControl

__all__ = [
    "PolicyPlan",
    "RobotSnapshot",
    "RuntimeEvent",
    "ViewerBridge",
    "ViewerClient",
    "ViewerControl",
]


def viewer_parameters(**options) -> dict:
    """补齐显示发布参数；不启动 Viewer 服务。"""

    values = {
        "enabled": False,
        "robot": "",
        "policy_label": "",
        "camera_hz": 5.0,
        "tianji_teleop_root": None,
        **options,
    }
    if values.get("tianji_teleop_root") is not None:
        values["tianji_teleop_root"] = Path(values["tianji_teleop_root"])
    return values
