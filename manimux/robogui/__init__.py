from pathlib import Path

from manimux.robogui.communication import PolicyPlan, RobotSnapshot, RuntimeEvent
from manimux.robogui.publisher import RoboGUIBridge, RoboGUIClient, RoboGUIControl

__all__ = [
    "PolicyPlan",
    "RobotSnapshot",
    "RuntimeEvent",
    "RoboGUIBridge",
    "RoboGUIClient",
    "RoboGUIControl",
]


def robogui_parameters(**options) -> dict:
    """补齐显示发布参数；不启动 RoboGUI 服务。"""

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
    from .control import control_parameters

    values["control"] = control_parameters(**values.get("control", {}))
    return values
