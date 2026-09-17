"""Compatibility identity and offline imports for the Tianji directory migration."""

import importlib
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("robots.tianji.tianji", "embodiments.robot.tianji_taccap.robot"),
        ("robots.tianji.kinematics", "embodiments.arm.tianji.kinematics"),
        ("robots.tianji.tianji_taccap_kinematics", "embodiments.robot.tianji_taccap.kinematics"),
        ("robots.tianji.vendor.marvin.fx_kine", "embodiments.arm.tianji.sdk.marvin.fx_kine"),
        ("robots.tianji.vendor.marvin.fx_robot", "embodiments.arm.tianji.sdk.marvin.fx_robot"),
        ("end_effectors.base", "embodiments.end_effector.base"),
        ("end_effectors.gripper", "embodiments.end_effector.gripper"),
        ("end_effectors.taccap.gripper", "embodiments.end_effector.taccap.end_effector"),
        ("end_effectors.taccap.geometry", "embodiments.end_effector.taccap.geometry"),
        ("sensors.taccap.camera", "embodiments.sensor.taccap.sensor"),
    ],
)
def test_old_modules_share_the_canonical_implementation(old, new):
    # Distinct module objects could duplicate the process-global control/IK locks.
    assert importlib.import_module(f"manimux.{old}") is importlib.import_module(f"manimux.{new}")


def test_base_classes_keep_identity_and_explicit_inheritance():
    from manimux.embodiments.end_effector import EndEffectorBase, GripperBase
    from manimux.embodiments.end_effector.taccap import TacCapGripper
    from manimux.embodiments.robot import RobotBase
    from manimux.embodiments.robot.tianji_taccap import TianjiTaccapRobot
    from manimux.end_effectors import EndEffectorBase as OldEndEffectorBase
    from manimux.robots.base import RobotBase as OldRobotBase
    from manimux.robots.tianji import TianjiRobot as OldTianjiRobot

    assert OldRobotBase is RobotBase
    assert OldTianjiRobot is TianjiTaccapRobot
    assert OldEndEffectorBase is EndEffectorBase
    assert issubclass(TianjiTaccapRobot, RobotBase)
    assert issubclass(TacCapGripper, GripperBase)
    assert issubclass(GripperBase, EndEffectorBase)


@pytest.mark.parametrize("legacy_first", [True, False])
def test_imports_and_construction_do_not_load_hardware_sdks(legacy_first):
    code = f"""
import sys
import ctypes
def forbidden(*args, **kwargs):
    raise AssertionError('Native library loaded during import/construction')
ctypes.CDLL = forbidden
if {legacy_first!r}:
    import manimux.robots.tianji
    import manimux.end_effectors.taccap
    import manimux.sensors.taccap
from manimux.embodiments.arm.tianji import TianjiSDKKinematics
from manimux.embodiments.robot.tianji_taccap import TianjiTaccapRobot
from manimux.embodiments.end_effector.taccap import TacCapGripper, TacCapGeometry
from manimux.embodiments.sensor.taccap import TacCapCamera
TacCapGripper(serial='offline', side='left', kp=8, kd=0.3)
TacCapGeometry()
assert 'xense.taccap' not in sys.modules
assert not any('.sdk.marvin' in name for name in sys.modules)
assert not any('backup' in name for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)
