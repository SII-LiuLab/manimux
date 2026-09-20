"""Component inheritance and offline imports at canonical implementation paths."""

import subprocess
import sys


def test_base_classes_keep_identity_and_explicit_inheritance():
    from manimux.embodiments.end_effector import EndEffectorBase, GripperBase
    from manimux.embodiments.end_effector.taccap import TacCapGripper
    from manimux.embodiments.robot import RobotBase
    from manimux.embodiments.robot.tianji_taccap import TianjiTaccapRobot

    assert issubclass(TianjiTaccapRobot, RobotBase)
    assert issubclass(TacCapGripper, GripperBase)
    assert issubclass(GripperBase, EndEffectorBase)


def test_imports_and_construction_do_not_load_hardware_sdks():
    code = """
import sys
import ctypes
def forbidden(*args, **kwargs):
    raise AssertionError('Native library loaded during import/construction')
ctypes.CDLL = forbidden
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
