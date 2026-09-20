"""Canonical Tianji components and hardware-free imports after alias removal."""

import importlib
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("package", "implementation", "name"),
    [
        ("robot.tianji_taccap", "robot.tianji_taccap.tianji_taccap", "TianjiTaccapRobot"),
        ("arm.tianji", "arm.tianji.kinematics", "TianjiSDKKinematics"),
        ("end_effector", "end_effector.base", "EndEffectorBase"),
        ("end_effector", "end_effector.gripper_base", "GripperBase"),
        ("end_effector.taccap", "end_effector.taccap.end_effector", "TacCapGripper"),
        ("end_effector.taccap", "end_effector.taccap.geometry", "TacCapGeometry"),
        ("sensor.taccap", "sensor.taccap.sensor", "TacCapCamera"),
    ],
)
def test_exports_share_the_canonical_implementation(package, implementation, name):
    # Distinct module objects could duplicate the process-global control/IK locks.
    public = importlib.import_module(f"manimux.embodiments.{package}")
    source = importlib.import_module(f"manimux.embodiments.{implementation}")
    assert getattr(public, name) is getattr(source, name)


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
from manimux.server.sensor.taccap.server import CameraServer
TacCapGripper(serial='offline', side='left', kp=8, kd=0.3)
TacCapGeometry()
assert 'xense.taccap' not in sys.modules
assert not any('.sdk.marvin' in name for name in sys.modules)
assert not any('backup' in name for name in sys.modules)
assert not any(name.startswith(('manimux.end_effectors', 'manimux.sensors.taccap'))
               for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)
