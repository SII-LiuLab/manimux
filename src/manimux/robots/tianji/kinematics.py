"""Compatibility import for :mod:`manimux.embodiments.arm.tianji.kinematics`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("manimux.embodiments.arm.tianji.kinematics")
