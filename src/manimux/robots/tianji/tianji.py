"""Compatibility import for :mod:`manimux.embodiments.robot.tianji_taccap.robot`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("manimux.embodiments.robot.tianji_taccap.robot")
