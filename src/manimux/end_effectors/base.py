"""Compatibility import for :mod:`manimux.embodiments.end_effector.base`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("manimux.embodiments.end_effector.base")
