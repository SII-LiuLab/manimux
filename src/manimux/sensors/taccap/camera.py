"""Compatibility import for :mod:`manimux.embodiments.sensor.taccap.sensor`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("manimux.embodiments.sensor.taccap.sensor")
