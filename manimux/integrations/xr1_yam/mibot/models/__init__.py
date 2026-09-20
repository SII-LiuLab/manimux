# Copyright (C) 2026 Xiaomi Corporation.
#
# Vendored from https://github.com/XiaomiRobotics/Xiaomi-Robotics-1 (Apache-2.0).
# The upstream module also imports BaseRunner, which drags in Lightning and
# DeepSpeed. Inference never touches it, so it is omitted here.
from mmengine import Registry

MIMODEL = Registry("MIMODEL")

from manimux.integrations.xr1_yam.mibot.models.VLA.XR1 import xr1  # noqa: E402

__all__ = ["MIMODEL", "xr1"]
