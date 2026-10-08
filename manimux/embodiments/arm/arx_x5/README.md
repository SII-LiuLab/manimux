# ARX X5 (2023)

`ArxX5Arm` loads its model offline. `ArxX5Controller` adapts the official X5
binding behind an owned process. The common arm assembly uses it through
`ArmController`; vendor operations never enter the runtime or RoboGUI.

See the [SDK adapter guide](../../../../docs/usage/can-arms.md) for configuration,
calibration, lifecycle and the explicit device-freshness limitation.

The kinematic `model.urdf` is derived from
[ARXroboticsX/X5](https://github.com/ARXroboticsX/X5/blob/9a255e38156e3f5a263f341df209ab028d49dad9/py/arx_x5_python/bimanual/script/x5.urdf),
revision `9a255e38156e3f5a263f341df209ab028d49dad9` (BSD-3-Clause; `LICENSE.txt`).
Visual and collision meshes are omitted; joint origins, axes, limits and inertia
are retained. Its broad source bounds are not calibrated device motion limits.
The default TCP is `link6`, in `base_link`, using the URDF's absolute arm-base
origin, not the SDK solver's zero-pose position reference.

Run `scripts/assets/prepare_arx_x5.py --help` to regenerate this model
from the pinned sources. SDK source and compiled libraries remain external.

`arm.py` owns offline model loading. `controller.py` owns parent-side calibration
and RPC lifecycle; `session.py` owns the native binding in the child process.
Installation and the pure-FK probe are documented in the
[SDK installation guide](../../../../docs/usage/robot-sdks.md).
