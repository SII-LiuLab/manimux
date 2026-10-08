# Standard PiPER

`PiperArm` loads its model offline. `PiperController` uses the pinned pyAgxArm
driver and wire codecs with an owned CAN receiver. This preserves packet receipt
times and propagates failures that the SDK's communication wrapper suppresses.

See the [SDK adapter guide](../../../../docs/usage/can-arms.md) for firmware,
calibration, receive-only operation, lifecycle and validation scope.

The shared visual and kinematic `model.urdf` and ten `assets/meshes/*.STL` files
are derived from
[agilexrobotics/piper_ros](https://github.com/agilexrobotics/piper_ros/blob/ac41fcbcdda598f01b51cf6175ed9a24d0dacadc/src/piper_description/urdf/piper_description.urdf),
revision `ac41fcbcdda598f01b51cf6175ed9a24d0dacadc` (MIT; `LICENSE.txt`). Meshes
are copied unchanged. ROS mesh references become relative paths; collision
elements are removed. Joint origins, axes, limits, inertia, visual origins and
materials remain unchanged. The model loads outside a ROS workspace and from
the installed ManiMux wheel without a device SDK.
The default TCP is `link6` in `base_link`. URDF export precision differs slightly
from the SDK's MDH table; the integration uses this URDF consistently for FK/IK.
Normalized opening expands into the model's nominal finger travel of
`[0, 0.035]` and `[0, -0.035]` m. Physical width calibration is independently
required by the controller; the visual mapping is not a calibration procedure.

See the [PiPER asset guide](../../../../docs/usage/piper.md) for offline preview,
replay, supported model scope and reproducible asset preparation. Run
`scripts/assets/prepare_piper.py --help` to regenerate this model and its meshes
from the pinned official source. No SDK source or native libraries are bundled.

`arm.py` owns offline model loading; `controller.py` owns communication and
feedback lifecycle. Shared station calibration lives in `arm/_calibration.py`,
independently of nominal finger geometry. See the
[SDK installation guide](../../../../docs/usage/robot-sdks.md) for dependencies
and the offline API/FK probe.
