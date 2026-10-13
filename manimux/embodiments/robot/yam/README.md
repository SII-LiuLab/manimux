# YAM robot assembly

`YamRobot(RobotBase)` assembles left/right YAM components and optional cameras from YAML. Each CAN channel has a
`YamController`; the robot layer coordinates named groups and motion phases without bypassing controllers to call the SDK.

```text
manimux/configs/embodiment/arm/yam.yaml          Arm configuration and SDK parameters
manimux/configs/embodiment/sensor/realsense.yaml Camera capture parameters
manimux/configs/embodiment/robot/yam_dual.yaml   Components, mounts and control groups
manimux/configs/embodiment/robot/yam_control.yaml Shared collection/deployment control parameters
manimux/configs/local/yam_example.yaml          CAN, camera serial and network service bindings
```

`left_arm` and `right_arm` each have seven coordinates. Built-in grippers are controlled with the arm; the end-effector field is `null`.
A camera `mount: null` means optical extrinsics are unavailable; configuration does not infer poses.
For hardware environment setup, see [the YAM component](../../arm/yam/README.md); for cameras, see
[the RealSense component](../../sensor/realsense/README.md).

## Lifecycle

Construction, model loading and FK/IK do not open devices. `connect()` performs no start motion by default.
Only `execute: true` together with `move_to_start_on_connect: true` enables movement to `start_joints`.
Start poses are specified by group name and retain the old driver's sequential arm order. Home first zeros both arms concurrently
while retaining gripper openings, then opens the grippers after both arms finish. Runtime still owns `home_on_close`.
`execute: false` disables trajectory submission and Home; explicit stop/shutdown cleanup may still submit a current-position hold.

Collection and inference construct this robot through `embodiments.robot.build_robot()` rather than
the old `robots/yam/` or `yam_dual` drivers. Existing experiments use `type: yam`, preserving action intervals,
control frequency, horizon, execution limits and model identity. The collection GUI reads device bindings from its own station
and passes them to the same components; leader/follower buttons and recording logic remain in the collection module.

## Camera ownership and configuration example

`manimux/configs/experiments/put_bottles/pi05/yam_pi05_joint.yaml` is a composed configuration example.
The separate camera service selects assembly cameras using `camera_server.cameras`; runtime reads network
sensors without also calling the robot's `start_sensors()`. Callers using robot sensors in-process must
explicitly manage `start_sensors()` / `close_sensors()`. Each device has one capture owner.

Inspect offline assembly without connecting hardware:

```python
from manimux.cli import load_config
from manimux.clock import SystemClock
from manimux.embodiments.robot import build_robot

config = load_config("manimux/configs/experiments/put_bottles/pi05/yam_pi05_joint.yaml",
                     local="manimux/configs/local/yam_example.yaml")
robot = build_robot(config["robot"], SystemClock())
print({name: model.num_coordinates for name, model in robot.kinematics.models.items()})
```

Raw high-rate CAN recording and SDK lock probes were removed. Normal trajectory and camera recording remain.
Offline tests cover SDK interfaces, named groups, Home phases, interruption and configuration contracts; they do not establish real-robot task success.
