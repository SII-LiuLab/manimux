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
while retaining gripper openings, then opens the grippers after both arms finish.
YAM declares `home_before_close=True`; runtime exits request Home independently of
legacy `home_on_close` values, model selection or inference strategy.
`execute: false` disables trajectory submission, Home and the assembly stop/hold command.
SDK connection/close behavior remains owned by the controller.

Collection and inference construct this robot through `embodiments.robot.build_robot()` rather than
the old `robots/yam/` or `yam_dual` drivers. Existing experiments use `type: yam`, preserving action intervals,
control frequency, horizon, execution limits and model identity. The collection GUI reads device bindings from its own station
and passes them to the same components; leader/follower buttons and recording logic remain in the collection module.

## Camera ownership and configuration example

`manimux/configs/experiments/example-yam-pi05/put_bottles/rtc.yaml` is a composed configuration example.
The separate camera service selects assembly cameras using `camera_server.cameras`; runtime reads network
sensors without also calling the robot's `start_sensors()`. Callers using robot sensors in-process must
explicitly manage `start_sensors()` / `close_sensors()`. Each device has one capture owner.

Inspect offline assembly without connecting hardware:

```python
from manimux.cli import load_config
from manimux.clock import SystemClock
from manimux.embodiments.robot import build_robot

config = load_config("manimux/configs/experiments/example-yam-pi05/put_bottles/rtc.yaml",
                     local="manimux/configs/local/yam_example.yaml")
robot = build_robot(config["robot"], SystemClock())
print({name: model.num_coordinates for name, model in robot.kinematics.models.items()})
```

Raw high-rate CAN recording and SDK lock probes were removed. Normal trajectory and camera recording remain.
Offline tests cover SDK interfaces, named groups, Home phases, interruption and configuration contracts; they do not establish real-robot task success.

### Fault shutdown

A healthy YAM returns to its zero Home after a runtime fault, including camera loss.
This is separate from the selected initial pose. Both controllers must have advancing
CAN and SDK feedback; Home arrival is verified before gripper opening and release.
After each Home phase, both controllers observe fresh feedback without an arrival
deadline. Position and speed must remain within the
release limits for `home_settle_duration_s` (0.25 seconds) continuously. This wait
does not resend a trajectory or relax limits. While waiting, logs report the largest
joint error and speed every five seconds; hardware-health failure stops the wait.
The retired `home_settle_timeout_s` setting in old snapshots is ignored with a warning.
If Home or health validation fails, automatic release is blocked. RoboGUI keeps a
Hardware recovery panel visible: confirm a clear Home path to retry Home, or confirm
physical support of ALL arms and secured payloads to release motor control. Terminal
input is not used. Lost GUI connections, stale confirmations and interrupts cannot
authorize release. Home retries recheck control health and measured arrival. Do not start another
runtime while this process still owns the controllers. This software gate does not
replace physical support, drive safeguards or validation of the Home path.
RoboGUI reports Home, resource closing, recording finalization and blocked release
separately. Its cleanup heartbeat indicates process status, not fresh motor feedback.
Both the runtime and dashboard processes must be restarted to load code changes;
reloading a browser does not update the dashboard server.
