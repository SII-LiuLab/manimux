# Tianji SDK kinematics

`TianjiSDKKinematics` implements `FlangeKinematicsBase` directly with the
official `vendor/marvin/fx_kine.py` and `libKine.so`. It imports neither the
legacy Tianji implementation nor a ManiMux SDK loader. Importing the class
does not load the vendor binding; constructing it initializes the offline
kinematics library, never a controller connection.

```python
import numpy as np
from manimux.robots.tianji import TianjiSDKKinematics

kin = TianjiSDKKinematics(arm="left")  # left = SDK 0/A, right = SDK 1/B
q = np.radians([21.8, -41.0, -4.74, -63.67, 10.15, 14.72, 7.68])
target = kin.fk_flange(q)
result = kin.ik_flange(target, seed_joints=q)
if result.converged:
    solved_joints = result.joints
else:
    print(result.reason)  # no fallback joint vector on rejection
```

- Inputs and results use seven J1..J7 angles in radians. Poses are 4x4
  `T_base_flange` transforms, with translation in metres in the selected arm's
  own base frame. No gripper coordinate, mounting transform or TCP offset is added.
- `config_path` defaults to the bundled M6 4.0 table. DH, PNVA limits and J6/J7
  interference parameters come from that table. Station overrides and control
  rate limits from the old implementation are not imported.
- IK calls the SDK's standard `ik` with ZSP type 0 and the supplied seed. It does
  not call `ik_nsp`, search alternative branches, smooth commands or provide
  differential IK. Future backends can implement the same flange interface.
- SDK failure flags and position limits are checked, followed by FK validation.
  Default acceptance tolerances are `position_tolerance_m=1e-5` and
  `orientation_tolerance_rad=1e-3`. Rejection returns `IKResult` with a reason;
  bad inputs and backend failures raise exceptions. A solution is not a checked
  motion path or a guarantee of a small joint step.
- The native library uses process-global arm slots. Calls through this wrapper
  are serialized, restore the instance's model, and remove SDK tool offsets.
  Do not concurrently use raw libKine clients that bypass this lock. Native
  initialization is repeated per operation; real-time suitability is unmeasured.

Validation: `tests/unit/test_tianji_sdk_kinematics.py` uses both a fake SDK and
the real Linux x86-64 library, without hardware. It covers units, rejection,
left/right round trips, and model/tool isolation across instances.

The new hardware driver and TCP composition are not implemented yet. The old
`tianji_dual` runtime entry is not restored by this kinematics-only implementation.

## Tianji + TacCap offline assembly

`build_tianji_taccap_kinematics` composes the official flange solver with
`TacCapGeometry`. Pass the installation transform explicitly:

```python
import numpy as np
from manimux.robots.tianji import build_tianji_taccap_kinematics, umi_follower_mount

model = build_tianji_taccap_kinematics("left", mount=umi_follower_mount())
# joint_1..joint_7 in radians, then gripper (0 closed, 1 open).
q = np.r_[np.radians([21.8, -41, -4.74, -63.67, 10.15, 14.72, 7.68]), 0.5]
target = model.fk(q)
result = model.ik(target, q, fixed_coordinates={"gripper": 0.5})
```

Create a separate instance with `"right"` for the other arm. Each instance uses
its own arm base, with local coordinate names `joint_1` through `joint_7` and
`gripper`. Future robot assembly must map its measured state and commands to this
same ordering; this factory does not register or replace the legacy driver.

`umi_follower_mount()` reproduces the installation in
`src/manimux/assets/end_effectors/umi_follower/end_effector.yaml`:
flange-to-tool translation [-0.01575, 0, 0.0505] m and extrinsic xyz RPY
[pi, -pi/2, 0]. Tool geometry separately supplies [0.11895, 0, -0.020995] m
with identity rotation. Their combined translation is [-0.036745, 0, 0.16945] m.
No legacy loader/functions are called and no native SDK tool offset is added.

This models the closed-pad midpoint of the existing UMI follower CAD variant as
fixed. The approximately +/-3 mm midpoint motion with opening is not calibrated
or modeled. Supply another mount and/or `TacCapGeometry(tcp_transform=...)` for
an installation with a different fixed calibration. The default TCP is named
`tianji_<side>_taccap_tcp`, relative to `tianji_<side>_base`.

Validation is offline: SDK FK/IK consistency does not validate physical tool
calibration, collision clearance, or hardware control. No TacCap SDK installation
is required for these geometry calculations.

## First-version robot assembly

`TianjiRobot` inherits `RobotBase` and owns ONE Marvin control session for one or
both configured arms. `TianjiArmConfig` binds each group's gripper, complete TCP
model, joint bounds and controller speed ratios. `left` maps to A/0; `right` maps
to B/1. Hardware state, command groups and `robot.kinematics` use the same names.
This is a direct-construction API; the legacy `tianji_dual` factory/config path
has not yet been migrated. No legacy backup functions are imported.

Assembly example (station values must be supplied, and this does not connect):

```python
from manimux.clock import SystemClock
from manimux.end_effectors.taccap import TacCapGripper
from manimux.robots.tianji import (
    TianjiRobot, TianjiArmConfig, build_tianji_taccap_kinematics,
)

clock = SystemClock()
# station is the installation's explicit per-side configuration dictionary.
arms = {}
for side in ("left", "right"):
    cfg = station[side]
    arms[side] = TianjiArmConfig(
        gripper=TacCapGripper(
            serial=cfg["serial"], side=side, kp=cfg["kp"], kd=cfg["kd"], clock=clock,
        ),
        kinematics=build_tianji_taccap_kinematics(side, mount=cfg["mount"]),
        joint_limits=cfg["joint_limits"],  # (lower, upper), each seven radians
        velocity_ratio=cfg["velocity_ratio"],
        acceleration_ratio=cfg["acceleration_ratio"],
    )
robot = TianjiRobot(ip=controller_ip, arms=arms, clock=clock)
```

Each TacCap group is `[joint_1, ..., joint_7, gripper]`. A bare arm may omit the
gripper if its complete manipulator model declares only seven joints. Configured
limits must match the installation and SDK model; command limits are checked
again even when a target came from IK. No implicit joint smoothing or step limit
is provided. The runtime remains responsible for trajectories and timing.

Offline usage, without connecting hardware:

```python
poses = robot.kinematics.fk(configuration)  # dict of complete per-arm vectors
results = robot.kinematics.ik(
    poses, configuration,
    fixed_coordinates={"left": {"gripper": 0.5}, "right": {"gripper": 0.5}},
)
```

FK returns TacCap TCP poses, including mount/tool offset once. Each pose is in
its own `tianji_left_base` / `tianji_right_base`, not one world frame. IK solves
arms independently and returns per-group `IKResult`; it does not execute any
solution or check inter-arm collision. FK/IK accept a non-empty subset of groups;
IK target/seed/constraint keys must match exactly. The current tool model retains
the fixed-TCP approximation documented above.

Hardware lifecycle:

- `connect()` requires configured arms already disabled, checks advancing
  feedback, then connects grippers. It does not enable motors, clear faults,
  calibrate or home. Unconfigured arms receive no commands.
- `get_state()` returns measured groups. Timestamp is the oldest host receipt
  across arm frames and gripper feedback; cached data is not stamped as new.
  Sequence counts returned snapshots. Missing/stale/faulted data raises; the
  runtime must stop on read failure, since there is no background monitor.
- `send_command()` requires every configured group. It validates all targets
  before writing, enables pending arms at measured joints, and confirms position
  mode plus configured ratios. Both arm targets use one SDK command batch;
  grippers dispatch afterward over separate links. Dispatch is not an atomic
  multi-device action or proof of target arrival. Failure attempts to stop all
  owned components. Grippers and robot must use the same clock domain.
- `stop()` requests servo-off for owned arms and invokes each gripper's stop,
  attempting other components even if one fails. This can release a grasp and is
  not an emergency stop. Subsequent commands re-enable.
- `close()` stops and closes owned resources; incomplete cleanup raises and can
  be retried. An arm whose disable cannot be confirmed retains the SDK connection.
- `home()` raises `NotImplementedError`: no home trajectory is defined yet.

Only one `TianjiRobot` control session is allowed per process because the native
SDK shares a connection and command buffers. Use this one object's left/right
configuration, not separate simultaneous robot instances. Construction and
imports do not load the control library. Validation uses fake control/gripper
SDKs and real offline kinematics only; hardware behavior remains unvalidated.
