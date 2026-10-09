# TacCap SDK source snapshot

The SDK lives in `sdk/TacCap-Gripper/`. `end_effector.py` provides the ManiMux
impedance and bounded-force position adapter; `geometry.py` provides independent
offline geometry.

- Upstream: https://github.com/XenseRobotics-AI/TacCap-Gripper
- Supplied checkout revision: `a5e41800a0c90ffb5e1fb2eb929dbd35db404403`
  (`v0.4.2`).
- Package version: `0.4.2`.
- License: Apache-2.0; preserved in the SDK's `LICENSE`.

This snapshot preserves the selected tracked source files from that exact upstream
tag without local SDK source changes.

Included: root build metadata, license, README/changelog, environment recipe,
`cpp/` sources/headers/tests/examples, `python/` sources/bindings/tests/examples,
`docs/`, firmware metadata/documentation, tracked `scripts/check_protocol_drift.py`,
and the vendor `.gitignore`. All 162 copied files were checked byte-for-byte
against the supplied checkout.

Excluded: Git metadata/CI, virtual environments, build directories, native
binaries, caches and the two flashable firmware images. Firmware-related source
APIs, documentation and the image manifest remain; firmware images referenced by
OTA examples are not shipped in this snapshot.

Build/install the SDK from this vendored source into the ManiMux hardware
Python environment. The build also needs the native OpenCV, spdlog/fmt and C++
toolchain dependencies described in `sdk/TacCap-Gripper/docs/INSTALL.md`.
Copying these sources does not install `xense.taccap`, and no compiled extension
or dependency on the original checkout is provided. Hardware environment installation and real-device validation remain separate steps.

## Offline geometry

`geometry.py` provides `TacCapGeometry`, a `FixedToolGeometry` implementation for
the existing UMI follower CAD variant. It declares one `gripper` coordinate,
normalized motor travel (0 closed, 1 open), and models the closed tactile-pad
midpoint as a fixed TCP. It excludes the flange mount and does not model the
approximately +/-3 mm midpoint movement with opening. A calibrated fixed TCP
can be provided via `tcp_transform=`. It imports no TacCap SDK and opens no device.

## Gripper driver

`TacCapGripper(GripperBase)` owns one follower selected by exact firmware serial
and left/right side. The constructor requires explicit `kp` (Nm/rad) and `kd`
(Nm*s/rad). `control_mode` is either `impedance` (the backward-compatible
default) or `force_position`. It does not import the native SDK or open hardware.
Install the vendored SDK into the runtime environment before calling `connect()`.
No path patches or external developer checkout imports are used.

- `connect()` opens serial with cameras off, validates the firmware position map,
  and reads feedback. It does not enable, calibrate, home, or clear motor faults.
- `get_state()` returns `GripperState(opening, timestamp)`. Opening is normalized
  motor travel, 0 closed and 1 open. Timestamp is host receipt in seconds from
  the supplied ManiMux Clock, not firmware sample time. Reads are synchronous
  (default 100 ms timeout), at most 30 Hz by default; more frequent calls return
  the cached measured state with its original timestamp. Firmware-internal
  cache age is not available through this read API.
- In `impedance` mode, `send_command(GripperCommand(opening))` enables on first
  command and submits a normalized impedance position target with zero
  feedforward. Submission has no ACK and is not evidence of target completion.
  Targets are not silently clipped or smoothed. The SDK maps targets through
  firmware calibration.
- In `force_position` mode, `connect()` additionally requires a persisted motor
  model and an audited, effective safety envelope. It builds
  `ForcePositionConfig.for_spec()` and rejects `grasp_torque_nm` above either the
  motor's continuous-stall rating or the firmware's effective continuous
  envelope. It never writes either record. The first authorized command starts
  `ForcePositionController` before enabling the motor; later commands update the
  same normalized position target. `grasp_torque_nm` is a bounded grip budget,
  not an extra policy coordinate or a promise that measured torque is constant.
  Targets at or below normalized opening `0.005` automatically use terminal
  force hold: after the target is reached, the SDK ramps the total MIT request
  to `grasp_torque_nm`. No additional action coordinate or driver option is
  required. With no object this loads the closed mechanical stop.
- While force-position control runs, `get_state()` reads the controller snapshot,
  never the synchronous motor API. It preserves stream sequence/timestamp identity,
  rejects stale or invalid feedback and uses the raw motor angle with the existing
  position map. Endpoint compression up to 5% of calibrated travel is exposed by
  `get_force_position_state().unclamped_opening` while the public normalized state
  remains in `[0, 1]`; larger excursions fail rather than being hidden.
- Invalid/out-of-range feedback and motor protection flags raise. A command
  failure requests motor disable. There is no autonomous monitoring thread;
  the SDK force-position mode does own its documented status/submit thread, while
  the robot/runtime must still poll feedback and stop on read failure.
- `stop()` disables an owned enabled motor, which can release a grasp. A later
  command re-enables; force-position stop first commands zero torque through the
  SDK controller. `close()` stops control before releasing serial; failed cleanup
  is reported and retains the connection for retry. Repeated successful close
  calls are harmless.

Camera/tactile acquisition, automatic calibration, and the legacy driver's
target-margin behavior remain excluded. The installation must select suitable
control gains or a force-position torque budget. Unit tests use a fake SDK; a
real-device acceptance run remains separate and requires explicit motion
authorization. Map `state.opening` to the kinematic `gripper` coordinate when
assembling the robot.

Select force-position control per task through the existing component hardware
override. Device serials remain in the private station file; the policy action
layout remains seven arm joints plus one normalized gripper coordinate:

```yaml
robot:
  options:
    component_hardware:
      left_end_effector:
        control_mode: force_position
        grasp_torque_nm: 0.35
        close_speed_radps: 0.6
      right_end_effector:
        control_mode: force_position
        grasp_torque_nm: 0.35
        close_speed_radps: 0.6
```

For the installed EL05 motors, `grasp_torque_nm` must not exceed the persisted
1.1 Nm continuous envelope. Start thin or soft object trials at a lower explicit
budget, such as 0.3 Nm, and tune from recorded force-position telemetry. Do not
change the policy-side opening mapping at the same time as the controller mode;
the SDK already adds its own `close_speed_radps` setpoint ramp.

Motor firmware qualification remains an installation acceptance step: the SDK
documents EL05 `1.0.5.0.4` or newer and RS00 `0.0.3.32` or newer. The driver does
not call `motor_version()` from `connect()` because that diagnostic may stop an
older motor. Verify or persist the version before authorizing the first physical
force-position motion; do not infer it from the motor model record.

## Shared camera SDK

The camera adapter lives in `embodiments/sensor/taccap/sensor.py`. Both adapters
import the installed `xense.taccap` package. The SDK source is kept only here;
the camera has no dependency on the gripper driver or its serial connection.
