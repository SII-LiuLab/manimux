# TacCap SDK source snapshot

The SDK lives in `sdk/TacCap-Gripper/`. `end_effector.py` provides the ManiMux
position-control adapter; `geometry.py` provides independent offline geometry.

- Upstream: https://github.com/Vertax42/TacCap-Gripper
- Supplied checkout base revision: `a20647ebafd1faafb90943eeeac2186abf847c0e`.
- Package version: `0.1.9`.
- License: Apache-2.0; preserved in the SDK's `LICENSE`.

This snapshot preserves the supplied working-tree contents of the selected
tracked source files, not a pristine export of the base revision. Two existing
local changes are intentionally retained:

1. `pyproject.toml`: declares NumPy, adds build/dev/example dependency groups and
   uv build settings. Its references to `scripts/setup_uv.sh` and `uv.lock` belong
   to the source checkout's local setup; those untracked files are not included.
2. `python/examples/gripper_control_test.py`: selects the right device explicitly
   instead of auto-opening the only gripper. This is an upstream-checkout example,
   not ManiMux's hardware binding or a script run during copying.

Included: root build metadata, license, README/changelog, environment recipe,
`cpp/` sources/headers/tests/examples, `python/` sources/bindings/tests/examples,
`docs/`, tracked `scripts/check_protocol_drift.py`, and the vendor `.gitignore`.
All 109 copied files were checked byte-for-byte against the supplied checkout.

Excluded: Git metadata/CI and assistant instructions, virtual environments,
build directories, native binaries, caches, firmware bundles and untracked local
setup files. Firmware-related source APIs and documentation remain; firmware
images referenced by OTA examples are not shipped in this snapshot.

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

## Position driver

`TacCapGripper(GripperBase)` owns one follower selected by exact firmware serial
and left/right side. The constructor requires explicit `kp` (Nm/rad) and `kd`
(Nm*s/rad); it does not import the native SDK or open hardware. Install the
vendored SDK into the runtime environment before calling `connect()`. No path
patches or external developer checkout imports are used.

- `connect()` opens serial with cameras off, validates the firmware position map,
  and reads feedback. It does not enable, calibrate, home, or clear motor faults.
- `get_state()` returns `GripperState(opening, timestamp)`. Opening is normalized
  motor travel, 0 closed and 1 open. Timestamp is host receipt in seconds from
  the supplied ManiMux Clock, not firmware sample time. Reads are synchronous
  (default 100 ms timeout), at most 30 Hz by default; more frequent calls return
  the cached measured state with its original timestamp. Firmware-internal
  cache age is not available through this read API.
- `send_command(GripperCommand(opening))` enables on first command and submits
  a normalized impedance position target with zero feedforward. Submission has
  no ACK and is not evidence of target completion. Targets are not silently
  clipped or smoothed. The SDK maps targets through firmware calibration.
- Invalid/out-of-range feedback and motor protection flags raise. A command
  failure requests motor disable. There is no autonomous monitoring thread;
  the robot/runtime must poll feedback and stop on read failure.
- `stop()` disables an owned enabled motor, which can release a grasp. A later
  command re-enables. `close()` disables before releasing serial; failed cleanup
  is reported and retains the connection for retry. Repeated successful close
  calls are harmless.

This first version excludes force control, camera/tactile acquisition, automatic
calibration, and the legacy driver's target-margin behavior. The installation
must select suitable control gains. Tests use a fake SDK only; the adapter has
not been exercised on hardware. Map `state.opening` to the kinematic `gripper`
coordinate when assembling the robot.

## Shared camera SDK

The camera adapter lives in `embodiments/sensor/taccap/sensor.py`. Both adapters
import the installed `xense.taccap` package. The SDK source is kept only here;
the camera has no dependency on the gripper driver or its serial connection.
