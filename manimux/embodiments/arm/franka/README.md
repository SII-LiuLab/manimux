# Franka FR3 component

This component implements `ArmBase` / `ArmController`; it does not import RPent,
RLinf, Ray or any policy framework. `robot/franka` assembles it through `RobotBase`
with an optional separate `end_effector/robotiq` component. Ordinary policy
adapters, inference strategies, executors and recording use the standard runtime.
An optional agent bridge belongs under `integrations/`, never in this driver.

## Model and coordinates

The bundled model is nominal FR3 v2.1, not Panda. Seven joints use radians and
`link0` → `link8` is the mechanical flange transform. The mounted tool transform
is composed once by `RobotModel`. The 8-coordinate Robotiq example uses seven
joint positions followed by normalized travel (0 closed, 1 open). Bare-arm
assemblies have seven coordinates and no implicit gripper.

`FrankaKinematics` loads offline and supplies bounded, seed-initialized numerical
IK. It checks its final position/orientation residuals (default 0.1 mm / 0.1 mrad).
IK does not establish collision-free motion. Meshes are display assets and are
not an enabled collision guard. An alternative `urdf_path` must retain joint1..7
and link0/link8; verify FK against measured controller poses before motion.

Assets are derived from [Franka Robotics' official description](https://github.com/frankarobotics/franka_description/tree/7aeeddc449edf8d62b594f9e36a81da53e7796f9),
revision `7aeeddc449edf8d62b594f9e36a81da53e7796f9`, under Apache-2.0. The generated
URDF retains the seven joint transforms/limits and flange. Official detailed visual
meshes, including their materials, are converted losslessly in triangle count to
binary glTF; the separate upstream collision meshes are unchanged. Visual meshes
are not loaded by the FK/IK solver. Inertials, ROS plugins and accessory frames are
omitted. Upstream LICENSE and NOTICE are in `assets/`; `visual-sources.json` records
source/output hashes and conversion provenance.

## SDK and lifecycle

Install the optional `franka` (read-only SDK) and `robotiq` extras in the robot
environment. Executing Linux x86_64 / CPython 3.12 sessions also install the
firmware-specific, SHA-256-pinned SDK profile at
`scripts/setup/franka-linux-cp312.txt`.
Select Franky/libfranka builds compatible with the controller firmware and exposing
`JointReferenceHandle` / `JointImpedanceTrackingMotion`; a package version alone
does not establish these capabilities. Executing connect checks them before I/O.
The tested Linux x86_64 / Python 3.12 station uses
[this Franky 1.1.3 build with libfranka 0.19.0](https://github.com/Brunch-Life/franky/releases/download/wheels-libfranka-0.19.0/franky_control-1.1.3-cp312-cp312-manylinux_2_28_x86_64.whl)
and pylibfranka 0.20.5. The ordinary PyPI 1.1.3 build is insufficient.
This is separate from a model server's environment. RPent is not required.

- Construction/model loading performs no hardware I/O.
- Both executing and receive-only sessions acquire the same local, per-address
  FCI lease before constructing their SDK. A second ManiMux connection fails
  without touching the controller. Stop retains the lease for Resume; successful
  close releases it, and incomplete stop retains ownership. Read diagnostics from
  the active runtime instead of opening another SDK connection. The lease covers
  the same configured address on this host, not raw SDK clients or other hosts.
- `execute: false` uses pylibfranka for reads. Franky is intentionally not used:
  its constructor writes collision thresholds. Send is rejected; stop/close
  never issue actuator commands in a read-only session.
- Executing connect uses Franky with real-time enforcement. Its constructor sets
  the explicitly configured collision thresholds. It does not home, activate
  FCI, unlock brakes, switch control modes or recover faults. The first target
  starts Franky's native persistent joint impedance tracking. Connect requires
  Idle; it does not take over an already moving controller.
- On Linux, SDK construction can promote its calling thread to real-time
  priority. The component restores the caller's prior scheduling policy after
  connect, including failed and read-only connects. It temporarily applies the
  SDK policy while starting Franky's native control thread so that thread
  inherits real-time scheduling, then restores the caller again. Camera,
  inference and recording work must not inherit the SDK's control priority.
- `send_commands` accepts absolute joint targets, checks model bounds and any
  explicitly configured host guards, and updates a native reference handle.
  Every accepted target, including the first, is submitted unchanged. A guard
  failure is rejected before the reference is updated. There is no driver-side
  clipping, retiming or hidden goal. Configure the existing executor's motion
  limits for rate shaping. No new point-to-point motion is created per tick.
  The runtime's recorded command is the SDK reference, not observed actuator
  torque or measured arrival. The runtime must stop on reported errors.
- Optional host guards default to `null` for ordinary VLA execution, matching the
  direct target dispatch used by other embodiments. Set a positive scalar or seven
  per-joint values for a bounded test. `max_target_delta_rad` bounds the distance
  from measured joints. `max_reference_step_rad` bounds each update against the last accepted reference
  (measured position at startup); elapsed idle time does not enlarge the allowance.
  This is a discontinuity guard, not a physical speed/acceleration/jerk guarantee.
  `measured_velocity_limit_rad_s` checks
  measured speed at host reads. Native gains and torque-rate limiting belong to
  Franky; controller collision protections, model joint bounds, stale-feedback
  checks and fault propagation remain active with these host guards disabled.
  The default gain vectors
  follow the [libfranka joint impedance example](https://frankarobotics.github.io/libfranka/0.15.0/joint_impedance_control_8cpp-example.html).
- State is measured `q`; repeated cached device timestamps preserve the original
  host timestamp/sequence. This timestamp is first host observation of a device
  sample, not a hardware acquisition timestamp. Expired samples are rejected.
- Stop calls libfranka's stop on owned active control and joins with a finite
  timeout. Only a matching self-requested preemption with no controller error is
  treated as normal completion. A failed join stays visible and cleanup can be retried. Stop does
  not home, recover faults or release brakes; a later command can resume from
  Idle. Close stops first and releases the SDK session.
- Home, startup moves and idle recovery are unsupported and not advertised.
  The runtime skips the base rejecting Home method during shutdown, and RoboGUI
  offers Finish rollout with Return Home disabled. Shared cleanup still stops
  owned control before saving a successful recording; failures remain visible.

Use `configs/examples/franka_joint_policy.yaml` for an ordinary XPolicyLab-backed
joint deployment, and `configs/local/franka_example.yaml` for station bindings.
Supply the selected model's actual cameras, horizon, action semantics and backend
identity. The example is an interface recipe, not a bundled trained policy.
Commands and deployment settings are documented in the public user guide.
