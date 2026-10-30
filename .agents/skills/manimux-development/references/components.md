# Components: hardware ownership and offline models

Source paths below are relative to the repository root. Read the current interface before copying
an implementation; controller/session work can evolve independently.

## Shared behavior across embodiments

The same base-class operation has the same public meaning on every supported
embodiment. Its SDK/RDK implementation may differ. This is a development contract,
not a claim that every existing driver already complies. Review YAM, Tianji and new
drivers against the interfaces and callers; do not copy a vendor exception as a rule.

| Operation / owner | Required public behavior | Vendor-specific implementation |
| --- | --- | --- |
| `ArmBase.load_model()` / offline model | Provide geometry, coordinate definitions and display resources without opening hardware. Adapters and RoboGUI use the same robot definition. | URDF/mesh loading, FK/IK solver and visual joint mapping. |
| `ArmController.connect()` / session owner | Establish the configured session without implicit homing or fault recovery. Document readiness and any explicitly enabled actuator side effects. | SDK construction, discovery, connection and device readiness checks. |
| `get_states()` / controller; `get_state()` / assembly | Return measured feedback with declared group order, units and sample identity. Never substitute commanded targets for measured positions. | SDK reads and conversion to `ArmState` / `RobotState`. |
| `send_commands()` / controller; `send_command()` / assembly | Accept the declared absolute joint-position targets. Tool coordinates retain their declared meaning. Returning means submission, not arrival. Surface dispatch failure. | SDK command encoding, configured limits and mode preparation; document internal interpolation and batch dispatch guarantees. |
| `stop()` / session owner | Stop ongoing execution on owned channels; attempt all owned channels and report failures. Document completion and the state required for subsequent commands. | Vendor stop/hold and mode transitions. Do not silently equate stop with homing or releasing actuator torque. |
| `home()` / robot assembly | Perform the explicitly defined home operation when supported. State the destination, tool behavior, completion guarantee and post-home control state. | Configured joint trajectory or a documented vendor Home operation. A RoboGUI initial pose does not define hardware Home. `home_targets(state)` reports the final grouped command target without I/O; unknown targets return `None`, never the rollout start. |
| `recovery_actions`, `hand_guiding_actions`, `drag_selections` / robot assembly; `clear_errors()`, `recover_home(stop)` | Optional idle recovery that RoboGUI may request between rollouts. `recovery_actions` declares supported actions; `hand_guiding_actions` maps cancellable action names to handlers accepting groups, a stop event and an active callback. The base maps `drag` to `recover_drag()` when advertised. `drag_selections` maps arm labels to robot groups. The session schedules each action on a fresh, disconnected assembly, cancels hand-guiding on lease loss, and signals shutdown before joining the worker. Home and hand-guiding handlers must observe cancellation and complete cleanup before returning. | Vendor fault reset, hand-guiding modes and durations, feedback monitoring, tool behavior and any preparation such as clearing faults before connecting. |
| `close()` / resource owner | Release owned resources, including after partial startup. Repeated cleanup must be safe; incomplete cleanup remains visible and retryable. Borrowers do not close shared sessions. | SDK shutdown, worker termination and connection release. |

Tianji Home targets live in the assembly's `home.joints_deg`. Its cosine trajectory
uses `hardware.home_motion`: `control_hz`, `peak_velocity_deg_s`, `tolerance_deg`,
and `settle_timeout_s` (also used while waiting for grippers to open). The shipped
assemblies retain 100 Hz, 9 deg/s, 0.5 degrees and 5 seconds. Experiment
`robot.options.hardware.home_motion` may override individual fields; these options
belong to the assembly and are not forwarded to the arm SDK.

Document these lifecycle details in the integration's runbook:

- **Read-only execution:** `execute: false` permits only documented read-only
  connections and feedback. It must not enable actuators, switch motion modes,
  send motion/hold commands or home. Check connect, stop and close as well as send;
  the current `RobotBase` send guard does not suppress controller lifecycle calls.
- **Continuing after stop or home:** show which SDK state the operation leaves and
  how the next authorized command becomes usable. Keep device mode preparation in
  the owning implementation; callers must not issue raw SDK recovery commands.
  Fault recovery remains explicit, not an automatic retry of failed motion.
- **Feedback freshness:** repeated reads of one cached sample retain its timestamp
  and sequence. Distinguish device time from host monotonic time. If the SDK exposes
  no sample identity or receipt time, document what can actually be measured instead
  of presenting read time as proven acquisition time.
- **Failure:** propagate a meaningful error from the boundary that owns it; do not
  swallow it and return ready/success. Reuse assembly cleanup and validation rather
  than duplicating broad catch/retry/check layers in every component.

Required operations cannot be stubbed out. Optional unsupported capabilities must
be declared and unavailable to normal callers/UI; an unexpected direct call should
fail explicitly. The current base `home()` raises `NotImplementedError`. Idle recovery
is declared through `recovery_actions`, `hand_guiding_actions` and `drag_selections`.
The serving session publishes `actions`, `hand_guiding_actions`, the active `task`,
and `selections`. RoboGUI shows clear, drag, release and idle Home from that declaration.
Drag and both Release variants use the same declared arm selections; single-arm
assemblies expose only their configured arm. A stale selection is rejected before
ending a rollout or starting recovery. Return Home operates on all configured groups,
independently of the hand-guiding selection.
The runtime publishes `home_available` from the implemented assembly Home operation;
inheriting the base rejecting method does not provide Home. Native external-action
owners do not expose in-rollout Home. RoboGUI disables Return Home when unavailable
and labels Finish as **Finish rollout** instead of **Finish & Home**. Ordinary runtime
shutdown homes connected assemblies with an implemented Home before closing resources.
The retired `home_on_close` option cannot override this policy. An external owner
retains any assembly-required `home_before_close` behavior, including YAM's measured
release gate. Franka has no Home implementation and stops/closes without homing.

For a focused review, trace configuration -> factory -> controller/assembly ->
runtime caller. Record the required behavior, actual SDK mapping, discrepancy and
owning file. Verify offline loading, read-only lifecycle, shared ownership, normal
dispatch, stop/home followed by further use, and partial-startup cleanup as relevant
with a fake SDK. State separately what still needs the real SDK or authorized hardware.

## Arm and shared controller

Start at `manimux/embodiments/arm/base.py`: `ArmBase`, `ArmController`, `ArmState`
and `ArmModel`. Put a vendor implementation and its assets under
`manimux/embodiments/arm/<name>/`; declare it in
`manimux/configs/embodiment/arm/<name>.yaml` via `implementation: module:Class`.

- `ArmBase.load_model(**options)` supplies geometry/visual resources without opening
  hardware. `num_joints` counts all coordinates owned by the component, including
  any integrated gripper; YAM's seven coordinates are not seven arm joints.
- The current controller interface owns `connect`, `get_states`, `send_commands`,
  `stop`, `close`; arm components expose their controller and channel. A shared
  controller is one resource owner, not one independently opened session per arm.
  Follow the assembly's lifecycle deduplication and batch dispatch. State the SDK's
  actual dispatch guarantees; a batch API does not prove simultaneous motion.
- `ArmController.command_bounds()` optionally returns per-channel `(coordinates, 2)`
  lower/upper pairs in command units, without hardware I/O; the default is empty.
  PiPER returns its station calibration bounds plus normalized gripper range.
  `RobotBase.command_bounds()` maps these into group order, appending [0, 1] for a
  separate gripper. Do not substitute model-training envelopes for device bounds.
- `validate_commands()` must be pure: `RobotBase.send_command()` runs it for all
  controller batches before either receive-only return or any physical dispatch.
  Device readiness remains in the execution path. `command_status()` returns a
  copied diagnostic of the last attempt: stage, completed and attempting channels,
  dispatch-attempt flag and group/channel mapping. A failing batch may be partly
  submitted; completed submission is not measured arrival.
- `ArmState` is measured feedback with its timestamp/sequence. A submitted target
  is not measured state, and command return is not arrival at the target.
- Keep SDK address/channel conversion in this component. Document connect, stop,
  home, dry-run and cleanup effects. Do not silently add fault clearing or homing.

Verify offline model construction, state/order/units and command dispatch using a
fake SDK. For shared controllers, verify lifecycle calls happen once per owner and
commands reach the intended channels. For lifecycle changes, cover partial startup
and cleanup. Do not edit an unrelated driver's session implementation to integrate
a new body.

## Gripper and other tools

Read `manimux/embodiments/end_effector/base.py` and `gripper.py` in that directory.
`EndEffectorBase` defines lifecycle and typed state/command capabilities;
`GripperBase` specifically uses `GripperState` / `GripperCommand` with normalized
travel: 0 closed, 1 open. This is not finger-pad distance or a universal tool format.

Put implementations under `manimux/embodiments/end_effector/<name>/` and component
YAML under `manimux/configs/embodiment/end_effector/`. Keep flange-to-TCP geometry in
the offline tool model (`manimux/kinematics/tool.py`); a tool-mounted sensor still
implements the sensor interface. A borrowed SDK connection must not be independently
closed by the borrower.

Current assembly/executor action handling supports zero or one scalar gripper
coordinate per group. A multi-coordinate hand or suction capability requires an
explicit end-to-end extension; inheriting `EndEffectorBase` alone is insufficient.
Test the actual capability's units, command mapping and measured feedback, not a
fake scalar adapter that discards unsupported coordinates.

## Assembly, coordinate layout and geometry

Read `manimux/embodiments/robot/base.py`, its sibling `__init__.py`, and
`manimux/embodiments/layout.py`. `RobotModel.from_config()` loads component models;
`build_robot(config, clock)` selects `robot.type` through built-ins, a
`module:factory` reference or the `manimux.embodiments.robot` entry-point group.
The result exposes `RobotBase` to the runtime, not a vendor-specific robot object.
Reuse generic assembly/model behavior; keep any required hardware assembly factory
under `manimux/embodiments/robot/<name>/`.

The component `implementation` selector is used by `RobotModel.from_config()` to
load offline models; it does not automatically construct hardware controllers.
The selected robot factory must create/reuse controllers, instantiate arms and
tools, and pass them to the existing assembly. Consume the station's `hardware`
and `component_hardware` bindings there. For example, `YamRobot` explicitly creates
`YamController`; changing only its arm YAML does not create another vendor's session.
Keep constructors inert and open SDK connections only through `connect()`.

The optional `controller: module:Class` field in an arm component YAML is consumed
by `manimux.embodiments.robot.can_arms:build_robot` for the X5/PiPER adapters.
It selects an `ArmController` for each independently owned CAN channel; the
generic `RobotModel` loader still loads only geometry. The factory uses the
existing station's component hardware bindings and `RobotBase` lifecycle/batching.
It does not introduce another runtime registry or loop. See the
[SDK adapter guide](../../manimux-station-setup/references/can-arms.md) for unsupported Home/read-only operations
and the X5 polling-time limitation.

The runtime uses `connect()`, `get_state() -> RobotState`,
`send_command(RobotCommand)`, `home()`, `stop()` and `close()`. Group names and
coordinate order must agree between state, command and offline model. The current
execution path sends joint-position targets; an EEF policy must be converted by
its action adapter before reaching this boundary.
On normal rollout completion, the runtime stops physical control before flushing
and finalizing the recording. A failed stop leaves an aborted partial episode;
the cleanup block still retries stop/close. Finalization must not keep control
active solely while saving data or publish success before stop succeeds.

Robot YAML in `manimux/configs/embodiment/robot/` names `components`, their YAML
references, mount relationships and `groups`. Component YAML declares coordinate
ownership, for example:

```yaml
# An arm with its integrated scalar gripper (fragment).
type: arm
implementation: manimux.embodiments.arm.yam:YamArm
action_layout: {arm_dofs: 6, gripper_dofs: 1}
options: {}
```

A bare arm declares zero gripper DOFs; an independent scalar gripper contributes
one gripper DOF. Groups concatenate arm then tool coordinates. Do not repeat a
global gripper count for heterogeneous groups. `assembly_action_contract()` and
`group_layouts()` resolve this for clients/adapters/executors; the model checks it
against actual geometry. `end_effector: null` means no separately attached tool,
not removal of an integrated arm's gripper.

`ManipulatorKinematicsBase` describes complete TCP kinematics;
`FlangeKinematicsBase` supports a bare arm composed with a tool. Read
`manimux/kinematics/base.py` and `composed.py` for the exact FK/IK signatures.
Robot geometry, tool offsets and coordinate definitions belong here. Solver
requirements must remain explicit through supported solver configuration/API;
never drop bounds, tolerances or failure behavior to fit a common signature.
RoboGUI display placement is not a calibrated arm-base transform.

Adapters must obtain this geometry from the resolved assembly/model, not require the
user to duplicate it as per-model calibration. If an integrated kinematics model
exposes TCP FK/IK but not a needed flange/tool relationship, first inspect its source
geometry and add reusable model access when available. The absence of that accessor
does not mean the physical station needs remeasurement. Do not assume every current
model already implements `FlangeKinematicsBase`. Installation-dependent calibration
must have its own verified station source; an unavailable measurement remains unknown.
Follow the shared [information ownership workflow](policies.md#resolve-geometry-and-station-facts-before-asking-the-user)
and test with nonidentity geometry to demonstrate that the selected model's values
reach the adapter without hardcoded defaults or hardware connections.

YAM IK settings belong in the arm component's `options`, shared by model adapters.
`pos_threshold` defaults to `0.0001` metres. An explicit YAML `null` disables only
final position-residual acceptance in chunk and step IK. Optimization still pursues
the original position objective and 0.1 mm convergence target; this is not a speed
setting. Orientation acceptance, finite-value checks and model joint limits remain.
Step IK still obeys its computation budget. Residuals can be arbitrarily large when
position acceptance is disabled; do not infer task accuracy from acceptance.
Keep the default enabled unless explicitly selected otherwise.

Verify the new group's actual layout, FK/IK and component command split; include
no-gripper or mixed layouts if supported. A known tool offset is a useful offline
fixture to check geometry propagation, not a new runtime calibration mechanism.

## Cameras and runtime sensors

Read `manimux/embodiments/sensor/base.py` and its sibling `__init__.py`.
`SensorBase` is inert until `start()`, `read()` returns a `SensorFrame` or named
bundle, and `close()` releases resources including after partial startup.

- Physical camera: implement under `manimux/embodiments/sensor/<name>/`. `build_camera()`
  selects the component implementation and supplies `name`, `clock` and configured
  options, including `camera_serial`. Device SDK startup belongs in `start()`.
- Runtime/network source: `build_sensor(config, clock)` selects `sensors[].driver`
  through `manimux.embodiments.sensor`; it returns the same sensor interface.
- Declare device defaults in `manimux/configs/embodiment/sensor/`; camera sets in its
  `cameras/` directory use readable names such as `realsense_3_views.yaml`.
  Serial numbers/endpoints belong in the private station file.

Trace physical binding -> assembly component -> camera-server source name ->
`sensors[].options.camera_names` -> `policy.adapter.camera_map` -> model input.
These are selection and naming boundaries, not three independent device definitions.
Reuse `manimux/servers/camera/server.py`; do not add per-vendor branches when the
component's `SensorBase` implementation can supply the frame.

Frames contain RGB uint8 HxWx3 data, capture timestamp and sequence. Preserve the
image and capture metadata together across repeated reads. Distinguish device
capture, host receipt and runtime read time; inspect the existing camera-server
driver/timestamp conversion before claiming all paths preserve acquisition time.
The timestamped PUB sensor keeps only the latest bundle, tolerates incomplete camera
startup until its deadline, and then requires every selected camera. Optional PUB
`camera_meta` is retained in each `SensorFrame`; station bridges must identify whether
timestamps describe exposure, host receipt or publication. Independent SDK processes
can publish through this existing protocol when their Python environments differ.
Validate frame age after receiving/deserializing the bundle. Bracket wall-clock
reads with monotonic samples when estimating/checking clock offset; thread
preemption must not turn a fresh frame into a future frame or mimic a clock jump.
Cached source timestamps remain unchanged, and actual stale/backward frames and
wall-clock jumps still fail explicitly.
Verify RGB ordering, cache behavior, selected source names and lifecycle with a
fake device. Optional RGB-D frames carry aligned metric `depth` plus `camera_meta` alongside
that capture's RGB/time/sequence. `SensorBase.read_rgbd()` rejects unsupported
sensors; the shared camera REQ `rgbd` command carries these extra fields while
`obs`/PUB stay RGB-compatible. RealSense requires aligned depth and `flip: false`.
Other channels still need a declared representation, not an RGB reinterpretation.

## Concrete implementation starting points

The [Franka component](../../../../manimux/embodiments/arm/franka/README.md)
implements a standalone seven-joint FCI controller and offline FR3 model.
`robot/franka` assembles it through `RobotBase`, optionally with the independent
[Robotiq component](../../../../manimux/embodiments/end_effector/robotiq/README.md).
The normal policy/adapter/executor path does not import RPent or RLinf. Franky owns
native persistent joint impedance tracking; the optional agent integration remains an upper
layer. Read-only Franka sessions use pylibfranka because Franky's constructor sets
collision thresholds. Home, startup motion and idle recovery are not implemented.
Both SDK paths acquire a local per-address FCI lease before connection. A second
ManiMux client is rejected before SDK construction, including receive-only probes.
Stop keeps the lease; close releases it only after successful shutdown. Diagnostics
must use the active owner's feedback. Raw SDK clients, address aliases and other
hosts require separate operator coordination; the local lease cannot exclude them.
Use the component runbooks for units, ownership, feedback and stop semantics.
Franka contains the SDK's calling-thread priority changes: connect restores the
caller's scheduling policy, while native control startup inherits the SDK's
real-time policy before the caller is restored. Keep general runtime work at its
original priority; do not lower the native control thread's real-time priority.
Accepted joint references are submitted unchanged, including startup. Host guards
for reference steps, measured-to-target distance and measured speed default to null;
bounded tests can explicitly configure a scalar or seven per-joint limits. Rate shaping
belongs to the shared executor. Native collision protection, model bounds, feedback
freshness and fault propagation remain active. There is no promised physical
acceleration/jerk bound from host checks. The selected Franky build must expose native
JointReferenceHandle and JointImpedanceTrackingMotion before any executing connect.
The Linux x86_64 / CPython 3.12 SDK profile is hash-pinned in
`scripts/setup/franka-linux-cp312.txt`; the package extra installs the read-only SDK.

| Component | Read first | Selecting configuration | Focused check |
| --- | --- | --- | --- |
| Arm/controller | `embodiments/arm/yam/arm.py` | `configs/embodiment/arm/yam.yaml` | Fake SDK: shared ownership, complete group dispatch, measured state |
| Independent gripper | `embodiments/end_effector/taccap/end_effector.py` | `configs/embodiment/end_effector/taccap.yaml` | Map normalized travel and preserve the borrowed/owned connection distinction |
| Assembly | `embodiments/robot/yam/robot.py`, `robot/base.py` | `configs/embodiment/robot/yam_dual.yaml` | Load model offline; compare group order and arm/tool coordinate split |
| RGB camera | `embodiments/sensor/realsense/sensor.py` | `configs/embodiment/sensor/realsense.yaml` | Real factory with fake SDK: RGB, capture metadata, lifecycle |
| Network sensor | `embodiments/sensor/camera_server/` | Experiment `sensors[].driver` | Preserve stream names and cached-frame metadata |

Paths in this table are below `manimux/`. Copy the boundary and lifecycle pattern;
device-specific gains, SDK calls, timeouts and geometry remain vendor choices.

For a bare arm component and a separate tool, the assembly fragment is:

```yaml
components:
  main_arm:
    config: ../arm/my_arm.yaml
  tool:
    config: ../end_effector/my_gripper.yaml
    parent: main_arm.flange
    # Example identity mount; replace with the actual flange-to-tool transform.
    mount: {xyz: [0.0, 0.0, 0.0], rpy: [0.0, 0.0, 0.0]}
groups:
  arm: {arm: main_arm, end_effector: tool}
```

The referenced component YAMLs each declare `type`, `implementation: module:Class`
and their model options. The arm declares its actual `action_layout`; the tool's
model contributes the supported tool coordinates. A new tool implements
`load_model`, `connect`, `get_state`, `send_command`, `stop` and `close` through
`EndEffectorBase`/`GripperBase`. It owns the scalar gripper state/command meaning;
the offline tool model owns flange-to-TCP geometry. Put actual addresses and serials
in the private station file. This is a configuration fragment, not a shipped driver.

## Hardware exit safety

An episode failure and actuator release are separate operations. Before granting
release, audit the loaded driver rather than relying on method names.

YAM declares the code-owned capability `home_before_close=True`. After a completed
connection, every runtime exit requests Home, including software failures, step limits
and legacy Finish requests with `home=false`. Configuration cannot disable it.
A stop exception does not by itself establish a hardware fault: Home checks all
controllers before motion and refuses unhealthy control. Other embodiments must
implement and validate this capability before opting in. YAM
requires both CAN feedback and SDK state to advance, and rejects motor error codes
other than the SDK's healthy `0x1`. After all controllers connect, initial-pose movement runs both arms concurrently,
just like Home; SDK connection/calibration itself remains sequential. Start/Home retain the SDK's 50-interval linear
waypoints, now checking feedback before each command and cancelling peer motion
on a move failure. These checks are outside the policy control tick. Home arrival is measured before opening grippers
and again before closing control. Camera health is rechecked after policy startup,
before robot connection; a camera can still fail during the subsequent connection.

`ArmController.validate_release()` is the device-owned release gate. YAM requires
stationary zero Home (six arm joints; gripper excluded); tolerances are explicit in
`embodiment/arm/yam.yaml`. `RobotBase.close()` validates every opened controller
before releasing any, including partial-connect cleanup. If validation fails, the
owning process retains control and waits for the persistent RoboGUI Hardware recovery
panel. The operator can confirm a clear Home path and retry Home, or confirm physical
support of ALL arms and secured payloads before releasing motor control. There is no
terminal input. Session/episode identity and a fresh gate token scope every request;
Home failures invalidate that token. Home retry rechecks every controller before close
and does not grant supported-release permission. GUI loss, timeout and interrupts do
not approve release; without a recovery channel the process remains blocked.
Direct YAM controller close repeats the check to cover other callers and late faults.
This retains existing control resources; it cannot restore a broken drive, protect
against power loss/SIGKILL, or establish collision clearance for a Home path.

An aborted Home is only retried after an explicit GUI action. The blocked state means
the current runtime still owns the hardware; do not kill it or launch another CAN owner. The ordinary RoboGUI Finish & Home action explicitly requests Home, independently
of `home_on_close` and idle recovery capabilities. Verify the actual GUI button request
through runtime shutdown, including a false legacy config: stop, Home, measured
arrival, then close. Other embodiments' explicit hand-guiding transitions may request
no Home; YAM's mandatory Home capability cannot be bypassed that way. Receive-only YAM may
still have an enabled SDK session: closing that session also requires the release gate.

Audit findings for other controllers must remain explicit: Tianji `stop()` requests
mode 0 (disable), PiPER close disconnects transport, and ARX X5 delegates stop/close to
its SDK process. None is automatically equivalent to YAM position hold. Their physical
braking/support behavior and fault-Home paths require vendor-specific validation; the
base no-op release hook is not evidence of physical safety for a new controller.
