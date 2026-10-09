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
| `home()` / robot assembly | Perform the explicitly defined home operation when supported. State the destination, tool behavior, completion guarantee and post-home control state. | Configured joint trajectory or a documented vendor Home operation. A RoboGUI initial pose does not define hardware Home. |
| `recovery_actions`, `drag_selections` / robot assembly; `clear_errors()`, `drag()`, `recover_home(stop)`, `recover_drag()` | Optional idle recovery that RoboGUI may request between rollouts. `recovery_actions` declares the supported actions (`clear_error`, `home`, `drag`; none by default) and `drag_selections` maps each drag label to robot groups. The serving session only schedules them, each on a fresh, disconnected assembly constructed with its configured options. Home implementations must observe the stop event and complete cleanup before returning. | Vendor fault reset, hand-guiding mode, tool behavior and any preparation, such as clearing faults before connecting. |
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
is declared through `recovery_actions`, which the serving session and the runtime read
instead of a robot type; some RoboGUI layouts still branch on robot names. These are
existing integration gaps,
not a complete generic capability mechanism. Resolve the affected caller contract
when adding a capability; do not invent a `supports_home` field that nobody reads.

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
[SDK adapter guide](../usage/can-arms.md) for unsupported Home/read-only operations
and the X5 polling-time limitation.

The runtime uses `connect()`, `get_state() -> RobotState`,
`send_command(RobotCommand)`, `home()`, `stop()` and `close()`. Group names and
coordinate order must agree between state, command and offline model. The current
execution path sends joint-position targets; an EEF policy must be converted by
its action adapter before reaching this boundary.

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
Verify RGB ordering, cache behavior, selected source names and lifecycle with a
fake device. Depth/tactile channels need a declared representation, not a silent
reinterpretation of the RGB frame.

## Concrete implementation starting points

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
