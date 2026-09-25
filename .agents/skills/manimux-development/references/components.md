# Components: hardware ownership and offline models

Paths are relative to the repository root. Read the current interface before copying
an implementation; controller/session work can evolve independently.

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
Viewer display placement is not a calibrated arm-base transform.

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
