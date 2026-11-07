# Flexiv Rizon4s dual-arm inference

Two independent RDK sessions use the standard ManiMux robot, adapters, executors,
recording and RoboGUI. Each arm contributes seven absolute joint angles in radians,
in RDK J1..J7 order. There is no dummy gripper. Physical left/right mapping comes
from station serials, never discovery order.

## Configuration

- Assembly: `manimux/configs/embodiment/robot/flexiv_dual.yaml`.
- Station template: `manimux/configs/local/flexiv_example.yaml`.
- Joint inference: `manimux/configs/experiments/example-flexiv/flexiv_joint.yaml`.
- Flange-pose inference: `manimux/configs/experiments/example-flexiv/flexiv_pose.yaml`.
- Offline display: `python -m manimux.robogui.dashboard --robot flexiv --demo`.

Copy the station template to a private file. Supply the two serials, matching SDK,
and optional host NIC IPv4 addresses in `interfaces`. `sdk_path` is an absolute
directory containing the binding and its shared dependencies. No teleoperation
repository is imported at runtime. Binding names are `left_flexiv`, `right_flexiv`,
`left_wrist_camera`, and `right_wrist_camera`.

Both inference examples require a trained policy service and camera streams
`left_wrist` and `right_wrist`. They are deployment templates, not trained policies
or simulations. Both have `execute: false`. No camera type/serial or checkpoint
is invented. Select camera components using the existing camera-server recipe
schema after installing cameras; model image keys use `policy.adapter.camera_map`.
The illustrative 20 Hz action grid and 16-step horizon must match the eventual
checkpoint. NRT command frequency is 50 Hz and must stay within 1..100 Hz.

Policy target tracking selects `hardware.control_mode`: `joint_position` (default)
or `joint_impedance`. Experiments can override it per arm through
`robot.options.component_hardware`. In impedance mode, `joint_stiffness_scale`
(default 1.0, range [0,1]) multiplies that arm's device-reported `K_q_nom`, and
`joint_damping_ratio` (default 0.7, range [0.3,0.8]) applies to all seven joints.
Both flat and grouped APIs set impedance after switching modes and before sending
the first target; Stop/resume reapplies the settings. Invalid parameters or a
failed mode/impedance setup prevent target dispatch. Connection and read-only
sessions do not switch modes or set impedance. Runtime metadata records the
selected mode, gains and the last successfully configured policy stiffness.

Home explicitly keeps joint position control, independently of policy mode.
Hand guiding retains its separate low-stiffness settings; the next policy send
reapplies policy impedance. This does not change target units, internal trajectory
velocity/acceleration limits, joint margins or gripper control. Configuration is
loaded by the runtime process; editing YAML does not switch an active controller.

After supplying the model and cameras, launch the existing runtime:

```bash
python -m manimux.robogui.dashboard --robot flexiv
python -m manimux serve --config manimux/configs/experiments/example-flexiv/flexiv_joint.yaml --local <station.yaml>
# Or select flexiv_pose.yaml with the matching model-side contract.
```

RoboGUI preset name: `flexiv`; runtime identity: `flexiv-dual`. The preset
places both bases on the top beam of a vertical support column with their z axes
pointing down and a common x/y orientation; the differing HOME joints reproduce
the photographed left/right poses. `xyz`/`rpy` are display-only, not calibrated
inter-arm geometry or hardware Home targets; adjust them to the measured station
mount. Recovery is implemented by the Flexiv assembly; see
`manimux/embodiments/robot/flexiv/README.md`. Home requires explicit assembly
targets. Hand guiding requires an explicit station setting and the flat RDK API.

## Policy contracts

Both paths use `xpolicylab_ws` with left/right layouts
`{arm_dofs: 7, gripper_dofs: 0}`. The existing wire codec supports this layout.

- Joint response: each step contains `left_arm_joint_state` and
  `right_arm_joint_state`, each seven absolute joint angles in radians.
- Pose response: each step contains `left_ee_pose` and `right_ee_pose`, each
  `[x, y, z, qw, qx, qy, qz]`. Translation is metres and rotation a unit quaternion.
  The target is the **bare flange in that arm's physical base_link frame**.
  The server handshake must declare
  `action_semantics: absolute_per_arm_base_xyz_wxyz`. Pose observations use the
  same offline FK and include these pose keys alongside measured joints.
- Neither contract contains gripper keys. Seven pose numbers are not seven joint
  angles; `action_format` and the adapter select the interpretation explicitly.

The shared PoseAdapter now accepts zero or one gripper coordinate. IK seeds each
row from the preceding solution and rejects the whole chunk on failure. The pose
example decodes in a separate process, outside the command loop. Specialized
model sampling is not implied. The executor limits/smooths joint targets; RDK
also interpolates them. This path does not use Cartesian impedance/force control
or reproduce teleop's nullspace controller. There is no inter-arm collision planner.

Do not select `execution.ik.mode: step` for this bare-arm assembly. Step IK
currently packs one trailing normalized gripper coordinate per group, so the
runtime rejects step mode for zero-gripper groups; use the pose template's chunk
decoding instead.

Nominal numerical IK costs roughly 25 ms per solve on this workstation; a moving
16-step dual-arm chunk decoded in about 0.7 s. Process decoding keeps that work
outside the command loop, but the deployment must still tolerate the resulting
hold and `max_plan_age_s`; benchmark the actual station before choosing the
horizon or control rate.

## SDK environment

ManiMux requires Python 3.11 or 3.12. A `flexivpy.cpython-310-*.so` cannot be loaded
in either. Build/install the binding for the hardware interpreter's ABI.

The working Rizon reference uses a custom flexivpy binding to RDK 1.9.3 with
firmware v3.11.2. Rebuild it with that Python's headers and pybind11 plus matching
RDK headers/static library/shared dependencies:

```bash
<python-3.12> -m pip install -e '.[xpolicylab]' pybind11
cmake -S <binding-source> -B <new-build-directory> \
  -DPYTHON_EXECUTABLE=<absolute-python-3.12> \
  -Dpybind11_DIR=<environment>/lib/python3.12/site-packages/pybind11/share/cmake/pybind11 \
  -DRDK_ROOT=<matching-rdk-install>
cmake --build <new-build-directory> -j2
```

Keep the resulting module and its deps directory together and set sdk_path. Use
a separate source/build copy to preserve an existing teleop installation. SDK
binaries are local dependencies, not distributed in this repository.

The driver also maps the grouped API (RobotInfo.single_arm_groups, ServoOn,
NrtJointPositionCmd) used by newer RDK releases. A configured group must select
exactly one single-arm group. API compatibility does not establish firmware
compatibility: newer RDK primarily targets Enlight/MICO. Use the SDK matching
Rizon firmware, rather than upgrading it to match an example.

## Lifecycle and limits

| Operation | Behavior |
| --- | --- |
| Offline model/factory | No SDK import, connection or actuation |
| Connect | Construct session, read limits/feedback; no Home, fault clear or motion-mode switch. Enables only when the Enable row applies. |
| Enable | Only with experiment execute=true AND station enable_on_connect=true; otherwise sending requires an already operational arm |
| State | Measured q; repeated device timestamp retains host time and sequence |
| Send | Reject or clip policy joints according to configured device-limit handling, check freshness/health, prepare NRT joint mode, submit zero terminal velocity and configured rate limits |
| Stop | Blocking RDK Stop to IDLE for owned active sessions; no torque-off or Home; next send prepares NRT again |
| Close | Stop if owned, release SDK object; failed stop retains the handle for retry |
| Read-only | No Enable/ServoOn, mode change, target, Stop, fault clear or Home, including failed-startup cleanup |

Feedback time is host **first observation of a new device timestamp**, not proven
acquisition or network receipt time. Frozen timestamps expire after feedback_timeout_s;
backward timestamps require reconnect. The assembly attempts both controllers on
stop/cleanup and reports failures. Sends are sequential and non-atomic. Native RDK
calls can block; this integration does not promise bounded emergency-stop latency.

Default controller limits: 0.15 rad/s, 0.3 rad/s², 2-degree joint-limit margin.
Examples set matching executor rate limits. `position_limit_abs: null` avoids
incorrectly clipping J6 at pi; the driver checks actual per-joint limits. Invalid
targets are rejected by default. To continue inference and execution when a policy
target reaches a limit, select `joint_limit_policy: clip`, `joint_margin: 0.0`
and `joint_clip_margin: 0.0` per arm under
`robot.options.component_hardware`. Immediately before SDK submission, finite
policy targets saturate to each device-reported joint interval inset by
`max(joint_margin, joint_clip_margin)`. With both margins zero, clipping uses
the SDK bounds exactly, without the previous 0.1-degree inward offset.
Targets inside that interval pass unchanged before optional dynamic shaping. This does not
wrap angles or use the nominal URDF limits.

`joint_target_clipped` logs the channel, joint numbers (1..7), requested/submitted
degrees and cumulative clipped-command count on entry, severity changes and at
most once per second during unchanged clipping severity. Recovery is logged when targets return inside the
interval. Runtime metadata includes the selected policy and resolved clip bounds.
The ordinary command recording remains the executor output before driver clipping;
the clipping log reports SDK-submitted values, not measured arrival. Raw model
predictions remain unchanged. Home targets remain strictly validated and are never
clipped. Nonfinite/malformed targets, stale feedback and SDK faults still fail.
Editing these settings requires a runtime restart to apply.

The Pi05 profile sets `joint_limit_tolerance: 0.0005235987755982988` (0.03 degree).
This is a diagnostic threshold for the controller's input target (executor output),
relative to raw SDK limits. It never expands the final command bounds. Clipping
within tolerance logs `severity=within_tolerance` at INFO; larger excess logs
`severity=beyond_tolerance` at WARNING immediately, even after a mild clip in the
same second. Both continue through hard clipping; large finite targets alone do
not abort execution. Metadata records the tolerance, successful command count
beyond tolerance and maximum requested excess. Failed sends do not update them.
Raw model horizons remain unchanged in their existing recording; these driver
counters are not counts of raw prediction rows or measured-limit excursions.

### Collection-aligned joint safety

The Pi05 Flexiv experiment also selects `joint_safety` per arm:

```yaml
joint_safety:
  slowdown_zone: 0.17453292519943295  # 10 degrees
  tracking_limit: 0.08726646259971647  # 5 degrees
  max_dt_s: 0.016
  tracking_timeout_s: 0.5
```

This implements the joint envelope from collection's `FlexivDiffIK.solve`:
outward speed decreases linearly in the last 10 degrees, using the nearer of
the previous submitted reference and measured position. Inward motion remains
available. The command rate is capped by the smaller of `max_velocity` and all
SDK `dq_max` values; elapsed command time is capped to avoid banking delayed
ticks. Each step intersects these bounds with the measured +/-5-degree tracking
interval. The first policy send, including after Stop, seeds from bounded measured
joints. Failed SDK sends do not advance the reference.

If the previous reference leaves the tracking interval or the step is infeasible,
the arm holds its last reference; persistent failure raises after 0.5 seconds and
the shared runtime stops execution. Teleop instead disengages its clutch and
rebases. Feedback may slightly exceed raw SDK bounds, but excess beyond the
5-degree tracking envelope fails. A device `recovery()` indication also fails;
there is no automatic recovery. Home bypasses dynamic policy shaping and keeps
strict target validation; hand guiding retains its independent limits.

Hard clipping remains an independent final command bound, inset by
`max(joint_margin, joint_clip_margin)`. This profile uses zero extra collection
margin and no hard inset. Dynamic shaping uses the same SDK bounds. It does
not add Cartesian IK, nullspace avoidance, workspace/collision checking or alter
raw model predictions. RDK still owns physical interpolation and acceleration;
a bounded submitted reference does not prove bounded measured motion.

`joint_target_guarded` reports requested/submitted angles at most once per second.
`joint_target_clipped` separately reports requested, hard-bounded and submitted
angles. Metadata records the safety settings, effective velocity, guard count
and last successful SDK submission. Tick command recordings remain executor
outputs, not driver-shaped targets or measured arrival. `joint_safety: null`
keeps the generic component's previous behavior; the Pi05 profile enables it.

The November 2026 stack-cube collection stores controller `Robot.actions().q_d`,
not the host's bounded IK command. Its recorded margin is zero. A 2-degree
deployment margin therefore rejects valid collection-region targets near J5's
-170-degree lower limit. Slight recorded q_d/encoder excursions and learned
predictions beyond the training range also require a final command clip; do not
expand the SDK limits to the dataset minimum.

## Geometry and future grippers

model.urdf and meshes/ are generated/copied from the official
[flexiv_description humble-v1](https://github.com/flexivrobotics/flexiv_description/tree/humble-v1),
revision `92ef7865d76585e6e08d291bdfe652d32f7740f4`, with rizon_type=Rizon4s,
no gripper/external force sensor and zero world mounting. Mesh references were
made relative. See LICENSE.txt (Apache-2.0). ROS/xacro is not needed at runtime.
FK uses base_link to flange. Bounded numerical IK accepts residuals <= 1e-4 m
and 1e-4 rad. Solver tolerances do not establish physical accuracy or continuous paths.

The shipped model is nominal. For calibrated arms, select private URDFs with each
assembly component's `options.urdf_path` (absolute path), retaining joint1..7,
base_link and flange. Use the same assembly in runtime and RoboGUI. Device joint
origins/limits and mounting transforms must not become shared model defaults.

RDK flange_pose can include mounting/world rotation. In the supplied 2026-10-12
saved station data, physical-base to RDK-world rotations are Ry(-90 degrees) for
left and Ry(+90 degrees) for right. Convert recorded RDK poses back to the declared
arm-base frame before training/deployment. This is station evidence, not a universal
Rizon transform. Validate independent poses before physical Cartesian deployment;
one saved pose does not validate every workspace configuration.

The bare-arm assembly remains available. The separate
`flexiv_omnigripper_dual.yaml` assembly adds independent OmniGripper components;
see `manimux/embodiments/end_effector/omnigripper/README.md` for its read-only
validation and uncalibrated flange-reference geometry.

## Validation

Local fake-SDK tests cover flat/grouped APIs, read-only lifecycle, enabling, limits,
stale/regressing feedback, stop/resume, retryable cleanup and dual-arm failures.
Loader/wire-codec tests decode joint and pose chunks, including unreachable-pose
rejection and regression coverage for the existing one-gripper PoseAdapter path.
They do not establish robot motion, camera readiness or trained-model task success.

The first live gate is a read-only feedback/FK probe. It refuses an executing
experiment or `enable_on_connect`, connects the configured serials, prints fresh
measured joints and base-frame flange poses, and closes without Enable, mode
change, target, SDK Stop or Home:

```bash
envs/flexiv/.venv/bin/python scripts/validation/flexiv_readonly_probe.py \
  --config .local/flexiv/joint.yaml --seconds 5
# Public templates need an explicit station:
# scripts/validation/flexiv_readonly_probe.py --config manimux/configs/experiments/example-flexiv/flexiv_joint.yaml \
#   --local <station.yaml>
```

## NRT hand guiding

Drag/Release use the existing flat `flexivpy` binding. Python owns connections,
mode changes, monitoring and cleanup. Drag refreshes a low-stiffness impedance
reference from measured joints at up to 100 Hz; Release uses zero stiffness and
monitoring. The assembly owns `hardware.drag_stiffness_scale` and
`hardware.hand_guiding_motion` (frequency, damping ratio, velocity/acceleration
limits). Guiding limits are independent of policy and Home limits. Mode, health
and feedback checks remain active, without applying policy target margins to
manual guiding. Stop uses blocking RDK Stop to IDLE. Active tool/load settings
remain unchanged. Physical acceptance of tuned parameters is still required.
