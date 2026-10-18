# Flexiv Rizon4s dual-arm inference

Two independent RDK sessions use the standard ManiMux robot, adapters, executors,
recording and RoboGUI. Each arm contributes seven absolute joint angles in radians,
in RDK J1..J7 order. There is no dummy gripper. Physical left/right mapping comes
from station serials, never discovery order.

## Configuration

- Assembly: `manimux/configs/embodiment/robot/flexiv_dual.yaml`.
- Station template: `manimux/configs/local/flexiv_example.yaml`.
- Joint inference: `manimux/configs/examples/flexiv_joint.yaml`.
- Flange-pose inference: `manimux/configs/examples/flexiv_pose.yaml`.
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

After supplying the model and cameras, launch the existing runtime:

```bash
python -m manimux.robogui.dashboard --robot flexiv
python -m manimux serve --config manimux/configs/examples/flexiv_joint.yaml --local <station.yaml>
# Or select flexiv_pose.yaml with the matching model-side contract.
```

RoboGUI preset name: `flexiv`; runtime identity: `flexiv-dual`. The preset
places both bases on the top beam of a vertical support column with their z axes
pointing down and a common x/y orientation; the differing HOME joints reproduce
the photographed left/right poses. `xyz`/`rpy` are display-only, not calibrated
inter-arm geometry or hardware Home targets; adjust them to the measured station
mount. Home and idle fault/drag recovery are not advertised.

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
| Send | Validate joints against device limits with margin, check freshness/health, prepare NRT joint mode, submit zero terminal velocity and configured rate limits |
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
targets are rejected rather than clipped.

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

When grippers arrive, add actual component drivers and calibrated flange mounts/TCPs
through existing end-effector interfaces, extend assembly construction, and update
the action contract/model data together. This assembly explicitly rejects an
unimplemented separate gripper.

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
# scripts/validation/flexiv_readonly_probe.py --config manimux/configs/examples/flexiv_joint.yaml \
#   --local <station.yaml>
```
