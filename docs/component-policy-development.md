# From components to policy actions

Use `configs/experiments/put_bottles/pi05/yam_pi05_joint.yaml` as the complete
example. An experiment selects a robot assembly, sensor sources, a policy client
and action adapter, an inference algorithm, and an executor. Machine bindings
remain in the private station file.

## 1. Describe the components

A component YAML declares its independent action coordinates:

```yaml
# Integrated arm with its own scalar gripper.
action_layout: {arm_dofs: 6, gripper_dofs: 1}
```

A bare arm declares only `arm_dofs`; an independent gripper declares
`gripper_dofs: 1`. The robot's existing `groups` mapping assembles these components.
For example, the left group may combine arm + gripper while the right group has
only an arm. Each group's coordinates are arm-first, followed by tool coordinates.
Joint names, units and geometry remain in its component model; the declared
width must match the model when it is constructed. A two-finger coupled gripper
usually has one independent opening coordinate, not two commands.

`assembly_action_contract()` reads these declarations without importing device
SDKs. `load_config()` resolves per-group `group_layouts` and `group_dims`; adapters
and clients receive the resolved layouts, and executors receive indices only for
groups that actually have a gripper. `RobotModel.action_layouts` checks the same
declaration against the assembled geometry. A connected robot's hardware dispatch
still follows component ownership: YAM sends all seven coordinates to its integrated
controller; a separate arm and gripper receive separate commands.

Existing `action_contract` declarations remain accepted for assemblies not yet
migrated, including Tianji. A uniform `gripper_dofs: 0` produces no gripper indices.
Current executors support no gripper or one normalized opening per group; multiple
independent tool coordinates require a corresponding executor/tool capability and
are rejected rather than partially controlled. The scalar `GripperBase` is not a
multi-finger hand or suction-command interface.

## 2. Keep geometry with the robot

`EdgeRuntime` passes `robot.kinematics` to the action adapter. DP, OpenWAM and XR1
use that grouped model. When constructed in an independent process without an
injected model, they load `RobotModel.from_config(robot["config"])`. No hardware
object crosses the process boundary, and model construction does not connect it.

Do not put another `kinematics: yam` or geometry copy in these adapters. Geometry
and solver defaults are configured on the arm component. Existing DP/OpenWAM
whole-chunk rejection and XR1 failed-waypoint hold behavior remain distinct.
Pi05 EEF already uses the grouped model. SAPolicy and Tianji-specific adapter and
solver migrations are deferred; SAPolicy retains its existing bounded IK path.

The offset geometry in `tests/unit/test_component_protocols.py` is a test fixture
only. There is no new runtime calibration or test-offset configuration.

## 3. Put framework wire conversion in the client

The policy plugin implements `reset(session_id)`, `infer(request)`,
`capabilities()` and `close()`. The existing plugin factory accepts a
`module:factory` reference. Learned model implementations still belong in
XPolicyLab under the repository contribution rules; a test backend is not another
native model integration.

The XPolicyLab client now accepts the following explicit option:

```yaml
policy:
  worker: xpolicylab_ws
  options:
    action_format: joint   # joint, pose, or native
```

For `joint`, `infer()` returns this internal structure:

```python
{
    "format": "joint",
    "actions": {
        "left_arm": left_targets,   # NumPy (horizon, group_dim)
        "right_arm": right_targets,
    },
    # Preserve any declared action_semantics and sampler metadata.
}
```

`JointAdapter` accepts absolute targets. LingBot's adapter additionally requires
its existing explicit relative-arm/absolute-gripper semantics and applies the
observation anchor. Equal matrix shapes do not imply equal action semantics.

For `pose`, each group has rows `[x, y, z, qw, qx, qy, qz, tool...]`. Translation
is metres. The selected adapter defines the reference frame and absolute/delta
meaning: DP, Pi05 EEF and OpenWAM consume absolute per-arm-base poses here. OpenWAM
continues to require its explicit semantics metadata. Pose-to-joint conversion,
seed selection and failure handling remain in the robot adapter.

Only the XPolicyLab codec knows its per-step dictionaries and field names. A
second framework client should translate its own reply into the same internal
structure, so it can reuse the adapter. Do not change normalization, joint order,
quaternion order, gripper convention, or absolute/delta meaning during translation.

Prepared requests can carry `model_state` and `model_info` for model-specific
semantic inputs such as EEF poses and measured history; framework clients encode
these into their own wire structure. The canonical observation/state/image types
are unchanged. SAPolicy and Tianji retain their old request attributes, handled
at the XPolicyLab client boundary while those adapters remain unmigrated.

`native` preserves the existing model-native payload. It remains the default for
unmigrated/specialized paths, including SAPolicy, Tianji EEF and XR1's 60-column
anchor-relative action representation. Migrated experiments explicitly select
`joint` or `pose`. External recipes using the migrated adapters must do likewise.
Backend identity and sampling capability checks remain in place.

## 4. Follow one action through the runtime

The runtime reads `RobotState` and camera `SensorFrame`s, asks the inference
strategy whether to submit, and calls `adapter.prepare_request()`. A policy worker
calls the framework client. The client translates the reply; the adapter converts
it into an `ActionChunk`. The strategy and `ActionTimeline` determine chunk handoff;
the executor generates commands at the robot control rate; `robot.send_command()`
dispatches them to owned components. Neither client wire conversion nor the
adapter changes `action_dt_s`, the robot control rate, or inference scheduling.

## Offline checks

Run from the repository root, without starting services:

```bash
envs/yam/.venv/bin/python -m pytest -q tests/unit/test_component_protocols.py
```

The tests exercise mixed arm/tool layouts through the executor, injected geometry,
independent model construction in spawned processes, default YAM FK/IK equivalence,
DP/OpenWAM rejection and XR1 hold, XR1 RTC conversion, sampler metadata, and two
clients feeding the same joint adapter. The second backend and shifted geometry
exist only in tests. They demonstrate the interface, not an actual RLinf/StarVLA
integration or real-robot validation.
