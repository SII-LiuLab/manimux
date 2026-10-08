# ARX X5 and PiPER SDK adapters

ManiMux provides controller adapters for **ARX X5 (2023)** and **standard PiPER**,
plus offline kinematic models and a shared arm assembly factory. Constructors and
`load_model()` never import a device SDK, open CAN or start a control process.
The implementation has been checked without hardware; physical connection,
calibration, stopping and task success still require device validation.

Use the [SDK installation guide](robot-sdks.md) for pinned versions and native
prerequisites. The ARX controller uses the official `arx_x5_python` binding;
the PiPER controller uses `pyAgxArm`. Neither selects a different SDK on error.

## Select the assembly

| Arm | Assembly YAML | Component names | Station template |
| --- | --- | --- | --- |
| X5 (2023) | `manimux/configs/embodiment/robot/arx_x5_dual.yaml` | `left_x5`, `right_x5` | [arx_x5_example.yaml](../../manimux/configs/local/arx_x5_example.yaml) |
| Standard PiPER | `manimux/configs/embodiment/robot/piper_dual.yaml` | `left_piper`, `right_piper` | [piper_example.yaml](../../manimux/configs/local/piper_example.yaml) |

Both assemblies expose `left_arm` and `right_arm`, each with six joint angles
in radians followed by normalized gripper opening: 0 closed, 1 open. Commands
are absolute joint-position targets. Independent CAN sessions are not an atomic
dual-arm dispatch. EEF policies still use ManiMux's existing action adapters;
controllers do not perform Cartesian conversion or model inference.

Use the factory through the existing `build_robot()` selector. For example, the
robot section of a PiPER experiment is:

```yaml
robot:
  type: manimux.embodiments.robot.can_arms:build_robot
  config: ../../embodiment/robot/piper_dual.yaml  # Relative to this experiment.
  group_dims: {left_arm: 7, right_arm: 7}
  options:
    execute: false
    home_on_close: false
```

Set the assembly path relative to the actual experiment file. Retain its policy,
camera mapping, action interval, scheduling and execution limits. This fragment
does not claim that an existing checkpoint is compatible with either arm.

## Bind and inspect a station

Copy the matching template into your private `station.yaml`, or select another
private file with `--local`. Read an existing station before modifying it and
identify the physical left/right CAN mapping. Placeholders intentionally fail
construction instead of opening a guessed interface.

Supply six device joint-limit pairs and each arm's measured gripper endpoints.
X5 encoder endpoints and `set_catch` command endpoints are separate calibration
fields: their units must not be assumed equal. PiPER's width-mode command and
feedback use metres. Commands outside joint bounds or opening `[0, 1]` fail;
there is no silent clipping or substitution of commanded targets for feedback.

For configuration inspection, without connecting:

```python
from pathlib import Path
from manimux.cli import bind_station
from manimux.clock import SystemClock
from manimux.embodiments.robot import build_robot

config = bind_station({
    "robot": {
        "type": "manimux.embodiments.robot.can_arms:build_robot",
        "config": Path("manimux/configs/embodiment/robot/piper_dual.yaml"),
        "group_dims": {"left_arm": 7, "right_arm": 7},
        "options": {"execute": False, "home_on_close": False},
    }
}, "manimux/configs/local/station.yaml")
robot = build_robot(config["robot"], SystemClock())
print({name: arm.channel for name, arm in robot.arm_components.items()})
robot.close()
```

## Lifecycle and feedback guarantees

| Operation | Standard PiPER | Official X5 (2023) |
| --- | --- | --- |
| `connect()` | Starts an owned CAN receiver and waits for all joint/gripper packets. No motion, reset or enable by default. | Requires `execute: true` and explicit `feedback_policy: sdk_poll`. Creates an owned SDK process, selects PROTECT before starting the vendor loop. Passive connection is unavailable. |
| Feedback | Oldest host receipt time across joint_12, joint_34, joint_56 and gripper. Cached reads retain timestamp and sequence. | Cached SDK polls retain timestamp and sequence. Timestamp means getter completion; the binding does not expose device sample identity or receipt time. |
| `send_commands()` | Checks fresh measured position and motor-enable feedback; sends `move_j` then width/force. Requires enabled motors; does not reset faults. | Prepares joint and calibrated gripper targets, then selects POSITION_CONTROL. Requires current SDK polling. |
| `stop()` | Attempts arm and gripper holds independently using their own fresh position feedback. Reports incomplete stops. Subsequent commands retain normal readiness checks. | Selects vendor PROTECT. A subsequent authorized command prepares targets before selecting POSITION_CONTROL. |
| `close()` | Stops/joins the owned receiver, closes the bus and detaches the driver. Cleanup errors preserve resources for retry. No disable, Home or reset. | Requests PROTECT then exits the owned process, releasing its threads and CAN handles. Forced termination reports an unconfirmed protective stop. |
| `home()` | Unsupported; inherited explicit `NotImplementedError`. | Unsupported; inherited explicit `NotImplementedError`. |

For PiPER, `execute: false` permits a receive-only connection and makes assembly
stop/close read-only as well. `enable_on_connect: true` is accepted only with
`execute: true`; it sends one enable request and waits for feedback, without Home
or fault reset. `move_j` retains firmware interpolation and `speed_percent`.
Its measured hold is not an emergency stop and submission does not prove arrival.
Stop attempts the arm and gripper independently: missing/stale gripper feedback
cannot block an arm hold, and an arm feedback or send failure cannot block a
gripper hold. Normal motion readiness checks (enable/status/fault feedback) do not
gate these stop attempts. A hold still requires fresh measured positions for its
own component; missing or stale positions are never replaced by old targets.
All failures are reported together after both components have been attempted.
A successful submission does not establish that a faulted motor physically stopped.
The adapter does not disable motors, reset faults or fall back to the SDK's
electronic emergency stop, which allows a raised arm to descend under damping.
Subsequent motion retains the normal feedback, enable and fault checks; recovery
remains explicit. Stale feedback, fault flags and failed dispatch/cleanup remain
visible errors.

For X5, the SDK's polling clock proves the process is reading, **not that motors
are still reporting**. Therefore the adapter rejects connection unless the caller
explicitly selects `sdk_poll`. It does not satisfy device-level freshness validation.
An independent feedback monitor or an upstream receipt-time API is needed before
claiming that guarantee. Native constructor/PROTECT effects also require physical
validation. The adapter never calls the upstream `SingleArm`, GO_HOME or the
independently maintained SDK's automatic motor-state clearing.

## Offline geometry and validation

Both component `model.urdf` files retain their respective upstream joint
origins, axes and inertia. X5 has a kinematic-only model; PiPER also bundles
the official visual meshes and an [offline RoboGUI preset](piper.md).
Neither model provides collision checking. Source pins and licenses are in
the component READMEs. By default `tcp_frame: link6` names
the flange frame; it does not invent a task-specific grasp centre or tool offset.
Select the checkpoint's actual TCP and calibrated bounds before EEF deployment.
The X5 source's broad `[-10, 10]` bounds are not physical device limits; set
`options.joint_limits` in its component recipe when using bounded hardware IK.

This contribution does not add live hardware RoboGUI presets. The current
RoboGUI's Home gating partly depends on robot names; it needs a capability-aware
extension before exposing Home controls for these assemblies. The ALOHA and PiPER presets provide offline display and replay.
A hardware deployment needs its own experiment and station configuration. Keep `home_on_close: false` for these factories.

Offline checks cover real pyAgxArm codecs with virtual CAN, X5 binding signatures,
fake X5 process lifetime, read-only guards, calibrated targets, cached/stale
feedback, stop followed by another command, failed/partial cleanup, actual
assembly loading and bounded FK/IK. They do not establish real-robot readiness.
The [RoboTwin ALOHA RoboGUI](aloha.md#preview-and-replay) remains a separate asset
preset and is not substituted for these physical arm models.

## Code map

Paths below are relative to `manimux/`. Start with the component model or
controller for the arm being changed; keep SDK calls inside its session owner.

| File | Responsibility |
| --- | --- |
| `embodiments/arm/_urdf.py` | Shared offline arm model loading and visual mapping |
| `kinematics/urdf_manipulator.py` | URDF FK and bounded numerical IK |
| `embodiments/arm/_calibration.py` | Required station endpoints and command validation |
| `embodiments/arm/piper/controller.py` | Owned CAN transport, packet receipt, readiness and PiPER commands |
| `embodiments/arm/arx_x5/controller.py` | Parent RPC lifecycle, calibrated commands and cached state |
| `embodiments/arm/arx_x5/session.py` | Child process that owns the native X5 binding and control modes |
| `embodiments/robot/can_arms/robot.py` | Assembly factory, configured controllers and station bindings |

The [asset preparation scripts](../../scripts/assets/README.md) regenerate public
models. The [SDK installation probe](../../scripts/setup/README.md) performs
reusable offline checks; local virtual-CAN and failure-injection tests remain
in the ignored development test workspace.
