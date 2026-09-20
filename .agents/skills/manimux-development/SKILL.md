---
name: manimux-development
description: Develop or review ManiMux robot drivers, cameras, policy adapters, runtime strategies, collection and Viewer features. Use to locate the right extension point and preserve collection, training and deployment compatibility.
---

# ManiMux Development

Paths below are relative to the ManiMux root. Read its `AGENTS.md` first.
YAM and RealSense component READMEs describe their current installation and interfaces.
Implement the requested feature at the existing extension point; do not turn an
integration into a framework rewrite or add unrelated checks.

## Choose the layer

| Work | Implementation and interface |
|---|---|
| Learned model, preprocessing, normalization, training, sampling | `XPolicyLab/policy/`; follow `XPolicyLab/AGENTS.md` and its model-integration skill |
| Robot SDK and hardware commands | `manimux/embodiments/robot/base.py` defines `connect`, `get_state`, `send_command`, `home`, `stop`, `close`; assemblies live alongside it |
| Camera device and runtime sensor | `manimux/embodiments/sensor/` owns `SensorBase`, `build_sensor` and device implementations; `read` returns one frame or a named bundle |
| Observation mapping and model actions to robot actions | `manimux/policy_adapter/base.py`, `manimux/policy_adapter/` |
| FK/IK and robot geometry | `manimux/kinematics/base.py`, `manimux/viewer/robots/base.py` and their body-specific implementations |
| Chunk scheduling and command generation | `manimux/runtime/inference.py`, `manimux/runtime/executors/base.py` |
| Collection GUI and recording | `manimux/collection/`; retain separate implementations per embodiment |
| Experiment UI, timelines and rollout evidence | `manimux/viewer/`, `manimux/recording/` |

Shared state, frame, request and action structures live in `manimux/types.py`.
Adapters inherit the default `prepare_request` hook from `PolicyAdapter`; runtime
calls `prepare_request(request)` and `decode_action(raw, context)` directly.

## Add or change an embodiment

- Put arms under `manimux/embodiments/arm/` and assemblies under
  `manimux/embodiments/robot/`. Use installed SDKs directly where possible. Vendored SDK source
  and native bindings are acceptable there; a separately published package is not required.
- The robot factory takes a plain parameter dictionary and a `Clock`. The main
  loader is `manimux.cli.load_config()`; the old top-level configuration classes
  have been removed. See `docs/code-organization.md` for migration boundaries.
  `manimux/plugins.py` currently supports
  a `module:factory` reference in `robot.type`, as well as built-ins and entry points.
  Robot factories return `RobotBase` and use the `manimux.embodiments.robot` entry-point group.
  Sensor factories live in `embodiments.sensor` and use `manimux.embodiments.sensor`.
  Import components directly; do not restore the removed top-level hardware namespaces.
  Follow the corresponding loader for kinematics and Viewer plugins.
- Specify named groups, joint ordering, units, gripper convention and any base/TCP
  transform in the implementation/config. Body-specific DOFs and SDK mappings belong
  in that body module, not in the shared scheduler. The current hardware runtime consumes
  joint-position commands; EE policies need an appropriate action adapter.
- Explain plainly whether connecting moves the robot, and what stop/home do.
  Do not assume another driver's defaults apply to the new one.
- Use a matching fake SDK or mock driver for the changed behavior. For a new body,
  exercise its actual group names and dimensions rather than copying a dual-arm fixture.

## Keep collection, training and deployment consistent

Collection does not need a cross-body GUI or a universal teleoperation implementation.
YAM's example is `manimux/configs/collection/yam/station.yaml` → `manimux/configs/collection/yam/control.yaml`
→ `manimux/configs/embodiment/robot/yam_control.yaml`; inference recipes reference that same control profile.
Declare action_dt_s in each experiment policy; the shared body profile supplies layout and limits.
Keep joint/gripper conventions and shared motion parameters
consistent across its collection and deployment. Preserve those meanings in training
conversion and checkpoint normalization. Executor smoothing is a separate choice.

Put complete experiments under `manimux/configs/experiments/<task>/` and deployment recipes
under `manimux/configs/policy/<model>/<embodiment>/`, without a `server/` layer. XPolicyLab
owns model defaults and serving. Private training configurations, launchers and notes
belong to the root's ignored `training/` workspace. Select the adapter class and its mappings
inline in `policy.adapter`; select inference and executor independently. Do not add
hand-written parameter validation or another configuration framework during refactors.
Document conversion units, reference frames, delta anchors and timing in code.
For learned-model work, use
`XPolicyLab/.agents/skills/xpolicylab-model-integration/SKILL.md`; use
`XPolicyLab/.agents/skills/xpolicylab-adapter-check/SKILL.md` when an adapter review is
actually requested. Do not duplicate those instructions.

## Camera and UI changes

Trace physical serial → camera-server name → `policy.adapter.camera_map` → model input.
The shared `SensorFrame` image is RGB uint8. Preserve a frame's image and timestamp
together; reading the same cached image again does not make it a new capture.
Inspect `manimux/embodiments/sensor/camera_server/driver.py` and
`manimux/embodiments/sensor/camera_server/timestamped.py` before assuming their timestamps
or frame sequences mean the same thing.

When editing generic UI, derive group displays from supplied data rather than assuming
left/right grippers. Existing fixed layouts are not templates for every body.
Run focused existing tests for the changed layer and report what was not exercised.
