---
name: manimux-development
description: Develop or review ManiMux robot drivers, cameras, policy adapters, runtime strategies, collection and Viewer features. Use to locate the right extension point and preserve collection, training and deployment compatibility.
---

# ManiMux Development

Paths below are relative to the ManiMux root. Read its `AGENTS.md` first.
Implement the requested feature at the existing extension point; do not turn an
integration into a framework rewrite or add unrelated checks.

## Choose the layer

| Work | Implementation and interface |
|---|---|
| Learned model, preprocessing, normalization, training, sampling | `XPolicyLab/policy/`; follow `XPolicyLab/AGENTS.md` and its model-integration skill |
| Robot SDK and hardware commands | `src/manimux/robots/base.py` defines `connect`, `get_state`, `send_command`, `home`, `stop`, `close`; implementations live alongside it |
| Camera device and runtime sensor | `src/manimux/sensors/base.py` defines `start`, `read`, `close`; implementations live alongside it |
| Observation mapping and model actions to robot actions | `src/manimux/policies/base.py`, `src/manimux/policies/`, `src/manimux/integrations/` |
| FK/IK and robot geometry | `src/manimux/kinematics/base.py`, `src/manimux/viewer/robots/base.py` and their body-specific implementations |
| Chunk scheduling and command generation | `src/manimux/runtime/inference.py`, `src/manimux/runtime/executors/base.py` |
| Collection GUI and recording | `src/manimux/collection/`; retain separate implementations per embodiment |
| Experiment UI, timelines and rollout evidence | `src/manimux/viewer/`, `src/manimux/recording/` |

Shared state, frame, request and action structures live in `src/manimux/types.py`.
Use `prepare_policy_request` for the optional adapter request hook; not every
adapter implements `prepare_request` directly.

## Add or change an embodiment

- Put its driver and SDK wrapper under `src/manimux/robots/`. Vendored SDK source
  and native bindings are acceptable there; a separately published package is not required.
- The robot factory takes a plain parameter dictionary and a `Clock`. The main
  loader is `manimux.cli.load_config()`; the old top-level configuration classes
  have been removed. See `docs/code-organization.md` for migration boundaries.
  `src/manimux/plugins.py` currently supports
  a `module:factory` reference in `robot.driver`, as well as built-ins and entry points.
  Follow the corresponding loader for sensor, kinematics and Viewer plugins.
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
YAM's example is `configs/collection/yam/station.yaml` → `configs/collection/yam/control.yaml`
→ `configs/robots/yam/common.yaml`; inference recipes reference that same control profile.
Keep each body's action interval, joint/gripper conventions and shared motion parameters
consistent across its collection and deployment. Preserve those meanings in training
conversion and checkpoint normalization. Executor smoothing is a separate choice.

Put tasks/checkpoints under the existing model/body `server` and `infra` task folders.
For learned-model work, use
`XPolicyLab/.agents/skills/xpolicylab-model-integration/SKILL.md`; use
`XPolicyLab/.agents/skills/xpolicylab-adapter-check/SKILL.md` when an adapter review is
actually requested. Do not duplicate those instructions.

## Camera and UI changes

Trace physical serial → camera-server name → `policy.options.camera_map` → model input.
The shared `SensorFrame` image is RGB uint8. Preserve a frame's image and timestamp
together; reading the same cached image again does not make it a new capture.
Inspect `src/manimux/sensors/camera_server/driver.py` and
`src/manimux/integrations/umi_dp_tianji/camera_sensor.py` before assuming their timestamps
or frame sequences mean the same thing.

When editing generic UI, derive group displays from supplied data rather than assuming
left/right grippers. Existing fixed layouts are not templates for every body.
Run focused existing tests for the changed layer and report what was not exercised.
