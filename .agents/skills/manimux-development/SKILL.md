---
name: manimux-development
description: Integrate or review ManiMux arms, grippers, cameras, robot assemblies, policy clients, action adapters, inference strategies, executors and Viewer features. Use to follow existing component interfaces, configuration ownership and offline validation instead of adding a parallel integration stack.
---

# ManiMux Development

Read the repository `AGENTS.md` and relevant nested instructions first. Code paths
below are relative to the repository root; `references/` links are relative to this
skill. This skill guides implementation and interface review, not hardware startup.

## What following the protocol means

A component follows its protocol when its caller can use the existing interface
without knowing the vendor or model name. Matching method names is insufficient:
inputs, outputs, units, action meaning, timestamps, resource ownership and failure
behavior must also agree. Different kinds of components have different interfaces;
a camera, a gripper and a policy client need not share one universal base class.

Integrate at the owning layer. Do not make an integration work by spreading device
names, wire-format parsing, geometry or experiment parameters through the main loop.
Do not fix this by adding another registry, configuration framework, generic wrapper
or repeated validation in every layer. Use existing loaders and check a constraint
where the corresponding data enters or changes meaning.

## Locate the extension before editing

1. Inspect the current working tree and the selected configuration. Follow its
   actual loader, interface and one relevant implementation; old examples can use
   compatibility paths. Preserve other developers' work.
2. Identify what is new: a physical component, an assembly of existing components,
   a checkpoint, an action representation, a backend framework, or a scheduling
   algorithm. A new checkpoint or station often needs only YAML changes.
3. Read only the relevant reference:
   - [Components](references/components.md): arms, shared controllers, grippers,
     offline geometry, assembled robots, cameras and runtime sensors.
   - [Policies](references/policies.md): learned models, backend clients, action
     adapters, formats and independent decoder processes.
   - [Runtime and configuration](references/runtime-config.md): inference,
     timelines, executors, Viewer/replay and where YAML parameters belong.
4. Before implementation, explain the proposed files, selected interface, important
   input/output semantics and a focused verification plan. Continue within the
   user's authorized scope; this is not an extra approval gate.
5. Implement, wire the real factory/config path, and verify behavior using the
   actual implementation with a fake SDK or synthetic service response where
   hardware/model execution is outside scope.

For an end-to-end worked example, read `docs/component-policy-development.md` and
`manimux/configs/experiments/put_bottles/pi05/yam_pi05_joint.yaml`. Read the YAML as
an example, not as permission to execute it or proof that devices are locally bound.

## Keep one chain understandable

```text
component YAML -> robot assembly -> RobotState / SensorFrame
  -> ObservationSnapshot -> InferenceRequest
  -> policy client / model service -> adapter -> ActionChunk
  -> inference strategy / ActionTimeline -> ActionHorizon
  -> executor -> RobotCommand -> robot assembly -> hardware components
```

Shared structures live in `manimux/types.py`. The client translates the backend's
wire format; the adapter handles robot action meaning and necessary FK/IK; the
strategy controls requests and chunk handoff; the executor generates commands;
the driver alone handles hardware. Viewer and recording consume runtime evidence.

## Review the integration, not just whether it runs

For each affected boundary, establish:

- **Discovery:** Which config field and factory select it? Can another component
  of the same capability replace it without editing `runtime/edge.py`?
- **Data:** What are the group names/order, shapes, units, coordinate frames,
  quaternion convention, gripper meaning and absolute/delta anchor?
- **Ownership:** Who opens and closes the device/session? Can offline model loading,
  adapter construction and Viewer replay avoid hardware connections?
- **Behavior:** Are reset, cached/stale data, unsupported capabilities and failures
  handled explicitly at the owning boundary? Were existing semantics preserved?
- **Evidence:** Does the test traverse the real loader or public interface? State
  what was verified offline and what still requires an SDK, checkpoint or hardware.

Choose checks relevant to the change; do not create a blanket validator or a test
that merely repeats implementation details. Useful existing examples under
`tests/unit/` include `test_component_protocols.py`, `test_yam_assembly.py`,
`test_composed_kinematics.py`, `test_orbbec_sensor.py` and `test_action_replay.py`.
Runtime tests commonly use `envs/yam/.venv/bin/python`; model-side tests need their
own model environment. Missing SDKs are evidence limits, not reasons to fabricate
a successful integration.

## Compatibility is not a template

Tianji controller/session work may change independently: read the current shared
interfaces and selected implementation, and keep session changes with their owner.
Do not duplicate Tianji internals into a new component or claim YAM/Tianji have been
fully validated as interchangeable. SAPolicy's existing bounded IK path and some
native payload paths are not yet migrated; preserve them unless migration is in
scope. Do not silently replace constrained IK with ordinary IK to unify signatures.

Describe remaining compatibility paths when they affect the task. Do not expand a
new integration into their cleanup. A directory or passing mock test alone does not
prove a backend, robot or sampling mode is supported.

## Development versus use

For device binding use `.agents/skills/manimux-station-setup/SKILL.md`; for paired
server/runtime configs and startup commands use
`.agents/skills/manimux-experiment/SKILL.md`; for saved rollout evidence use
`.agents/skills/manimux-result-analysis/SKILL.md`. Do not invent a second startup
workflow here. Teleoperation/demonstration collection is outside this repository;
retain runtime recording and offline replay.

Deliver a short explanation of the extension point, configuration, focused checks
and remaining limitations. Update the relevant component documentation when its
public interface changes. Write new comments and general documentation in English.
Commit/push only when requested; documentation work does not authorize services or
physical motion.
