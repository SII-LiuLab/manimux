---
name: manimux-development
description: Implement or review ManiMux components, model adapters, inference strategies, execution and RoboGUI features using the existing protocols. Use when configuration alone cannot supply the requested behavior.
---

# ManiMux Development

Read the [repository guide](references/agent-guide.md) and relevant nested/framework
`AGENTS.md` before editing. For a custom study using supported components, start with
[experiments](../manimux-experiments/SKILL.md); for another local hardware installation,
use [station setup](../manimux-station-setup/SKILL.md).

## Choose the owning layer

Trace the selected YAML through its loader, interface, working implementation and
caller before changing behavior. Read only the relevant reference:

| Change | Reference |
| --- | --- |
| Robot, arm, gripper, camera, SDK or kinematics | [Components](references/components.md) |
| Model framework, transport, action adapter or normalization | [Policies](references/policies.md) |
| Scheduling, timelines, recording, executor or RoboGUI | [Runtime](references/runtime-config.md) |
| Sampler capabilities or model/algorithm combinations | [Inference](../manimux-inference/SKILL.md) |
| Extension entry points and contributing | [Integration map](references/README.md) |

Model frameworks own weights, network/sampling and training. ManiMux owns transport,
adapters, strategy/timeline, execution, recording and GUI. Inspect upstream code before
classifying a model as a framework: XPolicyLab models stay under its `policy/<name>/`,
while a distinct framework stays a peer with its own ManiMux client. Do not duplicate
model implementations inside `manimux/` or migrate a working deployment incidentally.

## Implement against the contract

- Preserve unrelated work and state the edit scope. Reuse the existing factory,
  configuration loader and runtime loop. Validate data in the layer that owns it.
- Preserve units, frames, group order, input/output shapes, timestamps, action meaning,
  lifecycle and failure behavior across implementations of the same interface.
  Matching method names alone is insufficient. Fail explicitly for unsupported modes.
- For components, use the [operation contract](references/components.md#shared-behavior-across-embodiments).
  SDK device semantics must not leak into caller-side robot-name branches.
- XPolicyLab deployment recipes own `robot_action_dim_info` and `num_envs`; use its
  shared helpers and contribution rules. Preserve sampling flags through adapters and
  transport. Configuration cannot create a model-side sampler hook.
- Keep deployment constants in YAML. New capability needs an explicit protocol when
  it cannot fit the existing one; do not hide extra tool coordinates or weaken IK to
  imitate a supported interface.

## Complete and verify

Deliver the implementation, selecting YAML, a usable config/launch example and checks
through the real loader or public interface. Use fake hardware for lifecycle tests
and synthetic timed chunks for scheduling. Separate CPU/config checks, actual model
inference, connected-device evidence and physical task success. Local regression tests
are ignored by repository policy; do not force-add them.

Update the owning reference for interface changes and `guideline.html` for user-facing
settings/operations. Keep code comments and maintained documentation in English;
do not create a second guide under ignored `docs/`. Honor existing authorization;
development alone does not authorize robot motion or stopping live services.
Commit/push only when requested.
