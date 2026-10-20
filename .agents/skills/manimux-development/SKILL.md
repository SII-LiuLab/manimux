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
- Warmup must exercise the current resolved rollout configuration through shared
  planning/execution components, with only predicted hardware dispatch suppressed.
  Do not introduce model-specific warmup shortcuts or synthetic condition branches.
  Test dry-run isolation, real scheduling/IK, failures and the Start reset fence;
  see [the inference contract](../manimux-inference/SKILL.md#warmup-must-use-the-selected-configuration).
- Keep deployment constants in YAML. New capability needs an explicit protocol when
  it cannot fit the existing one; do not hide extra tool coordinates or weaken IK to
  imitate a supported interface.

## Physical hardware safety is a required review

Before changing robot lifecycle, runtime cleanup, recovery or hardware error handling,
trace the real loaded SDK through connect, stop, Home and close. Treat activation,
position hold, gravity compensation, drive disable and connection release as different
operations. A returned stop call or a passing mock test does not establish physical safety.

- Review the change across model entry points, inference strategies and embodiments,
  not only the model from the reported incident. Confirm each supported entry reaches
  the shared runtime; exercise registered strategies against the same exit contract.
  Record unsupported embodiment capabilities explicitly. Never copy Home logic into
  model adapters or enable a capability on an unreviewed controller. Config loading,
  backend integration and physical validation are separate coverage claims.
- Check partial startup (one arm connected), camera/model loss during connection and
  execution, rejected commands, stale CAN feedback, Home failure, normal Finish,
  Ctrl-C/SIGTERM, repeated interrupts and recording failures.
- For perception/model faults, stop task execution; an assembly that declares fault
  Home must check healthy feedback/control, Home, and verify measured arrival before
  release. Do not infer health from cached joints stamped at read time, automatically
  clear motor faults, or send Home through a failed controller.
- Home failure must not fall through into automatic torque release. Keep surviving
  support/control resources owned, block another Prepare, and require explicit
  confirmation of physical support before release. EOF, timeout and Ctrl-C are not
  release authorization. Never hide an unsafe close behind generic best-effort cleanup.
- Review every connected arm before closing the first one. Verify Home before opening
  grippers; secure payloads before any explicit manual release. Respect receive-only
  mode and inspect SDK constructors for actuator side effects and partial failures.
- Use fake-device fault injection first. Report software checks separately from
  hardware evidence; do not start, restart, move or release hardware merely to test a fix.
  Explain remaining limits (power loss, SIGKILL, failed drives and unvalidated paths).
- Distinguish target interpolation completion from measured arrival: YAM keeps the
  Home target and waits with fresh feedback, without an arrival deadline. Preserve
  hardware-health checks and report joint error/speed while waiting. Show blocking
  cleanup and supported-release waits in RoboGUI; a process heartbeat must never be
  presented as fresh robot feedback or enable Prepare before cleanup completes.

Read the [hardware exit contract](references/components.md#hardware-exit-safety)
and the embodiment's implementation before declaring a lifecycle change complete.

## Complete and verify

### Deployment handoff requires a complete run without hardware commands

Before handing a new or changed policy integration to an operator, the agent must
exercise observation acquisition/encoding, the real provider, response validation,
the configured adapter, asynchronous admission and executor through the final
would-be robot command. Keep hardware dispatch suppressed throughout, including
connect/Prepare, warmup, Home and cleanup. Use a demonstrably disconnected command
sink; do not construct a motor driver whose connection sends commands. Record which
observations are live, recorded or synthetic. A synthetic smoke test alone is not a
real-station qualification. If representative observations or provider access are
unavailable, state the missing verification rather than claim readiness.

The agent owns the diagnose/fix/retest loop before asking the operator to Start.
Persist raw provider responses **before** decoding or rejection, linked to session,
request sequence, input identity, prompt/model and timing. Preserve malformed output
as well as successful output; redact credentials and avoid dumping images into
console logs. Errors must identify field, expected/actual shape or range, invalid
indices/count and the saved artifact path. Do not silently clip or substitute invalid
actions just to pass. Use the provider's documented action semantics for any conversion.
Verify rejected-response persistence, timeout/error handling, request association and
no hardware sends. Run multiple consecutive representative requests, inspect admission
and command output, and report rejections/limitations rather than merely process exit.
Once this chain passes, the operator explicitly starts physical execution through
RoboGUI; software qualification is not a claim of physical task success.

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
