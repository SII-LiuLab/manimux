---
name: manimux-development
description: Integrate or review ManiMux robot components, policy frameworks, action adapters, inference strategies, executors and RoboGUI features using the existing protocols.
---

# ManiMux Development

ManiMux provides shared real-robot inference, execution and experiment operation.
Users choose their research and optional evaluation; integrations reuse the common
runtime and RoboGUI. Read the [repository guide](../../../docs/development/agent-guide.md)
before editing, plus relevant nested or framework `AGENTS.md` instructions.

## Choose the extension point

Trace the selected YAML through its loader, interface, one working implementation
and caller. Inspect upstream code before classifying a model versus a reusable
framework. A new checkpoint usually needs configuration, not a new backend.

Read only the matching public protocol guide (paths are relative to repository root):

- [Hardware components](../../../docs/development/components.md): arms/controllers,
  tools, assembled robots, offline FK/IK, cameras and sensors.
- [Policies](../../../docs/development/policies.md): models, peer frameworks,
  clients, wire codecs and observation/action adapters.
- [Runtime and configuration](../../../docs/development/runtime-config.md):
  scheduling, timelines, executors, display, records/replay and YAML ownership.
- [Model sampling matrix](../../../docs/advanced/inference-matrix.md): supported
  YAM model/algorithm combinations, configuration generation and validation limits.
- [Integration map](../../../docs/development/README.md): entry points and concrete recipes.

Binding an existing supported robot to another workstation uses
`manimux-station-setup`. Studying saved rollouts uses `manimux-experiments`.

## Plan around ownership

Preserve unrelated work and explain the affected files and data path before a
structural change. Resolve ambiguous ownership with the user; continue within an
already approved scope without asking again at every stage.

## Preserve the base-class contract

Implementations of the same interface must give each operation the same public
meaning. Matching method names is insufficient: preserve input/output types,
shapes, group order, units, frames, action meanings, timestamp semantics,
completion guarantees, lifecycle and failure behavior. Different component kinds
retain their own interfaces; an arm and a camera need not share one base class.

Vendor SDK/RDK calls, wire formats and device mode transitions may differ inside
each implementation. Callers must not need vendor-name branches to compensate for
those differences. For embodiment work, read the operation table and review steps
in [the component contract](../../../docs/development/components.md#shared-behavior-across-embodiments).

Review existing drivers, including YAM and Tianji, against that contract; their
presence in the repository is not proof of compliance. Report mismatches as
implementation gaps rather than redefining the protocol around an SDK. Required
operations must work; optional unsupported operations must be explicit to callers,
never silent no-ops or fabricated success. If the current capability declaration
or caller behavior is insufficient, identify the gap before changing the interface.

XPolicyLab deployment recipes own explicit `robot_action_dim_info` and `num_envs`;
pass the resolved policy config to the shared framework helpers. Do not recreate a
parent `env_cfg/` registry. Model representation and physical joint layout remain
separate responsibilities connected by the action adapter.

Use the existing loader and factory for each layer. Do not create a parallel main
loop, registry or config framework. Do not spread SDK calls, model codecs or
experiment constants into unrelated components. Validate data at the boundary
that owns its meaning, rather than repeatedly checking it throughout the stack.

If a capability does not fit an existing interface, identify that limitation and
propose the smallest explicit extension. Never hide extra tool coordinates, replace
specialized bounded IK with a weaker solver, or report sampling support without
model-side hooks. Compatibility paths are not templates for new integrations.

For an embodiment review, show the affected operation, required behavior, actual
SDK mapping, owning file and focused verification. Distinguish documented requirements
from verified implementation behavior. Keep unrelated drivers and ongoing migrations
outside the edit scope.

## Complete one usable integration

Deliver implementation, selecting YAML, a short launch/config example, and focused
validation through the actual loader or public interface. Use fake devices for
lifecycle/dispatch tests and synthetic chunks for scheduling tests. Separate offline
checks, actual model inference and physical task success. Tests remain local under
this repository's ignore policy; do not force-add them.

Update the matching public protocol guide when the interface changes. Keep one
maintained explanation, referenced by skills and user docs. Write new comments and
general documentation in English; update the Chinese homepage alongside English.
Development authorization does not authorize robot motion or stopping live services.
Commit/push only when requested.
