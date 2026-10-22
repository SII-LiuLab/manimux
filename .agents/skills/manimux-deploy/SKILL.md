---
name: manimux-deploy
description: Deploy a supported ManiMux model recipe on an installed local station, start its services and open RoboGUI for the operator. Also use for a hardware-free product demo or managed deployment status and shutdown.
---

# ManiMux Deployment

Take a fresh conversation from a model/task request to RoboGUI waiting for the operator.
Use the [managed launch guide](../../../guideline.html#managed-deployment) for commands.
This workflow manages one runtime `--config`, XPolicyLab WebSocket serving, cameras
and RoboGUI on Linux with tmux. Use `--external-policy` for an existing local/remote
model endpoint and `--external-camera` for existing camera-server network readers.
These dependencies are probed but never started, adopted or stopped; keep their existing
service owners and network tunnels available. Config-list studies and other model
frameworks retain their existing runbooks; do not silently convert them.

## Resolve and launch

1. Inspect current deployment state, the requested model/robot runbook, candidate runtime
   recipes, station bindings and installed interpreters. Reuse facts from files rather
   than conversation history. Use [station setup](../manimux-station-setup/SKILL.md) for
   missing bindings/environments and [inference](../manimux-inference/SKILL.md) for model
   compatibility. Ask only for unresolved physical mappings, artifact identity or task
   choices. Never substitute a checkpoint or change action semantics.
2. Run `manimux doctor --config <recipe> --local <station> --runtime-python <python>
   --policy-python <python> --json`; add `--camera-python` if needed. This checks
   configuration, artifact paths and ports without connecting devices or loading weights.
   For a new/changed policy integration, also satisfy the Development skill's
   [complete run without hardware commands](../manimux-development/SKILL.md#deployment-handoff-requires-a-complete-run-without-hardware-commands)
   before operator handoff. A healthy port or synthetic smoke test is insufficient.
   Exercise representative observations through the real provider, adapter, configured
   scheduler and executor into a disconnected command sink. Save raw rejected responses,
   diagnose failures and retest without actuator sends (including Prepare/Home/cleanup).
   Do not use real-robot Prepare to obtain this verification. Report missing evidence
   explicitly; the operator starts physical execution through RoboGUI after qualification.
3. For an authorized deployment, run `manimux up` with the same arguments. It starts
   owned services, probes external dependencies and starts the interactive runtime, then
   returns `gui_url`, logs, process identities and `attach_command`. Progress is on stderr;
   `--json` keeps stdout machine-readable. Do not add `--once` or press Prepare/Start.
4. Open `gui_url` using the host's browser-panel tool when available (in Codex,
   `open_in_codex` with a browser target). Otherwise use `up --open` for the default
   browser. Verify the rendered robot and operator panel; HTTP alone is insufficient.
   Report the selected model/task and hand over at **Prepare**, which can move hardware.
   Warmup and Start also remain with the operator unless explicitly authorized.

Use `--deployment-dir` consistently when choosing a non-default private state directory.
Repeated up reuses only an unchanged, healthy deployment owned by this tool. Undeclared
external port occupancy is a blocker, not permission to kill or adopt the service.
Inspect it and select the external-service flag when the recipe matches its contract,
or use its existing runbook.

## Observe and stop

`manimux status --json` reports process health, camera freshness, policy HELLO identity,
and runtime session state. Policy probes never reset or infer. `operator_control` includes
preparation, execution and cleanup; RoboGUI shows the precise stage. Only recorded or
operator evidence establishes task success.

Use `manimux down` for requested shutdown. Runtime exits first; a blocked exit retains
its dependencies and hardware ownership. Open RoboGUI Hardware recovery: the operator
confirms a clear path for Home retry, or physical support of ALL arms/payloads for
actuator release. Do not send terminal `RELEASE`; terminal input has been removed.
Never infer support from timeout or GUI disconnection. `manimux attach --service runtime`
is for logs, not the recovery workflow. Detach tmux with Ctrl-B then D.

## Product demo

For rehearsal, run `manimux up --demo --robot yam --gui-port 18086 --deployment-dir
.local/demo --json`, open its URL, and label footage **synthetic preview**. This starts
only RoboGUI, with isolated bridge ports and no station or weights. Follow the
[capture workflow](../../../guideline.html#deployment-demo-video) and VHS tape for
repeatable recording. Keep full real-robot takes and label accelerated waits.
