---
name: manimux-deploy
description: Deploy a supported ManiMux model recipe on an installed local station, start its services and open RoboGUI for the operator, including RPent agent execution. Also use for a hardware-free product demo or managed deployment status and shutdown.
---

# ManiMux Deployment

## Robot authorization — required for every task

Without explicit user permission, do not start or restart a robot, connect its
controller, enable actuators, or cause robot motion. This includes Prepare, Warmup
on connected hardware, Start, Home, gripper calibration, replay and recovery/release.
Invoking this skill or requesting integration, configuration, testing, evaluation or
video capture does not grant that permission. Reuse explicit authorization already
given for the same robot and operation; do not ask for it repeatedly.

Complete configuration, code and disconnected validation first. If permission is
missing, show the concrete robot, resolved configuration and intended operations,
then ask before the first hardware-affecting action. Configuration review is not
robot-start authorization. Leave physical controls with the operator by default.
`execute: false`, “dry run” and “warmup” are not evidence of isolation: constructors,
connect and cleanup can move hardware. Use disconnected models/fake devices offline.
A separately authorized GUI/model/camera service launch may proceed only after
verifying it will not connect or actuate the robot. Preserve existing service and
hardware ownership; do not stop, restart or release an active robot to test a change.

All models/backends must follow the [shared integration contract](../manimux-development/references/policies.md#mandatory-shared-contract-for-every-model-and-backend).
Reuse resolved embodiment/tool geometry and station calibration before asking for
missing physical facts. Do not introduce a backend-specific `native` action path
for convenience. Only a demonstrated inability to conform justifies requesting an
explicit user-approved exception; legacy implementations are not new templates.

Take a fresh conversation from a model/task request to RoboGUI waiting for the operator.
Use the [managed launch guide](../../guideline.html#managed-deployment) for commands.
This workflow manages one runtime `--config`, XPolicyLab WebSocket serving or a hosted
policy API, cameras and RoboGUI on Linux with tmux. Use `--external-policy` for an existing local/remote
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
3. For an authorized deployment, run `manimux up` with the same arguments as a
   background process and keep its handle. It starts owned services, probes external
   dependencies and starts the interactive runtime, prints `gui_url`, logs, process
   identities and `attach_command`, then holds until interrupted. Progress is on stderr;
   `--json` keeps the startup result machine-readable. Do not add `--once` or press
   Prepare/Start.
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

## RPent on a real station

When the user explicitly asks to use RPent, prepare and verify everything before handoff;
the user only starts and controls RoboGUI. Contracts are in the
[RPent reference](../manimux-development/references/rpent.md).

1. Planner environment: `third_party/RPent/.venv` (the peer's own directory, like model
   environments). Create it with `uv venv --python 3.11` and
   `uv pip install -e ./third_party/RPent -e .` when missing; the runtime keeps its robot
   environment (e.g. `envs/yam/.venv`).
2. Private configs under `manimux/configs/local/`, never committed:
   - motion profile from `configs/agents/rpent_<robot>_motion.example.yaml`. Fill each
     group's TCP workspace from measured data, e.g. FK of the station's teleoperation or
     rollout joints in the group's `base_frame`. Add `path_validator` only with measured
     extrinsics and table geometry; otherwise tell the user there is no collision check.
   - agent recipe: experiment, motion, task, `planner`/`model`/`reasoning_effort`.
   - RoboGUI experiment: `_base_` the public RPent variant for the requested VLA
     (`experiments/example-yam-dream1/put_bottles/rpent.yaml`,
     `experiments/example-yam-pi05/put_bottles/rpent.yaml`, each beside its direct `serial.yaml`
     on the same study `_base_.yaml`) or another runtime recipe plus `agent.config` (its `policy` is what
     the planner calls), and set `robot.options.execute: true`. With an
     agent profile the experiment runs only RPent; do not write `run.execution_mode`. Omit `agent.python` unless the planner lives outside `third_party/RPent/.venv`.
     Prefer a Serial base with the station's normal interpolation and executor (e.g. Dream-1
     `serial-slow10`: Smooth, 10 Hz rows at 100 Hz). Clear `chunk_policy_steps` and
     `max_chunk_policy_steps` when the base sets a chunk cap; keep blending off.
     Physical motion tools require a real execution configuration and measured arrival;
     `execute: false` is not a substitute for an offline fake-device test. Preparing an
     execution recipe or choosing RPent does not authorize connecting or starting the
     robot. Keep all offline probes disconnected and hand physical startup to the operator.
3. Verify without opening devices: resolve the experiment with `load_config`, build the
   robot offline and construct `MotionTools` from the motion profile, import
   `manimux.agents.rpent.toolkit` with `agent.python`, run `manimux doctor` and the
   camera/policy `--check` launchers. Run `scripts.validation.rpent_offline_probe` and
   `scripts.validation.rpent_finish_probe` (synthetic robot) with the planner interpreter;
   the latter must report zero robot commands after Finish.
4. Then `manimux up` with the RoboGUI experiment and open `gui_url`. A hosted VLA
   (`dream_api`) takes no `--policy-python`; its key (`policy.options.api_key_env`)
   must be exported by the user in the shell running `up`; never write it to a file. Hand over at
   **Prepare**: the user selects Execution **RPent**, then Prepare → Start; Pause holds
   and cancels the active call; Finish ends the rollout. Do not press these yourself.

Finish (RoboGUI or planner verdict) leaves the control loop before that tick's send,
fences the dispatch mailbox, fails the active call, rejects later planner requests and
terminates the planner process group. Robot stop/Home/release then follow the normal
assembly cleanup; a Finish with Home still moves the arm home.

## Observe and stop

`manimux status --json` reports process health, camera freshness, policy HELLO identity,
and runtime session state. Policy probes never reset or infer. `operator_control` includes
preparation, execution and cleanup; RoboGUI shows the precise stage. Only recorded or
operator evidence establishes task success.

For requested shutdown, send one SIGINT to the `up` process (the operator's Ctrl-C).
There is no separate stop command. Runtime exits first; a blocked exit retains its
dependencies and hardware ownership. Never send the second SIGINT yourself: it
force-kills services and drops actuator control, so it belongs to the operator after
physically supporting every arm. If the `up` process was lost, rerun the same `up`
command; it attaches to the running deployment. Open RoboGUI Hardware recovery: the operator
confirms a clear path for Home retry, or physical support of ALL arms/payloads for
actuator release. Do not send terminal `RELEASE`; terminal input has been removed.
Never infer support from timeout or GUI disconnection. `manimux attach --service runtime`
is for logs, not the recovery workflow. Detach tmux with Ctrl-B then D.

## Product demo

For rehearsal, run `manimux up --demo --robot yam --gui-port 18086 --deployment-dir
.local/demo --json` in the background, open its URL, and label footage **synthetic preview**. This starts
only RoboGUI, with isolated bridge ports and no station or weights. Follow the
[capture workflow](../../guideline.html#deployment-demo-video) and VHS tape for
repeatable recording. Keep full real-robot takes and label accelerated waits.
