---
name: manimux-station-setup
description: Bind a ManiMux deployment to a local robot, cameras, SDK environments and model/service paths. Use for first installation or another station using supported hardware; use experiments for study design and development for new drivers.
---

# ManiMux Station Setup

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

Read the [repository guide](../manimux-development/references/agent-guide.md) before
editing and use the [station walkthrough](references/station.md) for commands and
binding schemas. Paths are relative to the checkout. A clone does not install SDKs,
create CAN interfaces, download weights or identify the user's devices.

## Choose the launch workflow

For a supported single-config deployment on an installed station, use
[managed deployment](../manimux-deploy/SKILL.md) to check and start services together,
then open RoboGUI and hand over to the operator before Prepare.

Before giving runtime commands, establish whether the user wants to **manage
experiments** or **try once**. If context does not already answer this, ask the user.
Reuse the decision for an ongoing study. Follow the
[experiment workflow](../manimux-experiments/SKILL.md#start-from-the-users-experiment):
`run --config-list` manages experiments. `run --config` tries one concrete runtime
config interactively, including an existing experiment variant; records go to a
separate trial root and no study scoring/progress is enabled. No list or duplicate
config is needed. Optional `--once` connects immediately and exits after one rollout.
Use the same input flag/file and station for camera, generic policy and runtime.
Validate with `inspect --config` or `inspect --config-list` and service `--check`.
Camera/model services remain separate from the runtime command.

## Agent-assisted setup

Apply the shared [information ownership and review workflow](../manimux-development/references/policies.md#resolve-geometry-and-station-facts-before-asking-the-user)
for every model. Read the resolved assembly/groups, component/tool geometry, existing
station calibration and model contract before questioning the user. Intrinsic tool
geometry belongs to the embodiment/tool model; measured installation transforms belong
to reusable station calibration. Do not request the same calibration for each backend.
An inaccessible but already known geometry value needs a model API change, not another
measurement. Only genuinely missing physical facts need setup questions or measurement.
Never manufacture a calibration or copy another station's values to finish a template.

Own the local setup work; do not hand the user a template and ask them to fill YAML.
First inspect existing station bindings, intended Python environments, model assets,
OS device/interface listings and occupied service ports using non-actuating checks.
Reuse verified values. Ask only for facts that cannot be established reliably, such
as which physical camera is front/left/right, an unknown controller address, a remote
model host or the user's intended artifact location. Enumeration order is not placement.

After answers arrive, create or update the private station file, preserving unrelated
bindings. Prepare missing dependencies in the intended environment within the user's
authorized scope; do not replace working environments or substitute model weights.
If an SDK, credential or artifact requires user action, state the specific missing
item and continue independent setup rather than handing off the whole task.

Validate resolved device mappings, checkpoint/normalization paths, client/server
addresses and interpreter imports. Check existing listeners before suggesting ports;
do not silently kill processes, rename CAN interfaces or reconfigure networking.
For a config list, reject any station output override; recording.root belongs to the
config list. Help the user choose it once, then reuse it on subsequent launches. Keep secrets and machine bindings out of shared YAML.

Return the written station path, a short binding summary, remaining unverified items
and matched launch commands. Distinguish static validation from device connectivity
and actual motion. Setup alone does not authorize launching services or moving hardware.
Include each selected arm/camera and transform's source, resolved value and validation
status. Ask the user to check this concrete mapping after the agent has written it;
approval of that mapping is not permission to start the robot.

## Bind an existing deployment

1. Inspect the chosen experiment, robot assembly and existing station. Confirm only
   unknown hardware placement, addresses, model paths or execution settings.
2. Prepare the matching [hardware/model environments](references/environments.md).
   Existing `envs/*/.venv` directories are local assets. Install into the intended
   interpreter; root `uv sync` must not replace an independent hardware/model environment.
3. Create or update the private `manimux/configs/local/station.yaml`, using a matching
   template linked from the walkthrough. Preserve unrelated bindings. Component names
   must match the assembly. Another unit of the same robot does not need SDK edits.
4. Resolve the configuration with `resolve_local_path` and `load_config`; inspect
   addresses, component bindings, model paths and camera configuration without
   constructing a robot. Report unverified device connectivity separately.
5. Hand off interpreter, station, experiment, resolved bindings and the matching
   startup commands. Continue with [experiments](../manimux-experiments/SKILL.md)
   when the user is building a study rather than only connecting a station.

## Contracts that commonly cause mistakes

- YAM CAN names are actual Linux interfaces; `can_left` is not created by ManiMux.
  Left/right placement cannot be inferred from enumeration order. YAM's integrated
  grippers share arm CAN; TacCap gripper serials are separate from camera serials.
- Camera mapping is physical device → component → server stream → runtime sensor →
  `policy.adapter.camera_map`. A RoboGUI preview does not change model input mapping.
- Use one config list/station across runtime, camera and model launchers for studies.
  If services differ across entries, pass the same explicit `--experiment-id`; do not
  select an arbitrary child config for cameras. Legacy standalone examples use one
  experiment/station.
  Selection precedence is CLI `--local`, config list `local:`, then the default station.
  Standalone examples use experiment `local:` instead.
  CLI paths are cwd-relative; experiment references and station paths have their own
  file-relative bases. Read the walkthrough's scope table for unsupported entry points.
- Pi05 `paths.checkpoints` relocates a root; its recipe still selects checkpoint and
  normalization subpaths. UMI_DP/XR-1 artifact overrides have different contracts.
  Address binding must not change action semantics, timing, FK/IK, limits or weights.
- RoboGUI's Tianji preset is `--robot tianji`; the assembly/runtime identity is
  `tianji-taccap`. Match the deployment's identity rather than copying the CLI alias.
- Read `execute`, `move_to_start_on_connect` and the selected assembly's cleanup/Home
  behavior from the resolved configuration and current implementation.
  Prepare can cause motion; `execute: false` alone does not make a launch offline.
  Inspecting bindings or providing commands does not authorize service startup/motion.

## References by task

[Station schemas and launch scope](references/station.md) ·
[Environments](references/environments.md) ·
[CAN/ARX/PiPER adapters](references/can-arms.md) ·
[SDK installation](references/robot-sdks.md) ·
[Offline replay](references/replay.md).
For unsupported hardware, follow [development](../manimux-development/SKILL.md).
