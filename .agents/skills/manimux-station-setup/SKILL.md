---
name: manimux-station-setup
description: Bind a ManiMux deployment to a local robot, cameras, SDK environments and model/service paths. Use for first installation or another station using supported hardware; use experiments for study design and development for new drivers.
---

# ManiMux Station Setup

Read the [repository guide](../manimux-development/references/agent-guide.md) before
editing and use the [station walkthrough](references/station.md) for commands and
binding schemas. Paths are relative to the checkout. A clone does not install SDKs,
create CAN interfaces, download weights or identify the user's devices.

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
- Use one experiment/station across runtime, camera and supported model launchers.
  Selection precedence is CLI `--local`, experiment `local:`, then the default station.
  CLI paths are cwd-relative; experiment references and station paths have their own
  file-relative bases. Read the walkthrough's scope table for unsupported entry points.
- Pi05 `paths.checkpoints` relocates a root; its recipe still selects checkpoint and
  normalization subpaths. UMI_DP/XR-1 artifact overrides have different contracts.
  Address binding must not change action semantics, timing, FK/IK, limits or weights.
- RoboGUI's Tianji preset is `--robot tianji`; the assembly/runtime identity is
  `tianji-taccap`. Match the deployment's identity rather than copying the CLI alias.
- Read `execute`, `move_to_start_on_connect` and `home_on_close` from the experiment.
  Prepare can cause motion; `execute: false` alone does not make a launch offline.
  Inspecting bindings or providing commands does not authorize service startup/motion.

## References by task

[Station schemas and launch scope](references/station.md) ·
[Environments](references/environments.md) ·
[CAN/ARX/PiPER adapters](references/can-arms.md) ·
[SDK installation](references/robot-sdks.md) ·
[Offline replay](references/replay.md).
For unsupported hardware, follow [development](../manimux-development/SKILL.md).
