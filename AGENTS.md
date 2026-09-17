# ManiMux Agent Guide

ManiMux is a composable real-robot experiment platform. Keep model inference, runtime
scheduling, embodiment control, collection and experiment interfaces separate.
These instructions apply throughout this checkout; also read the instructions in any
submodule or nested directory before editing it.

## Task skills

Read only the skill relevant to the task; paths inside skills are relative to this
repository root. They describe existing code and workflows, not permission to run hardware.

| Task | Skill |
|---|---|
| Develop a driver, camera, adapter, runtime, collection or Viewer feature | [Development](.agents/skills/manimux-development/SKILL.md) |
| Bind an installation to local robots, cameras and SDKs | [Station setup](.agents/skills/manimux-station-setup/SKILL.md) |
| Select configs, give startup commands or run an experiment | [Experiment](.agents/skills/manimux-experiment/SKILL.md) |
| Analyze recorded rollouts, chunks, tracking or video | [Result analysis](.agents/skills/manimux-result-analysis/SKILL.md) |

## Model integration: XPolicyLab only

**Every new learned-policy integration or model reproduction must be implemented inside
`XPolicyLab/policy/<POLICY>/`, using the XPolicyLab adapter and serving conventions.**
Do not add another standalone native model implementation to ManiMux.

- Read [XPolicyLab/AGENTS.md](XPolicyLab/AGENTS.md),
  [the contribution standard](XPolicyLab/CONTRIBUTING.md) and
  [the reference adapter](XPolicyLab/policy/demo_policy/) before implementation.
- Reuse an existing policy directory when the model is already integrated. A new task,
  checkpoint or embodiment is not a reason to duplicate the model implementation.
- Keep upstream model source and reproduction changes under that policy directory,
  following XPolicyLab's vendored-source or pinned-submodule conventions. Preserve upstream
  licenses and attribution; document the upstream URL and revision in the policy README.
  Do not depend on an unrelated local checkout, absolute developer path or untracked symlink.
- Implement the actual model loading, preprocessing, normalization, sampling and output
  conversion there. A `model.py` that merely forwards to a legacy ManiMux native server,
  or imports its model implementation from `src/manimux/integrations/`, is not a migration.
- Do not add model weights, network implementations, processors, training pipelines or
  model-specific inference servers under ManiMux's `src/`, `scripts/` or `envs/`.
  Lightweight launchers that load config and start the XPolicyLab server are allowed.
- Use ManiMux's existing `xpolicylab_ws` worker. Do not introduce another per-model HTTP/TCP
  protocol or register a new native model worker to bypass the shared policy interface.

This rule concerns learned models. Robot drivers, embodiment/action adapters, runtime
strategies, executors, mock policies and collection leader policies still belong in ManiMux.
An isolated model environment or process is expected; a parallel native integration stack is not.

## Required structure and boundaries

```text
XPolicyLab/policy/<POLICY>/
    model.py + deploy.yml + deploy.py
    upstream model source / pinned source submodule
    installation, data, training and evaluation entry points
    README.md
configs/<model>/<embodiment>/server/<task>/
configs/<model>/<embodiment>/infra/<task>/
docs/<model>-<embodiment>-runbook.md
```

The policy files and scripts must follow the full XPolicyLab contribution standard,
including `Model(ModelTemplate)`, observation/action/batch/reset interfaces, standard
action dictionaries and `policy_name` matching the directory. Declare unsupported stages;
do not substitute fake training, dummy actions or silent fallbacks for an implementation.

| Layer | Responsibility |
|---|---|
| XPolicyLab model adapter | Model source, checkpoint loading, model transforms and sampler hooks |
| ManiMux policy / embodiment adapter | Observation mapping, robot groups, action semantics and necessary FK/IK |
| ManiMux runtime | Inference scheduling, chunk handoff, timelines and rollout lifecycle |
| Executor / RobotDriver | Command generation, configured limits and hardware communication |
| Collection / Robo GUI / recording | Demonstrations, experiment controls, visualization and execution evidence |

Model servers must not connect to cameras/CAN or command a robot. The hardware runtime
must not acquire model dependencies such as torch/JAX just to use a new policy.
Keep task, checkpoint, cameras, embodiment and runtime choices in configuration rather
than hard-coding one task or station into a model wrapper.

Reuse XPolicyLab's shared image, dimension and checkpoint helpers; do not duplicate them.
Preserve the checkpoint's RGB convention, joint order, gripper convention, absolute/delta
semantics, normalization, horizon and action interval across data conversion and inference.
Shared embodiment control profiles should align collection and deployment timing and motion
limits; execution smoothing remains an explicit, separate choice.

RTC, PAINT and other specialized sampling modes may be advertised only when their required
hooks are actually implemented in the model sampler. Keep capability negotiation and backend
identity checks; never disable them to make a mismatched checkpoint or unsupported mode run.

## Retired native integrations

The native MolmoAct2 and ABC servers, workers and deployment configs have been removed,
as has ManiMux's duplicate XR-1 model source. Do not restore parallel native model stacks.
XR-1 retains its embodiment adapter and NumPy action codec; its model runs in XPolicyLab.

MolmoAct2 has source under `XPolicyLab/policy/MolmoACT2/`, but that alone does not validate
the former YAM checkpoint and action contract. ABC has no replacement deployment here.
Their previous implementations remain in Git history; removal is not a claim of migration.
Reintroducing either deployment requires the XPolicyLab integration, paired configs, tests
and documented validation of checkpoint, transforms, timing and observation/action contracts.

## Validation and delivery

- Start with the XPolicyLab static/interface checks and relevant ManiMux unit tests.
  Verify camera mapping, action keys/shapes, reset behavior, paired backend identity and
  sampling capabilities before declaring an integration ready.
- Distinguish static checks, offline forward, server readiness and real-robot task success.
  Report unavailable checkpoints, hardware or validation honestly; never invent results.
- Do not start/stop camera, training, policy or robot services for a documentation change.
  Model integration work alone does not authorize physical robot motion.
- Keep weights, datasets, videos generated by validation, local source manifests, credentials
  and machine-specific experiment artifacts out of commits. Preserve useful reusable tests.
- `XPolicyLab` is a separate Git repository. When publishing a model integration, publish its
  changes to the user-confirmed submodule remote/branch before publishing the parent gitlink.
  Never publish an unreachable submodule revision or assume a developer branch.
- Commit or push only when requested. Keep unrelated edits and submodule pointers unchanged.
