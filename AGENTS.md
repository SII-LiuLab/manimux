# ManiMux Agent Guide

ManiMux is a composable real-robot experiment platform. Keep model inference, runtime
scheduling, embodiment control, collection and experiment interfaces separate.
These instructions apply throughout this checkout; also read the instructions in any
submodule or nested directory before editing it.

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

## Existing native integrations

`molmoact_http` and `abc_http` are legacy compatibility paths, not templates for new work.
The target for both is XPolicyLab. Check and reuse `XPolicyLab/policy/MolmoACT2/` for MolmoAct2;
do not assume that an existing directory already covers the local checkpoint and action contract.

When migrating a legacy model:

1. Move its model-side source and reproduction logic into the appropriate XPolicyLab policy.
2. Validate the real adapter and shared server independently of the old native server.
3. Compare checkpoint, transforms, observation/action contracts, timing and reset behavior
   against the old path; do not silently change runtime or control settings during migration.
4. Add matching server/infra configs, tests and documented commands before switching defaults.
5. Retire the native implementation only after the replacement is validated and the migration
   is in scope. Until then, identify it as legacy and do not claim migration is complete.

Do not break an existing working deployment merely to enforce a new directory layout.
Fixing a legacy bug does not authorize migrating unrelated models or stopping live services.

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
