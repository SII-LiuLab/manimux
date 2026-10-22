# ManiMux Agent Guide

ManiMux is an extensible real-world manipulation harness. Keep model inference, runtime
scheduling, embodiment control and experiment interfaces separate.
Teleoperation and demonstration collection are outside this repository; retain runtime
rollout recording and offline replay.
These conventions apply throughout this checkout. Repository skills load this guide
explicitly; read instructions in any submodule or nested directory before editing it.

Public entry points: [User guide](../../../../guideline.html) · [Integration map](README.md).
Skills route agents to these same maintained protocol guides.

## Engineering discussion rule

- Do not begin a response to a challenge with an automatic apology or concession.
  A question or counterexample is a request to examine the design, not an instruction
  to abandon it. Inspect evidence and state your own project-level judgment first.
- Separate observed behavior, requirements, proposals and unresolved assumptions.
  Explain the relevant code/config context, tradeoffs and concrete user workflow.
- Keep a recommendation consistent unless new evidence or a changed requirement
  warrants revision. When revising, identify the exact premise that changed and
  which parts of the earlier recommendation still hold. Correct confirmed mistakes
  plainly; this rule does not justify defensiveness or concealing an error.
- During architecture discussions, inspect existing ownership and representative
  upstream designs before inventing new layers. Discussion alone does not authorize
  configuration migration, service restarts or hardware motion.

## Documentation ownership

Root `README.md` explains project purpose, architecture and entry points. Directory
READMEs locate that directory's files and ownership boundaries; they are not duplicate
launch tutorials. Skills describe the agent's inspect/ask/write/verify workflow.
Maintain `guideline.html` as the single public user guide. Keep algorithm parameters,
launch commands and experiment setup there, using short sections and concrete examples.
Detailed component contracts and model runbooks live under the owning skill's
`references/`; keep historical reproduction evidence explicitly dated. Read only the
reference needed for the current task and verify runnable settings against current YAML.
`docs/` is an ignored local archive, not an active source or a fallback on a fresh clone.
The local root `experiments.md` holds this team's study decisions/results and stays private.
GitHub Pages copies `guideline.html` to `site/index.html`; no Markdown build is required.

## Product naming

Use **RoboGUI** as the visualization product name in documentation, UI labels and
launch messages. Public interfaces use `manimux-robogui`, `manimux.robogui`,
`robogui:` experiment settings and `robogui.yaml` presets. Use the same naming in
new integrations. Refer to `viser` only when discussing the underlying library
or its API, not as the ManiMux product name.

## Task skills

Read only the skill relevant to the task; paths inside skills are relative to this
repository root. They describe existing code and workflows, not permission to run hardware.

| Task | Skill |
|---|---|
| Deploy an installed station, open RoboGUI, or inspect/stop managed services | [Deployment](../../manimux-deploy/SKILL.md) |
| Integrate or review a component, policy client/adapter, runtime or RoboGUI feature | [Development](../SKILL.md) |
| Bind an installation to local robots, cameras and SDKs | [Station setup](../../manimux-station-setup/SKILL.md) |
| Track experiment progress, evaluate recorded rollouts, run PRM, or update experiment result tables | [Experiments](../../manimux-experiments/SKILL.md) |

For a new integration, the Development skill maps each extension to its existing
interface, owning directory, configuration and focused validation. Read its relevant
reference before copying an older integration; compatibility paths are not templates.

## First connection to a local robot

For another installation of a supported robot, start with
[local station setup](../../manimux-station-setup/references/station.md) and its matching template.
Keep machine-specific CAN interfaces, controller IPs, device serials and service addresses
in one private station file. Read an existing file before editing it, preserve component
names, and establish physical device mappings instead of assuming enumeration order means left/right.

The default private file is `manimux/configs/local/station.yaml`. Runtime startup and the
common camera/policy `--config` and `--config-list` entry points read it automatically; `--local` selects
another station. Follow the guide's scope table for entry points that still have separate
configuration. Configuration inspection does not authorize starting services or connecting
or moving hardware. Keep action semantics, FK/IK, timing and execution settings unchanged
when binding another installation of the same robot model.

Maintain code comments, user-facing text, configuration labels and documentation in English.
Preserve literal device paths and upstream legal notices; `README.zh-CN.md` redirects readers
to the English documentation.

## Learned models and backend frameworks

Classify an integration before choosing its directory. A learned model and a reusable
algorithm or serving framework are different extension types:

- Add a model, checkpoint or model reproduction to the framework that owns its runtime.
  When that framework is XPolicyLab, implement it under `XPolicyLab/policy/<POLICY>/`
  using the XPolicyLab adapter and serving conventions.
- Keep a distinct upstream framework with its own reusable runtime, dependency stack,
  model registry or serving contract peer to XPolicyLab. Do not nest the framework under
  `XPolicyLab/policy/` merely to reuse XPolicyLab's transport. Put its ManiMux client under
  `manimux/policies/<framework>/` and implement the existing `PolicyModel` contract.
- A new task, checkpoint, embodiment or one-model wrapper is not a new framework. Reuse
  the selected framework and change recipes or experiments instead of adding a backend.

Do not add model implementations or framework runtimes inside the ManiMux Python package.

For models owned by XPolicyLab:

- Read [XPolicyLab/AGENTS.md](../../../../XPolicyLab/AGENTS.md),
  [the contribution standard](../../../../XPolicyLab/CONTRIBUTING.md) and
  [the reference adapter](../../../../XPolicyLab/policy/demo_policy) before implementation.
- Reuse an existing policy directory when an XPolicyLab model is already integrated.
- Keep upstream model source and reproduction changes under that policy directory,
  following XPolicyLab's vendored-source or pinned-submodule conventions. Preserve upstream
  licenses and attribution; document the upstream URL and revision in the policy README.
  Do not depend on an unrelated local checkout, absolute developer path or untracked symlink.
- Implement the actual model loading, preprocessing, normalization, sampling and output
  conversion there. A `model.py` that merely forwards to a legacy ManiMux native server,
  or imports its model implementation from elsewhere in ManiMux, is not a migration.
- Do not add model weights, network implementations, processors, training pipelines or
  model-specific inference servers under ManiMux's `manimux/`, `scripts/` or `envs/`.
  Lightweight launchers that load config and start the XPolicyLab server are allowed.
- Use ManiMux's existing `xpolicylab_ws` worker for XPolicyLab models. Do not introduce
  another per-model HTTP/TCP protocol or native worker for a model already served by a
  supported framework. A peer framework may retain its native transport behind its
  `PolicyModel` client; capability, identity and reset behavior must remain explicit.

Robot drivers, embodiment/action adapters, runtime strategies, executors and mock policies
still belong in ManiMux. Model and framework environments may remain isolated; all backends
must reuse ManiMux's adapter, scheduling and execution layers instead of duplicating them.

## Required structure and boundaries

```text
XPolicyLab/policy/<POLICY>/  # when XPolicyLab owns the model
    model.py + deploy.yml + deploy.py
    upstream model source / pinned source submodule
    installation, data, training and evaluation entry points
    README.md
<FRAMEWORK>/                 # when adding a peer framework
    independently versioned framework source and deployment entry points
manimux/policies/<framework>/
    PolicyModel client + framework wire codec
manimux/configs/policy/<model>/<embodiment>/<task>/
manimux/configs/lists/<study>.yaml  # entries, trial plan and recording.root
manimux/configs/tasks/<task>.yaml  # reusable task semantics
manimux/configs/experiments/<model-checkpoint>/{_base_.yaml,<task>/<variant>.yaml}
manimux/configs/embodiment/initial_poses/<robot>.yaml
manimux/configs/inference/
manimux/configs/executor/
.agents/skills/manimux-inference/references/deployment/<model>-<embodiment>.md
training/  # Private configurations, launchers and notes; ignored by Git
```

Policy recipes select deployment artifacts, backend workers and inference parameters.
Each framework owns its model defaults and serving process; complete ManiMux runtime
choices live in experiments.
Keep policy recipes free of `server/`, `infra/` and `training/` subdirectories. Put
installation-specific training work in the root `training/` workspace; deployment
metadata needed for inference remains with the checkpoint and policy recipe.

XPolicyLab policy files and scripts must follow the full XPolicyLab contribution standard,
including `Model(ModelTemplate)`, observation/action/batch/reset interfaces, standard
action dictionaries and `policy_name` matching the directory. Declare unsupported stages;
do not substitute fake training, dummy actions or silent fallbacks for an implementation.

| Layer | Responsibility |
|---|---|
| Model framework adapter | Model source, checkpoint loading, model transforms and sampler hooks |
| ManiMux `policies/` client | Backend transport, wire codec, identity and capabilities |
| ManiMux `policy_adapter/` | Observation mapping, robot groups, action semantics and necessary FK/IK |
| ManiMux runtime | Inference scheduling, chunk handoff, timelines and rollout lifecycle |
| Executor / RobotBase | Command generation, configured limits and hardware communication |
| Robo GUI / recording | Experiment controls, visualization and execution evidence |

Model servers must not connect to cameras/CAN or command a robot. The hardware runtime
must not acquire model dependencies such as torch/JAX just to use a new policy.
Keep task, checkpoint, cameras, embodiment and runtime choices in configuration rather
than hard-coding one task or station into a model wrapper.

Within an XPolicyLab policy, reuse its shared image, dimension and checkpoint helpers;
do not duplicate them.
Preserve the checkpoint's RGB convention, joint order, gripper convention, absolute/delta
semantics, normalization, horizon and action interval across data conversion and inference.
Shared embodiment control profiles should align collection and deployment timing and motion
limits; execution smoothing remains an explicit, separate choice.

RTC, PAINT and other specialized sampling modes may be advertised only when their required
hooks are actually implemented in the model sampler. Keep capability negotiation and backend
identity checks; never disable them to make a mismatched checkpoint or unsupported mode run.

## Removed native and compatibility paths

The built-in MolmoAct2 and ABC model servers, HTTP clients, ManiMux action adapters and
experiments have been removed. Reuse `XPolicyLab/policy/MolmoACT2/` for MolmoAct2. Any future
ABC integration must also use XPolicyLab rather than restoring the deleted native or HTTP path.
Do not assume that an existing policy directory already covers a local checkpoint and action
contract.

When migrating a legacy model:

1. Move its model-side source and reproduction logic into the appropriate XPolicyLab policy.
2. Validate the real adapter and shared server independently of the old native server.
3. Compare checkpoint, transforms, observation/action contracts, timing and reset behavior
   against the old path; do not silently change runtime or control settings during migration.
4. Add matching policy recipes and experiments, tests and commands before switching defaults.
5. Retire the native implementation only after the replacement is validated and the migration
   is in scope. Until then, identify it as legacy and do not claim migration is complete.

Do not break an existing working deployment merely to enforce a new directory layout.
Fixing a legacy bug does not authorize migrating unrelated models or stopping live services.

## Validation and delivery

- Start with the selected framework's static/interface checks and relevant ManiMux unit tests.
  Verify camera mapping, action keys/shapes, reset behavior, paired backend identity and
  sampling capabilities before declaring an integration ready.
- Distinguish static checks, offline forward, server readiness and real-robot task success.
  Report unavailable checkpoints, hardware or validation honestly; never invent results.
- Do not start/stop camera, training, policy or robot services for a documentation change.
  Model integration work alone does not authorize physical robot motion.
- Keep weights, datasets, videos generated by validation, local source manifests, credentials
  and machine-specific experiment artifacts out of commits. Preserve useful reusable tests.
- `XPolicyLab` and peer framework submodules are separate Git repositories. Publish framework
  changes to the user-confirmed remote/branch before publishing a parent gitlink. Never publish
  an unreachable submodule revision or assume a developer branch.
- Commit or push only when requested. Keep unrelated edits and submodule pointers unchanged.

## Public branch changes

Changes to `main` must go through a pull request; do not push directly to it.
Keep local training variants and station-specific work on their designated branches
until explicitly selected for public release. Build cleanup PRs from `main` so they
do not pull in unrelated local experiments. Review references before removing scratch
files; keep maintained runtime implementations, licenses and actively used assets.
