# Documentation

[Project overview](../README.md) · [中文首页](../README.zh-CN.md)

## Start here

- [Connect your own robot](../manimux/configs/local/README.md): first steps for users and agents; one private station file for CAN/IP/USB and shared deployment service bindings.
- [Guideline](guideline.md): installation, hardware-free demos and complete YAM startup commands.
- [Python environments](../envs/README.md) · [XPolicyLab action metadata](../env_cfg/README.md): why these differ from experiment and station configuration.
- [Configuration guide](../manimux/configs/README.md): config layout, field meanings and shared control profiles.
- [Viewer tutorial](viewer-tutorial.html): rollout controls, camera views and experiment labels.
- [Action replay](action-replay.md): play named joint trajectories in an independent offline Viewer.
- [Experiment workflow](experiment-infra.md): persistent services, normal/experiment modes and saved evidence.
- [Architecture](architecture.md): policy, adapter, strategy, executor and robot boundaries.
- [Agent guide](../AGENTS.md): XPolicyLab-only model integration, legacy migration and validation rules.

## Components

- [XPolicyLab](../XPolicyLab/): model adapters and model-side sampling behind the shared
  policy interface. See the [integration runbook](xpolicylab-runbook.md).
- [PRM-as-a-Judge](../PRM-as-a-Judge/): offline model-based evaluation of recorded videos.
  See the [evaluation guide](prm-as-a-judge.md).

Both are version-pinned submodules. They are platform components, not additional policies
or inference strategies in the support counts below.

## Scope

Teleoperation and demonstration collection are outside this repository. ManiMux retains
policy deployment, rollout recording, offline replay and evaluation.

## Policies and deployment

Use each runbook's checkpoint, environment and action contract together. An available adapter
does not mean every checkpoint or inference-method combination has passed a real-robot trial.

- [Pi05 / OpenPI](pi05-yam-runbook.md), including paired put-bottles joint / joint+EE 30k configurations.
- [UMI DP on Tianji–TacCap](umi-dp-tianji-taccap-runbook.md), including checkpoint binding,
  TacCap camera service, Viewer and hardware runtime commands.
- [MolmoAct2](molmoact-yam-runbook.md) · [ABC](abc-yam-runbook.md).
- [GR00T N1.7](gr00t-yam-runbook.md) · [LingBot-VLA2](lingbot-vla2-yam-runbook.md).
- [Xiaomi XR-1](xiaomi-xr1-yam-runbook.md) · [OpenWAM](openwam-yam-runbook.md).
- [Xiaomi XR-1 on Tianji–TacCap](xiaomi-xr1-tianji-taccap-runbook.md), including the pass-ball step-50000 checkpoint and synthetic black ego view.
- [SAPolicy](sapolicy-yam-runbook.md) · [Shared XPolicyLab bridge](xpolicylab-runbook.md).
- Offline / simulation paths: [Cosmos3](cosmos3-offline-runbook.md),
  [Isaac 0.5](isaac05-offline-runbook.md).

## Support counts

The README badges count integration coverage, not task success, hardware validation of every
checkpoint, or support for every policy × embodiment × inference combination.

- **10 policy integrations:** eight model families have YAM deployment configurations:
  Pi05, MolmoAct2, ABC, GR00T, LingBot-VLA2, Xiaomi XR-1, OpenWAM and SAPolicy.
  Cosmos3 and Isaac 0.5 add two model-only / offline paths, not two more YAM-ready policies.
  Checkpoint variants, the generic XPolicyLab bridge are not counted separately.
- **Hardware assemblies:** YAM uses the component implementation; Tianji–TacCap migration
  boundaries are listed in [code organization](code-organization.md). Simulator drivers
  have been retired. A model checkpoint does not itself establish a hardware integration.
- **8 inference modes:** seven built-in strategies—ManiMux, RTC, ACT temporal ensembling,
  AAC, PAINT, AutoHorizon and DVAC—plus serial prefix execution. Serial is a scheduling mode
  of the ManiMux strategy, not an eighth registered strategy. Direct, Smooth and MPC are
  executors and are not counted as inference methods.

Sources: [model configurations](../manimux/configs/), [robot factories](../manimux/embodiments/robot/__init__.py),
[strategy registry](../manimux/runtime/inference.py) and [serial execution](serial-execution.md).

## Inference and execution

Method documents distinguish the upstream method, ManiMux adaptation and validation evidence.
Choose a method supported by the policy backend; these are not interchangeable sampler hooks.

- [ManiMux asynchronous scheduling](architecture.md) · [Serial prefix execution](serial-execution.md).
- [RTC contract](xpolicylab-runbook.md#rtc-规则) · [Chunk-step semantics](chunk-steps.md).
- [ACT temporal ensembling](act-temporal-ensemble.md).
- [AAC](reproductions/aac.md) · [Pi05 AAC](reproductions/aac-pi05.md).
- [Pi05 PAINT](reproductions/paint-pi05.md).
- [Pi05 AutoHorizon](reproductions/autohorizon-pi05.md) · [Pi05 DVAC](reproductions/dvac-pi05.md).
- [Braking execution](braking-execution.md) · [Independent IK](independent-ik-execution.md).

## Experiments and evaluation

- [Experiment design](experiment-design.md): controlled comparisons and evaluation planning.
- [PRM-as-a-Judge](prm-as-a-judge.md): offline video manifests, judge setup and reports.
- [Reference-layout overlay](evaluation-layout-overlay.md): reproduce an experiment's initial scene.
- [CAN setup](can-bus.md): YAM-specific hardware configuration.

Private training recipes, launchers and experiment notes live under the root
`training/` directory, which is ignored by Git and is not part of the public package.
XPolicyLab and upstream frameworks own model training and checkpoint loading.
