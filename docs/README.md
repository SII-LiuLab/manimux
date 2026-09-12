# Documentation

[Project overview](../README.md) · [中文首页](../README.zh-CN.md)

## Start here

- [Guideline](guideline.md): installation, hardware-free demos and complete YAM startup commands.
- [Configuration guide](../configs/README.md): config layout, field meanings and shared control profiles.
- [Viewer tutorial](viewer-tutorial.html): rollout controls, camera views and experiment labels.
- [Experiment workflow](experiment-infra.md): persistent services, normal/experiment modes and saved evidence.
- [YAM collection](yam-collection.md): the original-style collection GUI with ManiMux follower control.
- [Architecture](architecture.md): policy, adapter, strategy, executor and robot boundaries.

## Policies and deployment

Use each runbook's checkpoint, environment and action contract together. An available adapter
does not mean every checkpoint or inference-method combination has passed a real-robot trial.

- [Pi05 / OpenPI](pi05-yam-runbook.md), including paired put-bottles joint / joint+EE 30k configurations.
- [MolmoAct2](molmoact-yam-runbook.md) · [ABC](abc-yam-runbook.md).
- [GR00T N1.7](gr00t-yam-runbook.md) · [LingBot-VLA2](lingbot-vla2-yam-runbook.md).
- [Xiaomi XR-1](xiaomi-xr1-yam-runbook.md) · [OpenWAM](openwam-yam-runbook.md).
- [SAPolicy](sapolicy-yam-runbook.md) · [Shared XPolicyLab bridge](xpolicylab-runbook.md).
- Offline / simulation paths: [Cosmos3](cosmos3-offline-runbook.md),
  [Isaac 0.5](isaac05-offline-runbook.md), [ManiUniCon](maniunicon-sim.md).

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

## Experiments, evaluation and training

- [Experiment design](experiment-design.md): controlled comparisons and evaluation planning.
- [PRM-as-a-Judge](prm-as-a-judge.md): offline video manifests, judge setup and reports.
- [Reference-layout overlay](evaluation-layout-overlay.md): reproduce an experiment's initial scene.
- [YAM training pipeline](yam-training-pipeline.md) · [Pi05 joint+EE training](pi05-bottles-joint-ee-training.md).
- [CAN setup](can-bus.md): YAM-specific hardware configuration.
