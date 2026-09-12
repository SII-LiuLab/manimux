<div align="center">

# ManiMux
**A composable platform for real-robot experiments.**

Any Policy × Embodiment × Inference

[![Platform](https://img.shields.io/badge/Platform-7C3AED?style=flat-square)](#features)
[![Robo GUI](https://img.shields.io/badge/Robo%20GUI-0891B2?style=flat-square)](docs/viewer-tutorial.html)
<br/>
[![Component: XPolicyLab](https://img.shields.io/badge/Component-XPolicyLab-4F46E5?style=flat-square&logo=github&logoColor=white)](XPolicyLab/)
[![Component: PRM-as-a-Judge](https://img.shields.io/badge/Component-PRM--as--a--Judge-9333EA?style=flat-square&logo=github&logoColor=white)](PRM-as-a-Judge/)
[![Python 3.11 and 3.12](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?style=flat-square&logo=python&logoColor=white)](pyproject.toml)
<br/>
[![Policies: 10 integrations, including 2 model-only paths](https://img.shields.io/badge/Policies-10%20Integrations-2EA043?style=flat-square)](docs/README.md#support-counts)
[![Embodiments: 1 real robot and 1 simulation integration](https://img.shields.io/badge/Embodiments-1%20Real%20%2B%201%20Sim-2563EB?style=flat-square)](docs/README.md#support-counts)
[![Inference: 8 modes](https://img.shields.io/badge/Inference-8%20Modes-F97316?style=flat-square)](docs/README.md#support-counts)
<br/>
[![Collection: Teleop, UMI and DAgger; implementation status in roadmap](https://img.shields.io/badge/Collection-Teleop%20%7C%20UMI%20%7C%20DAgger-D97706?style=flat-square)](docs/README.md#collection-status)
[![Evaluation: human feedback and LLM judge](https://img.shields.io/badge/Evaluation-Human%20%2B%20LLM%20Judge-DB2777?style=flat-square)](docs/prm-as-a-judge.md)

[English](README.md) · [简体中文](README.zh-CN.md)

[**Features**](#features) · [**Demo**](#demo) · [**Architecture**](#architecture) · [**Guideline**](docs/guideline.md) · [**Documentation**](docs/README.md)

</div>

**ManiMux is a real-robot experiment platform for data collection, policy deployment and evaluation.**
It decouples policies, inference strategies, executors and embodiments, so researchers can compare
methods on a shared control stack, operate trials through a GUI, and inspect what the robot actually executed.

> Installation and startup commands: [Guideline](docs/guideline.md). Model and method details: [Documentation](docs/README.md).

## Features

| Feature | Status | What it provides |
|---|:---:|---|
| Composable deployment | ✅ | Policy × inference strategy × executor × embodiment |
| Inference methods | ✅ | Async, serial, RTC, PAINT and adaptive chunking |
| Robo GUI | ✅ | Rollout controls, cameras, 3D state, trajectories and chunk timelines |
| Teleop collection | ✅ | Leader policy + YAM GUI, with ManiMux follower control |
| Shared control profiles | ✅ | Common hardware settings, action timing and arm / gripper limits |
| Execution evidence | ✅ | Configs, observations, actions, commands, feedback, events and video |
| Evaluation | ✅ | Human labels + offline PRM / LLM judging |
| UMI / DAgger collection | — | [Collection roadmap](docs/README.md#collection-status) |

✅ denotes implemented functionality, not validation of every model / hardware combination.
[Support counts](docs/README.md#support-counts) also include model-only and simulation paths.

## Demo

![ManiMux rollout with live cameras, robot state and action-chunk visualization](assets/manimux-viewer-demo.webp)

## Architecture

```mermaid
flowchart LR
    observation["Cameras + robot state"] --> xpolicy["XPolicyLab"]
    observation --> native["Native policy adapters"]

    subgraph runtime["ManiMux execution"]
        strategy["Inference strategy + Timeline"] --> executor["Executor + Safety"]
    end

    xpolicy --> strategy
    native --> strategy
    teleop["Teleop GUI + LeaderPolicy"] --> executor
    profile["Shared control profile"] -.-> executor
    executor --> robot["RobotDriver + hardware"]
    executor -.-> viewer["Robo GUI"]
    executor -.-> records["Episode records"]
    viewer -.->|human labels| records
    records --> prm["PRM-as-a-Judge"]

    classDef component fill:#eef2ff,stroke:#6366f1,color:#312e81
    classDef control fill:#eff6ff,stroke:#3b82f6,color:#1e3a8a
    classDef interface fill:#ecfdf5,stroke:#10b981,color:#064e3b
    class xpolicy,native,prm component
    class strategy,executor,robot control
    class teleop,viewer interface
```

Teleoperation reuses execution interfaces without chunk scheduling; it retains its own GUI and recording format.

**GitHub:** [ManiMux](https://github.com/SII-LiuLab/manimux) · [XPolicyLab](https://github.com/Cuzyoung/XPolicyLab) · [PRM-as-a-Judge](https://github.com/YuyangLiu2003/PRM-as-a-Judge)

## Guides

- **Run:** [Guideline](docs/guideline.md) · [Configuration](configs/README.md).
- **Integrate:** [Components and policy runbooks](docs/README.md) · [Inference methods](docs/README.md#inference-and-execution).
- **Collect / evaluate:** [YAM collection](docs/yam-collection.md) · [Experiment workflow](docs/experiment-infra.md) · [PRM guide](docs/prm-as-a-judge.md).
- **Extend:** [Architecture contracts](docs/architecture.md).

Physical robots require matching model contracts and hardware safety measures; software checks do not certify safety or task success.
Upstream attribution: [notices](THIRD_PARTY_NOTICES.md) · [licenses](licenses).
