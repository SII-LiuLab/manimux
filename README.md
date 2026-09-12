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

[**Features**](#features) · [**Video**](#demo) · [**Architecture**](#architecture) · [**Quick Start**](#quick-start) · [**Documentation**](docs/README.md) · [**Citation**](#citation)

</div>

**ManiMux is a real-robot experiment platform for data collection, policy deployment and evaluation.**
Changing a policy should not mean rebuilding the robot control stack. ManiMux separates
**what to predict, when to execute, and how to control the hardware**, so models and chunking
methods can be compared through a consistent experiment workflow.

Collect demonstrations with a leader policy, run learned policies through configurable executors,
and inspect camera streams, trajectories and chunk handoffs in Robo GUI. Episode records connect
model predictions to issued commands and robot feedback, with human labels and offline judging
for evaluation. **XPolicyLab** provides policy integrations; **PRM-as-a-Judge** provides offline evaluation.

> 📖 Start with the [Guideline](docs/guideline.md) for installation and setup, or use the [Pi05 example](#quick-start) below. Detailed model and method guides live in [Documentation](docs/README.md).

## News

- **[2026-09-12]** YAM teleoperation collection now runs through ManiMux, with shared collection / inference control profiles. [Details](docs/yam-collection.md) · [Change](https://github.com/SII-LiuLab/manimux/commit/9d14670)

<a id="features"></a>

## ✨ Features

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

<a id="demo"></a>

## 🎬 Demo Video

Dual-arm YAM rollout with live cameras, 3D robot state and action-chunk handoffs.

![ManiMux rollout with live cameras, robot state and action-chunk visualization](assets/manimux-viewer-demo.webp)

[**▶ Open the MP4 recording**](assets/manimux_2026-09-05_23-17-35-00.00.03.144-00.00.34.914-seg1-00.00.02.596-00.00.34.966.mp4)

<a id="architecture"></a>

## 🧩 Architecture

**Policy inference and hardware execution are separate responsibilities.** The inference strategy
decides when and how to hand off a chunk; the executor turns its targets into robot commands.

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 180, "curve": "basis", "nodeSpacing": 24, "rankSpacing": 28, "padding": 14}}}%%
flowchart LR
    OBS["<b>OBSERVE</b><br/>Cameras · robot state<br/>Build policy inputs"]:::stage

    subgraph THINK["<b>PREDICT</b>"]
        direction TB
        XPOLICY["<b>XPolicyLab</b><br/>Pi05 · XR-1 · GR00T<br/>LingBot · OpenWAM"]:::xpolicy
        NATIVE["<b>Native adapters</b><br/>MolmoAct2 · ABC"]:::native
        XPOLICY ~~~ NATIVE
    end

    PLAN["<b>ADAPT & SCHEDULE</b><br/>Async · RTC · PAINT<br/>Serial · adaptive<br/><br/>Adapter → Timeline"]:::handoff
    ACT["<b>EXECUTE</b><br/>Direct · Smooth · MPC<br/><br/>Executor + Safety<br/>Control profile"]:::stage
    ROBOT(["<b>ROBOT</b><br/>RobotDriver<br/>Hardware"]):::robot
    TELEOP["<b>COLLECT</b><br/>YAM GUI<br/>LeaderPolicy"]:::collection
    REVIEW(["<b>REVIEW</b><br/>Robo GUI · records<br/>Human labels<br/>PRM-as-a-Judge"]):::side

    OBS --> THINK --> PLAN --> ACT --> ROBOT
    TELEOP --> ACT
    ACT -.-> REVIEW

    classDef stage fill:#F6F8FA,stroke:#8C959F,stroke-width:1px,color:#1F2328
    classDef handoff fill:#FFF4E5,stroke:#E36209,stroke-width:2.5px,color:#1F2328
    classDef side fill:#FFFFFF,stroke:#8C959F,stroke-dasharray:4 3,color:#57606A
    classDef robot fill:#1F2328,stroke:#1F2328,color:#FFFFFF
    classDef xpolicy fill:#8957E5,stroke:#6633B8,color:#FFFFFF
    classDef native fill:#2F6FEB,stroke:#1B4DB1,color:#FFFFFF
    classDef collection fill:#1A7F55,stroke:#125C3D,color:#FFFFFF
    style THINK fill:#FFFFFF,stroke:#8C959F,stroke-dasharray:5 4,color:#1F2328
```

Model servers never command hardware. Teleoperation bypasses chunk scheduling and reuses the
execution interfaces, while retaining its own collection GUI and recording format.

**GitHub:** [ManiMux](https://github.com/SII-LiuLab/manimux) · [XPolicyLab](https://github.com/Cuzyoung/XPolicyLab) · [PRM-as-a-Judge](https://github.com/YuyangLiu2003/PRM-as-a-Judge)

<a id="quick-start"></a>

## 🚀 Quick Start · Pi05 on YAM

This example runs **Pi05 pure-joint, step-30000, put-bottles with RTC**. It assumes the YAM and
OpenPI environments, checkpoint and local device configuration are already prepared;
see the [setup guide](docs/guideline.md#pi05-30k-on-yam). For a hardware-free first run, use the
[mock example](docs/guideline.md#hardware-free-start).

From the repository root, run these in **four separate terminals**. Reuse matching camera / Viewer
services if already running; collection and inference must not control the same robot simultaneously.

```bash
# Terminal 1: cameras
envs/yam/.venv/bin/manimux-camera-server --config configs/cameras.yaml

# Terminal 2: Viewer
envs/yam/.venv/bin/manimux-viewer --robot yam --host 127.0.0.1 --port 8086

# Terminal 3: pure-joint 30k model server
XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  scripts/servers/pi05_yam_server.py \
  --config configs/pi05/yam/server/put-bottles/joint-step30000.yaml

# Terminal 4: matching RTC runtime
envs/yam/.venv/bin/manimux serve \
  --config configs/pi05/yam/infra/put-bottles/rtc-joint-step30000.yaml
```

Open **http://127.0.0.1:8086**, then **Prepare → Start rollout → Finish & Home**.
Normal rollouts need no label; experiment rollouts require a human label before the next trial.
Keep the server and runtime configs paired: this example uses **joint**, not **joint+EE**.

## 📚 Guides

- **Run:** [Guideline](docs/guideline.md) · [Configuration](configs/README.md).
- **Integrate:** [Components and policy runbooks](docs/README.md) · [Inference methods](docs/README.md#inference-and-execution).
- **Collect / evaluate:** [YAM collection](docs/yam-collection.md) · [Experiment workflow](docs/experiment-infra.md) · [PRM guide](docs/prm-as-a-judge.md).
- **Extend:** [Architecture contracts](docs/architecture.md).

<a id="citation"></a>

## 📝 Citation

If ManiMux supports your experiments, please cite the repository. For experiments using its
XPolicyLab integration, please also cite the [XPolicyLab paper](https://arxiv.org/abs/2608.09892).

```bibtex
@misc{manimux2026,
  title = {{ManiMux}: A Composable Platform for Real-Robot Experiments},
  year = {2026},
  howpublished = {GitHub repository},
  url = {https://github.com/SII-LiuLab/manimux}
}

@article{community2026xpolicylab,
  title = {{XPolicyLab}: A Unified Standard and Open Ecosystem for Robot Policy Evaluation and Deployment},
  author = {{XPolicyLab Community} and Chen, Tianxing and Chen, Yue and Nian, Tian and others},
  journal = {arXiv preprint arXiv:2608.09892},
  year = {2026},
  doi = {10.48550/arXiv.2608.09892},
  url = {https://arxiv.org/abs/2608.09892}
}
```

---

Physical robots require matching model contracts and hardware safety measures; software checks do not certify safety or task success.
Upstream attribution: [notices](THIRD_PARTY_NOTICES.md) · [licenses](licenses).
