<div align="center">

# ManiMux
**统一控制底座，自由组合策略、推理与本体的真机实验平台。**

Policy × Runtime × Embodiment

[![Platform](https://img.shields.io/badge/Platform-7C3AED?style=flat-square)](#features)
[![Robo GUI](https://img.shields.io/badge/Robo%20GUI-0891B2?style=flat-square)](docs/viewer-tutorial.html)
<br/>
[![组件：XPolicyLab](https://img.shields.io/badge/Component-XPolicyLab-4F46E5?style=flat-square&logo=github&logoColor=white)](XPolicyLab/)
[![组件：PRM-as-a-Judge](https://img.shields.io/badge/Component-PRM--as--a--Judge-9333EA?style=flat-square&logo=github&logoColor=white)](PRM-as-a-Judge/)
[![Python 3.11 和 3.12](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?style=flat-square&logo=python&logoColor=white)](pyproject.toml)
<br/>
[![Policy：10 个接入，含 2 个仅模型路径](https://img.shields.io/badge/Policies-10%20Integrations-2EA043?style=flat-square)](docs/README.md#support-counts)
[![本体：1 个真机与 1 个仿真接入](https://img.shields.io/badge/Embodiments-1%20Real%20%2B%201%20Sim-2563EB?style=flat-square)](docs/README.md#support-counts)
[![推理：8 种模式](https://img.shields.io/badge/Inference-8%20Modes-F97316?style=flat-square)](docs/README.md#support-counts)
<br/>
[![数采：Teleop、UMI、DAgger；实现状态见路线图](https://img.shields.io/badge/Collection-Teleop%20%7C%20UMI%20%7C%20DAgger-D97706?style=flat-square)](docs/README.md#collection-status)
[![评测：人工反馈与 LLM Judge](https://img.shields.io/badge/Evaluation-Human%20%2B%20LLM%20Judge-DB2777?style=flat-square)](docs/prm-as-a-judge.md)

[English](README.md) · [简体中文](README.zh-CN.md)

[**Features**](#features) · [**视频**](#demo) · [**架构**](#architecture) · [**快速启动**](#quick-start) · [**文档**](docs/README.md) · [**引用**](#citation)

</div>

**ManiMux 将数采、策略部署与评测放在同一套真机控制底座上。**
换模型、换算法、换本体，不必从头搭建部署流程：通过配置组合 **Policy、Runtime 策略、
Executor 与本体**。标准接口分离模型推理与硬件控制，让接入能力跨本体复用，
而不是为每个“模型 × 机器人”单独维护一套代码。

**用 Robo GUI 管理实验，看清每一步执行。** 从准备、启动 rollout，到实时相机、3D 状态、
轨迹与 chunk 切换，再到回看模型预测、下发命令和机器人反馈，形成统一的实验流程。
**XPolicyLab** 负责模型接入，人工标注与 **PRM-as-a-Judge** 支持实验记录的评测。

**怎么采，就按同样的控制语义去执行。** 主从臂数采复用 ManiMux 的硬件接口与公共控制配置，
对齐动作时间、关节/夹爪约定和运动限幅，从控制层减少训推差异。
推理端需要的 smooth 等处理仍可显式选择，不再藏在另一套部署代码里。

> 📖 安装与环境准备见[使用指南](docs/guideline.md)，也可直接查看下方 [Pi05 全链路示例](#quick-start)；模型、算法与接口细节见[文档索引](docs/README.md)。

## News

- **[2026-09-13] Initial 版本正在开发。** 正在完善可组合的策略部署、主从臂数采与 GUI 实验管理，共用统一的真机控制底座。

<a id="features"></a>

## ✨ Features

| 功能 | 状态 | 提供什么 |
|---|:---:|---|
| 组合式部署 | ✅ | 配置驱动 Policy × Runtime 策略 × Executor × 本体 |
| 跨本体接口 | ✅ | 统一契约；已接入 YAM 真机与 ManiUniCon 仿真 |
| 可插拔推理 | ✅ | 异步、串行、RTC、PAINT 与自适应 chunking |
| Robo GUI | ✅ | 实验控制、相机、3D 状态、轨迹和 chunk 时间线 |
| Teleop 数采 | ✅ | 主臂 Policy + YAM GUI，从臂统一走 ManiMux |
| 采集—部署一致性 | ✅ | 共享硬件接口、动作时间与手臂 / 夹爪限幅 |
| 执行记录 | ✅ | 配置、观测、预测动作、下发命令、反馈、事件和视频 |
| 实验评测 | ✅ | 人工标注 + 离线 PRM / LLM Judge |
| UMI / DAgger 数采 | — | [采集路线图](docs/README.md#collection-status) |

✅ 表示已有实现，不代表所有模型 / 本体组合均已验证。
[接入数量](docs/README.md#support-counts)也包含仅模型和仿真路径。

<a id="demo"></a>

## 🎬 演示视频

双臂 YAM 真机 rollout：实时相机、3D 机器人状态与 action-chunk 切换。

![ManiMux 真机 rollout：实时相机、机器人状态与 action-chunk 可视化](assets/manimux-viewer-demo.webp)

[**▶ 打开 MP4 录屏**](assets/manimux_2026-09-05_23-17-35-00.00.03.144-00.00.34.914-seg1-00.00.02.596-00.00.34.966.mp4)

<a id="architecture"></a>

## 🧩 架构

**模型推理与硬件执行各司其职。** 推理策略决定何时生成、如何接续 chunk；
Executor 将目标动作变成机器人命令。

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 180, "curve": "basis", "nodeSpacing": 24, "rankSpacing": 28, "padding": 14}}}%%
flowchart LR
    OBS["<b>OBSERVE</b><br/>Cameras · robot state<br/>Build policy inputs"]:::stage

    subgraph THINK["<b>PREDICT</b>"]
        direction TB
        XPOLICY["<b>XPolicyLab</b><br/>Pi05 · XR-1 · GR00T<br/>LingBot · OpenWAM"]:::xpolicy
        NATIVE["<b>Legacy native</b><br/>MolmoAct2 · ABC"]:::native
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

模型 server 不直接控制硬件。数采绕过 chunk 推理调度、复用执行接口，
同时保留自己的采集 GUI 与保存格式。
新模型必须走 [XPolicyLab 统一接入路径](AGENTS.md#model-integration-xpolicylab-only)；
图中的 native 仅为迁移前保留的兼容入口。

**GitHub：**[ManiMux](https://github.com/SII-LiuLab/manimux) · [XPolicyLab](https://github.com/Cuzyoung/XPolicyLab) · [PRM-as-a-Judge](https://github.com/YuyangLiu2003/PRM-as-a-Judge)

<a id="quick-start"></a>

## 🚀 快速启动 · Pi05 on YAM

以下示例使用 **Pi05 纯 joint、step-30000、放瓶子任务与 RTC**。
需要先准备好 YAM / OpenPI 环境、checkpoint 和本机设备配置，见[环境指南](docs/guideline.md#pi05-30k-on-yam)。
没有硬件可先运行 [mock 示例](docs/guideline.md#hardware-free-start)。

从仓库根目录，在**四个独立终端**运行。已有匹配的相机或 Viewer 服务时可复用；
数采与推理不要同时控制同一个机器人。

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

打开 **http://127.0.0.1:8086**，按 **Prepare → Start rollout → Finish & Home** 操作。
正常 rollout 不强制打分，实验 rollout 需人工标注后再进入下一条。
server 与 runtime 的配置必须配套：这里是 **joint**，不是 **joint+EE**。

## 📚 使用指南

- **开始运行：**[完整指南](docs/guideline.md) · [配置说明](configs/README.md)。
- **模型与算法：**[组件和模型手册](docs/README.md) · [推理方法](docs/README.md#inference-and-execution)。
- **采集与评测：**[YAM 数采](docs/yam-collection.md) · [实验流程](docs/experiment-infra.md) · [PRM 评测](docs/prm-as-a-judge.md)。
- **扩展开发：**[架构与接口](docs/architecture.md)。

<a id="citation"></a>

## 📝 引用

如果 ManiMux 帮助了你的实验，欢迎引用本仓库。
使用 XPolicyLab 接入模型时，也请引用 [XPolicyLab 论文](https://arxiv.org/abs/2608.09892)。

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

真机需要匹配的模型契约与硬件安全措施；软件检查不等于安全认证或任务成功。
上游来源：[声明](THIRD_PARTY_NOTICES.md) · [许可](licenses)。
