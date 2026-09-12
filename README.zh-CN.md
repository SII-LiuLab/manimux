<div align="center">

# ManiMux
**从数采、推理到评测的组合式真机实验平台。**

Any Policy × Embodiment × Inference

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

[**Features**](#features) · [**演示**](#演示) · [**架构**](#架构) · [**使用指南**](docs/guideline.md) · [**文档**](docs/README.md)

</div>

**ManiMux 是一个覆盖数采、策略部署与评测的真机实验平台。**
它将 Policy、推理策略、Executor 与本体解耦，让研究者在同一套控制栈上比较不同方法，
通过 GUI 管理实验，并查看机器人实际执行了什么。

> 安装与启动命令见[使用指南](docs/guideline.md)；模型、算法与接口细节见[文档索引](docs/README.md)。

## Features

| 功能 | 状态 | 提供什么 |
|---|:---:|---|
| 组合式部署 | ✅ | Policy × 推理策略 × Executor × 本体 |
| 可插拔推理 | ✅ | 异步、串行、RTC、PAINT 与自适应 chunking |
| Robo GUI | ✅ | 实验控制、相机、3D 状态、轨迹和 chunk 时间线 |
| Teleop 数采 | ✅ | 主臂 Policy + YAM GUI，从臂统一走 ManiMux |
| 公共控制配置 | ✅ | 共享硬件参数、动作间隔与手臂 / 夹爪限幅 |
| 执行记录 | ✅ | 配置、观测、预测动作、下发命令、反馈、事件和视频 |
| 实验评测 | ✅ | 人工标注 + 离线 PRM / LLM Judge |
| UMI / DAgger 数采 | — | [采集路线图](docs/README.md#collection-status) |

✅ 表示已有实现，不代表所有模型 / 本体组合均已验证。
[接入数量](docs/README.md#support-counts)也包含仅模型和仿真路径。

## 演示

![ManiMux 真机 rollout：实时相机、机器人状态与 action-chunk 可视化](assets/manimux-viewer-demo.webp)

## 架构

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

数采复用执行接口，但不经过 chunk 推理调度，保留自己的 GUI 与保存格式。

**GitHub：**[ManiMux](https://github.com/SII-LiuLab/manimux) · [XPolicyLab](https://github.com/Cuzyoung/XPolicyLab) · [PRM-as-a-Judge](https://github.com/YuyangLiu2003/PRM-as-a-Judge)

## 使用指南

- **开始运行：**[完整指南](docs/guideline.md) · [配置说明](configs/README.md)。
- **模型与算法：**[组件和模型手册](docs/README.md) · [推理方法](docs/README.md#inference-and-execution)。
- **采集与评测：**[YAM 数采](docs/yam-collection.md) · [实验流程](docs/experiment-infra.md) · [PRM 评测](docs/prm-as-a-judge.md)。
- **扩展开发：**[架构与接口](docs/architecture.md)。

真机需要匹配的模型契约与硬件安全措施；软件检查不等于安全认证或任务成功。
上游来源：[声明](THIRD_PARTY_NOTICES.md) · [许可](licenses)。
