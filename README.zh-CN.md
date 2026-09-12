<div align="center">

# ManiMux
### 面向真机实验的组合式推理与执行框架

**换模型、换推理算法，不必重搭一套机器人控制栈。**

[![Python 3.11 和 3.12](https://img.shields.io/badge/Python-3.11%20%7C%203.12-306998?style=flat-square)](pyproject.toml)
[![模型接入：XPolicyLab 与原生适配](https://img.shields.io/badge/Policies-XPolicyLab%20%2B%20Native-306998?style=flat-square)](docs/README.md#policies-and-deployment)
[![推理：ManiMux、RTC 和 PAINT](https://img.shields.io/badge/Inference-ManiMux%20%7C%20RTC%20%7C%20PAINT-306998?style=flat-square)](docs/README.md#inference-and-execution)
<br/>
[![真机：双 YAM](https://img.shields.io/badge/Hardware-Dual%20YAM-52616b?style=flat-square)](docs/pi05-yam-runbook.md)
[![数采：主臂策略与 GUI](https://img.shields.io/badge/Collection-Leader%20Policy%20%2B%20GUI-52616b?style=flat-square)](docs/yam-collection.md)
[![评测：人工反馈与离线 PRM](https://img.shields.io/badge/Evaluation-Human%20%2B%20Offline%20PRM-52616b?style=flat-square)](docs/prm-as-a-judge.md)

[English](README.md) · [简体中文](README.zh-CN.md)

[**Features**](#features) · [**演示**](#演示) · [**架构**](#架构) · [**使用指南**](docs/guideline.md) · [**文档**](docs/README.md)

</div>

> **从这里开始：**[使用指南](docs/guideline.md)集中说明环境安装、无硬件体验、YAM 数采，
> 以及配套的模型 server / runtime 命令。首页只讲项目定位、能力与架构。

## 项目定位

模型能输出动作，不代表真机就能按预期执行。推理延迟、观测过期、chunk 切换和控制器参数
都会影响执行结果；如果每种算法各自维护一套部署代码，实验也很难公平比较。

**ManiMux 将 Policy、chunk 调度、Executor 和本体控制解耦。**
通过显式配置组合实验，在统一 Viewer 中操作，并保留动作、反馈与视频，
让研究者既能替换推理方法，也能看清机器人实际执行了什么。

**Any Policy × Embodiment × Inference** 是组合式接口的设计目标，不是任意组合零适配的承诺。
当前真机开发以双 YAM 为主；新的模型与本体组合仍需匹配动作契约、适配器并完成验证。

## Features

- **组合式部署：**模型、推理策略、执行器和本体适配器可独立替换，复用公共运行时；
  模型特有的预处理与采样逻辑留在模型接入层。
- **可插拔推理算法：**通过相同执行接口比较异步 ManiMux、串行执行、RTC、PAINT、
  Temporal Ensembling 和自适应 chunking；各方法分别说明后端要求与验证状态。
- **Viewer 控制实验：**在同一界面 Prepare、Start、Pause、Finish，结合实时相机、
  3D 机器人和轨迹叠加观察执行过程。
- **实时 chunk 可视化：**呈现推理进度、已执行动作、延迟裁剪、chunk 交接、
  RTC condition 关系和夹爪闭合目标，让调度过程可见。
- **结构化人工反馈：**区分普通运行与实验模式；实验结束后记录任务结果、
  smoothness 评分与失败标签，再进入下一条实验。
- **可追溯的执行记录：**保存解析后的配置、带时间戳的观测、预测动作、执行器命令、
  机器人反馈、事件和视频，用于实验对比与失败分析。
- **主臂 Policy 数采：**保留熟悉的 YAM 数采 GUI 和 episode 格式，
  从臂命令统一走 ManiMux；默认同步 30 Hz 采样与下发。
- **采集与推理共享控制参数：**通过本体公共配置对齐硬件参数、动作点间隔及独立的
  手臂 / 夹爪运动限幅，同时允许不同的 executor、输出频率和滤波选择。
- **离线视频评测：**接入 PRM-as-a-Judge，结合人工标签分析任务过程；
  judge 在离线运行，不进入真机控制环。

## 演示

![ManiMux 真机 rollout：相机、机器人状态与实时 action-chunk 时间线](assets/manimux-viewer-demo.webp)

*一个界面，观察场景、检查 chunk 切换、控制实验。*

## 架构

![ManiMux 架构：模型推理和主臂数采复用执行器与机器人接口](assets/manimux-architecture.svg)

- **推理路径：**观测 → Policy / Adapter → Strategy / Timeline → Executor → RobotDriver。
  模型只生成动作，不直接控制硬件。
- **数采路径：**YAM GUI → 主臂 Policy → Collection Backend → Executor → RobotDriver。
  数采不启动模型 server，也不经过 chunk 推理调度器。
- **共用控制契约，而非强行合并流程：**显式引用同一本体 profile 的配置共享控制参数；
  推理保留 Viewer 和运行时 Recorder，数采保留自己的 GUI 与保存格式。

接口细节见[架构文档](docs/architecture.md)；
同步 / 独立线程的数采边界见 [YAM 数采说明](docs/yam-collection.md)。

## 继续阅读

- **开始运行：**[使用指南](docs/guideline.md) · [配置说明](configs/README.md)。
- **模型接入：**[运行手册索引](docs/README.md#policies-and-deployment)，包括 Pi05、
  MolmoAct2、ABC、GR00T、LingBot-VLA2、Xiaomi XR-1、OpenWAM 和 SAPolicy。
- **算法实现：**[推理与执行文档](docs/README.md#inference-and-execution)。
- **实验评测：**[Viewer 教程](docs/viewer-tutorial.html) · [实验流程](docs/experiment-infra.md)
  · [PRM-as-a-Judge](docs/prm-as-a-judge.md)。
- **扩展开发：**[架构与接口](docs/architecture.md) · [完整文档索引](docs/README.md)。

## 安全与来源

ManiMux 会控制物理机器人。软件测试通过、部署可运行，不等于硬件安全或任务成功。
请核对所选配置与模型契约，保留物理急停，并遵循对应本体运行手册。
夹爪目标变红不代表传感器确认已经抓住物体。

上游来源与许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [licenses/](licenses)。
