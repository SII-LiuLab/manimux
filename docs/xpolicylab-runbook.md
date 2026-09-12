# XPolicyLab 集成手册

本文只说明 ManiMux 如何集成 XPolicyLab，不混放具体模型的部署命令。模型运行请看各自
runbook：

- [Pi05 + YAM](pi05-yam-runbook.md)
- [GR00T N1.7 + YAM](gr00t-yam-runbook.md)
- [Xiaomi XR-1 + YAM](xiaomi-xr1-yam-runbook.md)
- [LingBot-VLA2 + YAM](lingbot-vla2-yam-runbook.md)

配置统一按 `configs/<model>/<embodiment>/` 组织，具体命名规则见
[`configs/README.md`](../configs/README.md)。`configs/xpolicylab/yam/infra/smoke.yaml` 只用于
通用 WebSocket bridge 冒烟，不代表具体模型实验。

## 源码关系

`XPolicyLab/` 是指向 `Cuzyoung/XPolicyLab` fork 的 Git submodule。ManiMux 固定并记录
fork commit；fork 再从官方 `upstream` 获取更新。模型仍在各自独立环境和进程中运行，
ManiMux runtime 不 import torch/JAX 等模型依赖。

`--recursive` 也会初始化 XPolicyLab 内固定版本的第三方源码，例如官方 LingBot-VLA2 和
Cosmos3 使用的 `NVIDIA/cosmos-framework`；
不需要用户再手动 clone 到任意系统目录。

```bash
git clone --recursive https://github.com/SII-LiuLab/manimux.git
cd manimux
git submodule status
```

已有 checkout：

```bash
git submodule update --init --recursive
```

## 数据流

```text
YAM state + 三路 RGB + instruction
        -> ManiMux xpolicylab_ws worker
        -> msgpack/WebSocket INFER
        -> XPolicyLab policy/<MODEL>/model.py
        -> 模型原生 sampler
        -> XPolicy 标准 action keys
        -> ManiMux model-specific adapter
        -> ActionChunk(joint_position)
        -> ManiMux shared runtime + default/RTC strategy
```

边界保持明确：

- XPolicy model adapter 负责模型输入、输出、norm stats 和模型原生 sampler；
- ManiMux policy adapter 负责机器人 group、动作语义以及必要的 FK/IK；
- ManiMux shared runtime 负责时间线、执行、记录和 Viewer；InferenceStrategy 负责普通
  chunk 或 RTC 的请求时机与条件；
- 相机、CAN 和机器人 driver 不进入模型仓库。

## YAM embodiment 声明

XPolicy 的 `pack_robot_state` 从主仓库 `env_cfg/` 读取机器人维度：

```text
env_cfg/yam_dual.yml
env_cfg/robot/_robot_info.json
env_cfg/sim/yam_real.yml
```

YAM 的标准顺序是：

```text
left arm 6 + left gripper 1 + right arm 6 + right gripper 1 = 14
```

`robot.group_dims`、`group_prefixes` 和 `gripper_dofs` 必须与这个顺序一致。

## ManiMux 侧依赖

ManiMux 侧只需要线协议依赖：

```bash
uv pip install --python envs/yam/.venv/bin/python -e ".[xpolicylab]"
```

每个模型在自己的环境中启动 XPolicy server。不要把模型依赖装进 `envs/yam`。

## 新模型接入清单

**新模型及模型复现统一接入 `XPolicyLab/policy/<POLICY>/`，不再新增独立 native 路径。**
先阅读仓库级 [AGENTS.md](../AGENTS.md#model-integration-xpolicylab-only) 和
[XPolicyLab 接入规范](../XPolicyLab/CONTRIBUTING.md)。模型源码、加载、预处理、归一化、
训练适配与 sampler 留在 XPolicyLab；ManiMux 只保留硬件/动作适配、配置、轻量启动入口与
公共 runtime。不能只在 XPolicyLab 加一个代理壳，仍把真正的模型实现放在 ManiMux native server。

已有 `molmoact_http`、`abc_http` 暂为兼容路径，目标同样是迁入 XPolicyLab。
先验证替代实现，再切换配置和移除旧入口；不把“有一个模型目录”当成迁移完成。

每个模型必须有独立 server config、infra config 和 runbook。需要确认：

| 项目 | 必须从哪里得到 |
|---|---|
| observation keys | 模型源码、processor 或训练配置 |
| camera 数量与顺序 | checkpoint/model card/训练配置 |
| state/action 维度 | embodiment 配置与 stats |
| joint / EE / delta / absolute | 模型输出 transform |
| action horizon 与频率 | 训练配置或 checkpoint metadata |
| norm stats | 对应 checkpoint 的训练资产 |
| RTC 支持 | 模型原生 sampler 的正式 conditioning hook |

不知道这些契约时必须标记阻塞，不能创建看似可运行的假配置。

## RTC 规则

RTC 不是 WebSocket worker 自动提供的能力。只有模型 adapter 实现 `get_action_rtc`，并且
conditioning 真正进入模型原生 denoise/sampling 过程时，才能写 RTC config。否则 server
必须明确返回“不支持 RTC”，不能静默退化为普通推理。XPolicy server 会在 `HELLO_ACK`
声明 `sampling_modes`；PolicyWorker 将该能力交给 ManiMux，RTC 缺失时会在机器人连接前失败。

## 离线自测

```bash
PYTHONPATH=/home/ubuntu/manimux:/home/ubuntu/manimux/XPolicyLab \
  envs/yam/.venv/bin/python -m pytest \
  XPolicyLab/tests/unit/test_ws_infer_sampling.py \
  tests/unit/test_xpolicylab_plugins.py
```

这些测试不启动模型服务、GPU、相机、CAN 或真机。具体 checkpoint 的检查命令放在对应
模型 runbook 中。
