# 使用指南 / Guideline

[项目首页](../README.zh-CN.md) · [English overview](../README.md) · [完整文档](README.md)

首页介绍功能，本页集中放启动入口。所有命令从仓库根目录执行。
无硬件示例不需要 checkpoint；真机运行需要匹配的模型、设备配置和独立模型环境。

## Hardware-free start

项目支持 Python 3.11 / 3.12。安装开发环境并运行 mock：

```bash
git clone --recursive https://github.com/SII-LiuLab/manimux.git
cd manimux
uv sync --dev
uv run manimux run --config configs/mock.yaml
```

mock 使用模拟机器人、相机与 policy，运行 120 个控制 tick，记录写入 `data/`。
它不连接 CAN、实体相机或机器人。

独立 Viewer 演示：

```bash
uv run manimux-viewer --robot yam --demo --port 8086
```

打开 `http://127.0.0.1:8086`。这展示 Viewer 自带的演示数据，
不是前面 mock runtime 的实时画面。

数采 GUI 的无硬件预览：

```bash
uv sync --dev --extra collection
uv run python -m manimux.collection \
  --config configs/collection/yam/station.yaml --mock
```

打开 `http://127.0.0.1:8043`。硬件数采还需要 YAM / i2rt 与相机 SDK，
安装 `collection` extra 不会自动完成这些设备依赖的配置。

## Pi05 30k on YAM

以下使用已有的 `envs/yam/.venv` 和 OpenPI 环境，任务为
`Put bottles into the bin.`，checkpoint 为**纯 joint step-30000**。
环境和权重准备见 [Pi05 运行手册](pi05-yam-runbook.md)；
其他模型从[模型运行手册索引](README.md#policies-and-deployment)选择。

先检查配置中的本机 CAN 通道、相机序列号和权重路径是否属于当前设备。
不要同时让数采与推理控制同一个机器人；已有正确的相机或 Viewer 服务时可以复用。

分别在四个终端运行：

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

打开 `http://127.0.0.1:8086`，按 **Prepare → Start rollout → Finish & Home** 操作。
正常 rollout 不强制打分，实验 rollout 要求人工标注后才能进入下一条。
按钮、暂停和恢复的具体语义见 [Viewer 教程](viewer-tutorial.html)。

这套 RTC 的模型 horizon 为 50，动作点间隔为 `1/30 s`，`chunk_steps` 为 12。
12 是发起后续推理的执行门槛，不是把整个模型输出裁成 12 步；
等待后续推理完成期间仍会执行旧 chunk。

要用 **joint+EE 30k**，停止旧 server 和当前 runtime 后，配套替换两份配置：

- server：`configs/pi05/yam/server/put-bottles/joint-ee-step30000.yaml`；
- runtime：`configs/pi05/yam/infra/put-bottles/rtc-joint-ee-step30000.yaml`。

`policy backend identity mismatch` 通常说明当前端口上的模型与 runtime 预期不一致。
检查 task、checkpoint、训练配置与 norm stats；不要删除 `expected_backend` 来绕过保护。
模型 server 命令添加 `--check` 只核对路径和契约，不启动服务或机器人。

## YAM collection

使用已安装数采 extra 和硬件依赖的 YAM 环境：

```bash
envs/yam/.venv/bin/python -m manimux.collection \
  --config configs/collection/yam/station.yaml \
  --host 127.0.0.1 --port 8043
```

打开 `http://127.0.0.1:8043`，继续使用原 YAM 数采界面的选任务、预览、
Start Teleop 和录制流程。从臂底层改走 ManiMux，原 YAM-ABC-Reproduce 仓库不参与运行。
**Start Teleop 包含从臂对齐主臂的动作，不只是连接设备。**

默认同步 30 Hz，每次读取主臂后执行一次双臂目标；不额外启动 100 Hz 下发线程。
如需独立线程，选择 `configs/collection/yam/station-threaded.yaml`。
模式边界、相机配置、停止行为和保存格式见 [YAM 数采说明](yam-collection.md)。

## Configuration and outputs

- `configs/<model>/<embodiment>/server/<task>/`：checkpoint、归一化与模型服务。
- `configs/<model>/<embodiment>/infra/<task>/`：机器人、观测、推理调度、执行和记录。
- `configs/collection/<embodiment>/`：本体专用的数采 GUI、主臂和相机配置。
- `configs/robots/yam/common.yaml`：当前数采与两套 Pi05 放瓶子 RTC 30k 显式共用的控制参数。

公共配置对齐硬件参数、动作点间隔和手臂 / 夹爪运动限幅，
不强制数采与推理使用相同的滤波或执行频率。未引用 profile 的旧配置保持原行为。
修改配置不会热更新已连接的机器人；结束当前会话，再重启对应服务。
字段说明见 [配置参考](../configs/README.md)。

推理输出位于各配置的 `run.output_dir`，包含 session 与 rollout 记录。
数采默认输出为 `data/collection/episodes/<task>/<episode>/`。
模型目标、执行器命令和实际反馈并非同一种数据，比较时应使用匹配的字段与时间戳。

下一步：[实验设计](experiment-design.md) · [人工反馈与记录](experiment-infra.md)
· [离线视频评测](prm-as-a-judge.md)。
