# UMI Diffusion Policy 与 Tianji–TacCap

新入口将可复用本体、policy、本地工位和实验分开。仅迁移 Tianji–TacCap；
旧 Tianji/YAM 配置入口保留。Tianji Viewer 已接入新整机模型，见
[Viewer 配置与启动](viewer.md)；回零/拖动恢复尚未迁移。

## 配置职责

| 位置 | 内容 |
| --- | --- |
| `configs/embodiment/{arm,end_effector,sensor}/` | 组件实现、模型、控制和采集参数 |
| `configs/embodiment/robot/tianji_taccap.yaml` | 整机安装关系、左右组、显示资产 |
| `configs/policy/umi_dp/server.yaml` | 模型服务基础参数 |
| `configs/policy/umi_dp/adapter/tianji_taccap.yaml` | 观察映射、动作约定、动作时间 |
| `configs/experiments/runtime/tianji_taccap.yaml` | 原实验的调度、平滑、命令包络和运动限幅 |
| `configs/experiments/pass_ball/tianji_taccap_umi_dp.yaml` | 实验入口及配套模型/相机服务选择 |
| `configs/local/tianji_taccap.example.yaml` | 可复制的工位模板 |
| `.local/tianji_taccap.yaml` | 个人实际设备和服务绑定，不提交 Git |

`policy`、`execution`、`policy_server` 各可用 `config:` 引用一份基础 YAML，
实验中的同名字段覆盖基础值；字典逐项合并，列表整体替换。不递归继承。
这些引用以及 `robot.config` 都相对实验文件解析。旧 `control_profile` 的冲突规则不变。

local 中的 `robot.hardware` 绑定共享控制器，`robot.components` 按整机组件名绑定设备。
字段由组件决定：Tianji 用 IP/序列号，CAN 组件可以用 `channel`，串口组件可以用 `port`。
不需要的字段直接省略；一体化夹爪不需要伪造独立组件或填写 null。
此格式不依赖 Tianji，但 YAM 的具体驱动接入本轮未迁移，仍使用原配置。

`local` 不选择机械臂类型、安装关系、TCP 或执行开关。它只覆盖设备参数、服务地址与路径。
`paths.checkpoint`、`paths.output_dir` 相对 local 文件解析。可在实验中写 `local: 路径`；
CLI `--local` 优先，CLI 路径相对当前工作目录。未指定 local 时可以加载离线模型，
连接设备时仍需要实际绑定。

相机由独立 camera server 采集。实验将流名 `left_wrist/right_wrist` 映射到整机组件
`left_wrist_camera/right_wrist_camera`。camera server 和 runtime 读取同一份组件参数和
local 序列号；RGB、采集时间戳、历史帧顺序与原实验一致。夹爪序列号与相机序列号不同。

## 准备本地工位

从仓库根目录执行：

```bash
mkdir -p .local
cp -n configs/local/tianji_taccap.example.yaml .local/tianji_taccap.yaml
```

填写实际控制器 IP、左右夹爪与相机序列号，并在 `paths.checkpoint` 指定真实模型目录。
模板使用示例地址和占位序列号。当前开发工位的已知绑定已从旧配置迁入该私有文件；
不会自动探测或猜测物理左右位置、checkpoint 路径。

`services.policy.endpoint` 是 runtime 访问模型的地址；远程服务可另写 `bind_host: 0.0.0.0`。
`services.camera.endpoint/request_endpoint` 分别是 PUB/REP 客户端地址，服务端可另写
`bind_endpoint/bind_request_endpoint`。同机默认使用 127.0.0.1。

环境安装沿用 [原 Tianji runbook](umi_dp-tianji-runbook.md)。克隆仓库不会创建本地 venv；
TacCap 原生 SDK 仍需安装。硬件进程和模型进程使用各自的 Python 环境。

## 绑定 checkpoint

在安装了 UMI_DP/XPolicyLab 的模型环境执行：

```bash
envs/umi_dp/.venv/bin/python scripts/servers/umi_dp_tianji_server.py \
  --experiment configs/experiments/pass_ball/tianji_taccap_umi_dp.yaml \
  --local .local/tianji_taccap.yaml \
  --bind-runtime-config .local/pass_ball/run.yaml
```

这一步读取并核对真实 checkpoint 身份、horizon、动作时间和图像约定，不启动服务。
产生配对的 `run.yaml` 和 `run-server.yaml`，展开 policy/execution 引用并重定位整机路径，
保留绝对 local 引用。输出已存在时不会覆盖。改变 checkpoint 后需要重新绑定，不能
只修改路径绕过模型身份检查。仍支持原 `--config/--runtime-template/--checkpoint` 用法。

## 启动入口

以下命令用于正式连接，不是离线测试命令。硬件和 checkpoint 就绪后按服务角色分别运行。

模型环境启动已绑定的服务：

```bash
envs/umi_dp/.venv/bin/python scripts/servers/umi_dp_tianji_server.py \
  --config .local/pass_ball/run-server.yaml
```

设备所在电脑的 Tianji 环境启动 camera server，直接读取同一实验和 local：

```bash
envs/tianji/.venv/bin/manimux-camera-server \
  --experiment configs/experiments/pass_ball/tianji_taccap_umi_dp.yaml \
  --local .local/tianji_taccap.yaml
```

硬件 runtime 读取已绑定的实验：

```bash
envs/tianji/.venv/bin/python -m manimux run \
  --config .local/pass_ball/run.yaml \
  --local .local/tianji_taccap.yaml
```

`run` 会连接硬件并读取反馈；实验默认 `execute: false`、`end_effector_control: false`。
执行开关位于实验的 `robot.options`，local 不负责启用动作。Tianji 连接本身不回零；
首次实际执行命令时按已有逻辑使能。已有 camera server 的 `--config cameras.yaml` 用法保留。

## 保持的接口语义与验证范围

- 左右组 `left_arm/right_arm`，七个弧度关节和一个归一化夹爪开度。
- FK/IK 和模型目标均在各臂基座下；场景安装矩阵不进入控制计算。
- 保留官方 `ik + ik_nsp`、限位余量、J6/J7 干涉和原分支选择。
- 同进程复用整机运动学；action decode 子进程只加载同一装配文件的离线模型。
- 原实验 100 Hz 控制、30 Hz 动作点、首动作偏移、平滑和运动约束未变。

本次验证使用离线 libKine、假硬件和假的模型身份报告测试配置接线。
没有启动真实相机、机器人或模型服务，也没有验证实机任务成功率。
