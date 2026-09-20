# Xiaomi Robotics 1 / Tianji–TacCap 传球接入

本入口使用 `XPolicyLab/policy/Xiaomi_Robotics_1` 加载本地 step-50000
DeepSpeed 模型文件，并通过共享 `xpolicylab_ws` worker 服务 ManiMux。模型只接收两路
真实腕部 RGB；训练时的第三路 ego view 是纯黑，因此由 XPolicyLab 在预处理阶段生成
全零 `uint8 RGB`，不注册虚假相机，也不复用任意现场画面。

## 固定契约

- checkpoint：`mp_rank_00_model_states.pt`；8 个 optimizer state 不参与推理。
- normalization：同目录 `training_metadata/normalize.json`，形状分别为
  `mean/std=(30,60)`、`q01/q99=(1,60)`。
- observation：Tianji 左/右各 7 个关节与 1 个 `[0,1]` TacCap 开合量；两路腕部 RGB；
  第三路按左腕分辨率生成纯黑图，然后进入 XR-1 的公共 resize。
- action：30×60 anchor-relative EE delta。每一行都相对于发起请求时的 TCP pose，
  不是逐行累加；adapter 再用 Tianji 装配后的 TCP FK/IK 转为两组 30×8 joint position。
- TacCap 保持 checkpoint 的连续 `[0,1]` 开合量，不启用 close-latch 语义替换。
- `[16:20]` 的腰部/底盘槽位在 Tianji 上无对应执行器，明确丢弃并记录最大绝对值。
- 第一版只启用普通 ManiMux single-inflight；Tianji 的 XR-1 RTC condition codec 尚未实现。

## 文件

- 模型服务：`manimux/configs/policy/xiaomi-xr1/tianji/pass_ball/step50000.yaml`
- 安全关闭执行的实验模板：
  `manimux/configs/experiments/pass_ball/tianji_taccap_xiaomi_xr1_step50000.yaml`
- server launcher：`manimux/servers/xiaomi_xr1_tianji_server.py`

模板保持 `robot.options.execute: false`、Viewer 关闭。加载配置、检查 checkpoint 或启动
服务都不会令机器人运动。

## 1. 安装隔离的模型环境

模型依赖不能装进 Tianji 硬件 runtime 环境：

```bash
cd /path/to/manimux/XPolicyLab/policy/Xiaomi_Robotics_1
MIBOT_CONDA_ENV=mibot bash install.sh
```

Qwen processor 默认使用 `Qwen/Qwen3-VL-4B-Instruct`，首次启动需要 HuggingFace
可达或已有缓存。离线机器可通过 `--processor /absolute/local/processor` 指定本地目录。

## 2. 只检查 artifact 与配置

```bash
cd /path/to/manimux
conda run -n mibot --no-capture-output python \
  manimux/servers/xiaomi_xr1_tianji_server.py \
  --experiment manimux/configs/experiments/pass_ball/tianji_taccap_xiaomi_xr1_step50000.yaml \
  --local .local/tianji_taccap.yaml \
  --checkpoint /path/to/posttrain_pass_ball_50k_8gpu_epoch0_step50000_checkpoint \
  --check
```

传入 checkpoint 目录时 launcher 会唯一解析
`mp_rank_00_model_states.pt`，并自动使用相邻的
`training_metadata/normalize.json`。`--check` 只验证文件、shape、协议和黑图契约，
不加载 5B 模型。

## 3. 启动模型服务

```bash
cd /path/to/manimux
conda run -n mibot --no-capture-output python \
  manimux/servers/xiaomi_xr1_tianji_server.py \
  --experiment manimux/configs/experiments/pass_ball/tianji_taccap_xiaomi_xr1_step50000.yaml \
  --local .local/tianji_taccap.yaml \
  --checkpoint /path/to/posttrain_pass_ball_50k_8gpu_epoch0_step50000_checkpoint
```

服务启动后会发布 checkpoint、normalization、`packed_ee_delta`、black ego profile 和
action semantics 身份；ManiMux 在开始 rollout 前逐项核对，避免连到 UMI-DP 或错误 XR-1
checkpoint。

## 4. 相机与安全 dry run

另一个终端启动现有 TacCap camera server，只打开两路腕部相机：

```bash
cd /path/to/manimux
conda run -n xense-taccap --no-capture-output python \
  -m manimux.servers.camera.server \
  --experiment manimux/configs/experiments/pass_ball/tianji_taccap_xiaomi_xr1_step50000.yaml \
  --local .local/tianji_taccap.yaml
```

保持模板中的 `execute: false`，先运行 runtime 观察握手、推理延迟、IK 与保存记录：

```bash
conda run -n xense-taccap --no-capture-output python -m manimux run \
  --config manimux/configs/experiments/pass_ball/tianji_taccap_xiaomi_xr1_step50000.yaml \
  --local .local/tianji_taccap.yaml
```

实机执行必须在 dry run、起始姿态、两腕相机方向、TCP、夹爪 `[0,1]` 约定以及完整
30-step IK 都通过后，复制模板到 Git 忽略的 `.local/`，人工复核并只在该副本中设置
`execute: true`。本次接入没有启动硬件，也没有宣称真实传球成功。
