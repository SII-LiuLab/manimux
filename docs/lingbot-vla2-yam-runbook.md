# LingBot-VLA2 + YAM 运行手册

## 结论

LingBot-VLA2 已公开完整源码和推理入口，ManiMux 现在也有正式的
`XPolicyLab/policy/LingBot_VLA2/` adapter，不再复用旧的 Qwen2.5
`LingBot_VLA`。

当前本地 `checkpoints/pretrained/lingbot-vla-v2-6b` 是 foundation checkpoint，
不是 YAM 后训练模型。官方部署文档要求 `path_to_posttraining_ckpt`，所以 foundation
zero-shot 不是官方承诺的使用方式。为了测它是否已经学到跨本体先验，本项目提供独立
`server/base.yaml`；执行仍然复用标准 `infra/manimux.yaml`，不会制造第三种 runtime，
也不能把结构检查通过写成任务能力通过。

## 上游基线

- 官方源码：<https://github.com/Robbyant/lingbot-vla-v2>
- 当前集成 revision：`187f84061ba312acab3bca05a6ee26a8d75968da`
- 官方 foundation weights：`robbyant/lingbot-vla-v2-6b`
- 官方 RoboTwin 后训练 weights：`robbyant/lingbot-vla-v2-6b-robotwin`
- 源码许可证：Apache-2.0

官方 V2 使用 55 维 canonical state/action：14 arm joint、14 EE pose、2
gripper、12 hand、4 waist、2 head、3 mobile base、4 reserved。真正送进模型的
有效维度由训练 config、robot config 和 norm stats 一起决定，不是把 YAM 的 14
维数据随便塞进 55 维零向量。

论文把 Joint 与 EEF 作为两种 action space 分开消融：平均成功率分别为 55.0% 和
56.0%，并指出不同任务偏好不同表示；论文没有声称“joint+EEF 同时监督”更优。官方
代码的 55D feature/mask 机制允许同一 robot config 同时打开两块，因此本项目增加
joint+EEF 联合监督作为扩展实验，而不是冒充论文默认 setting。

联合监督时，模型输入仍只有 joint + gripper 14D；绝对 EEF state 只作为计算相对
EEF target 的锚点，不进入模型 state。YAM action 有 28 个有效维度：relative joint
12D、local-frame relative EEF 14D（每臂 `XYZ + quaternion(xyzw)`）和 absolute gripper 2D。官方 `joint_mask`
让同一个 flow-matching L2 loss 只平均这些有效维度；并不存在第二个单独加权的
“pos loss”。部署仍选 joint-only robot config，只执行 joint+gripper，EEF 是训练辅助目标。

## ManiMux / XPolicy 分层

```text
YAM + cameras
      |
      v
ManiMux canonical observation
      |
      v
XPolicyLab WebSocket protocol
      |
      v
LingBot_VLA2 adapter
  - 3 cameras -> official camera feature names
  - left/right 6 joints -> arm.position[12]
  - left/right gripper -> effector.position[2]
      |
      v
official LingbotVLAv2Server
  - FeatureTransform
  - checkpoint-specific norm stats
  - official flow sampling
      |
      v
model-native arm action[12] + absolute gripper[2]
      |
      v
XPolicy per-step dictionaries
      |
      v
ManiMux PolicyAdapter
  - absolute checkpoint: direct decode
  - relative checkpoint: request-time joints + arm deltas
      |
      v
canonical absolute joint ActionChunk
```

上游源码负责模型、FeatureTransform、normalization 和 flow sampling；XPolicy
adapter 只负责协议与 YAM 字段映射；ManiMux 只负责任务生命周期、调度、执行和
独立安全边界。

## 当前文件

```text
finetune server: configs/lingbot-vla2/yam/server/finetune.yaml
base server:     configs/lingbot-vla2/yam/server/base.yaml
infra:   configs/lingbot-vla2/yam/infra/manimux.yaml
rtc:     configs/lingbot-vla2/yam/infra/rtc.yaml
adapter: XPolicyLab/policy/LingBot_VLA2/model.py
sampler: XPolicyLab/policy/LingBot_VLA2/rtc.py
server:  XPolicyLab/policy/LingBot_VLA2/setup_eval_policy_server.sh
profile: XPolicyLab/policy/LingBot_VLA2/robot_configs/yam_dual_absolute.yaml
source:  XPolicyLab/policy/LingBot_VLA2/lingbot_vla_v2/  # vendored upstream source
check:   scripts/validation/check_lingbot_vla2_yam.py
audit:   scripts/validation/lingbot_vla2_yam_audit.py
prepare: scripts/datasets/prepare_lingbot_vla2_base_assets.py
joint+EEF train: scripts/training/train_lingbot_vla2_yam_joint_ee_cluster.sh
joint+EEF profile: configs/lingbot-vla2/yam/robot_configs/yam_dual_joint_ee_relative.yaml
stats:   src/manimux/integrations/lingbot_vla2_yam/norm_stats/yam_60ep.json
```

服务端不存在第二套 native 入口。标准 launcher 调用
`XPolicyLab/setup_policy_server.py`，后者只按 `policy_name: LingBot_VLA2` 加载
`model.py`；ManiMux 仅通过统一 WebSocket bridge 使用它。

官方 V2 sampler 本身没有在线 RTC 参数，但公开了 prefix KV cache 和
`predict_velocity(x_t, t)`。XPolicy adapter 因此实现了独立 sampler-level RTC：
保持官方 10-step Euler flow loop，在每个 denoise step 对 clean action estimate 做
VJP soft-mask guidance。它没有修改已生成 chunk，也不做 chunk splice。

## 显式部署配置

要把 server check 变成 `ready`，必须同时提供：

1. 仓库已经固定的官方 `lingbot-vla-v2` source checkout；
2. 用 YAM 数据 post-train 后的 `hf_ckpt/`；
3. official loader 要求的 `lingbotvla_cli.yaml`；
4. 与这份 YAM 训练数据匹配的 `norm_stats.json`；
5. 与训练完全一致的 YAM robot config；
6. 训练使用的 native control frequency 和 action horizon。

与其他模型一致，这些信息直接写在 server config：

```yaml
lingbot_vla2_root: lingbot_vla_v2
model_root: /path/to/checkpoints/global_step_15000/hf_ckpt
training_config_path: /path/to/lingbotvla_cli.yaml
qwen3vl_path: /path/to/qwen3_vl_4b_processor
robot_config_path: /path/to/robot_config.yaml
norm_stats_path: /path/to/norm_stats.json
action_horizon: 50
native_hz: 30.0
```

`robot_config.yaml` 决定模型输出是 absolute joint，还是 arm-relative + absolute
gripper。XPolicy 保留模型原生输出语义，ManiMux 再选择对应的 `PolicyAdapter`；server
不会把 relative action 偷偷转换成 absolute action。

官方 source 已直接纳入
`XPolicyLab/policy/LingBot_VLA2/lingbot_vla_v2/`。基础版本为官方 revision
`951475ae1b1d87553e7dc47c97b53a3d695c0d13`，YAM 扩展由 XPolicyLab 本身跟踪；
LingBot 不再要求 nested submodule 或个人 fork。

官方 loader 固定从 `checkpoint_path.parent.parent.parent / "lingbotvla_cli.yaml"`
读取训练 config。因此 checkpoint 建议保持如下目录：

```text
checkpoints/pretrained/lingbot-vla-v2-yam/
├── lingbotvla_cli.yaml
├── norm_stats.json
├── robot_config.yaml
└── runs/yam/hf_ckpt/
    ├── model.safetensors.index.json
    └── model-*.safetensors
```

`hf_ckpt` 向上三级后必须正好得到含 `lingbotvla_cli.yaml` 的目录，这是官方 loader
的路径约束。server config 中的 `native_hz` 和 `action_horizon` 必须使用采集/训练的
真实值。
`lingbotvla_cli.yaml` 必须保留 `model.post_training: true`、
`model.config_key: LingbotVLAV2Config`、55 维 canonical 配置、三相机顺序，以及
与 server config 完全相同的 `train.chunk_size`。

## 离线检查

以下命令只检查文件和 YAML/JSON 契约，不导入官方 torch 模型，不启动服务：

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/python scripts/validation/check_lingbot_vla2_yam.py
```

检查器读取 server config 中的显式路径；缺少任何权重 shard、训练配置、robot
config 或 norm stats 都会返回 `status: blocked`，且不会加载 GPU 模型。

检查器同时读取 `configs/lingbot-vla2/yam/infra/manimux.yaml`，要求：

- `policy.action_dt_s == 1 / native_hz`；
- `policy.horizon_steps == action_horizon`；
- relative checkpoint 必须使用 `policy.adapter: lingbot_vla2_yam`；
- baseline `execution.runtime == manimux`。

所以训练产物与执行时序不一致时会在模型加载前失败，而不是在真机循环中静默
拉伸动作。

RTC 配置使用同一个检查入口：

```bash
envs/yam/.venv/bin/python scripts/validation/check_lingbot_vla2_yam.py \
  --infra-config configs/lingbot-vla2/yam/infra/rtc.yaml
```

除相同的 Hz/dt/horizon 契约外，它还验证 sampler capability、`beta > 0`、delay
buffer，以及 `delay <= min_execute_steps <= horizon - delay`。

审计现有 foundation checkpoint 的 55 维投影：

```bash
envs/yam/.venv/bin/python scripts/validation/lingbot_vla2_yam_audit.py
```

## Base 权重能力实验

### Norm stats

LingBot 不能复用 XR-1 的 stats。XR-1 是 `30 x 60` anchor-relative EE delta；LingBot
在本实验中使用 `12` 个 absolute arm joints 加 `2` 个归一化 gripper。仓库内的
`yam_60ep.json` 由 `60` 个完整 YAM episode、`25,743` 条 transition 计算，分别包含：

- `observation.state.arm.position[12]`；
- `observation.state.effector.position[2]`；
- `action.arm.position[12]`；
- `action.effector.position[2]`。

四组特征均使用官方 real-robot config 的 `meanstd`。这让 state 输入和 action 输出具有
正确的 YAM 单位，但它们**不是 foundation checkpoint 的配对 post-training stats**，
因此只用于 base 权重能力诊断。换数据集或正式 finetune 时必须重新统计：

```bash
cd /home/ubuntu/manimux
PYTHONPATH=src envs/yam/.venv/bin/python -m \
  manimux.integrations.lingbot_vla2_yam.compute_norm_stats \
  --episodes /path/to/yam/episodes \
  --out /path/to/norm_stats.json
```

### 准备 Base 资产

以下命令不会复制 6B 权重，只在 ignored checkpoint 目录中创建相对符号链接，并从
固定 revision 的官方 `real_robot.yaml` 生成 loader 必需的 `lingbotvla_cli.yaml`：

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/python scripts/datasets/prepare_lingbot_vla2_base_assets.py

envs/yam/.venv/bin/python scripts/validation/check_lingbot_vla2_yam.py \
  --config configs/lingbot-vla2/yam/server/base.yaml \
  --infra-config configs/lingbot-vla2/yam/infra/manimux.yaml
```

第二条必须输出 `status: ready`、
`checkpoint_variant: lingbot_vla2_6b_base_with_yam_stats`、
`inference_status: not_verified`。`ready` 只表示 source、55D architecture、权重 shards、
YAM feature mapping、stats shape、50-step horizon 和 30 Hz 部署假设彼此一致。

### 安装模型环境

LingBot 使用和其他模型一致的独立 `uv venv`，固定放在
`envs/lingbot-vla2/.venv`。它是普通 venv，不受根目录 `uv.lock` 管理；不要对它运行
`uv sync`。安装由操作者执行：

```bash
cd /home/ubuntu/manimux/XPolicyLab/policy/LingBot_VLA2
bash install.sh
```

脚本使用 Python 3.12、PyTorch 2.8 CUDA 12.8 和 FlashAttention 2.8.3，并安装官方
depth 依赖、LingBot-VLA2 与 XPolicyLab。安装完成后先做 GPU/WS forward，不要直接启动
真机：

```bash
# terminal 1: XPolicy foundation base server
cd /home/ubuntu/manimux
bash XPolicyLab/policy/LingBot_VLA2/setup_eval_policy_server.sh \
  configs/lingbot-vla2/yam/server/base.yaml

# terminal 2: no-CAN forward probe
envs/yam/.venv/bin/python scripts/validation/xpolicylab_yam_forward_probe.py \
  --config configs/lingbot-vla2/yam/infra/manimux.yaml
```

只有 probe 返回有限的 `native_shape: [50, 14]` 和
`canonical_shape: [50, 14]`，才进入相机和双臂 YAM 实验：

2026-08-20 已在 RTX 4090 完成该检查：Python `3.12.13`、PyTorch
`2.8.0+cu128`、Transformers `4.57.3`、FlashAttention `2.8.3`；官方 6 个 shard
成功加载。第二次独立服务进程的首个 10-step denoise 为 `1.253 s`、WS 往返
`1.392 s`，随后两次稳态 denoise 为 `365 / 369 ms`、WS 往返 `392.3 / 396.1 ms`。
50 步在 30 Hz 下覆盖 `1.667 s`，所以稳态延迟预算足以持续供给默认 ManiMux。
三次均返回 `native_shape: [50, 14]`、`canonical_shape: [50, 14]` 且全部有限。该证据
确认 GPU 模型、XPolicy WS、YAM 字段映射和 action decode 已连通。随后默认 ManiMux
真机闭环也已完成；base 权重是否能完成任务仍单独判断。

## 真机运行：Base + ManiMux

首次只运行默认 ManiMux，不使用 RTC。清空双臂工作区、准备急停，然后按顺序打开四个
终端；不要同时启动第二个 LingBot server 或 RTC runtime。

### Terminal 1：LingBot 模型服务

```bash
cd /home/ubuntu/manimux
bash XPolicyLab/policy/LingBot_VLA2/setup_eval_policy_server.sh \
  configs/lingbot-vla2/yam/server/base.yaml
```

看到 `Model initialized ...` 后等待服务监听 `127.0.0.1:8501`。如需再次确认模型输出，
在另一个终端运行无 CAN probe：

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/python scripts/validation/xpolicylab_yam_forward_probe.py \
  --config configs/lingbot-vla2/yam/infra/manimux.yaml
```

### Terminal 2：三相机服务

已有 `5555` 相机服务时不要重复启动：

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/manimux-camera-server --config configs/cameras.yaml
```

确认三台 RealSense 均已打开，并看到 `REP bound` 与 `PUB bound`。

### Terminal 3：Viewer

```bash
cd /home/ubuntu/manimux
.venv/bin/manimux-viewer --robot yam --host 0.0.0.0 --port 8086
```

浏览器打开 `http://localhost:8086`。

### Terminal 4：CAN 检查与 ManiMux

先确认两路 CAN 都是 `ERROR-ACTIVE`：

```bash
for c in can_left can_right; do
  printf '%s: ' "$c"
  ip -details link show "$c" | grep -o 'ERROR-ACTIVE\|ERROR-PASSIVE\|BUS-OFF'
done
```

只有两路均为 `ERROR-ACTIVE` 才运行：

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/manimux run \
  --config configs/lingbot-vla2/yam/infra/manimux.yaml
```

连接后机械臂按配置用 `3.5 s` 移到起始姿态，结束时用 `3.5 s` 回 Home。正常停止时只在
runtime 终端按一次 `Ctrl-C`，等待回零和 Recorder 收尾，再依次停止相机、模型服务和
Viewer。新 rollout 保存在 config 指定的 `data/experiments/.../session-*/rollout-*`；未完整收尾的
记录带 `.partial` 后缀。2026-08-24 之前的探索记录统一归档在
`data/archive/pre-campaign-20260824/root-runs/`。

30 Hz 是 YAM 对照实验假设，不是从 foundation checkpoint 恢复出的训练频率。任务失败
首先记录为 base policy 结果，不能据此判定 adapter 或 ManiMux 断路。

### 已完成的真机记录

2026-08-20 的 `run-20260820T122849Z-b474d665` 使用
`LingBot-VLA2 YAM via XPolicyLab`、默认 `manimux` runtime、三路真实相机和双臂 YAM：

- 执行 `1600` 个 control tick，持续 `61.9 s`；
- 接受 `42` 个 `50 x 14` action chunk；
- 记录 `42` 个 plan boundary，无 plan rejection；
- runtime 正常结束并完成 Recorder 收尾。

因此默认 LingBot GPU -> XPolicy WS -> ManiMux -> 相机/CAN -> 双臂 YAM -> Recorder
链路记为已通过。记录里的 runtime success 只代表 rollout 生命周期完整结束，不代表
完成了 pick 任务，也不改变 foundation checkpoint 缺少 YAM post-training 的事实。

## Finetune 权重就绪后的命令

在 server config 中填写 checkpoint、training config、robot config 和 norm stats 的
绝对路径，并让 ManiMux infra 的 dt、horizon 与训练设置一致。无需修改 adapter 或
server 代码。
只有检查显示 `ready` 后，才允许按顺序启动：

```bash
# terminal 1: model server
cd /home/ubuntu/manimux
bash XPolicyLab/policy/LingBot_VLA2/setup_eval_policy_server.sh \
  configs/lingbot-vla2/yam/server/finetune.yaml

# terminal 2: cameras
# 使用现场已经验证过的 camera server 命令。

# terminal 3: ManiMux
envs/yam/.venv/bin/manimux run \
  --config configs/lingbot-vla2/yam/infra/manimux.yaml
```

这些 finetune 命令当前仍没有 GPU forward、server handshake、相机、CAN 或真机证据。

## RTC 实现与边界

完整路径如下：

```text
ManiMux RtcRuntime
  -> raw absolute joint condition [H, 14] + soft mask [H]
  -> XPolicy get_action_rtc
  -> arm[12] + gripper[2]
  -> checkpoint FeatureTransform normalization
  -> valid-action mask + padding to canonical [H, 55]
  -> every LingBot denoise step:
       clean = x_t - t * velocity(x_t, t)
       guidance = VJP(clean, (condition - clean) * weights)
       guided_velocity = velocity - scale(t, beta) * guidance
  -> official unnormalize/unpad
  -> absolute YAM joint chunk [H, 14]
```

纯 CPU 离线测试已覆盖 guidance 方向、zero-mask 等价于 native velocity、mask
范围、YAM 14D 拆分、55D padding/mask，以及 guidance 确实在 flow loop 内逐步
执行。

当前 base 资产已完成 GPU、官方模型 forward 和 XPolicy WS 验证；缺少的是匹配 YAM
post-training 的 checkpoint/stats，以及 RTC 真机实测。`rtc.yaml` 的
`initial_delay_steps: 12` 和 `min_execute_steps: 20` 当前只满足静态约束，不代表已完成
RTC 参数标定。因此默认先使用 `infra/manimux.yaml`，拿到 finetune 权重后再单独验证
RTC。
