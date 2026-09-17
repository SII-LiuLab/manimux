# GR00T N1.7 + YAM 运行手册

本文只覆盖 `robocurve/gr00t-n1.7-yam-molmoact2`。它是基于 NVIDIA
[Isaac-GR00T N1.7](https://github.com/NVIDIA/Isaac-GR00T) 的 YAM 微调权重，不是
`nvidia/GR00T-N1.7-3B` base 直接零样本上 YAM。

当前状态：**XPolicy adapter、checkpoint 契约、模型环境、Cosmos、GPU forward、
XPolicy WebSocket、默认 ManiMux、真实三相机、双臂 YAM 和 Recorder 已完成闭环验证。当前
checkpoint 在已跑场景未完成任务，这是 policy 质量结果，不是 infra 断路。**

## 契约

```text
3 RGB cameras + 14D current joint state + instruction
  -> XPolicyLab GR00T_N17 adapter
  -> GR00T N1.7 YAM checkpoint + checkpoint statistics
  -> 16 x 14 absolute joint positions at 30 Hz
  -> ManiMux ActionTimeline -> SmoothExecutor -> YAM
```

| 项目 | 值 |
|---|---|
| 官方模型代码 | `XPolicyLab/policy/GR00T_N17/gr00t_n17/`（vendored Isaac-GR00T N1.7） |
| XPolicy adapter | `XPolicyLab/policy/GR00T_N17/model.py` |
| 权重 | `/home/ubuntu/manimux/checkpoints/finetuned/robocurve/gr00t-n1.7-yam-molmoact2` |
| 发布来源 | `robocurve/gr00t-n1.7-yam-molmoact2` |
| normalization | checkpoint 自带 `statistics.json` 的 `new_embodiment` |
| cameras | `base_view`、`left_wrist_view`、`right_wrist_view` |
| state/action | `left_arm[6] + left_gripper[1] + right_arm[6] + right_gripper[1]` |
| action | 16 步、14D、absolute joint position |
| 时间语义 | 训练数据 30 Hz，`action_dt_s = 1/30` |
| denoise | checkpoint `num_inference_timesteps = 4` |

权重目录中的 `processor_config.json` 和 `statistics.json` 是模型契约的一部分。不能把
Pi05 的 stats、XPolicy 原 ARX modality config 或 base 模型默认 embodiment 替换进来。

## 1. 离线检查

这一步不导入 torch、不加载权重到 GPU：

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/python scripts/servers/gr00t_yam_server.py \
  --config configs/groot/yam/server/finetune.yaml \
  --check
```

检查会确认：两片权重存在、3 相机键一致、state/action 都是 14D、action 是 absolute、
horizon 是 16、30 Hz 声明一致，并且 YAM 的 q01/q99 stats 维度完整。输出中的
`contract_status: ready` **只表示这些静态契约通过**；只有模型环境与 Cosmos 都准备好后，
`runtime_status` 才会变成 `environment_and_cosmos_present_gpu_forward_not_verified`。这仍然
只代表本地文件完整，真实推理完成前
`inference_status` 始终是 `not_verified`。

## 2. 安装模型环境

当前本机 `gr00t_n17/.venv` 已完成依赖安装。新的 checkout 使用同一安装入口：

```bash
cd /home/ubuntu/manimux/XPolicyLab/policy/GR00T_N17
bash install.sh
```

GR00T N1.7 的 processor 还需要 gated `nvidia/Cosmos-Reason2-2B`。先完成 Hugging Face
授权；若使用本地模型，把 `configs/groot/yam/server/finetune.yaml` 中
`cosmos_model_path` 改为本地目录。

安装结束后重新运行第 1 步检查；不要在 `runtime_status` 仍为 blocked/operator action
时启动模型服务。

ManiMux 的 WebSocket worker 还需要通用 XPolicy 依赖。新环境按
[`xpolicylab-runbook.md`](xpolicylab-runbook.md) 安装 `.[xpolicylab]`；现有
`envs/yam/.venv` 已满足时无需重复安装。

## 3. 启动模型服务

Terminal 1：

```bash
cd /home/ubuntu/manimux
XPolicyLab/policy/GR00T_N17/gr00t_n17/.venv/bin/python \
  scripts/servers/gr00t_yam_server.py \
  --config configs/groot/yam/server/finetune.yaml
```

Terminal 2：先运行不接相机、不接 CAN 的单次 forward probe。它发送三张确定性的合成 RGB
图和配置里的 14D YAM 起始状态，并通过正式 XPolicy WebSocket 与 ManiMux adapter 检查
`16 x 14` absolute joint chunk：

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/python scripts/validation/xpolicylab_yam_forward_probe.py \
  --config configs/groot/yam/infra/manimux.yaml
```

只有输出 `"status": "ok"`、`"action_space": "joint_position"`、
`"canonical_shape": [16, 14]` 且动作全部有限，才算完成真实 GPU forward 和 WS 往返。仅看到端口
监听不算模型可用。

首次请求可能包含 CUDA/kernel/cache 冷启动，不能只用第一次延迟判断稳态。保持模型服务运行，
连续执行三次：

```bash
cd /home/ubuntu/manimux

for i in 1 2 3; do
  echo "===== Probe $i ====="
  envs/yam/.venv/bin/python scripts/validation/xpolicylab_yam_forward_probe.py \
    --config configs/groot/yam/infra/manimux.yaml
done
```

通过标准：三次都返回有限的 `native_shape` / `canonical_shape = [16, 14]`，且稳态
`round_trip_ms` 明显低于一个 16-step / 30 Hz chunk 的 `533.3 ms` 时长。

2026-08-20 实测：第一次冷启动请求 `632.2 ms`；随后三次为 `101.3 / 91.7 / 89.4 ms`。
因此 GPU、模型、WS、adapter 和稳态 latency gate 已通过；这仍不等于 ManiMux runtime、
Recorder 或真机闭环已经验证。

## 4. 相机、Preflight 与真机

Terminal 2 启动共享相机服务；已有 `5555` 服务时不要重复启动：

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/manimux-camera-server --config configs/cameras.yaml
```

先检查两路 CAN：

```bash
for c in can_left can_right; do
  ip -details link show "$c" | grep -o 'ERROR-ACTIVE\|ERROR-PASSIVE\|BUS-OFF'
done
```

然后运行只读 preflight。这个脚本会连接真实双臂和三相机并做两次 GR00T 推理，但会显式
关闭 `move_to_start_on_connect`、`home_on_close`，也不会发送模型动作：

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/python scripts/validation/pi05_base_yam_preflight.py \
  --config configs/groot/yam/infra/manimux.yaml
```

确认 `contract_checks` 全为 `true`，并人工检查 measured state、first action、
`max_first_joint_delta_rad`、gripper range、camera shapes 和 steady latency。脚本名保留了早期
Pi05 命名，但其实现使用传入配置构建通用 YAM policy/adapter，此处不会加载 Pi05。

清空工作区、急停在手并确认 preflight 输出后，才启动默认 ManiMux：

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/manimux serve --config configs/groot/yam/infra/manimux.yaml
```

真机 runtime 连接后会用 `3.5 s` 移动到配置起始位；正常 `Ctrl-C` 退出时会用 `3.5 s`
回 Home。不要在机械臂运动中关闭模型或相机服务；异常时优先物理急停。

2026-08-20 的两个受控 episode 分别记录了 91/74 个接受 chunk，无 plan rejection 或
invalid action，默认 ManiMux、三相机、双臂下发和 Recorder 因此已验收。两次都没有完成
pick 任务，只能记为当前 checkpoint 的闭环任务失败，不能倒推为 infra 未运行。

当前只提供默认 ManiMux 配置，不提供 GR00T RTC 配置。虽然 Isaac-GR00T N1.7 模型内部有
自己的 overlap/frozen-step RTC 分支，但当前 XPolicy `GR00T_N17` adapter 没有实现统一的
`get_action_rtc()` 契约，不能把默认推理包装成 RTC。

## 未验证项

- 该 checkpoint 的任务成功率与跨任务泛化；
- GR00T sampler-level RTC。
