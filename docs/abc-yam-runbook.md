# ABC + YAM 运行手册

## 已下载的官方 bottles 75k（2026-09-10 验证）

本机已有两份 ABC 权重：`abc_dit_xl_200k_model.pt` 和
`official_bottles_75k/bottles_75k.pt`，都在 `checkpoints/pretrained/abc/`。
下面这组命令明确选择 **bottles 75k**。ABC 的 HTTP worker 和 YAM adapter
已经注册，无需接入 XPolicyLab，也不经过 EEF→IK。

```bash
cd /home/ubuntu/manimux
envs/abc/.venv/bin/manimux-abc-server \
  --host 127.0.0.1 --port 8300 \
  --checkpoint checkpoints/pretrained/abc/official_bottles_75k/bottles_75k.pt \
  --norm-stats-path checkpoints/pretrained/abc/official_bottles_75k/norm_stats.json \
  --device cuda:0 --diffusion-steps 10 \
  --prompt 'put the plastic bottles in the bin'
```

实际下载的文件包含 `model`、`step=75000`、`norm_stats`。服务优先使用嵌入的
统计量；已经核验 state/actions 的 mean/std 与相邻 JSON 逐值一致。
上游 `prepare.py` 对应下载项是 `checkpoints/bottles_release_prep_75k.pt`。

复用已运行的相机和 Viewer，在结束其他 policy 的 runtime 后启动 ABC runtime：

若相机服务尚未运行，使用 640×480、与 teleop 一致的 RGB-only 配置启动：

```bash
envs/yam/.venv/bin/manimux-camera-server --config configs/abc/yam/cameras.yaml
```

旧通用 `configs/cameras.yaml` 省略尺寸，默认采集 640×360；这与当前成功 SA
路径的输入视野不同。ABC 服务保留自己的 letterbox 预处理，不使用 SA 的裁剪。
本次准备时，左 D405 在同时启用 RGB/深度时持续等待超时，RGB-only 可正常出帧。
相机配置新增可选 `enable_depth: false`；其他配置默认仍为 RGB+深度。
RGB-only 的 `read()` 明确返回 `(RGB, None)`，不伪造深度图，ZMQ RGB 协议不变。

```bash
envs/yam/.venv/bin/manimux serve \
  --config configs/abc/yam/infra/official-bottles-75k-smooth.yaml
```

由用户在 GUI Prepare / Start rollout。`official-bottles-75k-smooth.yaml` 采用
已经用于 SA 的公共执行器配置：100 Hz 控制，braking smooth，关节限速
0.6 rad/s、加速度 1.5 rad/s²；抓取接近阶段限速 0.25 rad/s。
夹爪独立限速 1.0/s、加速度 12.0/s²，连续 `[0,1]`，沿用抓取/释放位置保护。
这套保护只做 FK 判断到位，不对 DiT 输出解 IK；独立双臂 IK 解码关闭。
模型仍以 30 Hz 预测 30 步，源轨迹上限 25 步，剩余 0.3 秒请求下一次推理。
延迟裁剪仍可能减少实际执行行数，例如 200 ms 延迟会裁掉前 6 行，保留 19 行。
最长 18000 控制步（180 秒），连接时不自动移到起始姿态，结束时沿用双臂回 Home。

`official-bottles-75k.yaml` 保留为原来的 direct 对照配置：30 Hz 控制，
源轨迹上限 15 步，不使用 smooth 的限速与夹爪事件处理。本次准备的是 smooth 配置。

离线验证可独立重跑，不连接机械臂：

```bash
envs/yam/.venv/bin/python scripts/validation/abc_yam_offline_infer.py \
  --config configs/abc/yam/infra/official-bottles-75k-smooth.yaml \
  --episode ~/teleop_data/put_bottles_into_the_bin/20260905_155053_7fc15edb \
  --output data/diagnostics/abc_official_75k_20260910
```

本次录制数据第 0/150/300 帧经真实 HTTP worker→模型→adapter 均返回有限的
`30×14` 绝对关节动作，拆分为左右臂 `30×7` 后逐值一致，夹爪在 `[0,1]` 内。
10 步采样、FP32、RTX 4090，含 HTTP 的三次耗时约 203/184/211 ms。
报告和原始输出保存在输出目录的 `report.json`、`actions.npz`。
这验证了接口和推理链路，尚未证明此官方权重能在本机场景成功抓瓶子。

此次对照审计确认：ABC 与 Pi 的输入都来自请求时实测关节状态；左右臂各 7 维、
弧度绝对关节、夹爪 0 关/1 开的语义一致。ABC 保留自己的 CLIP/DINO 图像与状态
归一化，不能替换成 Pi/SA 的预处理。公共 `edge.py`、`timeline.py`、`smooth.py`
负责异步推理、延迟裁剪、减速、独立夹爪和收尾，没有 ABC 专属的旧控制循环。
差异来自旧 direct 配置没有选择 smooth。新配置对三组保存的真实模型输出做过
时间轴→smooth→SafetyGuard 离线探测（理想反馈，不模拟真机动力学），通过限速
与轨迹耗尽后的保持检查；ABC、timeline、executor 的现有测试均通过。
本次部署证据位于 `data/deployments/abc_official_75k_smooth_20260910/`。

## 通用 200k 配置

下文保留原有 200k 模型启动方式；不要与上面的 75k 命令混用。

和 MolmoAct 完全一样的四个服务，只有第 1 步（模型服务）和第 4 步的配置文件不同。
运行前清空机械臂工作区并准备好急停。

## 配置位置

```text
ManiMux: configs/abc/yam/infra/manimux.yaml
RTC:     configs/abc/yam/infra/rtc.yaml
```

以后增加其他本体时放在 `configs/abc/<embodiment>/`。

## 0. 一次性准备

ABC 用独立的 venv（它需要 CUDA 12.8 的 torch，和 MolmoAct 的 cu121 装不到一起）：

```bash
cd /home/ubuntu/manimux
uv venv envs/abc/.venv --python 3.12
uv pip install --python envs/abc/.venv/bin/python torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python envs/abc/.venv/bin/python -e ".[abc-yam]"
```

ABC 的任务指令走 CLIP 文本编码器，首次加载会下载 CLIP 资产到 `~/.cache/clip`。
没有外网时先手动放好这两个文件：

```
~/.cache/clip/ViT-B-32.pt
~/.cache/clip/bpe_simple_vocab_16e6.txt.gz
```

权重在 `checkpoints/pretrained/abc/abc_dit_xl_200k_model.pt`，归一化统计量
（`norm_stats`）已经打包在权重里，不需要额外的 json。

## ABC 是怎么推理的

一次推理的完整链路：

```
三相机 RGB (H,W,3 uint8)          左右臂关节 (14,)
        │                                │
        │  letterbox 等比缩放到 224x224   │  z-score 标准化
        │  居中补零 + ImageNet 归一化      │  (x-mean)/(std+1e-6)
        ▼                                ▼
   DINOv3 ViT-B/16 视觉编码        ┌──────────────┐
        │                          │  ABC-DiT XL  │
   任务指令 --> CLIP ViT-B/32 ---> │  32层/1536维 │
        (512维文本向量，带缓存)      │  2.02B 参数  │
                                   └──────┬───────┘
                                          │ 10 步整流流匹配采样
                                          ▼
                                  (30, 14) 归一化动作
                                          │ 反归一化 + 夹爪 clip[0,1]
                                          ▼
                    左臂 (30,7) <---- 拆分 ----> 右臂 (30,7)
                              绝对关节角(弧度) + 夹爪
```

单次推理实测 165-192 ms（10 步采样，RTX 4090，fp32）。

和 MolmoAct 在推理上的三个实质差异：

1. **任务指令走 CLIP 文本编码器**，不是 VLM 的 prompt。措辞要贴近训练时用过的
   任务名（权重里的 `sim_prompt_map` 记录了这些措辞）。CLIP 带 memo 缓存，
   换 prompt 只有第一次付编码开销，之后是查表。
2. **图像固定 224x224**，letterbox 在服务端做，所以相机给多大分辨率都行。
3. **相机名是硬绑定的，不只是顺序**：`top` = 前方场景，`left` / `right` = 左右腕。
   权重里的 `apool_queries.{top,left,right}` 按名字索引，接错位置模型直接失效。
   配置里已经映射好：`top_cam <- front_camera`。

## 1. ABC 模型服务

```bash
cd /home/ubuntu/manimux
envs/abc/.venv/bin/manimux-abc-server \
  --host 127.0.0.1 \
  --port 8300 \
  --checkpoint checkpoints/pretrained/abc/abc_dit_xl_200k_model.pt \
  --device cuda:0
```

看到 `Warmup OK` 和 `Uvicorn running` 后继续。

## 2. RealSense 相机服务

和 MolmoAct 用的是同一个相机服务，不需要改（可执行文件名里带 molmoact 只是
历史命名，它本身是通用的）：

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/manimux-camera-server --config configs/cameras.yaml
```

## 3. Viewer

和 MolmoAct 用的是同一个 Viewer：

```bash
cd /home/ubuntu/manimux
source .venv/bin/activate
manimux-viewer --robot yam --host 0.0.0.0 --port 8086
```

## 4. ManiMux 真机 runtime

先确认 `can_left` 和 `can_right` 都是 `ERROR-ACTIVE`（检查和重开命令见
[CAN 总线](can-bus.md)）：

```bash
for c in can_left can_right; do printf '%s: ' "$c"; ip -details link show "$c" | grep -o 'ERROR-ACTIVE\|ERROR-PASSIVE\|BUS-OFF'; done
```

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/manimux run --config configs/abc/yam/infra/manimux.yaml
```

## 和 MolmoAct 的差异

调参入口和 MolmoAct 完全一样（`execution.smooth` 控制速度和丝滑度，
`robot.options.*_duration_s` 控制起始/回零秒数）。只有这几项不同：

| | MolmoAct | ABC |
|---|---|---|
| 模型服务端口 | 8202 | 8300 |
| `policy.worker` / `adapter` | `molmoact_http` / `molmoact_yam` | `abc_http` / `abc_yam` |
| `policy.action_dt_s` | 0.05 | 0.033333（ABC 训练数据是 30 Hz） |
| `policy.horizon_steps` | 30 | 30（ABC-DiT 的 chunk_length 固定值） |
| 采样步数 | `num_steps: 10` | `diffusion_steps: 10`（低延迟可降到 5） |

机器人、相机、Viewer 三层用的是同一套插件，一行代码都没改。

## 需要注意的模型限制

- 这份权重是 ABC 官方的 **XDOF 工作站多任务预训练模型**（200k 步），不是针对
  我们这台 YAM 微调过的。它自带的 `norm_stats` 是 XDOF 数据集的统计量，所以
  零样本运行时关节偏置和夹爪标定不一定对得上，第一次跑必须低速、清空工作区。
- 任务指令的措辞会影响效果：CLIP 文本向量要贴近训练时用的任务名。权重里的
  `sim_prompt_map` 记录了训练用过的措辞，可以用它对照。
- 动作维度是 `[左臂6关节, 左夹爪, 右臂6关节, 右夹爪]`，关节是弧度绝对位置，
  夹爪归一化到 `[0, 1]`，和 MolmoAct 的 YAM 输出布局一致。

## 服务归属：哪些能停，哪些是共用的

MolmoAct 和 ABC 只有模型服务是各自独立的，相机和 Viewer 是同一套：

| 服务 | 端口 | 归属 |
|---|---|---|
| `manimux-molmoact-server` | 8202 | MolmoAct 专属，跑 ABC 时可以停掉释放显存 |
| `manimux-abc-server` | 8300 | ABC 专属 |
| `manimux-camera-server` | 5555 | **两者共用**，停掉 ABC 也没有相机 |
| `manimux-viewer` | 8086 | **两者共用** |

两个模型服务也可以同时开着（约 24 GB 显存），靠 `configs/*.yaml` 里的
`policy.options.server` 决定这次 rollout 连哪一个。

## 停止

正常运行时只在 ManiMux runtime 终端按一次 `Ctrl-C`，等待机械臂回 Home 和 Recorder
收尾；然后停止 ABC 模型服务。相机和 Viewer 如果还要给其他模型复用，可以继续保留，
否则最后再停止。异常运动时优先使用物理急停，不等待软件回零。
