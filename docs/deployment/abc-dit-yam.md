# ABC-DiT + YAM 运行手册

## 当前部署

官方 [ABC-DiT](https://abc.bot)（XL，2.02B 参数）通过 `XPolicyLab/policy/ABC_DiT` 和共享
`xpolicylab_ws` 接入，与 Pi05 一样使用通用 `JointAdapter`，不需要 ManiMux 侧专用适配器。
模型加载、图像预处理、归一化、CLIP 指令编码和采样都在该 policy 目录中；ManiMux 负责相机、
观测映射、调度、执行与记录。安装和 `deploy.yml` 键见
[ABC_DiT README](../../XPolicyLab/policy/ABC_DiT/README.md)。

| 参数 | 设置 |
| --- | --- |
| checkpoint | `checkpoints/pretrained/abc/abc_dit_xl_200k_model.pt`（官方多任务预训练 200k 步，未在本站微调） |
| SHA-256 | `79754ece…2ba175ddce41d018c9f51ba7`；与 `~/sa/checkpoints/sapolicy/abc_dit_xl_200k_model.pt` 为同一文件 |
| 归一化 | checkpoint 内嵌的 XDOF `norm_stats`（state / actions z-score） |
| 输入 | 三路 RGB：`top` ← `d405_front`，`left` ← `d405_left`，`right` ← `d405_right`；14 维关节状态 |
| 图像 | 640×480 → 等比 bicubic（antialias）缩到 224×168，上下各补 28 行零，再 ImageNet 归一化 |
| 输出 | `30 × 14` 绝对关节位置（弧度），每臂 `[6 关节, 夹爪]`；夹爪 0 闭合、1 张开，服务端裁到 `[0,1]` |
| 指令 | CLIP ViT-B/32 文本向量，取自 `run.task`（recipe `prompt` 非空时覆盖）；官方 bottles 指令 `throw plastic bottles in bin` |
| 动作间隔 / horizon / 采样步数 | 1/30 s / 30 / 10 |
| 噪声种子 | `inference_seed: 0`，每次 RESET 重置，连续请求推进 |
| 精度 | fp32（`fast_inference: false`，与官方部署命令一致；bf16 快速模式的输出有约 0.008 rad 的量化台阶） |
| RTC | 官方 ABC RTC：训练时动作前缀条件（`get_action_paint` → 上游 `sample_actions_rtc`，前缀 4 步）。另保留 PiGDM 推理时引导（`get_action_rtc`），实验不使用 |
| 端口 / 站点服务 | 8530 / `policy_abc_dit` |

相机名按名字绑定：权重的 attention-pool query 以 `top/left/right` 索引。左右腕交换后，
在 ABC 自己的验证集上关节误差从 0.058 rad 升到 0.122 rad，所以接错不会报错但模型失效。

## 实验配置

两份配置的执行部分与 Pi05 Joint 30k 对齐（Direct executor、不限速 profile、blend 0、100 Hz、
12000 控制步、10 布局 × 3 次）。RTC 完全按 ABC 官方部署命令（abc 仓库 deploy README，bottles）：

```text
--diffusion-steps=10 --rtc --rtc-prefix-length=4 --rtc-inference-lead-steps=7
--execute-chunk-dim=16 --prompt='throw plastic bottles in bin'
```

- Serial：`manimux/configs/experiments/put_bottles/abc_dit/yam_abc_dit_serial_200k.yaml`
- RTC：`manimux/configs/experiments/put_bottles/abc_dit/yam_abc_dit_rtc_200k.yaml`
- 模型 recipe：`manimux/configs/policy/abc_dit/yam/put-bottles/pretrained-200k.yaml`

| 项 | Pi05 Joint 30k | ABC-DiT 200k | 原因 |
| --- | --- | --- | --- |
| horizon | 50 | 30 | 模型固定 chunk 长度 |
| Serial K / RTC 每块执行 | 16 / s_min 16 | 16 / 16 | 官方 `--execute-chunk-dim=16` |
| RTC 方法 | PiGDM 推理时引导，β 5 | ABC 训练时前缀条件：前缀 4、lead 7 | 官方部署方式；见下文 |
| 起始位 | 零位、夹爪开 | ABC-130k bottles 的 ready 姿态（与 SAPolicy 相同） | 零位距 ABC state 均值约 3σ，ABC 数据中从不出现 |
| `run.task` | Put the bottles into the bin. | throw plastic bottles in bin | 官方 bottles 部署指令 |
| 服务 / 身份 | `policy` / pi05 | `policy_abc_dit` / abc_dit | — |

ABC 真机数据有两种任务名：`put_the_plastic_bottles_in_the_bin` 与
`throw_plastic_bottles_in_bin`，官方部署用后者。

### 官方 RTC 如何映射到 ManiMux

上游 `_RTCManager`：每块执行 E=16 步；在执行窗口结束前 lead=7 步取观测并开始推理；
把窗口最后 p=4 步作为新块第 0–3 行的硬前缀（训练时学过的前缀条件，checkpoint
`max_action_prefix: 8` 即前缀长度 0–7）；切换后从新块第 4 行执行。

ManiMux 现成的 `paint` 调度（`manimux/runtime/paint.py`）不需要修改：执行到第 16 行时取观测，
把推理期间将执行的 d=7 行（第 16–22 行）发给服务端，新块第 0 行对齐观测时刻并按实际时延裁剪。
`delay_buffer_size: 1000` 让初值 7 一直是延迟预测的最大值，lead 固定为 7；实际时延超过 7 步时
该块被拒收（`PAINT response advanced beyond its anchored prefix`），runtime 延续旧计划并上调预测。
`ABC_DiT.get_action_paint` 取这 7 行的最后 4 行作为硬前缀调用上游 `sample_actions_rtc`，
返回"前 3 行旧动作 + 新块"截到 30 行：第 0–6 行与旧计划完全一致，第 7 行起是新动作，
与上游在同一时刻、同一前缀上切换。唯一差异是第一块：上游在第 9 行取观测，这里在第 16 行。
`paint` 元数据记录 `method: abc_action_prefix`、`prefix_length: 4`，与 PAINT 论文的反演采样无关。

## 启动

站点文件需要 `services.policy_abc_dit.endpoint: ws://127.0.0.1:8530`（见
`manimux/configs/local/yam_example.yaml`）。四个终端，均在仓库根目录：

```bash
# 1: 相机
envs/yam/.venv/bin/python -m manimux.servers.camera.server \
  --experiment manimux/configs/experiments/put_bottles/abc_dit/yam_abc_dit_rtc_200k.yaml

# 2: RoboGUI
envs/yam/.venv/bin/python -m manimux.robogui.dashboard --robot yam --host 127.0.0.1 --port 8086

# 3: ABC-DiT 模型服务（fp32 加载约 10 s；端口打开前不要启动 runtime）
envs/abc/.venv/bin/python -m manimux.servers.abc_dit \
  --experiment manimux/configs/experiments/put_bottles/abc_dit/yam_abc_dit_rtc_200k.yaml

# 4: runtime（Serial 换成 yam_abc_dit_serial_200k.yaml，模型服务不用重启）
envs/yam/.venv/bin/python -m manimux serve \
  --config manimux/configs/experiments/put_bottles/abc_dit/yam_abc_dit_rtc_200k.yaml
```

模型命令加 `--check` 只解析路径和契约，不加载模型。Serial 与 RTC 共用同一个模型服务和身份。

checkpoint（8 GB）不进入 Git：新机器上从本机复制到 `checkpoints/pretrained/abc/abc_dit_xl_200k_model.pt`，
服务启动时校验 SHA-256。

`envs/abc` 是 ABC 的独立环境（Python 3.12、torch 2.11 cu128）。新机器上可以用
`bash XPolicyLab/policy/ABC_DiT/install.sh envs/abc/.venv` 创建。CLIP 资产放在
`~/.cache/clip/`（`ViT-B-32.pt`、`bpe_simple_vocab_16e6.txt.gz`），缺失时首次启动会下载。

## 验证记录（2026-10-07，RTX 4090）

以下都经过真实 `xpolicylab_ws` 客户端 → WebSocket → 模型服务 → `JointAdapter` 解码：

- 官方 bottles 预览集 10 条 val episode（8 真机、2 仿真），每条取 3 帧，共 30 个样本。
  预测 30 步与真值的平均关节误差 0.058 rad，原地不动基线 0.101 rad；夹爪误差 0.025 / 0.064。
  同一 val 帧上，bf16 快速模式与 fp32 的真值误差相同（0.0876 / 0.0874 rad）。
- RTC：把前 8 步条件设为真值 +0.05 rad，引导后前缀误差从 0.087 降到 0.038 rad。
- 时延（含传输）：默认 p50 68 ms，RTC 175 ms。fp32 下分别为 166 / 266 ms（进程内）。
- 种子：RESET 后同一观测的输出逐值相同；runtime 的 `expected_backend` 校验对两份实验均通过。
- 本站 3 条 teleop 放瓶子录制（640×480）：零样本预测并不比原地不动更接近人类示教
  （0.063 vs 0.052 rad，仅 20% 样本更好）；首步与当前状态的最大偏差 0.045 rad，不会跳变。

这些只证明链路和预处理正确，不代表本站真机任务成功。

## 已知限制

- 这是 XDOF 工作站的多任务预训练模型，未用本站数据微调。SAPolicy 排查过的站点差异同样存在：
  腕部相机视角（我们的接近水平，ABC 朝下看桌面）、夹爪外观、桌面/背景，
  以及我们的 YAM 底座多垫了 2.2 cm（见下节的安装补偿）。
- 首轮真机请低速观察、手放急停。需要限速时沿用现有 `executor.type: smooth`
  （如 SAPolicy 的 braking + 0.6 rad/s、1.5 rad/s²），不要改模型输出。
- fp32 前缀推理约 174 ms（含传输），在 lead 7 步（233 ms）以内；GPU 被其他进程占用时可能超出，
  表现为块被拒收。
- ABC 人类操作数据的单步关节变化 p99 达 0.07–0.24 rad（数 rad/s），模型按官方指令输出的
  动作也会这么快；Direct 执行器不会限速。

## 真机记录

**2026-10-07 第一次（PiGDM 推理时 RTC，s_min 15，bf16，指令 put the plastic bottles in the bin）**：
模型大多时间悬停。plan 61 突然以 2.7 rad/s 下冲，plan 62 虽带 RTC 条件却回到悬停，
交接一个 tick 内指令跳 0.353 rad，Direct 原样执行，表现为"静止—剧烈摇晃—静止"。
用录制观测离线重放：PiGDM 引导是否跟随条件取决于噪声种子，β 加到 100 也拉不回；
官方前缀条件在 3 个种子上都精确接上并连续延续。另有 bf16 量化导致交接小跳，中位数 0.0165 rad。
据此改为官方 ABC RTC + fp32。用同一场 rollout 的 15 个观测经真实客户端重放：
第 0–6 行与旧计划一致，第 7 行接缝中位数 0.0019 rad，最大 0.014 rad；时延 p50 174 ms。

## 安装高度补偿（试验）

我们的 YAM 底座立在 2.2 cm 垫板上，ABC 的直接放在桌面。关节空间模型复现的是 ABC 的关节姿态：
2026-10-07 官方 RTC rollout 中，抓取闭合时 grasp_site 在基座系的高度中位数 0.044 m，
与 ABC 官方真机数据的 0.045 m（152 次桌面抓取）相同，所以本站夹爪实际高出桌面 2.2 cm，
停在瓶子上方闭合（夹爪闭到 0.32–0.45，ABC 夹住瓶子时约 0.52–0.57）。

`yam_abc_dit_rtc_200k_mount22.yaml` 改用 `manimux.policy_adapter.joint_mount:MountOffsetJointAdapter`，
`base_offset_m: {left_arm: [0, 0, 0.022], right_arm: [0, 0, 0.022]}`：

- 请求：测得关节与 PAINT 前缀（RTC 条件同理）经 FK → 基座系 z +2.2 cm → IK，变成模型坐标；
- 动作：模型输出逐行 FK → z −2.2 cm → IK，变成机器人关节；姿态与夹爪不变，IK 失败则整块拒收；
- 起始位：ABC ready 姿态同样下移 2.2 cm，模型看到的仍是 ABC ready 姿态。

离线检查：每行精确下移 22.0 mm，水平偏差 < 0.03 mm；前缀往返误差 < 2e-3 rad；
接缝中位数 0.0020 rad；含转换的时延 p50 183 ms。转换在控制线程内完成，
每块约 15 ms 解码 + 6.5 ms 请求准备（PAINT 不支持 `action_decoding: process`）。

**2026-10-07 安装补偿 + 官方快速推理 + 官方 RTC（lead 7）**（`yam_abc_dit_rtc_200k_mount22.yaml`，
瓶子平放）：操作者确认全部抓住。左臂 9 次闭合中多数在机器人基座系 z 0.024–0.035 m（模型坐标约
0.046–0.057 m，对应 ABC 中位数 0.045 m），闭合 1.5 s 后开度 0.35–0.53（被瓶子撑住）。121 块全部接收，
无拒收；obs→commit 中位数 133 ms、最大 166 ms（延迟 3–4 步），其中补偿换算解码约 30 ms。
此前三个问题依次为：PiGDM 推理时 RTC 交接跳变 → 改官方前缀 RTC；fp32 时延 + 补偿换算超过 lead 7
导致 PAINT 拒收后停摆 → 官方快速推理（`FastRTCInferenceGraph`）；夹爪高 2.2 cm 未夹住 → 安装补偿。

仍未修复：`manimux/runtime/paint.py` 在一块被拒收后若"已执行 + 延迟 > horizon"会永久停止请求。
