# Pi05 + YAM 运行手册

本文只覆盖 Pi05 模型、checkpoint、输入输出契约和普通 ManiMux 基线。推理方法的原理、
参数、验证步骤和真机命令归各自的方法文档，不在模型 runbook 重复维护。

## 本地红球任务 checkpoint

本次训练从官方 `pi05_base` 初始化，使用以下 20 条 YAM episode 微调：

```text
/home/ubuntu/yam-abc-reproduce/data/episodes/pick_the_red_ball_up_and_place_it_into_the_box/
```

推理使用 step 1000 checkpoint 自带的同名 stats，不复用 Robocurve stats：

```text
checkpoints/finetuned/ziyang/pi05-yam-pick-red-ball-box-b384/1000/
  params/
  assets/yam_pick_red_ball_box_v1/norm_stats.json
```

- server：`configs/pi05/yam/server/finetune-pick-red-ball-box-step1000.yaml`；
- ManiMux：`configs/pi05/yam/infra/manimux-pick-red-ball-box-step1000.yaml`；
- 输入：三路独立 RGB、14 维 YAM state 和红球任务文本；
- 输出：`50 x 14` absolute joint positions；
- 时间语义：轨迹点 30Hz，底层下发 100Hz；
- 起始位：episode `20260812_193716_0751770e` 的第一帧，左右臂关节和夹爪逐值匹配；
- 当前证据：真实 checkpoint 的离线 GPU、XPolicy WebSocket、三相机、ManiMux 和双臂
  YAM 真机链路均已通过。模型能明显复现示教轨迹，但20条练手数据质量有限，尚不能稳定
  完成任务，因此记录为部署成功、policy质量有限，不记作任务成功率。

只检查路径与契约：

```bash
cd /home/ubuntu/manimux
XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  scripts/servers/pi05_yam_server.py --check \
  --config configs/pi05/yam/server/finetune-pick-red-ball-box-step1000.yaml
```

离线 GPU forward 会用上述 episode 第一帧的 14 维状态，不连接相机、CAN 或机器人：

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  scripts/validation/pi05_yam_offline_infer.py \
  --config configs/pi05/yam/server/finetune-pick-red-ball-box-step1000.yaml \
  --infra-config configs/pi05/yam/infra/manimux-pick-red-ball-box-step1000.yaml
```

模型服务：

```bash
XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  scripts/servers/pi05_yam_server.py \
  --config configs/pi05/yam/server/finetune-pick-red-ball-box-step1000.yaml
```

完成相机、CAN 和 preflight 检查后，真机 ManiMux 由操作者运行：

```bash
envs/yam/.venv/bin/manimux run \
  --config configs/pi05/yam/infra/manimux-pick-red-ball-box-step1000.yaml
```

## 螺丝刀 step-15000：七种算法、统一 Executor

这套配置使用同一份 step-15000 checkpoint、checkpoint-matched norm stats、任务文本、
三相机映射、起始位和 Recorder。模型原生输出均为 `50 × 14` absolute joint actions，
轨迹点间隔 `1/30 s`。算法只改变采样、chunk 选择和调度，不切换底层 executor。

所有配置位于 `configs/pi05/yam/infra/`，共同后缀为
`-assemble-screwdriver-step15000.yaml`：

| 算法 / 文件名前缀 | runtime | 算法配置 |
|---|---|---|
| `manimux` | `manimux` | single-inflight，剩余 0.4 s 发起补充推理，blend 4 步 |
| `rtc` | `rtc` | 最少执行 20 步，初始延迟 4 步，beta 9.1，blend 4 步 |
| `act-temporal-ensemble` | `act_temporal_ensemble` | coefficient 0.01，每 20 个 policy 步（约 0.667 s）请求一次 |
| `aac` | `aac` | 20 个候选，motion threshold 0.2，backward beta 0.99 |
| `paint` | `paint` | execution steps 12，初始延迟 10 步，延迟历史窗口 10 |
| `autohorizon` | `autohorizon` | 使用已接入的 JAX selector，由模型返回执行长度 |
| `dvac` | `dvac` | tail 5，alpha 2.0，滚动窗口 5，执行长度 1–50 |

统一的底层设置为 `executor: smooth`、100 Hz 控制、8 Hz cutoff、关节速度上限
`0.25 rad/s`、加速度上限 `0.5 rad/s²`、绝对位置上限 `3.14 rad`；左右夹爪均为
连续 `0–1`，速度上限 `1.0 /s`、加速度上限 `12.0 /s²`。
这些设置和原有螺丝刀 ManiMux / RTC 一致。ACT、AAC、PAINT、AutoHorizon、DVAC 按各自
契约保留 `blend_steps: 0`；blend 属于 Timeline 拼接，不是底层 SmoothExecutor 参数。

AAC 继续使用现有 `yam_60ep_ee_increment.json` 作为**候选评分用** EE 增量统计；
它不是螺丝刀任务专门重新估计的统计，也不替换 checkpoint 的动作归一化文件。
各方法保留已有的冷启动 timeout，输出分别写入
`data/experiments/pi05-assemble-screwdriver-step15000/<算法前缀>/`。

2026-09-08 补齐 ACT、AAC、PAINT、AutoHorizon、DVAC 五份螺丝刀配置。
这表示复用现有算法实现并完成配置/单元回归，不代表新增的五种组合已在该 checkpoint 上
完成 GPU 或真机测试；既有算法的实测记录见下方方法文档。历史红球和 Robocurve 配置保留不变。

先在 `/home/ubuntu/manimux` 启动同一个 Pi05 policy server：

```bash
cd /home/ubuntu/manimux
XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  scripts/servers/pi05_yam_server.py \
  --config configs/pi05/yam/server/finetune-assemble-screwdriver-step15000.yaml
```

模型服务无需随算法更换；在另一终端选择一种 runtime，例如 PAINT：

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/manimux serve \
  --config configs/pi05/yam/infra/paint-assemble-screwdriver-step15000.yaml
```

替换文件名前缀即可选择表中的其他算法，例如 RTC：

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/manimux serve \
  --config configs/pi05/yam/infra/rtc-assemble-screwdriver-step15000.yaml
```

七种 runtime 选择一种运行，共用上面的同一个 policy server。

Pi05 上的训练免推理方法由方法文档单独维护：

- ACT temporal ensemble：[`act-temporal-ensemble.md`](act-temporal-ensemble.md)；
- AAC：[`reproductions/aac-pi05.md`](reproductions/aac-pi05.md)；
- PAINT：[`reproductions/paint-pi05.md`](reproductions/paint-pi05.md)；
- AutoHorizon：[`reproductions/autohorizon-pi05.md`](reproductions/autohorizon-pi05.md)。
- DVAC：[`reproductions/dvac-pi05.md`](reproductions/dvac-pi05.md)。
  先用 `scripts/validation/xpolicylab_yam_dvac_probe.py --requests 3` 验证同一 session 内的滚动阈值，
  再决定是否开放真机命令。

以下章节记录先前 Robocurve 16-step checkpoint 的独立实验，不要与本地 50-step 配置混用。

## Robocurve 模型

YAM 微调权重位于：

```text
checkpoints/finetuned/robocurve/pi05-yam-molmoact2/  # robocurve/pi05-yam-molmoact2
  params/
  assets/yam-bimanual-merged/norm_stats.json
```

- 输入：base、left wrist、right wrist 三路独立 RGB，不拼成一张图；
- state：14 维，左 `6+1` 后右 `6+1`；
- 输出：`16 x 14` absolute joint positions，30Hz；
- flow sampling：OpenPI 默认 10 steps；
- stats：checkpoint 自带 quantile norm stats；
- 发布来源：`robocurve/pi05-yam-molmoact2`；
- server：`configs/pi05/yam/server/finetune.yaml`；
- ManiMux 30Hz 默认配置：`configs/pi05/yam/infra/manimux.yaml`；
- Pi-guided RTC 对照：`configs/pi05/yam/infra/rtc.yaml`；
- 50ms 拉伸时序对照：`configs/pi05/yam/infra/stretched-50ms.yaml`。

## 当前验证状态

已完成一次不连接硬件的真实 checkpoint GPU forward，输出为 finite 的 `16 x 14` absolute
joint actions。2026-08-20 又完成了一次真实 YAM rollout：操作者观察到策略明确朝 pick
任务执行，但旧实验配置的 `0.20 rad/s`、`0.40 rad/s²` 低速包络造成慢速追赶和抖动，
因此这次不能记作成功率。

该 episode 位于：

```text
data/archive/pre-campaign-20260824/root-runs/run-20260820T061701Z-8cfd2e00/episode-6bcee9a699eb/
```

记录中 143 次推理提交、141 个 plan 接受，常见 stale-prefix 只有 2–3 步，说明 16-step
horizon 没有被推理延迟耗尽。当前 ManiMux infra 配置使用与 MolmoAct 相同的 `smooth` executor
参数：8Hz cutoff、`0.25 rad/s`、`0.5 rad/s²`。Pi05 只保留模型契约要求的 30Hz
和 16-step horizon。这些是执行器
跟踪上限，不是模型动作好坏的阈值判定。

## 1. 离线检查

只检查路径和契约，不加载模型：

```bash
cd /home/ubuntu/manimux
XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  scripts/servers/pi05_yam_server.py --check \
  --config configs/pi05/yam/server/finetune.yaml
```

只做合成三相机 observation 的 GPU forward，不启动服务或机器人：

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  scripts/validation/pi05_yam_offline_infer.py \
  --config configs/pi05/yam/server/finetune.yaml
```

## 2. 模型服务

```bash
cd /home/ubuntu/manimux
XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  scripts/servers/pi05_yam_server.py \
  --config configs/pi05/yam/server/finetune.yaml
```

终端保持前台没有持续日志是正常的。确认 ready：

```bash
ss -ltnp | rg ':8500'
nvidia-smi
```

必须看到 `127.0.0.1:8500` 处于 `LISTEN`。

## 3. 相机与 Viewer

相机服务：

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/manimux-camera-server --config configs/cameras.yaml
```

已有 `5555` 服务时不要重复启动。Viewer 可选：

```bash
envs/yam/.venv/bin/manimux-viewer --robot yam --host 0.0.0.0 --port 8086
```

## 4. Preflight

读取真实三相机和当前关节并推理，但显式禁止 start-position 和退出回 Home，也不发送
模型动作：

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/python scripts/validation/pi05_base_yam_preflight.py \
  --config configs/pi05/yam/infra/manimux.yaml
```

保存它打印的 measured state、first action、shape、gripper range 和 steady latency。脚本只
报告契约，不给主观 `safe_for_live` 结论。

## 5. 真机 runtime

清空工作区、急停在手，并确认两路 CAN 都是 `ERROR-ACTIVE`：

```bash
for c in can_left can_right; do
  ip -details link show "$c" | grep -o 'ERROR-ACTIVE\|ERROR-PASSIVE\|BUS-OFF'
done
```

默认 ManiMux 实验保留 Pi05-YAM checkpoint 的 30Hz 时序：

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/manimux run --config configs/pi05/yam/infra/manimux.yaml
```

这个 ManiMux 基线将 16 个绝对关节点按约 0.50 秒执行。机器人控制环也是 30Hz，避免
改变 checkpoint 学到的时间语义；异步调度、Timeline、SmoothExecutor 和记录仍由
ManiMux 负责。

50ms 拉伸对照把同一组点按约 0.75 秒执行：

```bash
envs/yam/.venv/bin/manimux run --config configs/pi05/yam/infra/stretched-50ms.yaml
```

两份配置是独立对照实验，不要同时运行。2026-08-20 的 50ms 真实运行中，模型请求和
plan 接受都正常，但 replanning 明显变慢，操作者观察到“向前抖一下又回来”；因此 50ms
只保留为失败对照，不再作为默认真机配置。

连接后先用 3.5 秒移动到 YAML start pose，这不是模型动作。当前执行器比第一次真实 rollout
快很多；第一次使用新参数只做短 rollout，并始终准备急停。

## Chunk 边界诊断

普通 runtime 会在每个新 plan 接受时写一条 `plan_boundary` 到 episode 的
`events.jsonl`。它同时保存模型原始 chunk 首点、ManiMux 拼接后的首点、上一条命令和实测
关节；只增加诊断记录，不改变动作、速度或拼接算法。即使按 `Ctrl-C`，记录也会保留在
`.partial` episode 中。

完成一次 Pi05 和一次 MolmoAct2 短 rollout 后，只需告知“跑完了”；分析端会自动选择两种
policy 最新的有效 episode。也可以手动离线查看：

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/python scripts/validation/analyze_chunk_boundaries.py
```

## Pi05 YAM finetune + RTC

当前 step-1000 红球入盒实验仍使用同一个 finetune 配置，不需要启动第二份权重：

```bash
cd /home/ubuntu/manimux
XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  scripts/servers/pi05_yam_server.py \
  --config configs/pi05/yam/server/finetune-pick-red-ball-box-step1000.yaml
```

RTC 使用独立 infra 配置。重复评测时启动一次长期 session service：

```bash
envs/yam/.venv/bin/manimux serve \
  --config configs/pi05/yam/infra/rtc-pick-red-ball-box-step1000.yaml
```

它与 step-1000 Default config 使用相同的 `50 x 14` checkpoint contract、100Hz robot loop、
30Hz policy points、`0.25 rad/s`、`0.50 rad/s²` 和 3 秒 start/home。RTC pilot 使用
`min_execute_steps: 20`、`initial_delay_steps: 4`、`beta: 9.1`；运行时会根据真实 round trip
更新 delay forecast。只有 RTC scheduling 和 Pi-guided denoise condition 发生变化。

每次运行创建：

```text
data/experiments/pi05-red-ball-box-step1000/rtc/session-*/rollout-*/
```

Viser 在 episode 正常落盘后开放 `Task result`、`Smoothness (1-5)`、failure tags 和 note，
保存到 `rollout-*/evaluation/human-label.json`。这里的 task result 与 `result.json` 中表示 runtime
正常收尾的 `success` 完全分开。

Viewer 顶部先选择 `Experiment mode`：OFF 只控制 rollout，不要求 reward；ON 用于正式采集，
必须填写人工评测后才能开始下一条。正式实验同时填写可读的 `Layout / condition ID`。Prepare
时页面中的 task command 会真实发送给 Pi05，不只是显示文本。

`serve` 不加载模型、不启动相机，也不替代 Viewer。用户先分别启动 camera server、Pi05 model
server 和 `manimux-viewer`，再启动一次 `serve`。Viser 显示 service ready 后：

1. 选择 Experiment mode，确认 task；正式实验再填写 layout ID。
2. 点击 `Prepare new rollout`；ManiMux 创建全新 episode、连接机器人并移动到 start pose。
3. 等待页面显示 `PAUSED`，确认真机后点击 `Start rollout`。
4. 需要保持当前位置时点击 `Pause / Hold`，随后使用 `Resume rollout` 继续。
5. 完成或失败后点击 `Finish & Home`；等待 Recorder 落盘和机器人 Home。
5. 实验模式 ON 时填写并保存人工评测；service 回到 idle 后点击下一次 `Prepare new rollout`。

每条正式 config 以 10Hz 异步保存各相机 MP4；编码不会阻塞控制环。结束后检查
`videos/index.json` 的 `dropped_bundles` 和 `error`。完整证据目录见
[experiment infrastructure](experiment-infra.md)。

每条 episode 都创建新的 worker session，并重置 Timeline、RTC delay history、Executor、Recorder 和
Viewer trail；不会继承上一条 rollout 的推理状态。camera/model/viewer/service 进程保持运行。

只需要单条 rollout 或用于脚本兼容时，原 CLI 入口仍保留：

```bash
envs/yam/.venv/bin/manimux run \
  --config configs/pi05/yam/infra/rtc-pick-red-ball-box-step1000.yaml
```

## 停止

单次 `run` 在前台按一次 `Ctrl-C`，等待 Home 并退出。长期 `serve` 应先在 Viser 完成当前
rollout；回到 service idle 后，再在 serve 终端按 `Ctrl-C`。随后才停止 Viewer、模型和相机。
不要使用模糊 `pkill`。

## Pi05 base + RTC

base zero-shot 是单独实验：

```text
server: configs/pi05/yam/server/base.yaml
infra:  configs/pi05/yam/infra/base-rtc.yaml
```

它使用 50-step Pi-guided RTC，不代表 YAM 微调版的默认配置。不要用 base config 覆盖本文
的 16-step checkpoint 实验。base 权重仍位于 `checkpoints/pretrained/pi05/pi05_base`；它只
复用 Robocurve YAM checkpoint 内的 norm stats，因此配置会分别打印
`checkpoint_source` 和 `norm_stats_source`。
