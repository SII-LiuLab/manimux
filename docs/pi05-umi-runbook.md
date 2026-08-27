# Pi05 + UMI 运行手册

本文只覆盖 UMI 采集数据在 Pi05 上的训练链路和输入输出契约。YAM 机型的部署、方法对照
和真机命令仍在 [`pi05-yam-runbook.md`](pi05-yam-runbook.md)，两者不共用 checkpoint。

## 数据集

首个 UMI 数据集是双臂交接球：

```text
/home/jw/Desktop/dataset/exchange_ball_v0/   # LeRobot v3.0，15 episodes / 5170 frames / 30Hz
```

- `observation.state` 和 `action` 都是 20 维：左 `TCP(xyz + 6D 旋转) + gripper`，右同序；
- 相机：`observation.images.left_wrist`、`observation.images.right_wrist`，另有四路
  tactile（`*_tactile_left` / `*_tactile_right`）**当前不进模型**；
- 任务文本只有一个：`exchange ball`，训练时由 `prompt_from_task` 注入。

UMI 台架没有第三人称相机，而 pi05 固定暴露 `base_0_rgb`、`left_wrist_0_rgb`、
`right_wrist_0_rgb` 三个图像槽位。因此 `UmiInputs` 给 base 槽位填黑图并把
`image_mask` 置 False，这与 openpi 其它 policy 处理缺失相机的方式一致。

## 契约

- 训练配置：`pi05_umi`（全量微调）与 `pi05_umi_lora`（LoRA），都在
  `openpi/src/openpi/training/config.py`，都从官方 `pi05_base` 初始化。**单卡 32GB 只能跑
  LoRA**：全量微调的 train state（参数 + 梯度 + AdamW 二阶矩）约 50GB，在建 train state
  时就 OOM，和 batch size 无关。两者共用同一份 norm stats；
- 数据变换：`LeRobotUmiDataConfig` + `openpi/src/openpi/policies/umi_policy.py`；
- 动作：`50 x 20` **绝对** TCP 位姿与夹爪。旋转用 6D 表示，delta 只对平移有定义，
  所以 `use_delta_translation_actions` 默认关闭，需要时训练和推理必须同时打开；
- 推理侧 observation profile：`umi_native`（`XPolicyLab/policy/Pi_05/model.py`），
  只传两路 wrist 原始 HWC 帧和 20 维 state；
- `env_cfg_type: umi_dual`、`action_type: ee`，`arm_dim: [9, 9]`、`ee_dim: [1, 1]`，
  两份 robot-info 都要登记：`env_cfg/robot/_robot_info.json`（运行时与离线转换）和
  `XPolicyLab/utils/robot/_robot_info.json`（`utils/get_action_dim.sh`，训练路径）。

## 1. 数据落位

LeRobot 按 `HF_LEROBOT_HOME/<repo_id>` 解析数据集，`train.sh` 与
`compute_norm_stats.py` 必须指向同一份：

```bash
mkdir -p ~/.cache/huggingface/lerobot
ln -sfn /home/jw/Desktop/dataset/exchange_ball_v0 ~/.cache/huggingface/lerobot/exchange_ball_v0
```

数据根目录不在默认位置时，用 `OPENPI_HF_LEROBOT_HOME` 覆盖。数据集只在本地、没有
对应 Hub 仓库，若 LeRobot 仍尝试联网，加 `HF_HUB_OFFLINE=1`。

## 2. 归一化统计

这一步只读数据集，不需要 policy checkpoint，可以和权重下载并行：

```bash
cd XPolicyLab/policy/Pi_05/openpi
HF_LEROBOT_HOME=~/.cache/huggingface/lerobot HF_HUB_OFFLINE=1 \
  uv run scripts/compute_norm_stats.py --config-name pi05_umi
```

配置名是 `--config-name`，不是位置参数。产物为
`assets/pi05_umi/exchange_ball_v0/norm_stats.json`，训练时缺失会直接报错；
`pi05_umi_lora` 通过 `AssetsConfig` 复用这一份，不必重算。

这一步会顺带拉 PaliGemma 分词器（`gs://big_vision/paligemma_tokenizer.model`），gcsfs
不走系统代理，本机直连 DNS 解析失败。手动放进缓存即可，之后训练和推理都不再联网：

```bash
mkdir -p ~/.cache/openpi/big_vision
https_proxy=http://127.0.0.1:7897 curl -fL \
  -o ~/.cache/openpi/big_vision/paligemma_tokenizer.model \
  https://storage.googleapis.com/big_vision/paligemma_tokenizer.model
```

## 3. 训练

跑满 10k 步要一个多小时，放进 tmux，别用 `nohup`——从别的进程组起的后台任务会跟着
父调用一起被杀（表现是 step 中途 SIGKILL、日志没有 traceback、内存也没爆）：

```bash
cd /home/jw/Desktop/project/manimux
tmux new -s pi05-umi
OPENPI_TRAIN_CONFIG_NAME=pi05_umi_lora \
OPENPI_LEROBOT_REPO_ID=exchange_ball_v0 \
  XPolicyLab/policy/Pi_05/train.sh umi exchange_ball umi_dual ee 0 0
# Ctrl-b d 脱离，tmux attach -t pi05-umi 回来
```

六个位置参数是 `bench ckpt env_cfg action_type seed gpu_id`，checkpoint 落到
`XPolicyLab/policy/Pi_05/checkpoints/umi-exchange_ball-umi_dual-ee-0/<step>/`，
每个约 8.9GB，`assets/exchange_ball_v0/` 会一起写进去，服务端直接读它。

`pi05_base` 已在 `~/.cache/openpi/openpi-assets/checkpoints/pi05_base` 时，
`gs://` 权重路径会直接命中本地缓存，不需要覆盖。放在别处才需要：

```bash
--weight-loader.params-path /path/to/pi05_base/params
```

dataloader worker 数由 `OPENPI_NUM_WORKERS` 控制，默认 4。这台机器 30GB 内存，
checkpoint restore 本身占 ~12GB，openpi 原来的 8 个 worker 会被 OOM killer 直接
杀掉（退出码 137，没有 traceback）。

单卡实测约 2.2 it/s，10k 步 ≈ 1 小时 15 分；`pi05_umi_lora` 每 500 步存一次，
每个 checkpoint 8.9GB，跑满会占掉几十 GB，磁盘紧张时先调 `save_interval`。

## 3.1 训练曲线（wandb）

openpi 只内建 wandb，`train.sh` 默认打开。**没有登录凭据时 `wandb.init` 会卡在登录
提示**，所以先登录一次（凭据写进 `~/.netrc`，之后不用再管）：

```bash
XPolicyLab/policy/Pi_05/openpi/.venv/bin/wandb login
```

- `OPENPI_WANDB_PROJECT` 改 project 名，默认 `openpi`；run 名就是 checkpoint 目录名
  `umi-exchange_ball-umi_dual-ee-0`；
- 记录的是 `loss`、`grad_norm`、`param_norm`，外加 step 0 的 `camera_views` 拼图 ——
  可以顺便确认 base 槽位确实是黑图、两路 wrist 没有左右接反；
- 不想用 wandb 时设 `OPENPI_WANDB=0`。此时 loss 只在 stdout，且 tqdm 走的是块缓冲，
  要实时看得加 `PYTHONUNBUFFERED=1`。

## 4. 服务端检查

`configs/pi05/umi/server/exchange-ball.yaml` 里的 step 是占位值，先改成实际保留的
checkpoint，再只检查路径与契约：

```bash
XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  scripts/pi05_yam_server.py --check \
  --config configs/pi05/umi/server/exchange-ball.yaml
```

## 当前验证状态

已验证：

- `compute_norm_stats.py --config-name pi05_umi` 跑完 646 个 batch，state/actions 都是
  20 维，统计量量级正常；
- `train.sh` 端到端跑到第 523 步（随后手动中止），约 2.2 it/s，step 500 的 checkpoint
  正常落盘，`assets/exchange_ball_v0/` 一起写了进去；显存占用贴近 32GB 上限；
- 数据集契约、`umi_native` 编码和批处理有单元测试
  （`XPolicyLab/tests/unit/test_pi05_umi_encode.py`）。

未验证：没有跑完整训练（只到 523 步就停），没有评估过任何 checkpoint，没上真机，
因此不记任何成功率。wandb 只验证了开关接线，没有真正登录上传过。
`pi05_umi`（全量微调）在本机只确认过会 OOM，没有在大显存机器上跑通过。

真机还缺一段：现有 ManiMux infra（`configs/pi05/yam/infra/`）下发的是关节目标，而本
模型输出 TCP 位姿，需要先接通 IK 或 EE 控制通路才能谈 rollout。
