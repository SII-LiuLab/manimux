# RDT2 + UMI 接入计划

本文是把 RDT2 接进本仓库的执行计划，面向新 session 开工。**只覆盖本机开发**，git
推送权限的问题不在这里讨论（见文末「git 追踪」一节，本机阶段只需在 submodule 里
建分支提交）。

UMI 数据在 pi05 上的现状见 [`pi05-umi-runbook.md`](pi05-umi-runbook.md)，两条线不共用
checkpoint，也不共用数据格式。

## 为什么选 RDT2

我们的台架是**双腕鱼眼、无第三人称、双臂 20 维 EEF**。这在 XPolicyLab 现有的 42 个
policy 里是孤例——全部要么需要 head/third-person 相机，要么（X-VLA）只认 head。
pi05 也一样，`UmiInputs` 现在给 base 槽位填黑图 + `image_mask=False`。

RDT2 是唯一一个**观测和动作两侧都原生对齐**的开源模型：

| | 我们的 `exchange_ball_v0` | RDT2 要求 | 结论 |
|---|---|---|---|
| 相机 | 双腕，两路独立 | 双目拼接 `(384, 768, 3)` uint8 | 拼接 + resize |
| 帧率 | 30 Hz | 30 Hz，chunk=24（未来 0.8 s） | 直接对上 |
| 动作维度 | 20 | 20 | 对上 |
| 旋转表示 | 6D | 6D | 对上 |
| **动作臂序** | **左在前** | **右在前**（`d0-9`=右, `d10-19`=左） | **要交换** |
| **图像臂序** | 两路独立 | **左半=左臂，右半=右臂**（和动作相反） | **不要跟着交换** |
| **位姿** | **绝对 TCP** | **相对当前观测帧** | **要转换** |
| 夹爪 | **[0,1] 行程比例** | **绝对宽度 m，量程 [0, 0.088]** | 要换算 |
| 容器 | LeRobot v3.0 | **webdataset shards** | 要写转换脚本 |

臂序和夹爪单位这两条最容易静默出错——接反了不报错，只是训不出来。**尤其注意图像的
左右半和动作的臂序是反的**：动作右臂在前，图像左半却是左臂
（`configs/robots/*.yaml` 注释 `# robot configurations, 0->right 1->left`；
`deploy/inference_real_fm.py:249-250` 里 `left_stereo ← camera1_rgb`；
`models/rdt_inferencer.py:268` 按 `["left_stereo","right_stereo"]` 顺序拼接）。

对照另外两个候选：X-VLA 动作空间对得上（EE6D 20D）但预训练是 DROID/RoboMIND/AgiBot
的第三人称数据，视觉先验和我们不匹配；Hy-Embodied-0.5-VLA（`policy/Hy_Embodied_05_VLA`）
是 UMI 原生 + RoboTwin 2.0 榜首（90.9/90.1），adapter 里已经有 `umi_coord_frame`、
`with_absolute: false`、`PosRotMat6d` 这套，但它的 ego 相机是**进模型的**，我们没有。
Hy-VLA 是加了 ego 相机之后的首选，RDT2 是当前硬件下的首选。

## 上游事实（2026-08-26 核实）

**代码**：[thu-ml/RDT2](https://github.com/thu-ml/RDT2)，Apache-2.0，805 stars / 58 forks /
20 open issues。权重 2025-09 发布，arXiv [2602.03310](https://arxiv.org/abs/2602.03310)
2026-02 才补上，最近一次 push 2026-02-07。不是新东西，但**没有任何第三方 benchmark
数字或复现报告**——它的输入契约（腕部鱼眼 + UMI 夹爪）在主流 sim benchmark 里没有对应
观测源，上不了榜。

**权重**（全部 Apache-2.0）：

| HF repo | 大小 | 用途 |
|---|---|---|
| `robotics-diffusion-transformer/RDT2-FM` | 0.98 GB | flow-matching action expert ← **我们训这个** |
| `robotics-diffusion-transformer/RVQActionTokenizer` | 1.75 GB | 动作 tokenizer，**normalizer 也在这个 repo 里** |
| `robotics-diffusion-transformer/RDT2-VQ` | 16.6 GB | Qwen2.5-VL-7B 底座 —— **FM 路径也必须加载它**，不是可选项 |
| `datasets/robotics-diffusion-transformer/BimanualUR5eExample` | — | 双臂 UR5e 示例 shards，用来对格式 |

normalizer 文件名：`umi_normalizer_wo_downsample_indentity_rot.pt`（`indentity` 这个拼写
错误是上游自带的，不是笔误；在 `RVQActionTokenizer` repo 里，不用去 README 里那个清华的
URL 下）。已落盘并验过内容：`ParameterDict`，键为 `action`、`robot0_eef_pos`、
`robot0_eef_rot_axis_angle`、`robot0_gripper_width`、`robot1_*`，外加
`robot0_eef_pos_wrt1` / `robot1_eef_pos_wrt0` 这类跨臂相对项，每项含
`input_stats` / `offset` / `scale`。注意它用的是**轴角**，而 action 向量里是 6D。

**RVQ 对 FM 路径唯一的作用就是这个 normalizer。** `MultiVQVAE` 只出现在 VQ 路径
（`vqvae/`、`deploy/inference_real_vq.py:255`），`deploy/inference_real_fm.py` /
`models/rdt_inferencer.py` / `rdt/train.py` 里 grep 不到。

**但 FM 路径必须加载 16.6 GB 的 `RDT2-VQ`（Qwen2.5-VL-7B）**，这条推翻了「只训 370M
action expert」给人的轻量印象：

- `scripts/finetune_rdt.sh:21` 写死 `VISION_LANGUAGE_MODEL_NAME_OR_PATH="robotics-diffusion-transformer/RDT2-VQ"`
- `rdt/train.py:131-136`、`models/rdt_inferencer.py:116-121` 都 `Qwen2_5_VLForConditionalGeneration.from_pretrained(...)`
- `deploy/inference_real_fm.py:102` 里 `--pretrained_vision_language_model_name_or_path` 是 `required=True`
- 决定性证据：`models/rdt_inferencer.py:227-239` 取 `outputs.past_key_values[selected_layers]`，
  配合 `configs/rdt/post_train.yaml` 的 `selected_layers: [0..13]` —— action expert 直接吃
  Qwen 的 **14 层 KV cache**，所以 `pre_compute_txt_embeddings_dir` 那条省显存的旁路
  **替代不了**它（预存文本 embedding 没有图像条件下的逐层 KV）。

不需要的：独立的 SigLIP / DINO 视觉编码器 —— `models/rdt_inferencer.py:47-48` 硬编码
`self.image_transform = None; self.vision_encoder = None`。需要的：
`Qwen/Qwen2.5-VL-7B-Instruct` 的 **processor**（`rdt/train.py:127`、
`models/rdt_inferencer.py:123` 写死了仓库 ID），RDT2-VQ 目录里自带同名文件。

**10,000 小时 UMI 预训练语料没有开源**，只能拿 ckpt，复现不了预训练。

**官方显存表**：

| Mode | RAM | VRAM | 我们的机器 |
|---|---|---|---|
| Inference | > 32 GB | ~16 GB | **内存不够，30 GB** |
| **Fine-Tune RDT2-FM (action expert)** | — | **~16 GB** | **显存宽裕** |
| Fine-Tune RDT2-VQ (LoRA) | — | > 32 GB | 卡在门槛上 |
| Fine-Tune RDT2-VQ (Full) | — | > 80 GB | 不可能 |

这张表里 FM 那行的「~16 GB 显存」**已经包含冻结的 Qwen2.5-VL-7B 底座**（见上）。真正
的瓶颈不是显存而是**系统内存**：加载 16.6 GB 的 backbone 时 CPU 侧峰值 RSS 很可能顶到
30 GB 上限。

**动作布局**（README 逐字，注意右臂在前）：

```
[0-2]   RIGHT ARM  end effector position x, y, z   (m)
[3-8]   RIGHT ARM  end effector rotation 6D
[9]     RIGHT ARM  gripper width                   (m)
[10-12] LEFT ARM   position
[13-18] LEFT ARM   rotation 6D
[19]    LEFT ARM   gripper width
```

训练数据里的夹爪就是**裸的 [0, 0.088] 米绝对宽度**（实测 shard 的 d9 max 正好 0.08800），
**不做相对化**。`ckpt/RVQ/README.md:59` 逐字：
*"RIGHT ARM gripper width, normalized to [0, 0.088], 0.088 means fully open"*。

⚠️ **README 里那步 `/0.088*0.10` 不是单位换算，是他们那台机器的夹爪标定，我们不能套。**
`grep -rn "0\.088"` 全仓扫描的结果：它**只**出现在部署侧输出
（`deploy/inference_real_fm.py:408,414,423`、`inference_real_vq.py:500,507,516`，同一文件
里套了三次，是上游自己的 bug）和 README 的部署片段里；`data/`、`rdt/`、`train.py` 里
一处都没有。它对应的是他们 `configs/robots/eval_bimanual_fr3_config.yaml:57` 那个
`open_width: 0.12` 的 Franka 夹爪。**我们的目标区间就是 [0, 0.088]，不要再乘。**

**webdataset shard 结构**（实测自 `shard-31-000001.tar`，不是抄 README）：

每样本 4 个成员，key 是**全局递增整数**（不是从 0 开始，组内首片还不连续）：

| 成员 | shape / dtype | 说明 |
|---|---|---|
| `{k}.image.jpg` | 解码后 `(384, 768, 3)` uint8 | 两路腕部鱼眼横向并排，各 384×384；**左半=左臂 robot1，右半=右臂 robot0** |
| `{k}.action.npy` | `(24, 20)` float32 | 见下面的逐维表 |
| `{k}.action_token.npy` | `(27,)` int16，值域 0-1023 | RVQ token，pos 18 + rot 6 + grip 3 |
| `{k}.meta.json` | dict，只有一个键 | `{"sub_task_instruction_key": "<episode 路径>"}` |
| `instructions.json` | 扁平 dict | episode 路径 → 指令字符串。注意是**复数**，上游 `configs/datasets/example.yaml` 里写的 `instruction.json` 是它自己的不一致 |

每个 shard 上限 20000 样本，超出开新片。

**action 20 维实测**（由 `vqvae/models/multivqvae.py:169-173` 与
`data/umi_video_dataset.py:399-430` 确证）：

| 维 | 含义 | 实测范围 |
|---|---|---|
| d0-d2 | robot0（**右臂**）EE 位置，**相对增量** m | ±0.21 / ±0.16 / ±0.15 |
| d3-d8 | robot0 旋转 6D，相对 | identity 附近 |
| **d9** | robot0 夹爪宽度，**绝对 m** | 0.00194 – 0.08800 |
| d10-d12 | robot1（**左臂**）EE 位置，相对增量 | ±0.17 / ±0.16 / ±0.19 |
| d13-d18 | robot1 旋转 6D，相对 | identity 附近 |
| **d19** | robot1 夹爪宽度，绝对 m | 0.00088 – 0.08800 |

三条容易踩的实测结论：

1. **6D 旋转的「零」是 identity `[1,0,0,0,1,0]`，不是全零**。实测 row 0 的 d3-d8 均值
   = `[1.0, -6e-5, 1.4e-4, 6e-5, 1.0, 1e-5]`。判断「是否为 delta」不能断言旋转块为 0。
   **而且这 6 个数是旋转矩阵的前两**列**，不是前两行**：`data/umi/pose_util.py:152-156`
   的 `mat_to_rot6d` 取 `mat[...,:,0]` / `mat[...,:,1]`，`rot6d_to_mat` 用
   `np.stack((b1,b2,b3), axis=-1)`。行列搞反不会报错，只会得到转置的旋转。
2. **相对位姿的锚点是当前观测帧，不是 chunk 的 row 0**
   （`data/umi_video_dataset.py:407-418`：`convert_pose_mat_rep(..., base_pose_mat=pose_mat[-1], ...)`）。
   row 0 已经是下一个控制步，所以接近零但不为零——实测 `mean|·| ≈ 1 mm`，max 14 mm。
3. **夹爪维完全不做相对化**（`umi_video_dataset.py:423` 直接切原始 action）。
4. **本体感受（proprio）不进模型**：`deploy/inference_real_fm.py:393` 发的是
   `np.zeros(20)`，README:311-313 也明说。观测只有图像和语言。

`action_token` 是给 RDT2-VQ 训练用的。FM 路径不读它（`MultiVQVAE` 在 FM 相关文件里
grep 不到），但为了和上游格式一致，转换脚本仍然要写这个字段——先确认
`scripts/finetune_rdt.sh` 走的 dataloader 是否强制要求它存在。

## 本机约束

- 单卡 32 GB 显存 / **30 GB 系统内存**（内存是当前最紧的资源，见
  `manimux-host-oom-kills-fcitx5`：python OOM 会连带杀掉 fcitx5）
- 无直连外网，代理 `127.0.0.1:7897`，CN 主机直连。**HF 权重不走代理的办法**：
  `hf-mirror.com` 只能给元数据，大文件会 302 到被劫持的 `us.aws.cdn.hf.co`；
  `hf-mirror.net`（Cloudflare）自己吐 LFS 字节，可用，但要逐文件对
  `hf-mirror.com` 官方元数据里的 `lfs.oid` 校验 SHA256。详见 memory
  `manimux-host-network-egress`
- 数据：`/home/jw/Desktop/dataset/exchange_ball_v0/`，LeRobot v3.0，
  **15 episodes / 5170 frames / 30 Hz**

## 执行步骤

### 步骤 0：拿权重 + 内存验证（阻塞项）

**权重部分已完成**（2026-08-26）。上游代码 shallow clone 在
`~/Desktop/project/RDT2`（仓库外，不进 submodule），权重落盘：

| 路径 | 大小 | commit sha |
|---|---|---|
| `ckpt/RDT2-FM` | 931 M | `f803808e3f79c77bba8bb4f3326e83d04e60783a` |
| `ckpt/RVQ` | 1.7 G | `a6a614ae049cc3b9fa6bd4f5fadaeee3375d4dec` |
| `ckpt/Qwen2.5-VL-7B-Instruct-processor` | 11 M | `cc594898137f460bfe9f0759e9844b3ce807cfb5` |
| `data/ur5e_example` | 1.5 G（只取 2 个 shard + 元数据） | `5550bc13fc38348360626a030dc585c8d2d5298f` |
| `ckpt/RDT2-VQ` | 16.6 G | 下载中 |

`hf` CLI 在本机对 hf-mirror 那条路走不通（元数据 HEAD 返回 `Repository Not Found`），
改用 curl，全程不走代理：

```bash
# 元数据 + lfs.oid
env -u http_proxy -u https_proxy -u all_proxy curl -s --resolve hf-mirror.com:443:160.16.86.14 \
  "https://hf-mirror.com/api/models/<repo>/tree/main?recursive=1&expand=1"
# 字节
env -u http_proxy -u https_proxy -u all_proxy curl -fL -C - --resolve hf-mirror.net:443:104.21.21.35 \
  -o <out> "https://hf-mirror.net/<repo>/resolve/<sha>/<file>"
```

每个 LFS 文件都对官方 `lfs.oid` 校验过 SHA256，与 HuggingFace 官方逐字节一致。
示例数据集完整是 36.5 GB / 45 个 shard，**只下了 2 个**——它只是格式参照物，不是训练数据。

**内存验证还没做，这才是真正的阻塞项。** 官方说 inference 要 > 32 GB 内存，我们只有
30 GB，而且现在知道了 FM 路径必须加载 Qwen2.5-VL-7B，所以这条风险比原先估计的更大。
按 `examples/DEPLOYMENT_TIPS.md` 跑 `deploy/inference_real_fm.py` 的离线推理，目标只有
一个：**看 RSS 峰值会不会超过 30 GB**。跑之前先 `htop` 关掉别的占内存的东西，并且知道
OOM 会连带杀掉 fcitx5。

内存不够的应对，按代价排序：加 swap（最省事，慢）、`device_map` / 分片加载减少 CPU 侧
峰值、量化 backbone（会改变数值行为，只适合验证链路）、换机器。**「只加载 FM 不加载 VQ
主干」这条已经排除了**——KV cache 那条依赖决定了它不成立。

### 步骤 1：LeRobot v3.0 → webdataset 转换 —— **已完成**

`scripts/lerobot_to_rdt2_shards.py`（本仓库，不进 submodule）。用 openpi 那个 venv 跑，
它已经有 pyarrow / av / cv2：

```bash
XPolicyLab/policy/Pi_05/openpi/.venv/bin/python scripts/lerobot_to_rdt2_shards.py \
    --dataset /home/jw/Desktop/dataset/exchange_ball_v0 \
    --out /home/jw/Desktop/dataset/exchange_ball_v0_rdt2
```

产物：`shards/shard-000000.tar`（5170 样本，294 MB）、`instructions.json`（15 条）、
`dataset.yaml`（可直接当 `--webdataset_config`）。整个转换 18 秒，流式写出，内存平稳。

`--self-test` 不需要数据集，只跑纯变换的断言（臂序自逆、6D↔矩阵往返、identity、
夹爪端点、以及「左臂运动必须落进 d10-12」「左半必须是左腕」这两条方向性断言）。

**不生成 `action_token.npy`。** FM 的 dataloader 逐字只取三个字段——
`rdt/dataset.py:33-39` 的 `.map()` 里只有 `image.jpg` / `action.npy` / `meta.json`，
`collate_fn`（`rdt/dataset.py:89`）也只用这三个，而且 `states` 直接是
`torch.zeros(...)`，再次印证 proprio 不进模型。

五件事的落地：

1. **动作臂序交换**：`swap_arm_order`，自逆函数，有往返测试；
2. **绝对 → 相对**：锚点是**该帧的 `observation.state`**，chunk 取 `action[i:i+24]`，
   `rel = R_anchor^T (p - p_anchor)`、`R_anchor^T R`（`pose_repr_util.py:37-38` 的逆）。
   夹爪不做相对化；
3. **双目拼接**：各 resize 到 384×384 后水平拼接，**左半左腕**；
4. **夹爪**：`full_open_normalized`（`pos=1.0 → 0.088`）为默认，
   `--gripper-mapping stroke_absolute` 可切到 `pos * 0.096`。**必须和
   `policy/RDT2/deploy.yml` 的 `gripper_mapping` 一致**；
5. **尾部**：默认 `--tail pad`（重复最后一帧动作补满 24），`discard` 可选。
   pad 保住了每条 episode 最后 23 帧，15 条共 345 样本（6.7%）。

#### 验证结果

对着上游 `shard-31-000001.tar` 逐字段比对：

| | 我们的 | 上游参照 |
|---|---|---|
| 成员 | `image.jpg` / `action.npy` / `meta.json` | 同上 + `action_token.npy` |
| action | `(24, 20)` float32 | `(24, 20)` float32 |
| image | `(384, 768, 3)` uint8 | 同 |
| d9 右夹爪 | 0.01716 – **0.08800** | 0.00194 – **0.08800** |
| d19 左夹爪 | 0.01990 – 0.08043 | 0.00088 – 0.08800 |
| row0 位置 `mean\|·\|` | 0.00146 m | 0.00100 m |
| row0 rot6d 均值 | `[0.9999, 1.5e-4, -5.3e-4, -1.1e-4, 0.9997, -2.1e-4]` | `[1.0, -6e-5, 1.4e-4, 6e-5, 1.0, 1e-5]` |

d19 上界 0.08043 不是 bug：左夹爪在这批数据里最大只开到 0.9143，×0.088 = 0.0805。

两个端到端检查：

- **相对→绝对往返**：348 个 chunk（episode 0，避开尾部 padding）重建回绝对位姿，
  最大位置误差 **1.5e-08 m**、最大旋转矩阵元素误差 **4.2e-08** —— float32 精度极限；
- **左右半归属**（这条只能靠像素证）：直接解码 `left_wrist` / `right_wrist` 原视频的
  第 151 帧重新拼一次，和 shard 里 `150.image.jpg` 比对——
  **左半 vs 左腕平均差 0.82（纯 JPEG 量化），vs 右腕 64.30**。右半同理。方向是对的。

### 步骤 2：微调 RDT2-FM

```bash
# configs/datasets/exchange_ball.yaml
name: manimux/exchange_ball
type: single
shards_dir: /path/to/shards
kwargs:
  instruction_path: /path/to/instructions.json
  normalizer_path: ckpt/RVQ/umi_normalizer_wo_downsample_indentity_rot.pt
```

改 `scripts/finetune_rdt.sh` 里的 dataset config 路径和 `<repository-path>`，然后跑。
FM 是 action expert 全参微调，bf16，~16 GB 显存。

README 建议 **训练不超过 5 个 epoch 以免过拟合**。我们只有 15 条数据——过拟合是必然，
这一步的目标是**跑通链路**，不是出结果。别拿这一版的成功率下任何结论。

放 tmux 里跑，别用 `nohup`（原因见 `pi05-umi-runbook.md` 第 3 节）。

### 步骤 3：接进 XPolicyLab —— **已完成（wiring 层面）**

adapter 已经写好，在 submodule 分支 `rdt2-integration`，提交 `cb95b56`（未 push）：

```
XPolicyLab/policy/RDT2/{model.py,deploy.yml,README.md,install.sh,process_data.sh,train.sh}
XPolicyLab/tests/unit/test_rdt2_umi_conventions.py     # 32 个单测
```

`deploy.py` / `eval.sh` / `__init__.py` / `setup_eval_*.sh` 与 `demo_policy` 逐字节相同。
`umi_dual` 两份 robot-info 都已存在，`bash utils/get_action_dim.sh <root> umi_dual` 输出 `20`。

参考来源：观测侧抄 `policy/Pi_05/model.py:657` 的 `encode_umi_obs`（双腕、无第三人称
profile + 相机 key 候选列表）；relative→absolute 和 `_rotm_to_quat_wxyz`（4 分支
Shepperd）复用 `policy/Xiaomi_Robotics_1/model.py:739-799`；chunk 节奏的 key 名
`exc_action_size` / `exc_action_interval` 取自 `policy/Hy_Embodied_05_VLA/deploy.yml:44-50`
（仓库里这个概念有 5 种叫法，选了注释最清楚的一个）。`H_RDT` / `RDT_1B` 检查过，都是
joint-only + 三相机、整块返回，没有可复用的东西。

**验证到哪一步**：静态检查、32 个单测、以及 debug loop 两遍（含 `DEBUG_OBS_ENCODED=1`）
都过了，10 个 episode 无 traceback。但 **debug 模式下故意不加载模型**
（`debug_load_model: false`）——FM 要拉 7B backbone，30 GB 内存下 OOM 会杀掉输入法，而
wiring check 本来就不需要权重。stub 带红色横幅、每步保持当前观测位姿，但**仍然走完整
的真实解码路径**（臂序交换、夹爪换算、相对→绝对、6D→四元数）。所以除了网络前向，
其余都验过了；`_build_upstream_policy()` 和 `_forward()` 的真实分支是**完全未测试的代码**。

复现命令：

```bash
cd XPolicyLab/policy/RDT2
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo exchange_ball demo umi_dual ee 0 0 0 rdt2 rdt2
EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=1 bash eval.sh RoboDojo exchange_ball demo umi_dual ee 0 0 0 rdt2 rdt2
```

conda env `rdt2`（python 3.11 + `pip install -e XPolicyLab` + pytest，没装 torch）。
两个环境坑：shell 里的 `VIRTUAL_ENV=.../manimux/.venv` 会顶掉 conda env，要先清掉并把
`.venv/bin` 从 PATH 摘出去；`utils/setup_env_client.sh:23` 在 conda activate **之前**就用
外层 python 读 yaml，所以外层也得有 pyyaml。这两条是既有问题，不是 RDT2 引入的。

#### ⚠️ `ee_pose` 宽度：框架内部两套契约打架

- `utils/process_data.py:176` 的 `unpack_robot_state` 按 `arm_dim` 切，umi_dual 下
  `left_ee_pose` 是 **9** 维（xyz + 6D）；
- `debug_env_client.py:248` 的 `validate_robot_state_dict` **硬编码 `"ee_pose": 7`**。

实跑确认 9 维会被拒（`dim mismatch: expected 7, got shape (9,)`）。RDT2 adapter 因此默认
`ee_pose_format: quat`（7 维），另留 `rot6d`（9 维）可切换，观测侧两种宽度都接受。

**这条对 pi05 那条线也成立**：`policy/Pi_05/model.py` 走的就是 `unpack_robot_state`，
所以它的 `umi_dual + ee` debug loop 大概率过不了。pi05 至今没跑过 debug loop，所以还
没暴露。真机接通 EE 通路时这个分歧必须先定下来。

## 已知风险

1. **内存 30 GB < 官方要求 32 GB** —— 步骤 0 就是为这个设的，先撞这堵墙。**这条比原先
   估计的更严重**：已确认 FM 路径必须加载 16.6 GB 的 Qwen2.5-VL-7B backbone，而且
   「只加载 action expert」的省内存旁路不成立。
2. **零样本跨本体不适用于我们**。README 明说需要
   *"purchase the designated end effector and camera, and 3D print the corresponding
   camera stand and flange"*。我们的 YAM 夹爪几何不一致，零样本不成立，只能微调。
3. **数据量**。15 episodes ≈ 3 分钟。对照：HiFi-UMI 给 π0.5 post-train 用了
   **每任务 3,200 条 / 10–20 小时**，RDT2 预训练 10,000 小时。换 backbone 换不掉这个变量，
   补数据的优先级高于换模型。
4. **无第三方验证**。RDT2 至今没有独立复现或 benchmark 数字，我们是在自己趟。
   保留 pi05 那条线作为对照，别把它删了。
5. ~~我们自己数据的 6D 旋转是行还是列~~ —— **已解决，两边都是列，不需要转置**。
   采集侧 `teleop/drivers/umi_source.py:46-53` 的 `rot6d_to_mat` 注释逐字：
   *"r6[0:3] and r6[3:6] are the first two \*columns\* of R (that is the convention
   bi_taccap_gripper documents and emits) ... recorded data is already orthonormal to
   ~4e-8"*，`:232` 按 `r1..r6` 顺序取。RDT2 侧 `data/umi/pose_util.py:152-156` 同样是列。
   所以 `rot6d_layout: cols` 对我们的数据也成立。
6. **`gripper_mapping` 是建模选择**。0.096 压成 0.088，微调能吸收多少不知道；
   转换脚本必须和 adapter 选同一个。
7. **tracker→TCP 共轭没实现**。上游用
   `C = inv(T_tracker_to_tcp) @ T_tracker_to_policy` 共轭相对位姿
   （`deploy/umi/real_world/real_inference_util.py:191-226`），常数是写死的 UMI 夹爪
   几何。我们的 TCP 定义不同，且在自己数据上微调应该把这个帧关系吃进去，所以省略了。
   如果哪版微调是在 UMI 工具帧里训的，就得补回来。
8. **normalizer 里是轴角，action 里是 6D**，两者的关系没有追。`RDTInferencer` 内部
   做 unnormalize，adapter 不碰，但写转换脚本时要确认自己喂进去的是哪种表示。

## git 追踪（本机阶段）

- **RDT2 上游仓库放在本项目之外**（如 `~/Desktop/project/RDT2`），不作为 submodule
  嵌进来，避免再加一层 sub-submodule。
- **adapter 代码写在 `XPolicyLab/policy/RDT2/`**，属于 submodule 的内容。
  分支 `rdt2-integration` 已从 detached `524fd1d`
  （= `origin/manimux-policy-integration-20260820` 的 tip）建好，adapter 在提交
  `cb95b56`，未 push。pi05 那批未提交改动原样带过来了，没有并进这个 commit。
- **submodule 里没配 git identity**（`git config user.name` 为空，历史提交都是
  `Cuzyoung <gongzy23@...>`）。这次是用 `git -c user.name=... -c user.email=...` 一次性
  绕过的，下次提交还会撞同一个错，可以 `git config --local` 设一下。

- **数据转换脚本写在本仓库 `scripts/`**，属于主仓库的内容。
- 一次逻辑改动 = submodule 一个 commit + 主仓库一个 commit（含 gitlink 指针 bump）。
  本机阶段只提交不推送。推送时的顺序和权限见团队约定。

## 参考

- [thu-ml/RDT2](https://github.com/thu-ml/RDT2) · [arXiv 2602.03310](https://arxiv.org/abs/2602.03310)
- README 本地副本（本次会话抓取，491 行）：会话 scratchpad 下 `rdt2_readme.md`，
  失效后重新 `curl https://raw.githubusercontent.com/thu-ml/RDT2/main/README.md`
- [HiFi-UMI (arXiv 2607.25895)](https://arxiv.org/abs/2607.25895) —— UMI 数据 post-train
  π0.5 的动作约定和数据量参照
- [Hy-Embodied-0.5-VLA (arXiv 2606.14409)](https://arxiv.org/abs/2606.14409) ——
  加 ego 相机之后的备选，adapter 已在 `XPolicyLab/policy/Hy_Embodied_05_VLA/`
- 本仓库：[`pi05-umi-runbook.md`](pi05-umi-runbook.md)、[`xpolicylab-runbook.md`](xpolicylab-runbook.md)
