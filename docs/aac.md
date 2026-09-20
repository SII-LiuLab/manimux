# Adaptive Action Chunking (AAC)

逐文件、逐公式和逐级验证证据见
[`docs/reproductions/aac.md`](reproductions/aac.md)。本文保留为架构与使用概览。

ManiMux 的 AAC 接入追踪两个官方仓库：

- multi-sample GR00T server：`Adaptive-Action-Chunking/gr00t-multi-sample` commit
  `11e926b0f34cf6acfcb92c0fe6127a1bdc7b856a`；
- entropy/chunk selector client：`Adaptive-Action-Chunking/robocasa` commit
  `fed3e6b5eb348160dd0570f326f726758fee9056`。

## 官方方法的动作契约

官方代码基于 **GR00T N1.5**，一次 backbone forward 后将 feature batch 扩展为
`N=20`，从不同初始高斯噪声并行生成 20 个 16-step chunk。RoboCasa/LIBERO client 按
7D action 计算：

```text
3D end-effector position
+ 3D end-effector rotation
+ 1D gripper close
```

每个时间步分别计算 position/rotation 的 sample-covariance Gaussian differential entropy
和 gripper 的 Bernoulli entropy，然后计算各前缀的平均 entropy。官方 elbow 是
`max(argmax(diff(prefix_mean)) + 1, 2)`；motion floor 选择第一段 movement magnitude 大于
`move_th=3.0` 的前缀，最终长度取两者最大值。候选默认选第 0 条，也提供 `mean` 与
`backward` selector。

## YAM joint-output、incremental-EE adaptation

当前接入的 GR00T N1.7 与 Pi05 checkpoints 都输出 14D absolute joint position，不是官方
AAC 评测的 7D EE action。我们不直接用 joint distance 代替 EE metric，而是复用
`manimux.embodiments.arm.yam.kinematics.YamKinematics`，把每个候选、每个时间步的左右关节分别转换成
grasp-site pose，再构造逐步 EE action。第一步是 `measured pose -> action[0]`，后续是
`action[t-1] -> action[t]`：

```text
YAM absolute joints
  -> existing YAM FK
  -> left/right incremental EE position[3] + rotation-vector[3] + gripper[1]
  -> 与当前 YAM 数据匹配的固定 min-max stats（仅 entropy / selector）
  -> 每只手独立计算官方 Gaussian(position) + Gaussian(rotation) + Bernoulli(gripper)
  -> 左右手 scalar entropy 取平均

平移增量在各机械臂 base frame 中计算；旋转采用官方左乘组合约定
`R_delta = R_target @ R_previous.T`。motion magnitude 使用未归一化的物理增量并逐臂计算：

||sum(delta position)|| + ||compose(delta rotation)|| + 0.2 * gripper_toggle

最后对左右臂 magnitude 取平均，再执行官方 motion floor 和
`max(entropy_elbow, motion_floor)`。
```

官方 entropy 和 selector 使用 checkpoint 的 normalized EE action。YAM checkpoint 只有
joint stats，因此 ManiMux 从与该 YAM 数据域匹配的 60 条示教中离线统计 EE 增量 min/max，
按官方 `2 * (x - min) / (max - min) - 1` 公式归一化。统计文件是
`manimux/policies/xpolicylab/norm_stats/yam_60ep_ee_increment.json`；更换本体、FK、
joint 顺序或训练数据域时必须更换，不能跨本体复用。

官方 `move_th=3.0` 属于 RoboCasa EE controller action 尺度，不能直接解释成 YAM 的米/弧度。
YAM 60 条示教的 16-step 双臂平均 motion magnitude 中位数是 `0.156`、75 分位是 `0.259`，
所以首轮配置使用 `motion_threshold: 0.2`。这是有记录的数据域标定值，不是论文在 YAM 上
验证过的阈值；真机实验仍需要记录实际选择分布。

## 分层位置

```text
GR00T N1.7 or Pi05 official backbone + denoise
  -> XPolicy multi-sample action-head batch (20 native joint chunks)
  -> ManiMux XPolicy bridge + shared YAM FK
  -> dual-arm mean of official EE entropy / motion / selector
  -> selected 2..16 step native joint chunk
  -> ManiMux XPolicyAdapter -> canonical ActionChunk
  -> AacInferenceStrategy -> Timeline -> Executor -> Safety -> Robot
```

- GR00T 在官方 action head 的 post-backbone feature 处扩 batch；Pi05 在官方 prefix KV cache
  建好后扩 batch。两者都只增加显式 multi-sample hook，denoise loop 不改。
- `manimux/policies/xpolicylab/aac.py` 复用共享 FK，并实现官方选择顺序与双臂平均。
- `scripts/datasets/compute_yam_aac_ee_stats.py` 从指定本体数据离线生成固定 EE 增量 stats；AAC config
  必须显式声明 `inference.aac.ee_stats_path`，缺失时拒绝启动。
- XPolicy WebSocket 声明 `aac` capability，避免普通模型误跑 AAC。
- ManiMux bridge 在选择后附加 `chunk_id / entropy_elbow / motion_floor`，Runtime 将这些
  标量写入 `plan_accepted` 事件，便于离线解释每次自适应决策；metadata 不参与动作解码。
- ManiMux `AacInferenceStrategy` 等待被选 chunk 结束后再取新 observation，保持官方同步
  cadence；推理期间真实机器人只能 hold，AAC 本身不是 RTC。
- `allow_short_horizon` 只允许 AAC 返回 `2..H` 步，默认 XPolicy adapter 仍要求固定 horizon。

## 当前验证边界

已验证：FK joint→逐步 EE 增量、base-frame/左乘旋转约定、固定 min-max、双臂平均、公式单测、
候选 selector、可变 horizon、XPolicy sampling dispatch、capability handshake、配置加载和
Python 编译。

已验证 Pi05 `N=20, H=50` 的真实 4090 forward：首次 JIT `7036.7 ms`，三次 warm round
trip 为 `530.1/509.9/515.4 ms`。未验证 GR00T N1.7 `N=20` 的真实 GPU forward、Pi05
长时间显存稳定性、YAM EE motion threshold、相机/CAN/真机行为。配置位于
`manimux/configs/experiments/<task>/{yam_groot,yam_pi05}_aac*.yaml`，在真机 gate 完成前不标记为硬件可运行。Pi05
逐文件审计见
[`docs/reproductions/aac-pi05.md`](reproductions/aac-pi05.md)。
