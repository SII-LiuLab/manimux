# ManiMux 真机推理实验设计

> 状态：研究协议草案。先在双臂 YAM 上完成指标校准和小规模 pilot，再冻结正式协议。

## 1. 立意

ManiMux 的第一目标是成为面向真机部署的 policy-free inference infrastructure：模型、推理
算法、本体和任务可以通过 config 组合，研究者不需要为每个 VLA 重写相机、异步请求、
chunk 执行、安全、记录和可视化。

实验部分服务于两个相互独立但互相支撑的贡献：

1. **Infra contribution**：同一套运行时接入多种 Policy、推理算法和机器人，并保存可比较的
   action lineage 与真机结果。
2. **Empirical contribution**：冻结 Policy 和真机条件，只替换推理算法，量化推理算法如何改变
   任务成功、运动连续性和 chunk 回放；进一步研究这种影响与 Policy、任务和微调数据量的交互。

如果实验结果支持，最终希望回答：

- 推理算法是否会在不改变 checkpoint 的情况下显著改变真机行为？
- 改善发生在任务成功、运动连续性，还是只发生在视觉观感？
- 某种方法是否只适合特定 Policy、延迟、horizon 或任务压力？
- 更好的推理是否能提高低数据量微调模型的实际可用性？
- 一次失败主要来自 Policy 原始 chunk、推理 Infra，还是机器人跟踪？

## 2. 实验边界

第一阶段只做真实双臂 YAM，不使用仿真结果代替真机结论。计划比较的 Policy 是 Pi05、XR-1、
LingBot-VLA2 和 Cosmos3，但只有完成匹配 YAM 数据的 post-training、动作语义审计和真实 rollout
后，模型才进入正式任务能力表。

实验遵守以下边界：

- 一次正式对照只改变一个变量。
- 相同对照使用同一 checkpoint、norm stats、任务指令、相机、起始布局和机器人控制包络。
- Runtime-only 方法留在 ManiMux；sampler/denoising hook 留在 XPolicy 的模型实现旁边。
- 不把有限 action chunk、服务连接成功或 rollout 正常结束写成任务成功。
- episode 是统计单位；100 Hz control ticks 不是独立样本。
- 人工任务标签是主结果；PRM 和运动学指标是辅助证据。

## 3. 研究问题与验证顺序

| 阶段 | 研究问题 | 固定项 | 改变项 | 进入下一阶段的条件 |
|---|---|---|---|---|
| E0 指标校准 | 指标能否识别已知回放和停顿？ | 一段参考轨迹 | 人工注入回拉、停顿、边界跳变 | 指标排序符合注入强度 |
| E1 Motivation | 推理算法本身影响多大？ | Pi05、checkpoint、任务、布局 | Default、ACT、IT-RTC、PAINT | 至少一个结果指标有稳定差异 |
| E2 Task stress | 差异在哪类任务上出现？ | Policy、算法 preset | 精度、连续性、反应、双臂任务 | 找到对推理敏感且可重复的任务 |
| E3 Policy interaction | 结论是否依赖 Policy？ | YAM 数据、任务和控制设置 | Pi05、XR-1、LingBot、Cosmos3 | 形成 universal 与 sampler-aware 两张表 |
| E4 Data efficiency | 推理是否改变微调效率？ | 训练 recipe、评测布局 | 10/15/20/25/30 demos、算法 | 得到数据量与算法的交互曲线 |
| E5 Embodiment | 结论是否跨本体？ | 筛选后的任务和方法 | 机器人本体 | 只复现最关键结论 |

不要从完整笛卡尔积开始。E1 和 E2 先淘汰没有信息量的算法与任务，再扩大 Policy 和数据量。

## 4. 最小实验单元

最小实验单元不是一条 rollout，而是一个 **matched rollout block**：

```text
同一个 task_id + operator-randomized layout_id + policy checkpoint
                    ↓
按随机顺序各运行一次待比较算法
                    ↓
按初始 top-camera 参考图恢复布局并完成重复
                    ↓
每条 episode 立即标注，整个 block 完成后再随机新布局
```

操作者可以在任务规定的工作区内自由摆放物体，但一次随机摆放只产生一个 `layout_id`。同一个
matched block 内的所有算法必须复原这一布局；不能给每个算法各自随便摆一次，否则布局差异会和
推理算法差异混在一起。算法执行顺序在每个 repeat 内独立随机化。

建议第一个 pilot：

```text
Policy      Pi05 当前 YAM 微调 checkpoint
Task        红球放入盒子
Algorithms  Default / ACT / IT-RTC / PAINT
Layouts     5 个 operator-randomized layout_id
Repeats     每个布局完整恢复并重复 2 次
Total       4 × 5 × 2 = 40 episodes
```

这 40 条只用于校准指标、估计方差和筛选方法，不作为最终显著性结论。正式实验的 episode 数根据
pilot 方差、置信区间或 sequential comparison 再决定。

### 4.1 YAM pilot v1 setting

第一轮 YAM pilot 使用下面的统一设置。时间是协议单位；`max_control_steps` 只是由控制频率换算出的实现值。

| 项目 | 冻结值 | 说明 |
|---|---:|---|
| Robot control | `100 Hz` | 所有算法共用同一底层控制频率 |
| Rollout upper bound | `60 s` | 成功、明确失败或 safety stop 可以提前结束，但必须记录结束原因 |
| ManiMux limit | `6000 ticks` | `60 s × 100 Hz`，不是 6000 个独立统计样本 |
| Video | `30 FPS` | 三路相机按同一 recording 配置保存 |
| Reset/Home | YAM zero Home | `Finish & Home` 后回零位；下一条 rollout 再进入任务固定 start pose |
| Task start pose | task/checkpoint matched | 同一 task 的所有 Policy/算法保持一致，不与物体 layout 随机化混用 |
| Layout | operator-randomized, block-matched | 用户自由摆放一次，同一 block 通过 top-camera 参考图恢复 |
| Trials | `10 / algorithm / task` | `5 layouts × 2 repeats`；pilot 后再依据方差扩充 |

数采和执行可能使用不同频率，不能直接比较裸 `step`。例如当前 YAM 数采是 `30 Hz`：若同样采用
`60 s` 上限，对应最多 `1800` 帧；只有数采本身也是 `100 Hz` 时，`5000 steps` 才表示 `50 s`。
训练数据保存实际完成长度，评测 config 统一保存 `timeout_s`、`control_hz` 和换算后的
`max_control_steps`。

## 5. Task 设计

任务按推理压力设计，不按物体名字堆数量。每个任务必须有可复位的初始布局、明确成功条件和可记录
的失败类型。

| 优先级 | Task | 主要推理压力 | 成功定义 | 附加客观量 |
|---|---|---|---|---|
| P0 | 红球放入盒子 | 基础 pick-place、现有 checkpoint 验证 | 球完全进入目标盒且未掉落 | 最终球/盒相对位置 |
| P1 | 试管或圆柱插架 | 精细对准、chunk 接缝 | 物体进入指定孔并保持 | 插入深度、碰撞次数、末端误差 |
| P1 | 开放容器/托盘运输 | 连续性、抖动、急停 | 到达目标区且内容物未超阈值洒出 | 剩余珠子或质量、最大倾角 |
| P1 | 目标移动后的恢复 | 观测过期、闭环反应 | 目标移动后仍完成抓取或放置 | 恢复时间、旧方向继续距离 |
| P2 | 双臂长物体搬运/交接 | 双臂同步和单臂异常 | 物体稳定进入目标位姿 | 左右臂不同步、物体姿态误差 |
| P3 | 毛巾折叠 | 柔性接触、长时序 | 角点和折叠区域达到阈值 | 角点误差、覆盖 IoU |

第一批正式任务建议使用“插架、开放容器运输、目标移动恢复”。毛巾折叠有价值，但初始布料状态和
成功定义难以标准化，应在刚体任务验证协议后再加入。开放容器先使用珠子或大米，不直接用液体。

受控扰动必须写入 task protocol，例如“EE 进入指定区域后，将目标沿 x 方向移动 5 cm”，并保存
扰动时间戳。不能由操作者临场决定。

## 6. 三项正式结果指标

主表只保留三项指标。延迟、hold、tracking error 和 CBA 分解仍计算，但只作为诊断列或附录，
避免用大量相关指标挑选对自己有利的结果。

### 6.1 Task Success Rate，TSR

- `success / failure / invalid` 三态；`invalid` 不计入分母并必须写原因。
- task-specific rubric 在采集前冻结。
- UI 的 task success 与 runtime 的正常退出分开保存。
- 正式视频复核时隐藏 Policy 和算法名称；至少抽取一部分由第二位评审者复标。

### 6.2 Human Smoothness Score，HSS

人工对完整视频打 1–5 分，使用固定锚点：

| 分数 | 定义 |
|---:|---|
| 1 | 危险跳变、长时间卡住或频繁明显回拉 |
| 2 | 多次停顿/回拉，明显影响任务执行 |
| 3 | 可完成动作，但 chunk 接缝或抖动清晰可见 |
| 4 | 只有少量轻微接缝，不影响任务 |
| 5 | 连续、自然，肉眼几乎看不到 chunk 切换 |

操作者可以在 rollout 后立即打分，作为工程反馈；论文主结果优先使用随机化、隐藏方法名的视频复评分。

### 6.3 自动运动指标暂不冻结

第一轮 pilot 只固定保存完整轨迹、chunk 边界、命令、真机状态、推理耗时和人工标签，不预先把
“反向、停顿或高频运动”定义成坏行为。叠衣服等任务可能先夹取、后摇动；同一种运动模式在不同
任务阶段具有不同语义。自动指标必须先在 matched rollout 上与人工标签和视频逐条对齐，再冻结公式。

## 7. 诊断量：不进入主排行榜

当 TSR、HSS 或 CRR 出现差异时，再用以下量解释原因：

| 诊断量 | 作用 |
|---|---|
| Policy Seam Error | raw chunk 开头与上一条执行轨迹的差距，判断 Policy 是否先产生接缝 |
| Infra Intervention Magnitude | committed chunk 与 raw chunk 的差距，判断 Infra 改了多少 |
| Executed Seam Error | measured state 在边界附近的跳变，判断真机最终承受多少 |
| Boundary-conditioned motion | 检查异常是否稳定集中在 chunk 边界，而不是把任务本身的停顿或反向判错 |
| Command/state alignment | 必要时解释底层执行是否偏离命令，不预设为主排行榜指标 |
| Inference Cost | round-trip latency、model evaluations、显存和 gap 次数，表示获得效果的代价 |

这些量目前只是候选分析方向，不写入主排行榜。先完成每种算法五次 pilot，再根据轨迹、视频、成功
标签、人工流畅度和 failure tags 检查哪些量真正对应人看到的问题。

## 8. 推理算法调参协议

规模化实验前，每种方法必须先得到一套对它自身公平的 preset。不能用论文默认参数跑到底，也不能在
正式 test rollout 上反复试到成功。

### 8.1 调参单位

主实验使用一套 `algorithm × policy × embodiment` preset。不同 Policy 的 horizon、sampler 和延迟
不同，因此允许有不同 preset；同一 Policy 下不能针对每个 test task 单独调参。

### 8.2 固定与可调参数

| 固定，不允许为某个算法改变 | 方法拥有，可以调 |
|---|---|
| checkpoint、norm stats、task instruction | temporal weight、query interval |
| 相机、起始布局分布、控制频率 | repaint/inversion 参数 |
| Robot velocity/acceleration limits | delay、prefix、execution horizon |
| action semantics、action dt 的训练语义 | entropy/variance threshold |
| 成功 rubric、评测预算 | 方法论文定义的 selector 参数 |

如果更改机器人速度或 `action_dt`，它必须成为一个独立实验变量，不能悄悄归入算法调参。

### 8.3 Dev/Test 隔离

1. 每种算法使用相同数量的 dev blocks 和同一组 dev layouts。
2. 先排除不安全、持续 gap 或无法稳定运行的配置。
3. 在剩余配置中优先选择 dev success 更高的 preset；成功相近时，再选择 HSS 更高、CRR 更低者。
4. 将最终 YAML、git revision 和参数 hash 冻结后，才进入 test layouts。
5. latency 和 model evaluations 作为成本同时报告；不能用无限计算换质量而不说明。

### 8.4 调参表

| Policy | Algorithm | Candidate preset | 关键参数 | Dev blocks | TSR | HSS | CRR | Cost | 决定 |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| Pi05 | ACT | `act-q3` | `query_interval=3` | TBD | TBD | TBD | TBD | TBD | TBD |
| Pi05 | IT-RTC | `rtc-default` | delay/prefix settings | TBD | TBD | TBD | TBD | TBD | TBD |
| Pi05 | PAINT | `paint-3n` | inversion/repaint settings | TBD | TBD | TBD | TBD | TBD | TBD |

## 9. Policy 与算法矩阵

不能假设所有方法都能原样作用于所有 Policy。正式结果拆成两张表：

### 9.1 Universal Runtime

| Policy | Default | ACT | YAM task checkpoint | 正式评测资格 |
|---|---:|---:|---|---|
| Pi05 | ✅ | ✅ | 已有低质量 20-demo checkpoint | Pilot only，需高质量数据 |
| XR-1 | ✅ | 待验证 | 待完成匹配训练 | 待定 |
| LingBot-VLA2 | ✅ | 待验证 | 待完成匹配训练 | 待定 |
| Cosmos3 | 待验证 | 待验证 | 尚无 YAM task checkpoint | 不进入当前主表 |

### 9.2 Sampler-aware Methods

| Policy | IT-RTC | PAINT | AAC | AutoHorizon | DVAC | 说明 |
|---|---:|---:|---:|---:|---:|---|
| Pi05 | ✅ | ✅ | ✅ | ✅ | ✅ | 当前主要算法研究载体 |
| XR-1 | 待审计 | 待审计 | 待审计 | 不默认支持 | 待审计 | 只有官方 sampler hook 对齐后才进入 |
| LingBot-VLA2 | 待审计 | 待审计 | 待审计 | 不默认支持 | 待审计 | 不用外观相似的后处理冒充论文算法 |
| Cosmos3 | 待审计 | 待审计 | 待审计 | 待审计 | 待审计 | 先完成官方推理和 YAM checkpoint |

## 10. 数据效率实验

建立至少 30–40 条高质量示教池，并生成三个 stratified subset seeds。每个 seed 内保持：

```text
10 ⊂ 15 ⊂ 20 ⊂ 25 ⊂ 30
```

训练时固定数据预处理、batch size、optimizer、epoch 数和 checkpoint 选择规则。只固定 gradient steps
会让小数据集被重复看到更多次，从而混淆“数据量”和“训练次数”。

第一轮只做：

```text
Pi05 × 5 data budgets × {Default, E1 最佳算法} × 2 tasks
```

确认存在稳定交互后，再在 XR-1 和 LingBot 上复现 `10 / 20 / 30` 三个关键点。

## 11. 统一 Experiment UI

最终使用一个 Viser UI 管理实验上下文、运行状态、episode 标注和结果查看。当前 Viewer 已有
Start/Resume、Pause/Hold、Finish & Home、相机、预测轨迹和 achieved trail；在此基础上增量扩展，不新建一套模型专属 UI。

### 11.1 页面结构

```text
┌ Experiment Setup ────────────────────────────────────────────┐
│ task / layout_seed / policy / checkpoint / algorithm preset │
│ current block / randomized order / config hash              │
├ Live Rollout ───────────────────────┬ Runtime Diagnostics ───┤
│ cameras + robot + predicted plan   │ latency / plan / gap   │
│ achieved EE trail                  │ safety / recorder      │
├ Post-rollout Annotation ────────────┴────────────────────────┤
│ success: yes / no / invalid                                │
│ smoothness: 1 2 3 4 5                                      │
│ failure tags + operator note                               │
├ Results ─────────────────────────────────────────────────────┤
│ rollout table / grouped summary / video review / PRM        │
└──────────────────────────────────────────────────────────────┘
```

### 11.2 分阶段实现

1. **UI V1：多-rollout 控制、标注与结果浏览**。用户启动一次 `manimux serve --config ...`；
   Viser 负责 Prepare、Start、Finish、人工标注和下一条 episode。
2. **UI V2：实验队列**。用户仍在终端部署 camera、model server 和 ManiMux service；UI 只读取
   service 发布的 config/episode metadata，并提示 matched block 的下一格实验。
3. **UI V3：结果审阅**。在同一个 Viewer 中浏览 episode、视频、人工标签、自动指标和 PRM，不增加
   camera/model/runtime 的网页启动器。

Viser 永远不下载模型、不启动 Policy Server、不切换 checkpoint，也不任意修改真机 config。保存路径
由 `manimux serve --config ...` 决定，UI 只显示路径并向已经结束的 episode 写 evaluation sidecar。
原 `manimux run` 继续作为单 episode CLI 使用。

当前 V1 已支持显式实验模式开关。OFF 用于普通部署与 debug，不要求 reward；ON 用于正式采集，
每条 finalized rollout 必须保存人工标签后才能 Prepare 下一条。task command 和 layout ID 在 Prepare
时冻结到该 rollout。完整操作与数据契约见 [experiment infrastructure](experiment-infra.md)。

## 12. Episode 评测数据契约

当前 `result.json.success` 表示 runtime 生命周期正常结束，不是任务成功。不能覆盖或重新解释它。
评测结果作为 sidecar 原子写入 episode：

```text
session-*/
  session-manifest.json          # config snapshot、config SHA256、git SHA
  rollout-001/
    meta.json                    # task、layout、algorithm、Policy Server fingerprint
    data.zarr/
      ticks/                     # state、reference、executor output、command
      plans/000000/
        canonical_raw/           # canonical policy chunk before strategy
        infra_output/            # chunk after inference strategy
        committed/               # final Timeline horizon
    videos/
      <camera>.mp4
      index.json                 # timestamps、drops、encoder error
    events.jsonl
    result.json                  # runtime lifecycle
    evaluation/
      human-label.json           # experiment mode ON 时由操作者保存
      automatic-metrics.json     # pilot 后冻结，当前不生成
      prm-v1.json                # 可选离线 PRM 结果
```

当前 `human-label.json`：

```json
{
  "task_result": "success",
  "smoothness_score": 4,
  "failure_tags": [],
  "operator_note": "minor seam near placement",
  "reviewer_id": "operator-01",
  "review_mode": "live",
  "label_schema": "human-label-v1",
  "created_at": "ISO-8601"
}
```

正式视频复核写新的 evaluator/version，不覆盖 live 标注。聚合器按 episode id 合并 runtime、manual、
automatic metrics 和 PRM；缺失值保持空，不猜测为失败或零分。

## 13. 预制结果表

### 13.1 Rollout-level Table

| Experiment | Block | Seed | Task | Policy | Algorithm | Valid | Success | HSS | Latency | Failure tag | Rollout |
|---|---|---:|---|---|---|---|---|---:|---:|---|---|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 13.2 Main Summary Table

| Task | Policy | Demos | Algorithm | Preset | N valid | TSR ↑ | HSS ↑ | Latency/Cost | Learned metric TBD |
|---|---|---:|---|---|---:|---:|---:|---:|---:|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

主表必须带置信区间或 bootstrap interval，不只写平均值。失败、invalid 和安全停止分别计数。

### 13.3 Attribution Table

| Rollout | Raw seam | Infra intervention | Executed seam | Stage/context | 解释 |
|---|---:|---:|---:|---|---|
| TBD | TBD | TBD | TBD | TBD | Policy / Infra / robot / unclear |

Attribution Table 只用于解释主结果，不用于事后挑选新的“胜负指标”。

## 14. PRM 与人工评测

PRM-as-a-Judge 作为完全离线模块读取视频，生成 progress curve、regression、stagnation 和 success
quality；它不参与控制，不修改 episode，也不替代 task success。

接入前先用人工标注的 milestone/回退视频验证 PRM 在 YAM 视角上的可靠性，并同时保存 raw 与
processed progress curve。可参考 [PRM-as-a-Judge](https://github.com/Yuheng2000/PRM-as-a-Judge)
及其[项目页](https://prm-as-a-judge.github.io/)。

## 15. 下一步执行清单

- [ ] 每种算法先跑 5 次 pilot，并保存轨迹、人工标签和视频证据。
- [ ] 为 Pi05 的 Default、ACT、IT-RTC、PAINT 建立等预算 dev tuning 表。
- [ ] 冻结四套 `algorithm × Pi05 × YAM` preset 和 config hash。
- [ ] 定义 5 个红球任务 layout seeds，随机化 40 条 pilot 的运行顺序。
- [x] 实现 Viewer 的 post-rollout `success / smoothness / tags` 标注面板。
- [x] 实现 rollout human-label sidecar writer。
- [x] 实现实验模式 ON/OFF、task/layout 冻结和 reward gating。
- [x] 保存 canonical raw、infra output、committed horizon 与异步多相机视频证据。
- [ ] 根据 pilot 数据提出候选指标，并用人工标签验证相关性和失效案例。
- [ ] 再冻结插架、开放容器运输和动态恢复三个正式 task protocol。
- [ ] 只把 E1/E2 中有信息量的方法扩展到更多 Policy 和数据量。

## 16. 参考

- Balasubramanian et al., [On the analysis of movement smoothness](https://pmc.ncbi.nlm.nih.gov/articles/PMC4674971/)
- PRM-as-a-Judge, [project](https://prm-as-a-judge.github.io/) and [code](https://github.com/Yuheng2000/PRM-as-a-Judge)
- [Beyond Binary Success: Sample-Efficient and Statistically Rigorous Robot Policy Comparison](https://arxiv.org/abs/2603.13616)
- [What Are We Actually Benchmarking in Robot Manipulation?](https://arxiv.org/abs/2606.04233)
