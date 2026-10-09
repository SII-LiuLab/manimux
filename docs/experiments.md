# ManiMux 实验登记：Settings 与逐任务结果

> 2026-09-29 · v0.1 草案。Pi05 抓瓶子 Serial K=16、RTC 最小源进度阈值16与β=5、双方 blend=0、100 Hz 插值 + Direct、不限速已写入 YAML；推理配置收敛到 `inference/aligned/`。配对入口已接通自动预热、手动Start前RESET、seed0与RTC自动延迟初值。保留用户手调的控制步数预算，不新增严格墙钟时限。仅完成源码、静态与配置检查，完整 setting 尚未冻结，未新增实测结果。最新进度见 §7.1。
> 本文是当前实验的登记入口；[实验记录](usage/records.md)说明对照原则与数据格式，[研究流程](usage/research.md)说明操作。

**2026-10-01 默认运行基线：主循环 `robot.control_hz=100`，RoboGUI `--render-hz=30`。**
当前已对齐的 Pi05 抓瓶子 Serial / RTC 均采用此基线；200 Hz 仅为已完成的性能诊断，RTC 入口已恢复 100 Hz。
RoboGUI 默认以 30 Hz 刷新最新状态，接收与绘制分线程，不补播连续旧状态；这不是模型动作频率或相机帧率。
后续新增或对齐的实验沿用该基线，偏离时显式登记；历史未对齐入口不因本次决定自动改写。

**2026-10-01 分支整合：**实验登记继续保存在 `docs/experiments.md`；Pi05 抓瓶子 Joint 30k Serial / RTC 与 pretrained RTC 入口显式配置 10 个布局 × 3 次、强制参考图的 study template。Free rollout 不受模板限制。RoboGUI 保留 30 Hz 最新状态显示，并增加历史 rollout 回放入口；pretrained recipe 补齐动作维度与环境数量。配置/代码整合不代表新增真机结果。

## 1. 本轮范围与记录方式

- **先在抓瓶子任务确定 Serial / RTC setting；Pi05 增加丢硬币任务。SAPolicy 不进入本轮。**
- 一个 task 一张主结果表。同一模型主行用 **Serial**，下面用 **+ RTC / + ManiMux / + PAINT** 等子行；子行表示替换推理方法，复用同一 checkpoint。
- 表中 `—` 表示未测或缺失，不能当成 0。部署入口待绑定不代表没有训练好；有入口也不代表已通过真机实验。
- 本文维护任务、setting 决定、结果及证据链接；**运行值仍由已有 experiment / policy / inference / executor YAML 持有**，公共参数沿用 `config:` 引用。
- 每个最终 setting 使用唯一 ID，绑定 task、policy、本体、checkpoint、动作表示和算法。确认后填配置路径及版本；每轮实际采用值以该 session 的 resolved config 和 backend identity 核对。
- 顺序：**核对现值 → 确定 Serial / RTC 候选 → 写入匹配 YAML → dev 验证 → 冻结 setting → test → 回填同一张表**。调参记录与正式 test 结果分开。

## 2. 主表列定义

一个 task 一张结果表，横向只放 **模型/推理方法、Human、Chunk seam、PRM 指标**。共同 setting、10 个布局 × 3 次的评测预算和输出路径统一写在表外，不再添加 N_task 或 Setting 列。以下是报告格式草案，评测口径随 setting 一起冻结。

**评分UI分两类（2026-09-29）：**通用情况使用`Completed / Not completed`；已定义的真机任务通过experiment的`evaluation.config`绑定自己的评分规则。当前支持通用/任务型二分类和任务型计数，不按任务prompt猜测规则。瓶子是计数任务之一，显示`Bottles in bin`的0–6选择，自动显示k/6及完整成功/失败。无效尝试通过独立`Invalid trial`标记排除，不当作第三种完成度。现有普通rollout免评分、experiment rollout可保存或跳过评分的流程保持；两类评分只决定评分页面内容。

| 分组 | 列 | 定义与报告方式 |
|---|---|---|
| Human | H-SR ↑ | `success / (success + failure)`，填 `n/N (%)`；任务判据先冻结 |
| Human（抓瓶子） | H-Score ↑ | 每条记最终入桶数k（0–6），得分k/6；主表填有效人工计数样本的平均得分×100%。H-SR另按六瓶全部入桶的episode比例计算 |
| Chunk seam | L-Mean / L-P95 / L-Max；R-Mean / R-P95 / R-Max ↓ | 左右手各三列，独立计算，单位 mm；Mean 与 P95 按 episode 等权汇总，Max 分别保留该手最坏交接 |
| PRM | M25 / M50 / M75 / SR / MP / PPL / CRA / STR / DRR / FNS / SQS | 全部 11 项横向列入各 task 主表；定义与适用样本见 §2.2 |

**跳变口径：**比较“新 committed plan 首点”与“旧 committed plan 在新 plan 起始时刻的采样值”，经同一本体 FK 转为末端位置，计算欧氏距离。它是**参考轨迹接缝**，不称为实测机械臂瞬时跳变。

**三项统计：**每条 rollout 的左右臂分别计算有效交接值的 Mean、P95 和 Max。P95 是第 95 百分位数（采用线性插值），不是 95% 置信区间。主表 Mean/P95 分别对各 rollout 的 Mean/P95 等权平均；主表 Max 取本组全部有效交接中的最大值，同时在明细记录对应 rollout 和边界。主表 P95 因此表示“每条 rollout 的 P95 的均值”，不把长 rollout 的更多交接当成更多独立实验；三列使用相同有效边界集，无有效值填 `—`。

报告前必须检查连续运行段、实际接手边界和旧计划是否耗尽。Serial 等待期间的末点 hold 单独标记，不与 RTC 有重叠的交接混称；没有有效边界填 `—`。现有画图脚本提供逐交接字段，统一跨算法有效性筛选和聚合仍需补齐。

任务、seam、PRM 的有效样本可能不同，证据中分别列 `N_task / N_seam / N_prm`。录像故障不自动抹掉已有人工任务结论；空 PRM 曲线不填零分。PRM 的模型版本、视角、采样时间、prompt、goal/reference 和后处理必须固定。全部 11 项 PRM 指标进入主表；M25/M50/M75/SR 用百分数，其他量保留无量纲小数。

有数据后报告区间：SR 报二项比例区间；连续量按 matched layout/block 重采样，避免把同一 rollout 的接缝或 control ticks 当成独立实验。暂不预设最优数字，不加粗空格或缺失值。

来源：[人工标签](../manimux/evaluation/manual.py)、[交接分析](../scripts/validation/plot_chunk_handoffs.py)、[PRM 指标](../PRM-as-a-Judge/eval/prm_judge/metrics.py)。

### 2.1 跳变分析：现有计算与出图

现有脚本直接计算每次候选交接的 `left_position_seam_mm`、`right_position_seam_mm`，并保存 chunk/request 编号、切换时刻和裁剪信息。主表的 Mean/P95/Max 需要按上述有效边界规则另行汇总；画图脚本本身尚未输出这些汇总值。当前不计算姿态角、速度、加速度或 jerk 接缝。

- **逐交接图** `chunk-00-to-01-eef-xyz.png`：左右臂 XYZ 共 6 子图，对照旧/新 chunk，标出切换线、过期前缀、重叠区域与 committed 接缝点。
- **总览图** `rollout-<label>-all-handoffs-overview[-NN].png`：每页最多 12 次交接。
- **多页 PDF** `rollout-<label>-all-handoffs.pdf`：收集全部逐交接图。
- **数值明细** `handoff-summary.csv` / `.json`：每次候选交接一条，供筛选、聚合及回溯。

另有 [终端诊断脚本](../scripts/validation/analyze_chunk_boundaries.py)，检查重规划间隔、跟踪差、chunk 首点相对上一 command 的差、blend 改变量及反向/回拉率。它不是上述末端位置接缝的统计器，不出图、不进主表；不能把其混合 group 的数值直接当成纯关节 rad 指标。

### 2.2 PRM：完整输出与主表取值

PRM 生成**按时间采样的原始/处理后进度曲线**，不是自动命名“抓取/搬运/放置”等语义阶段。当前工具的模型汇总有以下 11 项无量纲指标；阈值、采样与后处理尚需随 judge profile 冻结。

| 字段 | 含义 | 当前主表 |
|---|---|---|
| M25 / M50 / M75 ↑ | 达到 25% / 50% / 75% 最大预测进度的 episode 比例 | 已列入 |
| SR ↑ | 按进度阈值或指定标签汇总的成功率；默认进度阈值 0.99 | 已列入 PRM SR；不能替代 Human SR |
| MP ↑ | 最大预测进度；主表对有效 episode 等权平均 | **PRM MP** |
| PPL ↑ | 进度路径效率，`MP² / Σabs(Δprogress)` | 已列入 |
| CRA ↓ | 相对历史最高进度的平均回退损失，按 MP 归一化 | 已列入 |
| STR ↓ | 相邻采样进度变化小于阈值的比例，默认阈值 0.005 | 已列入 |
| DRR ↑ | 最大回退之后恢复了多少；汇总仅计发生真实回退的 episode | 已列入 |
| FNS ↑ | 失败样本距成功的接近程度；汇总仅计失败 episode | 已列入 |
| SQS ↑ | 由效率、回退与停滞合成的成功质量；汇总仅计成功 episode | 已列入 |

DRR/FNS/SQS 的样本集合不同，需同时保存各自 N。SR、FNS、SQS 的成功分类受 `success_source` 影响；缺失标签可能回退到进度阈值，不能据此补造人工结论。

回填时仅聚合该行明确选定的 task、setting、dev/test cohort；PRM manifest 的 `model` 分组键须区分不同模型与推理 setting。可读 `run_summary.json.metrics.groups` 中匹配该 task/setting 的组，不直接照搬可能跨 task 汇总的 leaderboard 或 Excel Model_Mean。FNS/SQS 由 `trace_summary` 派生，并不在普通逐 case `metrics` 中；`success_source=label` 时也不能照搬其中按阈值计算的 `metrics.SR`。没有适用样本的条件指标填 `—` 并记录 N=0，不照搬工具的零值。

- **数值/证据**：`per_case.jsonl`、每 case 的 `result_summary.json`、Dopamine 的 `pred_vllm.json`，以及 `run_params.json`、`discovery_manifest.json`、`run_summary.json`、`metrics.xlsx`、`report.md`。
- **启用可视化时**：Dopamine 的 `reward_vis.mp4` 展示视频与 progress/hop；`visualizations/cases/*.png` 对照原始/处理后进度曲线和成功阈值；`visualizations/report.html` 提供视频/曲线联动、指标与里程碑、回退恢复等图表；另有 CSV 索引及可视化 Markdown。PNG 依赖 matplotlib，生成失败须记录。
- **表格覆盖**：每 task 主表横向列出 Human SR、Mean/P95/Max 接缝和全部 11 项 PRM 指标；原始数据、逐 episode 明细和图在表外的路径登记中链接。当前没有新评测结果，主表仍为 `—`。

## 3. 逐任务结果表

表头简写：**H-Score** 为人工平均完成度（当前用于六瓶任务），**H-SR** 为整条任务人工成功率，**P-SR** 为 PRM 成功率；H-Score、H-SR、P-SR、M25/M50/M75 单位为 %。**L-Mean / L-P95 / L-Max** 为左手，**R-Mean / R-P95 / R-Max** 为右手，六列独立填写，单位 mm；后面的 M25 至 SQS 均为 PRM 指标。每个 task 的共同实验条件、样本数和证据在表外统一登记。

### Table 1. 抓瓶子放入箱子 · YAM · 首轮主任务

共同实验条件见 §4–5；每个模型 × 方法 10 个固定布局 × 每布局 3 次，共 30 条。进度见 §8，输出路径见 §9。

任务标识：`put_bottles`。2026-09-29用户确认PRM任务提示为 **`put bottles into the bin`**，采用 **Top单视角**；提示、视频来源、judge权重、采样间隔和目标图均作为评测输入参数，不写死到模型或指标函数。现有policy运行指令仍由experiment的`run.task`持有，评测提示单独存证，不能改写历史rollout中的实际指令。

**人工评分已确认：每条固定6个瓶子，每个最终进入桶内的瓶子计1/6。**记录原始计数`completed_count=k`（整数0–6）与任务目标数`target_count=6`，逐条得分为`k/6`；例如4瓶为4/6≈66.7%。只有k=6为完整`success`，0–5为完整任务的`failure`并保留部分得分。按结束时实际入桶的不同瓶子计数，不把同一个瓶子反复入桶累计计分，不增加未确认的终态保持秒数。

抓瓶子主表同时显示 **H-Score=100×mean(k/6)** 与 **H-SR=100×完整成功条数/有效成功失败条数**；前者是平均完成度，后者是六瓶全成功率。invalid与缺失标签分别留档；历史只有二分类的标签可按原rubric用于SR，但不得推测瓶子数或给缺失H-Score补零。RoboGUI计数输入及保存已接入代码：保持`human-label-v2`兼容，增加可选`evaluation`规则快照、`completed_count`、`target_count`及有效尝试的`score`。未选数量保持未标注，不能默认算0；invalid不生成score。保存以对应episode的`meta.json.evaluation`为准，矛盾数量/成功标签会报错。当前仅静态及配置检查，未运行RoboGUI/行为测试；批量H-Score汇总已接入task报告（§9.4），按匹配的rubric分别聚合；旧二分类标签不补计数。

两份Pi05 Serial/RTC入口共用[瓶子评分配置](../manimux/configs/evaluation/put_bottles.yaml)：`kind=count`、`task_id=put_bottles`、`rubric_id=put_bottles_6_v1`、`target_count=6`。未配置evaluation时使用`kind=binary`通用评分；定义其他任务可复用同一入口绑定新的规则，不在RoboGUI新增瓶子特判。已定义任务也可选择binary规则，显式给出完整任务/规则身份。规则在runtime初始化时规范化，冻结进meta与RoboGUI开始/状态/结束消息；session默认规则不会覆盖尚待评分的旧rollout。

**PRM采样：**先采用PRM-as-a-Judge官方工具默认`frame_interval=72`，保留传参覆盖。对当前30 FPS编码录像，等效每2.4秒采样一次（约0.417 Hz）；这不是72 FPS，也不是不依赖录像帧率的固定时间频率。官方指南建议初次运行保持默认，并给出短任务加密采样的36帧示例，未宣称72帧对所有任务最优。来源：[官方Advanced Guide](https://prm-as-a-judge.github.io/advanced.html)，本地CLI对应`--frame-interval`。同一任务比较组固定最终生效值，保存编码FPS与采样帧索引。

**PRM参数入口：**权重用`--prm-path`；采样用`--frame-interval 72`；任务提示用manifest的`task`；Top录像用manifest的`video`，省略腕部视频字段；目标图用manifest的`goal_image`（或显式`--goal-fallback`）。当前YAM的Top映射为`d405_front`，必须从各rollout的`videos/index.json`定位实际文件，不能猜文件名。现有Robo-Dopamine adapter会把唯一Top视频复用于模型的三个相机槽位，应记录为单视角输入。目标图路径可传参；有图时采用独立确认的成功状态，同任务比较组保持一致；无图时明确记录blank占位，不静默使用待评分轨迹末帧。统一task-dir入口的参数透传尚待§10实现，现有PRM CLI/manifest已支持上述参数。

| Model | H-Score ↑ | H-SR ↑ | L-Mean ↓ | L-P95 ↓ | L-Max ↓ | R-Mean ↓ | R-P95 ↓ | R-Max ↓ | M25 ↑ | M50 ↑ | M75 ↑ | P-SR ↑ | MP ↑ | PPL ↑ | CRA ↓ | STR ↓ | DRR ↑ | FNS ↑ | SQS ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Pi05 · Serial** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| + RTC | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| + ManiMux | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| + PAINT | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **XR1 · Serial** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| + RTC | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| + ManiMux | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **LingBot-VLA2 · Serial** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| + RTC | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **OpenWAM · Serial** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| + ManiMux | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **DP · Serial** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| + ManiMux | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

`+ ManiMux` 指现有普通异步调度，不能与 Serial 混为一个 baseline。OpenWAM / DP 当前接入未提供 RTC sampler hook，因此没有预填 RTC 结果行；未来实现后再增补。

Pi05 主组暂选 Joint 30k；Joint+EE 训练后 Joint 输出、同 checkpoint 的 EEF 输出应另加具名动作表示对照，不能合并进上述 Pi05 主组。现有 PAINT 瓶子入口使用 15k，不能直接作为 30k 的 `+ PAINT` 结果。

### Table 2. 丢硬币 · YAM · Pi05 补充任务

本 task 统一实验条件；每个模型 × 方法 10 个固定布局 × 每布局 3 次，共 30 条；具体布局和共同条件待冻结；输出路径见 §9。

已有 Pi05 coin 50k 产物和离线推理记录；当前没有对应 ManiMux task recipe / experiment。离线记录的 prompt 是 `put_the_coin_into_the_bin`，正式指令、投放目标、起始状态和成功判据待确认；不从口语“丢”推断必须抛掷。

| Model | H-SR ↑ | L-Mean ↓ | L-P95 ↓ | L-Max ↓ | R-Mean ↓ | R-P95 ↓ | R-Max ↓ | M25 ↑ | M50 ↑ | M75 ↑ | P-SR ↑ | MP ↑ | PPL ↑ | CRA ↓ | STR ↓ | DRR ↑ | FNS ↑ | SQS ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Pi05 · Serial** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| + RTC | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| + ManiMux | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| + PAINT | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

先在瓶子 dev 布局选定 Pi05 推理 preset，再核对 coin checkpoint 的动作/时间约定。相同约定可复用该 preset；不能在 coin test 上重新挑参数。必须改动时记录新版本及原因，并单列实验。

### Table 3. 组装螺丝刀 · YAM · 后续预留

本 task 统一实验条件；每个模型 × 方法 10 个固定布局 × 每布局 3 次，共 30 条；具体布局和共同条件待冻结；输出路径见 §9。

已有 15k recipe / RTC 入口；当前不启动该任务。各模型的专用 Serial 入口、task rubric 和预算待绑定。

| Model | H-SR ↑ | L-Mean ↓ | L-P95 ↓ | L-Max ↓ | R-Mean ↓ | R-P95 ↓ | R-Max ↓ | M25 ↑ | M50 ↑ | M75 ↑ | P-SR ↑ | MP ↑ | PPL ↑ | CRA ↓ | STR ↓ | DRR ↑ | FNS ↑ | SQS ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Pi05 · Serial** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| + RTC | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **XR1 · Serial** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| + RTC | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **LingBot-VLA2 · Serial** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| + RTC | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

### Table 4. 传球 · Tianji · 后续预留

本 task 统一实验条件；每个模型 × 方法 10 个固定布局 × 每布局 3 次，共 30 条；具体布局和共同条件待冻结；输出路径见 §9。

独立本体 setting，不套用 YAM 的控制包络；UMI-DP / XR1 的模型和任务条件须分别核对。XR1 × Tianji 当前 adapter 明确不接收 RTC condition，不预填其 RTC 结果行。

| Model | H-SR ↑ | L-Mean ↓ | L-P95 ↓ | L-Max ↓ | R-Mean ↓ | R-P95 ↓ | R-Max ↓ | M25 ↑ | M50 ↑ | M75 ↑ | P-SR ↑ | MP ↑ | PPL ↑ | CRA ↓ | STR ↓ | DRR ↑ | FNS ↑ | SQS ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **UMI-DP · Serial** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| + RTC | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **XR1 · Serial** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

### Table 5. 红球放入盒子 · YAM · 历史任务预留

本 task 统一实验条件；每个模型 × 方法 10 个固定布局 × 每布局 3 次，共 30 条；具体布局和共同条件待冻结；输出路径见 §9。

只登记匹配红球任务的 Pi05 1k；`pick_red_object` 目录还包含其他物体/指令，不把目录里的所有实验自动汇总为同一 task。

| Model | H-SR ↑ | L-Mean ↓ | L-P95 ↓ | L-Max ↓ | R-Mean ↓ | R-P95 ↓ | R-Max ↓ | M25 ↑ | M50 ↑ | M75 ↑ | P-SR ↑ | MP ↑ | PPL ↑ | CRA ↓ | STR ↓ | DRR ↑ | FNS ↑ | SQS ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Pi05 · Serial** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| + RTC | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

## 4. 抓瓶子：模型身份与当前入口

本节是 2026-09-28 的配置/产物清点，不是新推理验收。训练完成依据用户确认；所列本地文件及配置本轮仅只读查看，未重新校验全部权重 hash。

| 模型组 | 产物 / recipe | H；动作间隔 | 推理步数现值 | 当前缺口 |
|---|---|---|---|---|
| Pi05 Joint 30k | [joint-step30000](../manimux/configs/policy/pi05/yam/put-bottles/joint-step30000.yaml) | 50；1/30 s | 10 | Serial / RTC 执行层和blend已对齐；其余冻结项见下一节 |
| XR1 EE 30k | [finetune-put-bottles-step30000](../manimux/configs/policy/xiaomi-xr1/yam/finetune-put-bottles-step30000.yaml) | 30；1/30 s | 5；模型当前也固定 5 | 瓶子 Serial 入口待绑定 |
| LingBot-VLA2 30k | `checkpoints/finetuned/ziyang/lingbot-vla2-put-bottles-b64-step30000` | 模型 chunk_size=50；训练 30 FPS | 产物 config 为 10，部署生效值待绑定 | 产物已存在；瓶子 recipe / experiment 待绑定 |
| OpenWAM 30k | [finetune-put-bottles-step30000](../manimux/configs/policy/openwam/yam/finetune-put-bottles-step30000.yaml) | 32；1/30 s | 待核对 effective sampler | Serial 当前额外 blend=8，尚未统一 |
| DP EEF 100k | [eef-step100000](../manimux/configs/policy/dp/yam/put-bottles/eef-step100000.yaml) | 输出 6；1/30 s | 待核对 checkpoint 内嵌 cfg | Serial 当前 K=6，不可强行改成超出输出的 Pi05 K=16 |

Pi05 coin 产物：`checkpoints/finetuned/pi05/pi05_coin_20260928_v2_w32/50000`。已有离线记录为 H=50、denoise=10、14D absolute joint/gripper；训练采样 30 Hz。正式部署 recipe、动作转换和 dt 仍需绑定，本次未复验模型。

**对齐规则（2026-09-29 用户确认）：**同一模型的 Serial / RTC 使用同一 checkpoint、normalization、模型输入、H、动作 dt 和实际推理步数。不同模型不强求相同步数、H或算力，保留各自训练/推理约定并报告；不把算法自身的调度、裁剪差异当成需要抹平的问题。

配对实验还需固定推理设备、精度、batch 和服务负载，记录预热、随机种子以及实测 latency / model evaluations。相同 denoising 步数也不保证 RTC 与 Serial 的计算开销相同。

## 5. 第一张 setting 卡：Pi05 抓瓶子 Serial / RTC

来源：[Serial 30k](../manimux/configs/experiments/put_bottles/pi05/yam_pi05_serial_joint_step30000.yaml)、[RTC 30k](../manimux/configs/experiments/put_bottles/pi05/yam_pi05_rtc_joint_step30000.yaml)、[Serial preset](../manimux/configs/inference/aligned/yam_serial_chunk16.yaml)、[RTC preset](../manimux/configs/inference/aligned/yam_rtc.yaml)。

**推理配置归档规则：后续完成对齐的 YAML 统一放在 `manimux/configs/inference/aligned/`，对应实验通过 `inference.config` 引用。**本轮先绑定上述 Pi05 Joint 30k 两个入口；其余入口逐项对齐后再迁入。`aligned` 表示推理参数的对齐版本，不代表完整实验 setting 或真机验证已经完成。

**下表逐项标记确认状态；确认决定不等于 YAML 已更新或已通过实测。**候选 ID 为 `B-P05-S-v0` / `B-P05-R-v0`，先用于 dev，不进入正式结果。

| 参数 | 当前 Serial | 当前 RTC | 首轮建议 / 理由 |
|---|---|---|---|
| checkpoint / norm stats / 输入 | Joint 30k；三路 D405 | 相同 | 保持相同；冻结确切 task prompt 和相机映射 |
| `policy.action_dt_s`；H；`num_steps` | 1/30 s；50；10 | 相同 | 两者保持；不通过动作时间缩放调快机器人 |
| `policy_server.inference_seed` | 0 | 0 | 共同Joint30k recipe显式指定；Pi05模型加载及每条新rollout的RESET复位噪声序列，连续请求推进；已接通源码，未运行模型端回归 |
| `run.warmup_before_start` | true | true | Prepare后持续真实推理；用户随时手动Start，排空预热请求并等待RESET应答后才取新观测正式推理；没有固定时长或稳定门槛 |
| `robot.control_hz` | 100 Hz | 100 Hz | **两者已确认并写入 YAML**：30 Hz 模型动作点经 Timeline 线性插值，在 100 Hz 执行循环中更新目标命令 |
| RoboGUI `--render-hz` | 30 Hz | 30 Hz | 2026-10-01 确认；独立 RoboGUI CLI 默认值为30，无需额外传参；不改变模型动作时间轴 |
| executor / control profile | `direct` + `yam_control_unlimited` | 相同 | 2026-09-29 第1–5项：移除 Smooth recipe 引用，保留共同不限速 profile |
| arm 速度 / 加速度限制 | null；null | null；null | 不额外限速、限加速度；保留 1/30 s 模型动作时间轴 |
| gripper 速度 / 加速度 / 单独关爪速度限制 | null；null；null | 相同 | 三项全部关闭；SDK 的夹持力处理是另一层，见参数清单 |
| Smooth 低通 / tracking mode | 不执行 | 不执行 | 当前选 Direct；loader 补出的 Smooth 默认字典不参与命令生成 |
| `run.max_control_steps` | 12000 | 12000 | 用户手动调整，同一比较组保持相同；RUNNING控制tick包含推理等待/hold，Pause不计。100 Hz下名义120 s，实际时长可能更长；不增加严格墙钟timeout |
| 调度 | `algorithm=manimux`，`inference_schedule=serial` | `algorithm=rtc` | 保留方法差异；Serial 等旧前缀结束后才请求，RTC 与旧计划执行重叠 |
| Serial `chunk_policy_steps` | 16 | 不作为固定执行长度 | **Serial K=16，用户最新确认并写回**；模型仍输出H=50，再执行前16点 |
| RTC `min_execute_policy_steps` | 不适用 | 16，显式写在 `rtc` 下 | **RTC s_min=16，用户最新确认并写回**；保持源进度语义，不再借用 Serial 的 `chunk_policy_steps` 别名，含义见下一节 |
| RTC `initial_delay_policy_steps` | 不适用 | null，自动初值 | 用预热有效样本初始化；各采样分支首次成功耗时单独存证、不进校准。提前Start且没有校准样本时明确提示，初值为0，后续按真实提交更新 |
| RTC `delay_buffer_size` | 不适用 | 10 | 初始候选保留 10 |
| RTC `beta` | 不适用，无生效β | 5.0 | **用户已确认并写回** aligned RTC preset；Serial不使用RTC条件引导 |
| `blend_policy_steps` | 0 | 0 | **用户已确认并写回**；两组均无额外 chunk 交接混合，100 Hz 线性插值仍保留 |
| `action_start_mode` | `first_step_when_ready` | `drop_infer_latency` | 已按用户要求改名；仍按观测到可执行时刻的真实耗时裁剪，不改变方法语义 |
| `max_plan_age_s` | 2 s | 2 s | 两者保持；按观测年龄验收 |
| 请求 deadline / WS RPC 超时 | 2 s / 10 s | 相同 | 用户确认保持现状；2 s为响应验收deadline，不取消远端计算，RPC仍可能等待到10 s |
| 初始状态 / 录像 | 零位、夹爪开；30 FPS | 相同 | 冻结相同初始状态、摆放参考图和录像配置；视频时间以记录 timestamp 为准 |

2026-09-29最新决定：删除额外的chunk提交等待参数。解码结果通过检查后，在当前commit时刻直接生效；延迟统计和RTC裁剪只使用到提交时刻的真实已耗时长，不再人为附加等待时间。

用户最新决定保留 `run.max_control_steps`，由用户手动调整，同一比较组使用相同值；不新增严格wall-clock timeout。当前12000控制tick包含RUNNING期间的推理等待和hold，Pause不计；循环超时等会使实际时长超过名义120 s。每方法数量已明确为10个布局×3次=30条；抓瓶子人工rubric已定为每瓶1/6、六瓶全入桶才完整成功。PRM提示、Top视角及首轮72帧采样已登记在Table 1；实际judge/goal参数及其余profile须在运行时解析并留档，具体dev/test布局版本仍需存证。

第11项已通过真实CLI loader核对：两份入口解析后的 `policy`、`policy_server`、`robot`、`executor`、`sensors`、`camera_server`、`recording` 和task均相同，H=50、denoising steps=10、action dt=1/30 s；这只证明配置对齐，不代表模型或真机验证。第12项保留算法区别：Serial从返回后的第一点开始，RTC按 `drop_infer_latency` 裁掉过时点。

第13项的两个超时职责不同：`policy.timeout_s=2`从runtime构造请求时的调度时刻起算，worker开始处理前检查是否已过deadline，runtime收到响应后再检查worker记录的完成时刻。该时间包含排队、输入打包、通信、模型推理及返回处理，不是纯GPU forward，也不是机器人任务时长；后续动作适配/轨迹提交还有独立检查。到第2秒不会取消远端推理，Serial/RTC也不会仅因该deadline到达就解除in-flight。

`policy.options.request_timeout_s=10`从WebSocket发送完成后开始限制等待匹配回复的时间，不含之前的打包/发送耗时。等待超时后worker返回错误，runtime处理后才允许下一次请求；没有发送远端取消指令，旧计算仍可能占用模型锁，使后续请求排队。每次新请求有新的message_id，旧的晚到回复会被跳过，不会充当下一请求的结果。本批只解释，暂不改数值。

例：请求构造后0.3 s返回通过2 s时限检查；3 s才完成则被拒绝为`stale_or_expired_response`，不是在第2秒已经取消；发送后一直收不到匹配回复则约10 s后触发WS等待超时。等待期间先执行旧计划，耗尽后继续发送last_command。可能原因包括尚未预热的JIT冷编译、GPU/CPU争用、模型服务排队或网络异常；当前未据日志判定本机发生了哪一种。

另须区分：`max_plan_age_s=2`在提交时检查观测年龄，RTC的`drop_infer_latency`还会裁掉过期行。50个30 Hz点的末点位于49/30≈1.633 s，所以即使响应在2 s内完成，RTC也可能已无剩余动作而报`no_future_horizon`；Serial不做这种延迟裁剪。这不是WS超时。

第15项通过共同[模型recipe](../manimux/configs/policy/pi05/yam/put-bottles/joint-step30000.yaml)控制 `inference_seed: 0`，独立于训练/checkpoint命名的 `seed`。当前Pi05噪声为flow matching的初始高斯噪声；Prepare时RESET，自动预热结束后再次RESET并等待应答，正式请求从seed0序列开始，连续推理继续推进，不是每个chunk重复同一份noise。模型实际报告seed并纳入backend identity，旧服务缺少该字段时不会静默通过校验，需重启模型服务应用改动。正式运行中的Pause/Home/Resume不重置模型RNG。当前只接通Pi05；其他policy需逐后端接入和验证，不能据此宣称全模型seed已统一。

共同控制 profile 为 [yam_control_unlimited](../manimux/configs/embodiment/robot/yam_control_unlimited.yaml)，两组执行器改为 [DirectExecutor](../manimux/runtime/executors/direct.py)，不引用 Smooth recipe。手臂、夹爪速度/加速度和单独关爪速度全部为 null；Timeline 线性插值保留。Direct 不含 Smooth 的低通、±3.14位置裁剪及夹爪0–1裁剪；当前空 command_safety 映射下，上层全局±3.14检查也不启用，shape/finite检查仍保留。SDK自身限位、PD和夹持力处理继续生效。该默认 setting 先绑定本轮 Pi05 Joint 30k 配对入口，其他历史实验不自动迁移。共享值由 [loader](../manimux/cli.py) 注入并检查冲突。

Serial 对 RTC 比较的是整个推理执行方案，包含调度和条件采样两部分变化。若要单独归因于 RTC conditioning，后续增加同执行条件的异步无 conditioning 对照，即主表中的 `+ ManiMux`。

完整参数盘点见 [§11](#parameter-audit)：区分 YAML、Python 默认值、固定行为与当前未启用项；低通、hold、随机数、预热和 SDK 夹爪处理均不能只从 RTC YAML 判断。

## 6. RTC 到底能调哪些参数？

**Pi05 最小源进度阈值 16、β=5 已写入 aligned RTC preset；Serial / RTC 的额外 blend 均为0。**延迟窗口保留10；`initial_delay_policy_steps: null`选择预热自动初值，已替代本组临时手填4。H、动作间隔、推理步数以及 executor 不是同一模型下可以随 RTC 单独改变的自由项。

| 参数 | 控制什么 | 本轮处理 |
|---|---|---|
| `inference.rtc.min_execute_policy_steps` | 下一次推理的源 chunk 执行索引阈值；影响更新时机与重叠长度 | Pi05 最新确认 16；XR1 当前 15，处理其 setting 时再同步确认 |
| `inference.rtc.beta` | sampler 的条件引导强度参数；Pi05 中是时变 guidance scale 的上限，不是轨迹混合比例 | 本轮 Pi05 已改5；XR1 旧入口仍为4.25，待其 setting 对齐后绑定 aligned preset |
| `inference.rtc.initial_delay_policy_steps` | 新 rollout 延迟窗口的初始样本；不是人为增加推理延迟，也不是永久下限 | 本组Pi05为null，使用预热校准样本；无有效样本时明确提示并从0开始。XR1旧入口仍为8，尚未对齐；数字初值兼容旧配置 |
| `inference.rtc.delay_buffer_size` | 保留多少个已接受计划的延迟样本，预测取窗口最大值 | 当前两者 10；窗口越长，历史慢响应影响持续越久 |

所有步数单位为 **policy action step**，不是控制 tick，也不是 denoising iteration。30 Hz 下 16 个 step 的名义时间约为 0.533 s；不能据此声称 RTC 每 0.533 s 固定切换一次。Serial K也已统一为16，但参数语义不同。

额外 blending 和计划有效期也会影响结果，已在 §5 单列。普通调度的 `refill_threshold_s` / `inference_schedule` 不控制 RTC 请求时机，不能列作 RTC 的调参旋钮。

### “最小步数”的准确含义

令 H 为源输出 horizon，d 为当前延迟窗口最大值，m 为 `min_execute_policy_steps`：

```text
m 未指定时：m = max(1, floor(H / 2))
触发阈值：s_trigger = min(max(m, d), H - d)
当前进度：s = min(source_offset + trimmed_steps + timeline.cursor(now), H)
无在途请求且 s >= s_trigger 时，尝试发起下一次推理。
只有 d <= s <= H - d 时，才能构造 RTC overlap condition。
```

**这里计入的是当前 chunk 返回时已经发生的耗时，不是假定未来 latency 已知。**例如 H=50、m=16、30 Hz，观测到提交共耗时130 ms，实际裁掉4个过时源动作点；历史窗口最大延迟为180 ms，对应预测 d=6，所以触发阈值仍是16。提交后410 ms的一次检查中，cursor=12，源进度为4+12=16，可发起下一次推理。裁掉的4点没有由这个新 chunk 执行；下一请求使用的预测6也不是本次实际裁剪数。cursor按提交后的时间计算，不是机器人实际完成的动作数或100 Hz命令次数。

请求期间旧轨迹继续执行，直到新结果通过提交才切换chunk；达到阈值只触发下一次推理。Serial K=16 与 RTC m=16 **分别控制固定前缀长度和源进度触发阈值，数值相同不代表相同实际开环执行长度或推理频率**。只有 d≤16 且 H−d≥16 时，触发阈值才是16；更大延迟时仍按上面的完整公式计算。

### 延迟预测的实际代码口径

下面是当前Pi05配对入口的实现；预热交互见§7.2。

- `initial_delay_policy_steps: null`使用预热校准窗口：普通和conditioned分支各自第一次成功调用只单独记录耗时，其后最近10个成功样本除以action dt向上取整。Start的策略reset会装入这些样本；后续Pause/Home的策略reset也保留该预热初值。提前Start而尚无有效样本时窗口为`[0]`，UI和metadata明确标为未校准，不回退到隐藏的4。
- 每个计划成功提交后，以观测时刻和请求开始时刻中较早者到提交时刻的时长，除以 action dt **向上取整**。这包含观测年龄、请求准备、传输、模型推理和解码，而不只是 GPU forward；提交后没有额外等待。
- 将该样本加入窗口，超出容量则丢掉最旧值；下一次预测取窗口最大值。预热样本或无校准时的0会随正式成功提交逐渐被替换。拒绝或失败的响应不更新窗口。
- 未来延迟仍可能超过预测。返回后的实际裁剪使用真实已耗时长；若计划已过期或无剩余动作，则拒绝。等待期间旧计划耗尽，就按已统一的逻辑继续发送 last command。

若 `2d > H`，无法满足 overlap 条件；当前实现会记录 infeasible 并可能发出不带 condition 的请求。这样的 rollout 不能默默当作完整 conditioned RTC 成功验收；需要记录 conditioned 请求比例、延迟和 gap 原因。第一条无前序计划的请求本来就不带 condition。

源码：[RTC 调度与延迟统计](../manimux/runtime/rtc/strategy.py)、[condition mask](../manimux/runtime/rtc/mask.py)、[Serial 调度](../manimux/runtime/inference.py)、[Pi05 sampler](../XPolicyLab/policy/Pi_05/openpi/src/openpi/models/pi0.py)。

### 官方参数与本轮选择

核对来源为作者论文 [v2 Appendix A.5 / Table 4](https://arxiv.org/html/2506.07339v2#A1.SS5)，不是第三方实现的默认值。

| 参数 | 论文真机实验 | ManiMux Pi05 抓瓶子 |
|---|---|---|
| β | 5 | aligned YAML 为5，已确认并写回 |
| 最小执行步数 | 25 | **16，最新确认** |
| 延迟窗口 b | 10 | 当前 10，建议保持 |
| H | 50 | 当前 50，建议保持 |
| 去噪步数 n | 5 | 当前 10；是否改为 5 另行确定，Serial / RTC 必须成对修改 |
| 初始延迟 d_init | 算法输入；超参数表未给统一数值 | 本组null自动校准，无有效样本时显式从0开始；这是ManiMux约定，不能称为官方默认 |

β 是手动给定的常数上限；实际 guidance scale 随去噪时间按公式计算，不逐步手工指定、不在线学习 β。论文 A.2 通过消融选择 5，并指出过高 β 在少步去噪下可能造成不稳定。建议先使用该有出处的起点，不宣称它已是本平台各模型的最优值。

官方 Algorithm 1 用延迟历史窗口的最大值预测下一次延迟；soft mask 使用 Eq. 5 的指数衰减。作者公开仓库 [real-time-chunking-kinetix](https://github.com/Physical-Intelligence/real-time-chunking-kinetix) 是仿真实验代码，不能把其中的固定模拟延迟当作真机自适应调度的部署默认值。

公开代码亦核对到固定版本 `9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b`：[RealtimeMethodConfig](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b/src/eval_flow.py) 的 `max_guidance_weight=5.0`、`prefix_attention_schedule=exp`。该仿真评估显式扫描 delay / execution horizon；上表的真机数值来自论文，而非将 Kinetix 的 H=8 配置移植为 Pi05 setting。

## 7. 决定与下一步

下表保留各次登记时的状态；当前实现进度以§7.1和§7.2为准。

| 日期 | 决定 | 状态 |
|---|---|---|
| 2026-09-28 | 排除 SAPolicy；先抓瓶子；Pi05 增加丢硬币 | 用户已确认 |
| 2026-09-28 | 每 task 一张论文式主表；模型主行，算法子行 | 用户已确认 |
| 2026-09-28 | 主结果表不放 N_task、Setting；共同实验条件、数量和路径放在表外 | 用户已确认 |
| 2026-09-28 | 跳变增加 P95 和 Max；左右手各自独立列出 Mean/P95/Max，单位 mm | 指标已选；统计口径见 §2 |
| 2026-09-28 | Human + seam + PRM 三类分析；主表横向列出全部 11 项 PRM 指标 | 用户已确认完整列；具体 judge profile 待冻结 |
| 2026-09-28 | 每个 task 下，每个模型 × 推理方法采用 10 个固定布局，每布局 3 次，共 30 条；完整保存输出路径 | 用户最新决定，替代原 20/30 候选；具体布局待冻结，见 §9 |
| 2026-09-28 | 取消人工平滑度评分；人工只记录任务结论、失败标签和备注 | 用户已确认；新标签 v2 不再保存 HSS，历史标签保留 |
| 2026-09-28 | Pi05 抓瓶子 Serial / RTC 首轮候选，见 §5 | 执行频率已落地；其余候选参数仍待讨论 |
| 2026-09-28 | Pi05 抓瓶子 RTC `min_execute_policy_steps=12` | 历史决定；已被2026-09-29最新确认的16替代，完整 preset 未冻结 |
| 2026-09-28 | Pi05 抓瓶子 Joint 30k Serial / RTC 均采用 30 Hz 动作点 → 100 Hz 插值、平滑与命令下发 | 用户确认两者插值；Serial 原为 100 Hz，RTC 已改为 100 Hz；尚未实测下发频率 |
| 2026-09-28 | 默认 setting 关闭执行层手臂和夹爪全部速度/加速度限制，包括单独关爪限速；保持模型动作时间轴 | 用户已确认；Pi05 抓瓶子 Serial / RTC 已引用共同 unlimited profile；低通与 hold 语义仍待确定 |
| 2026-09-28 | 核对原作者论文；β 首轮建议由保留 9.1 改为 5 | 建议，待确认；官方值与本地现值分列 |
| 2026-09-29 | 停用 Smooth 执行整形，Pi05 抓瓶子 Serial / RTC 切 Direct，保留100 Hz插值与不限速profile | 第1–5项已写回配置；未运行真机；hold分支未改 |
| 2026-09-29 | 所有方法在RUNNING且没有可用reference时保持最后发送的command | 后续确认已实现共同hold分支；Pause/Home保持原实测复位行为；未运行行为测试或真机 |
| 2026-09-29 | blend全部0、β=5、seed默认0；同模型对齐、同任务严格时限、同本体底层一致、每次Home起步、相机冻结、增加预热与记录 | 历史反馈已登记；严格墙钟时限已被后续“保留手调控制步数预算”决定替代。其余按每批5项推进，具体实现状态见下表 |
| 2026-09-29 | 完成对齐的推理 YAML 放入 `inference/aligned/`；本轮 Pi05 双方 blend=0、RTC β=5，显式 min12 | 历史批次：第6–10项配置已写回，两个入口已切换引用；当次CLI loader含station的只读加载与diff检查通过。min12已被后续16决定替代；未运行模型或真机 |
| 2026-09-29 | action start模式改名为`drop_infer_latency`；RTC初始延迟由预热后实测自动估计 | 模式命名已同步代码和仓内YAML，裁剪行为不变；自动估计与第16项预热共同实现，当前仍用临时初值4 |
| 2026-09-29 | Pi05 抓瓶子 RTC `min_execute_policy_steps` 由12改16，保持源进度语义；Serial K12不变 | 历史批次：aligned RTC配置与30k入口RoboGUI标签已更新，当次CLI loader确认RTC min16、β5、blend0、`drop_infer_latency`及Serial K12；Serial K12随后被16替代。未运行行为测试、模型或真机 |
| 2026-09-29 | Pi05抓瓶子Serial也统一K16；RTC m16、β5保持，Serial无生效β | 用户最新决定；Serial preset改名为`aligned/yam_serial_chunk16.yaml`，模型仍输出50点；起点和裁剪保留各自算法语义 |
| 2026-09-29 | 取消严格墙钟时限需求，保留用户手动调整 `run.max_control_steps`；同一比较组使用相同值 | 第14项按最新反馈收敛；当前双方12000，不增加timeout代码，RUNNING等待/hold计步、Pause不计，实际时长可能超过120 s |
| 2026-09-29 | 第11–13项核对同模型配置、保留算法起点区别、解释2 s deadline与10 s RPC超时 | 真实CLI loader确认两份入口模型/硬件/相机/记录/task配置相同，H50、denoise10、dt1/30；超时数值暂不改。未运行模型或真机 |
| 2026-09-29 | Pi05显式inference_seed=0，每条新rollout复位噪声序列，同条连续请求推进 | Joint30k recipe与backend identity已接通Model/Policy.reset_rng；JAX恢复key，PyTorch使用独立generator生成初始noise。模型端回归与真机未运行；自动预热后的RESET待第16项 |
| 2026-09-29 | 用户确认超时保持现状，不再讨论删除 | 保留2 s响应验收、2 s轨迹年龄及10 s通信等待；不限制已接受chunk的后续执行时长，不修改运行逻辑 |
| 2026-09-29 | 预热持续到用户手动Start，不设置等待时长或自动稳定门槛 | Prepare后自动持续推理、显示耗时，由用户判断何时开始；RTC沿用实测延迟窗口，Start切换时隔离预热结果并RESET seed。需求已确定，代码待实现 |
| 2026-09-29 | 抓瓶子固定6瓶，逐瓶1/6；PRM提示与Top视角确定，其余评测输入参数化 | Table 1增加H-Score并保留H-SR；PRM task=`put bottles into the bin`，Top单视角，首轮采用官方工具默认frame_interval=72。权重、目标图等通过CLI/manifest传入并固定同组生效值；RoboGUI计数及统一入口尚待实现 |
| 2026-09-29 | 评分UI区分通用二分类与已定义真机任务 | 已接通evaluation.config、冻结metadata、RoboGUI按规则显示与v2兼容标签保存；瓶子配置绑定6瓶计数，其他未绑定任务保持通用二分类。仅静态/配置验证，RoboGUI及真机未运行，批量H-Score尚未接入report |

当前Pi05 Joint30k配对入口已接通Serial K=16、RTC m=16和β=5、双方blend=0、30 Hz动作点 → 100 Hz插值、Direct、不限速、共同last-command hold、seed0，以及第16项持续自动预热/自动延迟初值/Start前RESET。第13项继续保留2 s验收、2 s轨迹年龄及10 s等待回复；预算仍由用户手调控制步数，不新增严格墙钟时限。代码与配置已落地，模型端、RoboGUI交互和真机行为尚未运行验收；接下来用这两个候选进行dev验证，登记实际judge/goal与布局版本。完整setting及§10的三类指标统一评测仍未验收。

**实现范围限于`B-P05-S-v0` / `B-P05-R-v0`。**表中的Pi05 ManiMux/PAINT与XR1、LingBot-VLA2、OpenWAM、DP仍待各自配置、sampler seed/RESET及部署绑定对齐；不迁移历史入口，不将本批结果扩称为全模型完成。缺少瓶子recipe/experiment的模型也不能仅凭表格预留行启动。

每次回填结果附：setting 版本、准确入口/recipe、checkpoint 与 norm 身份、代码版本、resolved config、layout/block 与 rollout 清单、三类指标各自样本数、无效/缺失原因及完整分析路径。机器地址和设备序列号留在私有 station，权重、视频和 rollout 不提交到仓库。

### 7.1 2026-09-29 用户逐项反馈与批次进度

沿用对话中1–22的编号，**每轮只讨论/处理5项**。下面完整保存用户反馈，不把后续已确认选择重新问一遍，也不把候选实现当成已完成。

| 原编号 | 用户反馈 / 决定 | 实现状态 / 待办 |
|---|---|---|
| 1 | 保留30 Hz模型动作与100 Hz插值执行 | 两份Pi05瓶子入口已配置 |
| 2 | 手臂/夹爪全部不限速、不限加速度 | 共同unlimited profile已配置 |
| 3 | 额外execution整形也关闭，使用Direct | 本批两份入口已切Direct，低通不执行 |
| 4 | 随Direct统一臂爪执行 | 本批已移除Smooth路径；SDK自身处理仍存在 |
| 5 | 所有方法在无轨迹空档继续发送last_command | [edge.py](../manimux/runtime/edge.py)的共同RUNNING空档分支已接通。首次由连接后实测位置初始化，成功发送后才更新；Pause/Home仍按原语义处理。未运行行为测试 |
| 6 | blend全部设0；核查为何记忆中是0 | 本轮双方aligned preset均为0，两个Pi05入口已引用；原Serial为0，原RTC为4，Git提交d28c697及其父版本均已有4，并非本轮新增 |
| 7 | 忽略condition与实际轨迹差异议题 | 按用户要求不开展额外修改 |
| 8 | 解释原min12为何计入latency；未来latency不可预知；后续决定改为min16 | 已核对源进度=已裁剪前缀+提交后cursor；实际trim已知，下一请求延迟仍预测。公式与例子见§6；最新m=16已写回，保持同一源进度语义 |
| 9 | 详细解释RTC latency预测；预热后自动估计初值，不要求手填 | 本组已改null自动初值：预热各分支首次成功耗时单列，其后最近10个成功样本ceil取整、窗口max；无校准样本提前Start时明确从0开始。正式成功提交继续滚动更新，失败/拒绝不更新。见§6与§7.2 |
| 10 | β改成5 | aligned RTC preset已写为5.0，两类RTC提交blend均为0；未运行模型/真机 |
| 11 | 同一模型不同方法一致；不同模型不强求一致 | 真实CLI loader确认Pi05配对入口的policy、policy_server、robot、executor、sensors、camera_server、recording和task均相同；H50、denoise10、dt1/30。配置核对不等于模型/真机验证 |
| 12 | 起点/裁剪属于算法区别 | 保留Serial `first_step_when_ready`与RTC `drop_infer_latency`，不作为需要统一的缺陷 |
| 13 | 已解释并确认超时保持现状 | 保留2 s deadline、2 s max_plan_age与10 s通信等待；验收/提交之后的执行时长不计入2 s。超时不取消远端，请求返回前保持in-flight |
| 14 | 最新决定：不做严格墙钟上限，保留用户手调控制步数 | 保留run.max_control_steps，同一比较组使用相同值，当前双方12000；RUNNING等待/hold计步、Pause不计，实际时长可能超过120 s；不加timeout代码 |
| 15 | 所有默认seed0；解释与flow初始noise及config的关系 | Pi05已接通recipe.inference_seed=0→Model→Policy.reset_rng，metadata与backend identity带seed；Prepare及预热结束时RESET，Start等待应答后正式请求从seed0开始，连续infer推进；正式Pause/Home/Resume不复位。仅静态/配置检查，其他policy尚待各自接入 |
| 16 | Prepare后持续自动预热，用户观察稳定后手动Start；预热输出不得发送给robot | 两个入口均启用run.warmup_before_start；普通/RTC conditioned真实预热、耗时展示、请求隔离、RESET应答和自动延迟初值已接通，无固定等待时长/次数/稳定门槛。预热不进入正式轨迹、控制计数或录像；未运行行为验收，见§7.2 |
| 17 | 同本体的所有实验底层SDK设置保持一致 | 当前Pi05配对入口共用SDK与执行配置；已接通robot_backend存证，包括安装包/加载模块身份、连接后的PD/限位/夹爪力字段及有效夹爪开闭边界。SDK未提供的字段与校准过程来源明确记为unavailable，不能声称完整校准过程已捕获；未运行硬件 |
| 18 | 每次实验从Home开始 | Prepare仍按共同start_joints回零/张爪，start_duration_s=2；meta.initial_state记录连接后实测姿态及相对目标误差，formal_start_state记录首次RUNNING tick的新鲜实测状态，手动Home事件也记录状态证据。没有新增到位阈值或Start验位门控，实际到位尚未实测；不能用上一条结束Home代替本条起点证据 |
| 19 | 冻结当前相机/输入设置 | Pi05配对入口保留相同三路D405映射与输入约定，最终配置写入session-manifest；相机采集与录像时间戳的既有缺口按第20项保留 |
| 20 | 录像时间/采集时间缺口暂不处理 | 降低优先级，不扩展修复；保留已知限制说明 |
| 21 | 可记录补充的复现信息 | 已接通session的source_provenance/deployment_artifacts与resolved config，以及episode的policy_backend/robot_backend/initial_state/formal_start_state/inference_seed/warmup。policy_backend.model.execution报告实际后端、框架版本、参数dtype/device；compute/noise dtype未知时明示，不用参数dtype冒充全链路精度。源码dirty与内容摘要不是运行代码的原子快照；不声称已计算所有权重完整hash或捕获GPU负载。未生成本轮真机证据 |
| 22 | 说明那些未启用参数到底在哪 | 当前Pi05配对入口无需调这些项：普通异步refill默认值在[runtime/__init__.py:113](../manimux/runtime/__init__.py)，仅由[inference.py:139非Serial分支](../manimux/runtime/inference.py)消费；Smooth默认项在[smooth.py:638](../manimux/runtime/executors/smooth.py)，Direct不使用；异步解码默认项在[runtime/__init__.py:119](../manimux/runtime/__init__.py)，当前inline joint路径不启用。其他算法/执行器仍可能使用，保留对应代码 |

本批两份executor已切Direct，runtime的RUNNING无reference分支已统一。hold例子：上一目标1.0、实测0.8；无未来reference时Serial和RTC都继续发1.0。异步方法只要新计划及时接上就不会进入空档；首次请求、耗时突增、失败或拒绝仍可能触发。当前只做源码、静态与配置检查，没有启动SDK/相机/模型或运行行为测试。继续保留命令安全检查，不通过重置其历史绕过约束。

### 7.2 Start 前预热：Pi05 配对入口已接通

两个入口均配置`run.warmup_before_start: true`：**Prepare后自动持续预热，不增加Warmup按钮，不设固定等待时长、最少次数或自动稳定门槛；用户观察耗时后自行点击Start。预热模型输出不发送给机器人。**当前实现要求RoboGUI Start控制和inline动作解码，未扩展到其他模型的历史入口。

流程：`Prepare → 自动Warming up → 手动Start → 排空预热请求 → RESET应答 → 新观测的正式推理/执行`。预热结果直接显示在左侧Action chunks和中央3D轨迹，标注`Warmup preview · no execution`；左侧同时显示最近10次成功请求的E2E耗时曲线。右侧只保留简短状态与错误，不再堆列耗时数字；不自动判定稳定或解锁Start。

1. [PolicyWarmup](../manimux/runtime/warmup.py)使用独立adapter、负请求ID和所选策略的专用预热请求接口，复用正式模型后端及输入形状。解码动作仅作为带`warmup_preview`标记的RoboGUI预览发布：左侧展示最近两次推理的动作格及输出中的夹爪开闭预测，中央通过FK绘制最新左右臂预测路径。预览不提交Timeline、不推进正式请求/执行历史、不驱动executor，也不改变画面中的实测机器人姿态；预热期间机器人沿用实测姿态hold，Prepare/Home仍属于原有机器人生命周期。
2. Serial持续普通推理；RTC按成功请求交替覆盖ordinary/conditioned分支。condition使用当前实测姿态构造的占位轨迹和合法mask，只用于覆盖真实采样分支，绝不执行；它不另设手填延迟4。错误单独记录并显示，不算成功或校准样本。
3. 每分支首次**成功**调用耗时写入`first_inference_ms`，不加入校准窗口；UI的`recent_inference_ms`仍显示包含首次调用的最近10次成功耗时。随后成功请求进入容量10的`calibration_latency_ns`，包含观测年龄、请求准备、传输、采样和解码，不附加人为等待；RTC按action dt向上取整并取窗口max。`initial_delay_policy_steps: null`启用该初值；不要求等满窗口，提前Start且没有有效样本时明确记录`latency_calibrated=false`、样本数0，RTC从0估计开始。
4. Start立即停止新增预热请求，等待本地在飞结果，并在同一WS连接排空超时请求的晚到回复，再发RESET、等待应答；期间继续hold。RESET错误不会进入正式推理，UI显示原因，可Pause后再次Start重试或Finish。RESET只复位模型历史及seed0序列，保留编译缓存；策略reset重新装入预热校准窗口，预热动作和观测历史不继承。
5. RESET成功的tick仍只hold，下一tick重新读取状态与相机帧才开始正式推理。预热单列为`warmup_*`事件和`meta.warmup`，不进入正式计划计数、控制tick预算或录像；生命周期`wall_time_s`仍包括Prepare、预热与Pause，不能当作正式运动时长。正式运行后的Pause/Home/Resume不重新预热或重置模型RNG。
6. 正式RTC继续以成功提交更新延迟窗口；失败/拒绝不更新。预热减少冷启动影响，但Start仍有一次正常首推理，未来负载、网络或解码延迟仍可突增，因此保留last-command hold。没有样本的提前Start不会被误报为已完成延迟校准。
7. Start、Finish或预热退出时立即清除预览；同一条已结束试验的迟到预览不能恢复显示。Finish在Home及记录收尾之前停止本条rollout的policy transport worker，禁止新请求；下一次预热只在用户再次Prepare后开始。

接入点：[session.py](../manimux/session.py)管理Prepare，[warmup.py](../manimux/runtime/warmup.py)隔离预热，[edge.py](../manimux/runtime/edge.py)管理hold与正式切换，[worker.py](../manimux/policies/worker.py)提供异步RESET应答，[ws_client.py](../manimux/policies/xpolicylab/ws_client.py)排空同连接请求，[dashboard.py](../manimux/robogui/dashboard.py)展示状态。当前仅完成源码、静态与配置检查；没有运行pytest、模型推理、RoboGUI服务或真机验收。

跨Finish/Prepare的旧请求由[共享WS服务](../XPolicyLab/client_server/ws/model_server.py)的连接及RESET代际检查隔离。ManiMux连接显式启用`cancel_pending_on_disconnect`：Finish关闭连接后，尚未进入模型的排队/预处理请求被拒绝；已经进入模型的一次调用可能完成，但结果被丢弃，不再显示或执行。其他未启用此选项的客户端保持原有断线重试语义。RESET前接收但尚未进入模型的旧请求也会被拒绝，避免修改新试验的模型状态。应用本批改动需要重新启动runtime/RoboGUI及模型服务；这里只登记实现，没有代用户重启服务。

## 8. Agent 评测与进度登记

使用 [manimux-experiments skill](../.agents/skills/manimux-experiments/SKILL.md) 读取本文 setting、核对实际 rollout、执行离线分析并回填。Skill 不保存另一份参数默认值。

下面统计**选定 setting 对应的 cohort**，不自动纳入同名模型的历史试跑。`—` 表示尚未核对，不代表 0；当前目标是每组 10×3=30 个预定试验；具体布局冻结且逐条身份可核对后才填写完成比例。

| Setting | 阶段 | N_target | Attempts / finalized / partial | N human / seam / PRM | 下一项 | Cohort / evidence |
|---|---|---|---|---|---|---|
| B-P05-S-v0 | K16；seed0/预热已接通，待dev验收 | 30（10×3） | — | — | 验证Prepare→Start→Finish及记录，登记布局/judge证据 | 待生成，路径登记见 §9 |
| B-P05-R-v0 | m16、β5；自动延迟初值已接通，待dev验收 | 30（10×3） | — | — | 验证两采样分支、延迟窗口和首次正式请求，登记布局/judge证据 | 待生成，路径登记见 §9 |

每轮评测还需在对应 setting 的 evidence 中登记 **metric profile / judge profile**：指标版本与有效边界类别、judge 模型/权重身份、视角/时间采样、prompt/goal/reference、推理模式、后处理和样本清单。当前 PRM profile 尚未冻结，agent 应先准备输入和列出缺项，不能自行采用默认分数作为正式结果。

记录覆盖及缺口见 [记录字段说明](usage/records.md#saved-evidence)。Human、seam、PRM 分别检查可用性：某项不足时保留 `—` 和原因，其余项可以独立推进。

## 9. 评测数量与输出路径登记

### 9.1 数量

- 用户已确认：每个真机任务下，**每个模型 × 推理方法各采用 10 个固定布局 × 每布局 3 次 = 30 条**。布局用 `01`–`10`，重复编号用 `1`–`3`；所有比较组使用同一组布局、重置规则和任务判据，每次重复重新摆放。每个布局/重复 block 内交错或随机化方法顺序，不固定总是 Serial 先跑。
- 30 表示预定试验槽位，不是 30 次成功。失败计入已评测及人工 SR 分母；invalid、partial、未标注、PRM 失败分别留档，不自动补测凑分。若人工明确安排补测，保留原 attempt 和原因，在同一布局/重复槽位下登记新 attempt，不能覆盖原始记录或事后挑选成绩。有效样本数与已执行槽位数分别报告。
- 不因 seam/PRM 数据缺失而重跑直到得到足额好分数；分别报告 `N_task / N_seam / N_prm`，以及条件指标 `N_DRR / N_FNS / N_SQS`。同一 rollout 的多个 chunk 不算多条实验。

| Task | 本体 | N_target | 数量状态 | 输出登记状态 |
|---|---|---|---|---|
| put_bottles | YAM | 30（10×3） | 数量、6瓶及每瓶1/6判据已定；正式布局版本待存证 | Pi05 配置目标目录已登记；正式 cohort 待生成 |
| coin | YAM | 30（10×3） | 待冻结 | recipe / experiment 和输出路径待绑定 |
| assemble_screwdriver | YAM | 30（10×3） | 后续任务，待冻结 | 待登记 |
| pass_ball | Tianji | 30（10×3） | 后续任务，待冻结 | 待登记 |
| 红球放入盒子（匹配指令） | YAM | 30（10×3） | 历史任务预留，待冻结 | 待登记；历史试跑不自动计入 |

### 9.2 路径登记表

表中“配置目标目录”只说明 YAML 将数据写到哪里，不表示已收集正式结果；下列绝对路径按当前 checkout `/home/ubuntu/manimux` 解析。其他机器同时记录 host/storage 标识。保留原始目录，不搬动历史 rollout；每次新 session/分析产生后补上精确路径。

| Task / setting | 配置目标目录（run.output_dir） | 正式 cohort 清单 | Seam / 汇总分析目录 | PRM run 目录与报告 |
|---|---|---|---|---|
| put_bottles / B-P05-S-v0 | `/home/ubuntu/manimux/data/experiments/pi05-put-bottles-joint-step30000/serial` | 待生成 | 待生成 | 待生成 |
| put_bottles / B-P05-R-v0 | `/home/ubuntu/manimux/data/experiments/pi05-put-bottles-joint-step30000/rtc` | 待生成 | 待生成 | 待生成 |
| put_bottles / 其余模型与方法 | 各 setting 冻结时从实际 YAML 登记 | 待生成 | 待生成 | 待生成 |
| coin / Pi05 各方法 | 待绑定 | 待生成 | 待生成 | 待生成 |
| assemble_screwdriver / 各模型与方法 | 待逐 setting 登记 | 待生成 | 待生成 | 待生成 |
| pass_ball / 各模型与方法 | 待逐 setting 登记 | 待生成 | 待生成 | 待生成 |
| 红球放入盒子 / 各方法 | 待逐 setting 登记 | 待生成 | 待生成 | 待生成 |

同一个 task 的每个实际 setting/version/split 分别新增登记行，不能把多组输出合到一个“最新结果”路径。路径与证据在以下三处互相链接：**本节路径表 ↔ §8 进度行 ↔ §3 对应 task**。主结果表不再重复这些字段。

### 9.3 每条 rollout 的精确索引

Agent 每轮建立唯一分析目录，推荐 `data/analysis/<task>/<setting>/<split>/<analysis-id>/`，并在路径表填入它的绝对路径。其中 `cohort.jsonl` 每个尝试一行，枚举所有 session 下的完整 rollout 路径；不能只记录 `rollout-001`、通配符或 output root。重复评测时创建新分析版本，保留旧清单。

每行最少记录：

- `task`、`setting_id`、`split`、`host`、`attempt_id`、`layout_id`、`repeat_id`、block 和参考图路径/hash（缺失明确标记）；
- `episode_dir`、`session_manifest`、`meta`、`result`、`data_zarr`、`events` 的绝对路径；
- `videos_index`、各视角 `video_paths`、`human_label` 的绝对路径；
- `seam_output_dir`、`prm_manifest`、`prm_run_dir`、`prm_case_id`、逐 case 输出路径；
- final/partial 状态及 Human/seam/PRM 各自的有效性、缺失或排除原因。

不存在的文件写 `null` 与原因，不伪造路径；尚未分析的 seam/PRM 字段同样为 `null`。同时保存 `provenance.json`、`episode-metrics.json`，串起 cohort、metric/judge profile、各自分母、汇总数值、图和报告。以上是 **agent 的分析交付约定**，不是 recorder 当前已自动生成的文件；原始录像、轨迹和人工标签保持在原 episode 目录。

### 9.4 每个 task 的 HTML 结果报告（已实现）

入口与配置见 [报告使用说明](advanced/evaluation-reports.md)。每个task的本地索引固定为
`/home/ubuntu/manimux/data/evaluation_tasks/<task>/task.yaml`，报告固定输出到
`/home/ubuntu/manimux/data/analysis/<task>/reports/<UTC时间>-<唯一ID>/index.html`。
同task的`latest.json`保存最新完成版本的绝对HTML路径；旧版本保留。

报告包含Human SR/H-Score、切换后10/20/30步轨迹图与表、10×3布局槽位、逐条rollout与缺失原因，
同目录保留`summary.json`、`episode-metrics.json`、`episodes.csv`、`cohort.jsonl`、
`provenance.json`、`task.yaml`和SVG。统计先逐rollout计算，再按rollout等权汇总。
现场/视频人工评分分开，计数rubric分开，所有候选轨迹均明确标为非正式执行指标。

2026-09-29已生成以下task报告：`put_bottles`、`coin`、`stack_blocks`、
`stack_blocks_small`、`assemble_screwdriver`、`pass_ball`、`redball`。
瓶子报告仅选用2026-09-22的`rollout-002`，独立归入`legacy-pi05-rtc-20260922 / historical`，
该条旧标签有现场success，没有逐瓶计数；不填入B-P05-S/R-v0正式成绩。
其余报告当前无指定评测cohort，显示待评测；数采示范不自动计入评测。

## 10. 统一离线评测接口：实现方案

**状态：task报告入口已实现；完整数值评测编排仍未完成。** `python -m manimux.evaluation --task-dir <目录>` 现可读取显式task索引，汇总Human SR/H-Score及已有10/20/30步轨迹产物，生成独立HTML、图表和数据文件（§9.4）。正式执行边界筛选、PRM运行/导入、正式表格回填仍待接通；下述完整流程图是目标职责，不能据此认为报告CLI已经运行FK或PRM。

### 10.1 现有十个布局

本轮只读确认：`data/evaluation_layouts/put_bottles_into_the_bin/01.png`–`10.png` 全部存在，均为 640×480，十个文件 hash 不同。已有独立采集工具及 Viser 参考蒙版；这证明图片已保存，不代替对物体摆放、机位和正式任务条件的确认。

图库分组 `put_bottles_into_the_bin` 必须在后续 task 文件中显式对应实验 task `put_bottles`；图库名称和模型 Task command 都不自动充当规范 task ID。

**已接入代码，静态检查通过，尚未做 RoboGUI/真机验收：** 在参考布局选择 Task、编号 `01`–`10`，在 New rollout 选择 `Experiment repeat=1/2/3`，点击 `Prepare experiment rollout`。Prepare 一次冻结指令、位置、重复次数及参考图身份；取消独立可编辑 layout 框。位置/Task/重复次数在准备和运行期间锁定，蒙版透明度仍可调。参考图缺失时不提交 Prepare；普通 rollout 不要求参考图并清空正式实验身份。

冻结内容经 RoboGUI → session → runtime 同时写入 `meta.json` 与 RoboGUI 状态：`layout_id`、`repeat_id`、`reference_layout={task,path,sha256}`。图像解码和 hash 来自同一份文件字节；控制轮询不再读取图库。`Run / Recorded layout / repeat` 显示本次身份；重连后图库版本不匹配时隐藏参考并提示。独立尝试继续使用现有 episode 路径与 session/episode ID，不新增第二套 attempt UUID；重复选择同一位置/次数仍产生独立 rollout，后续盘点需显式处理重复。

这里冻结的是图片身份，不会复制或禁止独立采集工具覆盖原 PNG；正式采集期间不覆盖参考图。旧 rollout 缺少这些字段时保持未知，不补猜编号、次数或 hash。一次性 `manimux run` 若设置 `run.experiment_mode=true`，也必须显式提供上述身份字段，缺失或非法值会在创建机器人/传感器前报错。

### 10.2 唯一数据流与接口

```text
Task 目录：task.yaml（任务与数据索引；引用已有 runtime YAML 和 rollout 路径）
  └─ python -m manimux.evaluation --task-dir <task-directory>   [报告入口已实现；下列全流程分步接入]
       ├─ 核对 task / 模型 / 方法 / 布局 01–10 / repeat 1–3 / attempt
       ├─ human-label.json → Human SR
       ├─ data.zarr + 离线 FK + 有效交接证据 → L/R Mean、P95、Max + 图
       ├─ videos + task prompt + judge profile → 现有 PRM 后端 → 11 项指标
       └─ analysis/<analysis-id>/ → 逐条指标、任务汇总、图与表格
```

沿用已有 `manimux/evaluation/`，不另建平行的 `manimux.eval` 实现，不引入通用插件/注册框架。该接口只做离线评测，不打开机器人或相机，不改原始 episode。任务目录作为索引和报告入口，显式引用当前分散的 rollout 根目录；历史数据保持原位，不要求为了评测搬迁录像。

`task.yaml` 只持有机器需要读取的评测协议和来源：稳定 task ID、layout 库和参考图身份、10×3 计划、成功判据/超时、模型方法对应的已有配置与 raw roots、metric/judge profile 和 PRM 解释器。运行参数仍归已有 experiment/policy/inference/executor YAML 管理，实际值从 session resolved config 核对，不另抄一份 RTC/训练参数。实验 Markdown 展示已确认约定与生成的结果；指标计算不能依赖 agent 临时编写算式或解析自然语言猜参数。

### 10.3 三类指标各自读什么

- **Human**：读取真实 `evaluation/human-label.json`；成功/失败用于 SR，invalid/缺失分别计数，不能从 runtime `success` 推断任务结论。六瓶任务已接入人工计数与v2可选字段；后续批量report需校验冻结rubric并读取`completed_count`、`target_count`，另计算H-Score。计数必须由人保存，不能从PRM推测或给旧标签补造。
- **左/右跳变**：读取 `data.zarr/plans/*/committed` 的关节轨迹，用记录本体的离线 FK 得到末端 XYZ；比较新 chunk 首点与旧 chunk 在交接时刻的采样位置，得到 mm 距离。结合 `ticks.plan_id`、时间与分段事件确认实际使用过的交接；参考轨迹接缝不等于实测位移，也不从视频估计。左右手各自计算三个统计量，口径沿用 §2。
- **PRM**：读取已保存视频、评测任务提示和固定judge profile；抓瓶子当前选择Top单视角、`put bottles into the bin`、72帧间隔，均由任务配置/评测参数传入，不写死在adapter。其他任务若采用多视角，依据`videos/index.json`的时间戳检查/处理同步，不能直接假设相同帧号表示同一时刻。只生成manifest并调用现有PRM后端，复用其11项公式；PRM模型依赖留在独立环境，不导入硬件runtime。权重路径、目标图及其余profile参数保存最终解析值；初始布局参考图不自动充当任务完成目标图。

### 10.3.1 切换后 10 / 20 / 30 步相似度（已接入离线诊断）

2026-09-29：共享实现为 `manimux/evaluation/trajectory.py`，现有
`scripts/validation/plot_chunk_handoffs.py` 默认同时导出相似度；`--metrics-only`
跳过图片生成。`--robot-config` 现为必填，必须选择与记录的本体及工具一致的离线 FK
配置，不能默认用 Tianji 解释 YAM 轨迹。分析不连接硬件，输出禁止放在原始 episode 内。

```bash
envs/yam/.venv/bin/python scripts/validation/plot_chunk_handoffs.py \
  --episode <absolute-rollout-path> \
  --robot-config <matching-offline-assembly.yaml> \
  --out <new-analysis-directory> \
  --metrics-only --windows 10 20 30 --stationary-speed-mm-s 0.01
```

**数值口径**：比较相邻 accepted plans 的 `committed` 关节参考轨迹。以新轨迹
`start_time_ns` 为 t0，以其记录的 `committed.dt_ns` 为模型步长，取
`t0 + k*dt, k=1..N`，另取 k=0 作起点与速度差分锚点。两条轨迹在相同绝对时间上各按
自身 dt 插值关节，再用同一臂 FK 得到 XYZ；不按 chunk 行号直接比较，也不按100 Hz控制步
计数。30 Hz 时三个完整窗口约333 / 667 / 1000 ms，需要覆盖 N+1 个含锚点的采样位置。

每臂独立输出：`seam_mm`（起点）、`position_rmse_mm`（前N个未来点的位置RMSE）、
`endpoint_mm`（第N步误差）、`shape_rmse_mm`（各减去自身t0位置后的RMSE）、
`velocity_rmse_mm_s` 和 `velocity_cosine`（两段速度序列展开后的余弦）。任一轨迹的
窗口RMS速度不超过CLI阈值时，cosine记null并记录原因，其余指标仍保留。XYZ指标不包含
末端姿态或夹爪一致性。重叠不足、非有限输入、缺组等均记录缺失原因，不外推或补点；JSON
为null，CSV为空。窗口的seam也只在该完整窗口有效时输出，不取代已有独立接缝指标。

输出 `overlap-metrics.json` / `.csv`（逐交接、逐臂、逐窗口）及
`overlap-summary.json`（单rollout统计）。每个窗口记录候选数、数值有效数、每个指标的有效数
和缺失原因；另外提供三个窗口共同有效交接集合的汇总，防止窗口越长而样本集合变化混淆趋势。
报告保存plan/request身份、dt、选中参考数组hash、指标源码hash、FK配置hash与输入metadata hash；
FK配置hash不覆盖其引用资产，不自动证明历史运动学一致性。

**证据边界**：本入口仍是候选诊断，逐条标 `execution_eligibility=not_checked`，总报告
`formal_evaluation=false`。它不确认ticks实际接管、暂停/Home分段、hold或未执行计划；
正式筛选由§10.4接入后完成。Serial无未来重叠不能补成常值去凑得分。数值反映计划一致性，
不能据此宣称实测轨迹平稳；RTC条件前缀也可能提高该数值。跨rollout正式汇总仍待实现。

历史验证输入：Pi05 RTC `session-20260922T072222Z-f58fde5b/rollout-002`（旧setting）。
69个plan形成68组相邻候选；一组旧轨迹不覆盖新起点，10/20/30步的数值可计算交接数为
67/67/66。该记录不能计入当前aligned setting的正式成绩。
产物：`data/analysis/pi05-joint-20260922-rollout-002-overlap-windows-20260929/`。

### 10.4 最小实现范围与验收

1. **记录身份与分段证据**：layout、repeat、参考图身份已接入现有 RoboGUI/session/runtime metadata 路径，尚待交互验收；有时间戳的暂停/恢复分段仍待实现，需结合已保存 tick plan ID 判断交接。旧数据缺证据则标未知，不猜 repeat 或拼接跨段边界。
2. **任务读取和数值评测**：在现有 `manimux/evaluation/` 增加统一入口与读取/计算模块，盘点所有尝试和 10×3 槽位、汇总 Human SR，规范有效接缝分类及左右 Mean/P95/Max。提取现有画图脚本中可复用的 FK/绘图逻辑，让旧 CLI 调用同一实现，不复制第二套 seam 公式。
3. **PRM 适配**：增加薄适配模块，完成视频质量/对齐检查、manifest 导出、独立环境调用与结果导入；保存实际 PRM run/case 路径和 profile 身份。读取已有匹配结果可以复用；profile 或视频改变则不得复用旧分数。
4. **统一报告（HTML部分已实现，见§9.4）**：当前生成HTML、JSON/CSV、cohort、provenance与SVG；正式指标筛选、`table.md`及Markdown回填仍待完成。目标是由一个 report 模块生成 `cohort.jsonl`、`episode-metrics.json`、`summary.json`、`table.md` 和图。每个 metric 带有效数量、来源及缺失原因；写入新 analysis 目录，精确索引原始证据。固定 task/model/method 行映射后更新 Markdown 对应表格及表外进度/路径，不重写人工说明。

离线验收覆盖重复/缺失槽位、Human v1/v2、已知 FK 接缝、暂停/hold/未执行计划排除、空条件指标、PRM profile 不匹配与缺视频。先用已知轨迹和已有记录验证数字及路径，再单独做真实 PRM 推理验收；离线检查不代表真机任务成功。实现后 skill 只负责选择协议、调用此入口和检查结果，不能保留与入口竞争的临时计算流程。

<a id="parameter-audit"></a>

## 11. Pi05 抓瓶子 Serial / RTC 参数盘点

**以下保留2026-09-28切Direct之前的审计快照**，范围为 §5 的 Joint 30k 两个入口，包含实际 CLI 的 station 合并、Python 默认值和安装版 SDK；不是全模型参数穷举或正在运行进程的状态证明。2026-09-29当前配置与决定以§5和§7.1为准；表中的Smooth8 Hz等不再代表当前执行路径。审计未运行模型、测试或硬件。

该快照发现了低通、hold、blend/β、随机数、预热与时间预算等问题；最新反馈及处理状态已逐项记在§7.1。后续阅读必须区分审计时存在的处理和已经切换为Direct后的实际行为。

### 11.1 两处行为是否属于 bug？

- **continuous 夹爪绕过手臂低通：是现有设计，不自动构成 bug。**它直接取插值后的夹爪参考，走独立限速后覆盖手臂滤波结果。现在独立限速也已关闭；是否给夹爪增加滤波是新的执行选择，须双方一致。
- **审计时计划耗尽后的 hold 不一致**：原Serial保持上一条command，原RTC保持measured state，可能产生目标回撤。2026-09-29用户后续确认后，RUNNING无reference已统一保持last_command；Pause/Home语义保留。源码/静态检查不等于已通过行为测试或证实真机抖动。
- **其他隐藏值不一概视作 bug。**例如 mask 公式、PD、夹爪力处理可以是固定实现；随机数不随 rollout reset、无显式预热则是复现实验口径的缺口，应显式决定并记录。

<details>
<summary>展开：A. 已生效的运行、调度与执行参数</summary>

当前共同来源：[Serial](../manimux/configs/experiments/put_bottles/pi05/yam_pi05_serial_joint_step30000.yaml)、[RTC](../manimux/configs/experiments/put_bottles/pi05/yam_pi05_rtc_joint_step30000.yaml)、[unlimited profile](../manimux/configs/embodiment/robot/yam_control_unlimited.yaml)、[Smooth recipe](../manimux/configs/executor/yam_smooth_control_profile.yaml)。`默认`表示 Python 补齐，不表示 YAML 已显式声明。

| 字段 / 行为 | Serial | RTC | 归属与解释 |
|---|---|---|---|
| `action_dt_s` / `horizon_policy_steps` / `control_hz` | 1/30 s / 50 / 100 Hz | 相同 | experiment；保持模型原时间轴 |
| `algorithm` / `inference_schedule` | manimux / serial | rtc / 默认 deadline | RTC 不使用普通 schedule 控制请求时机 |
| `chunk_policy_steps` | 12 → `max_chunk_policy_steps` | 12 → `rtc.min_execute_policy_steps` | inference；别名相同，语义不同 |
| `rtc.initial_delay_policy_steps` / `delay_buffer_size` / `beta` | 不生效 | 4 / 10 / 9.1 | RTC recipe；β=5 仍只是待确认建议 |
| `blend_policy_steps` | 0 | 4 | inference；与插值、低通是三种不同处理 |
| `action_start_mode` | first_step_when_ready | skip_elapsed_steps | inference |
| `max_plan_age_s` | 2 s | 相同 | 按 observation time 计算 |
| `policy.timeout_s` / `startup_timeout_s` | 2 / 30 s | 相同 | 接受响应 deadline / 本地 worker 启动等待 |
| `policy.options.request_timeout_s` | 10 s | 相同 | WS RPC 超时；2 s deadline 不会取消远端推理 |
| `policy.action_decoding` | inline | 默认 inline | 当前 joint adapter 在控制循环中解码 |
| `policy.adapter.allow_short_horizon` | 默认 false | 默认 false | 要求输出完整 H=50 |
| `executor.type` / `smooth.tracking_mode` | smooth / 默认 legacy | 相同 | 当前不是 braking 跟踪 |
| `smooth.cutoff_hz` | 8 Hz | 相同 | 手臂一阶低通；100 Hz 下 α≈0.3345 |
| `smooth.max_velocity` / `max_acceleration` | null / null | 相同 | 已按用户决定关闭 |
| `smooth.mode` / `max_step_dt_s` | 默认 per_joint / null | 相同 | 限速模式与 dt 上限；当前无有限速率整形 |
| `smooth.position_limit_abs` | 3.14 rad | 相同 | 仍有位置裁剪及安全检查 |
| `smooth.gripper.mode` / `closed_value` / `open_value` | continuous / 0 / 1 | 相同 | 夹爪目标为归一化开度，不是 rad |
| `smooth.gripper.group_indices` | left_arm:6 / right_arm:6 | 相同 | 从 assembly 的 6 arm + 1 gripper 布局注入 |
| `gripper.max_velocity` / `max_acceleration` / `max_closing_velocity` | null / null / null | 相同 | 已全部关闭；含单独关爪速度 |
| `release_guard` / `grasp_guard` | 默认 null / null | 相同 | 均未启用 |
| `executor.command_safety` | 上下界、速度、加速度映射均空 | 相同 | 独立逐关节 envelope 未配置；shape/finite/统一位置边界仍检查 |
| `run.max_control_steps` | 12000 | 相同 | RUNNING ticks，含等待、不含 Pause；标称 120 s，非严格墙钟上限 |
| 初始姿态 / `start_duration_s` | 双臂 [0,0,0,0,0,0,1] / 2 s | 相同 | experiment；归零、夹爪张开 |
| `move_to_start_on_connect` / `home_on_close` / `execute` | true / true / true | 相同 | 生命周期与实际下发 |
| `home_duration_s` / `home_gripper_release_duration_s` | 2 s / 默认 1 s | 相同 | Home 两阶段：手臂回零，再张爪 |
| assembly `hardware.stale_timeout_s` | 0.2 s | 相同 | YAM 返回主机读缓存时间，不能据此证明电机反馈时效 |

实现来源：[参数补齐](../manimux/runtime/__init__.py)、[policy 默认值](../manimux/policies/base.py)、[Smooth](../manimux/runtime/executors/smooth.py)、[SafetyGuard](../manimux/runtime/safety.py)、[YAM 生命周期](../manimux/embodiments/robot/yam/robot.py)。

</details>

<details>
<summary>展开：B. 固定在代码里的调度、插值和等待规则</summary>

- **RTC 触发及延迟**：§6 给出完整公式。min12 统计源 chunk 进度，包含已裁掉的点；延迟为接受计划端到端年龄除以 action dt 后向上取整；预测固定取最近 10 个已接受样本的最大值。首请求无 condition；不满足 `d≤s≤H−d` 时可发无条件请求，不能把整条 rollout 自动视为全程 conditioned RTC。
- **RTC condition 的来源**：使用解码后、Timeline blend 前的轨迹；后续 blend 和 Smooth 改变实际目标，因此 condition 不等于最终 100 Hz command。[strategy.py](../manimux/runtime/rtc/strategy.py)
- **mask**：前 d 点为 1，其余重叠区域固定指数公式 `c×expm1(c)/expm1(1)`，没有另一个可调衰减系数。[mask.py](../manimux/runtime/rtc/mask.py)
- **Timeline**：固定线性插值；RTC 过期裁剪用 ceil。blend4 以最后发出的 command 为锚点，新轨迹权重依次 1/4、1/2、3/4、1；有 condition 的请求也混合。Serial 前12点加最后一周期 hold，标称 12/30 s；RTC 普通轨迹跨度为 `(N−1)×dt`。[timeline.py](../manimux/runtime/timeline.py)
- **手臂低通**：`α=dt/(1/(2πfc)+dt)`；legacy 先低通，再速率整形，再位置裁剪；continuous 夹爪独立取 reference，覆盖上述结果。Smooth horizon 固定2点，legacy 只用首点。[smooth.py](../manimux/runtime/executors/smooth.py)
- **请求与超时**：最多一个在途请求；2 s 只拒绝迟到响应，在途标志仍等响应回来才清除，所以可能等到 10 s transport 超时。worker 的请求/响应队列容量固定1，latest-wins，进程用 spawn。[worker.py](../manimux/policies/worker.py)
- **循环时钟与 hold**：相机读取、状态读取、inline decode、发送和记录在同一个循环。超时后从完成时刻再等一个 dt，不补发追赶。Serial 无未来轨迹时保持上一 command；legacy RTC 保持 measured state。Pause/Home 清轨迹和策略、丢弃旧响应，RTC 延迟恢复初始4；不同时重置远端模型 RNG。[edge.py](../manimux/runtime/edge.py)
- **标签问题**：RTC RoboGUI label 当前仍为 `RTC max 12`，实际是 minimum source-progress trigger，标签待修正；不能据标签解释算法。

</details>

<details>
<summary>展开：C. Pi05 模型、采样、随机数与预处理</summary>

模型拥有这些参数，不能写进 executor 代替模型设置。来源：[recipe](../manimux/configs/policy/pi05/yam/put-bottles/joint-step30000.yaml)、[Pi_05 adapter](../XPolicyLab/policy/Pi_05/model.py)、[OpenPI 配置](../XPolicyLab/policy/Pi_05/openpi/src/openpi/training/config.py)、[模型选择/加载](../XPolicyLab/policy/Pi_05/openpi/src/openpi/policies/policy_config.py)。

| 参数 / 约定 | 当前值及来源 |
|---|---|
| checkpoint / stats | recipe 的 Joint 30k 与同产物 assets；source/variant 为声明身份，本次未复核权重 hash |
| `train_config_name` / `observation_profile` | pi05_yam / yam_native |
| `action_type` / `env_cfg_type` | joint / yam_dual；输出14维，每侧6关节+1夹爪 |
| `action_horizon` / `num_steps` | 50 / 10，真实传入 sampler；Serial 仍生成50行再截前12 |
| sampler / `num_samples` / noise | Euler、从 t=1 到0，dt=−1/10；单样本；标准高斯 `[1,50,32]` |
| RTC guidance | beta 实际9.1；固定 scale=`min(beta,(t²+(1−t)²)/max(t(1−t),eps))`，clean estimate 的 VJP |
| RNG | Policy 默认 `jax.random.key(0)`；每 infer split；Model.reset 不重置 RNG，recipe 无生效的每 rollout seed |
| warmup | 追踪到的 launcher → Model 路径无显式 warmup；普通和 conditioned 请求需分别确定预热规则 |
| 后端 / 精度 | 当前产物有 params、无 model.safetensors，选择 JAX；参数载入 bfloat16，不代表图像、noise 等全链路 BF16 |
| GPU / JAX 环境 | recipe 未绑定设备；启动环境/JAX 决定；ManiMux launcher 未固定显存预分配或设备环境变量 |
| 架构默认 | pi05=true；action_dim32；gemma_2b + gemma_300m；max_token_len200；discrete_state_input=true |
| 真实 prompt | 来自 experiment/RoboGUI 的任务指令 `Put the bottles into the bin.`；recipe.task_name 不是实际 prompt |
| 文本变换 | 去两端空白，下划线/换行换空格；不转小写；state 256 bins；超出200 tokens截断 |
| 图像 | 三路 RGB uint8 → 保比例黑边 resize224×224（linear）→ float32 [−1,1]；推理无训练增强 |
| `use_delta_joint_actions` | true；12个arm关节相对观测state，两个夹爪保持绝对；输出加回state，最终发送绝对joint |
| normalization / padding | q01/q99 quantile，分母epsilon=1e−6；14维pad到32，输出保留14；动作不按quantile范围裁剪 |
| batch | 当前正常 WS 路径实际单观测，batch1；`eval_batch=false` 不是这里的batch调节入口 |

随机数/预热来源：[policy.py](../XPolicyLab/policy/Pi_05/openpi/src/openpi/policies/policy.py)；采样公式：[pi0.py](../XPolicyLab/policy/Pi_05/openpi/src/openpi/models/pi0.py)；预处理：[yam_policy.py](../XPolicyLab/policy/Pi_05/openpi/src/openpi/policies/yam_policy.py)、[transforms.py](../XPolicyLab/policy/Pi_05/openpi/src/openpi/transforms.py)、[tokenizer.py](../XPolicyLab/policy/Pi_05/openpi/src/openpi/models/tokenizer.py)。

训练模板的 seed、batch8、trainsteps10000、optimizer 等不是此30k产物的真实训练记录；不能拿部署载入的模板默认值冒充训练事实。推理 seed/设备/预热也不会因随意往 deploy YAML 添加同名字段就自动生效。

</details>

<details>
<summary>展开：D. 相机、录像、布局与计时</summary>

已按实际 CLI 的 `_load_config`（包含默认 station）和只读 `camera_config` 核对；没有连接相机。设备序列号、CAN、地址仍归私有 station。

| 字段 / 设置 | 当前值 | 归属 / 影响 |
|---|---|---|
| camera map | head←front；left_wrist←left；right_wrist←right | experiment.policy.adapter；三路 D405 |
| `width` / `height` / `fps` | 640 / 480 / 30 | camera recipe，双方相同 |
| `exposure_us` / `white_balance` | 10000 / 4000 | 手动设置并关闭对应自动项 |
| `enable_depth` / `align_depth` / `flip` | false / false / false | 当前只用 RGB |
| `warmup_frames` / `background` | 15 / true | 相机预热与后台缓存，不是模型预热 |
| 相机 `max_frame_age_sec` / `read_timeout_ms` | 0.30 s / 1200 ms | USB采集缓存与读取 |
| 客户端 `max_frame_age_sec` / `request_timeout_ms` | 0.5 s / 500 ms | runtime 的 camera_server sensor |
| 相机首帧等待 / 未来时间容差 | 固定1.5 s / 0.05 s | Python内常数 |
| `recording.video_fps` / codec / queue | 30 / 默认 mp4v / 默认8 bundles | 采样、编码和背压；满队列丢 bundle并计数 |
| `robogui.camera_hz` | 默认5 Hz | 显示刷新，与采集/控制/录像频率不同 |
| camera server PUB / heartbeat | CLI 默认1/30 s / 10 s | runtime 用REQ取帧，PUB不是控制时钟 |
| `run.experiment_mode` | 默认false；RoboGUI可选正式实验 | Prepare时冻结布局、repeat和参考图hash |
| 预算 / 输出目录 | 每模型×方法10×3；各自raw root | §9登记；timeout/rubric/PRM profile未冻结 |

来源：[相机 recipe](../manimux/configs/embodiment/sensor/cameras/realsense_3_views.yaml)、[RealSense](../manimux/embodiments/sensor/realsense/sensor.py)、[网络 client](../manimux/embodiments/sensor/camera_server/client.py)、[camera server](../manimux/servers/camera/server.py)、[录像](../manimux/recording/video.py)。

**数据限制**：当前网络 sensor 在读取图像后以收到响应的主机时刻构造 SensorFrame，没有把服务端各相机采集时间完整透传；三路取最新缓存也不是硬件同步触发。录像按收到的 bundle 采样，MP4 为固定30 FPS，掉帧不自动补时长；时间解释须结合 index 和掉帧信息，不能仅凭 MP4 frame/FPS 当真实任务时间。[driver.py](../manimux/embodiments/sensor/camera_server/driver.py)

</details>

<details>
<summary>展开：E. 本机 i2rt SDK 内的执行参数</summary>

安装版 **i2rt 1.1.2，来源 revision `5d47b358bafb30c65e397f2ece506550a0db4594`**（dist-info 声明），代码根 `envs/yam/.venv/lib/python3.12/site-packages/i2rt/`。以下为本机源代码值，不是 SDK 实测频率或所有工位的共同默认；SDK源码未做全树hash复核。当前 station 双臂仅覆盖 channel。

| 设置 | 本机有效值 / 行为 |
|---|---|
| arm PD | kp=[80,80,80,10,10,10]；kd=[5,5,5,1.5,1.5,1.5] |
| gripper PD | kp20、kd0.5；linear_4310 |
| 重力补偿 | 开启，arm factor=[1,1.1,1.1,1.2,1,1]；爪重力项0 |
| initial zero_gravity_mode | 默认true；首个joint command后正常PD；idle grav_comp_kd=[.1,.1,.1,.3,.05,.05] |
| **夹爪力限制** | **get_yam_robot硬编码50 N**；SDK会再次修改夹爪目标，不属于已关闭的速度限制 |
| 夹爪力处理常量 | 力矩平均0.1 s、buffer1000；判堵0.5 Nm且速度<0.3 rad/s；解除0.2 Nm或张开请求；补偿+0.3 Nm；目标平滑α=0.1 |
| 夹爪映射 | motor_stroke6.57 rad、gripper_stroke0.096 m、sign1；归一化开度映射到标定端点 |
| 自动标定 | 新建SDK实例默认校准；每方向最长2 s、torque0.5 Nm、0.05 s检查、变化<0.01 rad连续3次、换向间隔0.3 s |
| joint limits | SDK XML各关节范围向两侧扩0.15 rad，command再次clip；实测监测额外0.1 rad buffer；不同于上层统一±3.14 |
| start/Home插值 | SDK固定50区间、51次发送，sleep(duration/50)；绕过Smooth；start双臂依次、Home阶段双臂并行 |
| 命令语义 | command_joint_pos更新缓存；速度目标/外加torque置0，再由PD及重力补偿控制；不是发送差分速度 |
| SDK/CAN周期 | 各线程运算后sleep0.001 s / 0.0005 s；实际频率取决于耗时；CONTROL_FREQ=250只用于带宽估算 |
| 其他有效边界 | 重力计算abs>25 Nm报错；clip_motor_torque默认inf；电机越±π会调整offset±2π |
| 协议 / 恢复 | MIT、p16反馈、buffered reader=false、auto_recovery=false |

SDK 来源：`robots/get_robot.py:133`、`:192`、`:298`；`robots/config/yam.yml:11`、`linear_4310.yml:19`；`robots/motor_chain_robot.py:543`、`:615`；`robots/utils.py:599`、`:719`；`motor_drivers/dm_driver.py:543`。ManiMux 接入：[arm.py](../manimux/embodiments/arm/yam/arm.py)。

XML原始J1–J6范围（rad）分别为 `[-2.61799,3.14159]`、`[≈0,3.66519]`、`[0,3.14159]`、`[-1.69297,1.5708]`、`[-1.5708,1.5708]`、`[-2.0944,2.0944]`。电机协议位置16bit，速度/PD/torque12bit；DM4340编码速度±10 rad/s、torque±28 Nm，DM4310为±30 rad/s、±10 Nm。这些是设备协议边界，不能当成另一个可随实验关闭的 Smooth 限速。

可由 component hardware 传入但当前未覆盖的 SDK 项：`ee_mass`、`ee_inertia`、`gravity_comp_factor`、`gripper_limits_override`、`gripper_kp/kd`、`zero_gravity_mode`、`sim`、`enable_auto_recovery`、`use_coulomb_friction`。Coulomb补偿当前关闭，其幅值默认不参与本轮。

**记录边界**：runtime保存的command是传给SDK的输入，SDK位置clip、夹持力处理后的最终电机目标没有作为同一个command字段保存。正式报告应固定SDK版本、PD、力处理和标定口径；不能把上层command称为电机最终目标。

</details>

<details>
<summary>展开：F. 当前不生效的默认项与配置保存范围</summary>

- `inference.refill_threshold_s=.4`：当前Serial/RTC均不用；RTC的默认`inference_schedule=deadline`也不驱动RTC请求时机。
- `policy.trajectory_duration_s=null`：未启用；一旦指定，会用 `duration/(H−1)` 改写实际action间隔。`policy.inference_delay_s=0`用于FakePolicy，此WS模型不用；`policy.device=cpu`不是远端JAX设备选择。
- `independent_group_decoding=false`、`decode_budget_ms=40`、`expected_decode_s=0`、`decode_forecast_size=0`、`decode_forecast_mode=max`：当前inline joint路径不启用对应分组/异步解码预算。
- gripper `.35/.85`阈值、`min_closed_s=0`、`open_confirm_s=0`：continuous且无guards时不控制锁存/张闭。`release_guard`默认容差0.02 m、timeout2 s；`grasp_guard`默认0.02 m、0.08726646 rad、settle0.15 s、开度稳定阈值0.01、timeout2 s、approach速度null；两者当前都是null。
- MPC、braking、temporal ensemble、AAC、PAINT和AutoHorizon未启用；补全配置里出现这些默认字典不代表参与此次执行。Smooth的第二个参考点也不被legacy用作前馈。
- Pi_05上游deploy.yml不会自动合并进当前ManiMux launcher；其中seed/eval_batch/checkpoint_num不能当成生效值。当前JAX分支不使用PyTorch compile/device选项；正常INFER也不随意透传`sampling.noise/num_steps`。
- 默认`run.experiment_mode=false`、空layout/repeat、RoboGUI host/port/标签是启动与界面信息；正式尝试身份以Prepare冻结记录为准，不将RoboGUI描述当成算法真值。

`session-manifest.json` 已存展开的config、入口文件hash和父仓库/XPolicyLab Git SHA；rollout meta另存task、身份、Smooth和backend声明。[保存逻辑](../manimux/cli.py)。它尚不能完整证明SDK内部默认、实际GPU/精度/预热、每条RNG、相机生效选项/标定值、checkpoint hash及dirty源码状态；正式setting冻结时需要补足这些证据，不以同一个Git SHA掩盖未提交差异。

本次验收：两份入口经实际CLI loader（含station）读取成功，双方Smooth与motion_limits一致；五项速率限制均为null；`git diff --check`通过。没有执行真机、模型推理或行为测试；hold、seed、warmup、标签和记录缺口尚未修改。

</details>
