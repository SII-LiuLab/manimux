# Configuration layout

模型运行配置按 `configs/<model>/<embodiment>/{server,infra}/` 组织；
任务相关配置继续按 task 分子目录，避免将所有实验堆在 `infra/` 下：

```text
configs/
  <model>/
    <embodiment>/
      server/
        <task>/
          <checkpoint>.yaml
        <checkpoint-or-backend>.yaml
      infra/
        <task>/
          <runtime>-<checkpoint>.yaml
        <experiment>.yaml
  collection/
    <embodiment>/
      station.yaml
      control.yaml
  robots/
    <embodiment>/
      common.yaml
```

`server/` 只描述模型服务、checkpoint、norm stats 和模型原生采样参数；`infra/` 描述
ManiMux 的机器人、传感器、policy wire、执行器、Viewer 和记录。文件只创建实际存在的
角色，不再使用 `live` 后缀。名称直接表达实验差异，例如 Pi05 的 `base.yaml` /
`finetune.yaml`，以及统一的 `manimux.yaml` / `rtc.yaml` runtime。

命名轴不能混用：`server/{base,finetune}.yaml` 表示 checkpoint 变体，
`infra/{manimux,rtc}.yaml` 表示推理 runtime。base 权重加本体 projection stats 仍叫
`server/base.yaml`，不能在 infra 层新增 `zero-shot.yaml`。XR-1 只使用 XPolicy server；
不再提供平行的 native runtime 配置。

公共配置不属于任何模型，继续放在顶层：

- `cameras.yaml`：共享相机服务；
- `robots/`：机器人 driver 配置；
- `robots/yam/common.yaml`：显式引用时共享的 YAM 控制参数；
- `collection/<embodiment>/`：本体专用、config 驱动的数采入口；
- `mock.yaml`：全 mock runtime；
- `maniunicon_meshcat.example.yaml`：通用仿真示例。

新增本体时不要复制模型目录，例如 Pi05 的 ALOHA 配置应放到
`configs/pi05/aloha/`，而不是创建新的 `pi05-aloha` 模型目录。

## 采集与推理的公共控制参数

配置通过顶层 `control_profile` 显式引用公共 YAML，路径相对引用它的配置文件解析。
公共文件只允许一层引用，不递归继承；局部配置与公共值冲突时直接报错。
这不是全仓库默认覆盖，未引用 profile 的旧实验保持原行为。

YAM 的公共部分包括 driver、关节分组、CAN 通道、夹爪类型/力、动作点间隔、
`command_safety` 和独立的 arm/gripper `motion_limits`。当前同步数采与两套
放瓶子 Pi05 RTC 30k 配置共享 `configs/robots/yam/common.yaml`。

- 采集默认同步 30 Hz Direct，每次主臂采样下发一次双臂目标。
- RTC 默认 100 Hz Smooth，但 policy 动作点仍为 30 Hz。
- 共用运动限幅；Smooth 滤波、输出频率和推理调度由各自配置选择。
- 手臂软件速度/加速度限幅当前为 `null`；夹爪闭合目标限速为 `1.0 /s`，
  打开目标直接切换。硬件保护不会因此被关闭。
- 共享 `motion_limits` 支持 Direct / Smooth；MPC 不支持这组配置时明确拒绝。

具体字段、配置边界和记录语义见 [YAM 数采说明](../docs/yam-collection.md#shared-control-profile)。

## 第一份配置：`configs/mock.yaml`

ManiMux 配置使用严格 schema：拼错字段或写入未知字段会直接报错，不会静默忽略。
Beginner 先从全 mock 配置理解七个顶层区域。

### `run`：这次实验是什么

| 字段 | 含义 |
|---|---|
| `task` | 发送给 policy 的任务指令，同时写入 Recorder 和 Viewer。fake policy 不读取语义，但真实 VLA 会读取。 |
| `output_dir` | 每次启动创建 `session-<timestamp>-<id>/` 的根目录，config manifest 和 rollout 都写在这里。 |
| `max_steps` | 最多执行多少个 control tick；mock 中 `120 / 100 Hz ≈ 1.2 s`。 |
| `experiment_mode` | Viewer 实验模式的默认值。默认 `false`；也可在 `serve` 就绪后、Prepare 之前用显眼按钮切换。 |
| `layout_id` | 物体摆放或条件编号；正式实验建议显式填写，普通运行可留空。 |

### `robot`：控制哪具身体

| 字段 | 含义 |
|---|---|
| `driver` | RobotDriver 插件名；`mock_dual_arm` 只在内存里模拟状态和命令。 |
| `control_hz` | ManiMux 主控制环频率；`100 Hz` 表示每 `10 ms` 一个 tick。它不是模型推理频率。 |
| `group_dims` | canonical action 分组和每组维度。名称、顺序和维度必须与 adapter、executor、robot 一致。 |

### `sensors`：policy 能看到什么

`sensors` 是列表，可以声明多路命名传感器。

| 字段 | 含义 |
|---|---|
| `name` | observation 中的传感器键；adapter 按名称查找相机。 |
| `driver` | SensorDriver 插件名；`mock_camera` 生成确定性的 RGB 图。 |
| `width` / `height` | 该传感器输出图像的宽和高。mock 输出 shape 为 `(height, width, 3)`。 |
| `fps` | 传感器声明的目标帧率。当前 `mock_camera` 不自行限频，而是在每个 control tick 被读取；真实 camera-server driver 才按服务数据更新。 |

### `policy`：谁来想、一次想多远

| 字段 | 含义 |
|---|---|
| `worker` | PolicyModel 插件名。模型在独立 spawn 进程中运行；`fake` 生成确定性的双臂波形。 |
| `adapter` | PolicyAdapter 插件名，负责 observation 编码和 action 解码；`identity` 直接接收 fake model 产生的 canonical chunk。 |
| `device` | 模型设备提示；`fake` 不使用它，真实模型插件可以使用。 |
| `action_dt_s` | action chunk 相邻两个点的时间间隔。`0.05 s` 等于 policy 轨迹点 `20 Hz`，不等于 robot 的 `100 Hz` 控制环。 |
| `timeout_s` | 一次推理请求的 deadline；响应晚于 deadline 会被丢弃。 |
| `horizon_steps` | 每个 action chunk 的点数。`20` 点、间隔 `0.05 s`，首末点覆盖 `(20-1)×0.05=0.95 s`。 |
| `inference_delay_s` | fake model 专用的模拟推理耗时；调大它可以观察异步推理和过期响应。 |
| `expected_backend` | 可选的 policy-server 身份契约。声明稳定的 `server` 和/或 `model` 元数据；worker 握手后、机器人连接前按递归子集严格匹配，防止端口上部署了错误 checkpoint。 |

### `execution`：什么时候推理、怎样执行

| 字段 | 含义 |
|---|---|
| `executor` | `smooth` 或 `mpc`。mock 默认只使用下面的 `smooth` 参数；`mpc` 块此时不参与执行。 |
| `runtime` | 推理 strategy：`manimux`、`act_temporal_ensemble`、`rtc`、`aac`，或第三方插件。它不更换 Robot、Timeline、Executor 和 Safety。 |
| `refill_threshold_s` | 当前 Timeline 剩余时间低于该值时提交下一次推理请求。 |
| `commit_lead_s` | 新 chunk 被接受后，从“当前时刻 + lead”开始生效，给原子切换留出极小调度余量。 |
| `max_plan_age_s` | 从 observation 时间算起，chunk 超过该年龄就以 `plan_too_old` 拒绝。 |
| `blend_steps` | 新 chunk 裁掉过期前缀后，前多少步从当前 measured command 线性过渡到模型轨迹。`0` 表示不融合。 |

`refill_threshold_s` 和 `inference_schedule` 只属于默认 strategy。`runtime: rtc` 有自己的
`H/s/d` 调度；`runtime: act_temporal_ensemble` 使用 `query_interval_steps`；`runtime: aac`
在自适应短 chunk 执行完后同步请求下一组候选；`runtime: paint` 使用论文的异步 `s/d`
prefix contract。这些 strategy 若配置
默认 strategy 的调度字段会直接校验失败，避免出现“写了参数但实际没生效”。

`policy.expected_backend` 不应锁定每次进程都会变化的 `server_instance_id`，通常只声明
`server: xpolicylab_policy_server`，以及 `model` 下的 `policy_family`、`task_name`、
`checkpoint_variant`、checkpoint/norm-stats 来源与解析后的路径。实际握手可以包含更多字段；
config 声明的每个字段必须存在且完全相等，否则 rollout 在 `RobotDriver.connect()` 之前失败。

`execution.temporal_ensemble` 仅在 `runtime: act_temporal_ensemble` 时生效：

| 字段 | 默认值 | 含义 |
|---|---:|---|
| `coefficient` | `0.01` | 官方 ACT 的指数权重系数 `w_i ∝ exp(-coefficient × i)`；预测按旧到新排列。 |
| `query_interval_steps` | `1` | 每隔多少个 **policy action step** 请求一个新 chunk。`1` 是官方 ACT temporal aggregation 频率；Pi05 示例用 `4`，即约 `4 × 33.3 = 133 ms`。 |

ACT strategy 要求 `blend_steps: 0`：重叠 chunk 已由 ACT 公式融合，再做线性 seam blend
会二次修改算法输出。当前实现的公式和默认系数来自官方 ACT commit `742c753`，ManiMux
只把同步的逐步查询改成非阻塞、可参数化的 policy-step 查询调度。

`execution.aac` 仅在 `runtime: aac` 时生效：

| 字段 | 默认值 | 含义 |
|---|---:|---|
| `num_samples` | `20` | 一次视觉编码后并行采样的候选 chunk 数，跟随官方 AAC。 |
| `motion_threshold` | `3.0` | 未归一化 EE motion floor 阈值，必须按本体动作单位标定；YAM 60 条示教配置使用 `0.2`。 |
| `ee_stats_path` | required | entropy/selector 使用的固定 EE 增量 min-max；必须匹配本体、FK、joint 顺序和训练数据域。 |
| `chunk_id_selector` | `"0"` | `"0"` 选第一条候选（官方默认），也支持官方 `mean` 和 `backward` selector。 |
| `backward_beta` | `0.99` | `backward` selector 对重叠未来动作的指数权重。 |

AAC 要求 `policy.options.allow_short_horizon: true` 和 `blend_steps: 0`。官方实现面向 GR00T
N1.5 的 7D EE pose/gripper action；`configs/{groot,pi05}/yam/infra/aac*.yaml` 是明确标注的
YAM adaptation：模型仍输出 14D absolute joints，但 `aac_kinematics: yam` 会先用共享 YAM
FK 转成左右逐步 EE 增量，使用 config 指向的固定 stats 做官方 min-max，再分别执行官方
score 后取平均。stats 只参与 AAC 评分，最终执行仍是原始 joint chunk。它不是官方论文
的 Pi05/YAM 配置。

`execution.paint` 仅在 `runtime: paint` 时生效：

| 字段 | 默认值 | 含义 |
|---|---:|---|
| `execution_steps` | `10` | 论文中的执行窗口 `s`：旧 chunk 至少执行到该 index 才提交下一次异步推理。 |
| `initial_delay_steps` | `4` | 首次 PAINT 请求使用的延迟 `d`；之后用已完成请求的真实耗时滚动更新。 |
| `delay_buffer_size` | `10` | 延迟历史窗口；使用窗口内最大值，避免真实推理超过已锚定 prefix。 |

PAINT 要求 `d <= s <= H-d` 且 `blend_steps: 0`。ManiMux 将旧 chunk 的 `A[s:s+d]` 送给
XPolicy；XPolicy sampler 完成论文的 naive forward、backward Euler、prefix repaint 和
final forward。禁用 seam blend 是为了避免旧 chunk 在成为下一次 prefix condition 前被
Timeline 二次改写。若响应
实际需要丢弃的步数超过 condition 的 `d`，runtime 会拒绝该响应而不是执行未锚定动作。

`runtime: autohorizon` 没有可调 method 参数。XPolicy 按官方默认值读取 Pi05 action expert
第三个 denoise step 的 self-attention，并返回 `execution_steps`；ManiMux 同步执行完整 chunk
的这个前缀，耗尽后才请求下一次推理。它要求 `blend_steps: 0`，并且不使用
`inference_schedule` 或 `refill_threshold_s`。当前 Pi05/YAM 路径是官方算法的 JAX attention
port；selector 和默认参数固定到 AutoHorizon commit `c7504f1`，与官方 PyTorch 的数值 parity
仍需单独验证。

`execution.dvac` 仅在 `runtime: dvac` 时生效：

| 字段 | 默认值 | 含义 |
|---|---:|---|
| `tail_steps` | `5` | 论文中的 `L`，只统计最后几个 clean-action estimates。 |
| `alpha` | `2.0` | 阈值 `tau = mean + alpha × std` 的局部标准差倍数。 |
| `rolling_window_size` | `5` | 论文中的 `m`，保留最近几个 policy call 的方差序列。 |
| `min_execution_steps` | `1` | 论文中的 `N_min`。 |
| `max_execution_steps` | policy horizon | 论文中的 `N_max`；按 Equation 7 仅用于没有 threshold crossing 的分支。 |

DVAC 要求 `blend_steps: 0`，同步执行服务器选出的稳定 prefix，耗尽后才重新请求。当前实现
严格使用论文公式和默认值，但论文没有公开代码，也没有定义空 rolling buffer；Pi05/YAM
明确采用当前方差自举首请求，并只统计 14 个有效 normalized action 维度，不统计 OpenPI
padding。完整审计见 `docs/reproductions/dvac-pi05.md`。

`execution.smooth` 仅在 `executor: smooth` 时生效：

| 字段 | 含义 |
|---|---|
| `cutoff_hz` | 一阶低通截止频率；越低越平滑但跟随越慢。 |
| `max_velocity` | 每个 canonical 标量每秒允许的最大变化量。对关节位置通常理解为 `rad/s`。 |
| `max_acceleration` | 每个 canonical 标量每秒速度允许的最大变化量，关节通常为 `rad/s²`。 |
| `position_limit_abs` | 通用绝对位置包络 `[-limit, +limit]`；SafetyGuard 和 executor 都会使用。它不是机器人逐关节精确限位表。 |

`execution.mpc` 仅在 `executor: mpc` 时生效：

| 字段 | 含义 |
|---|---|
| `horizon_steps` | MPC 每个 tick 向前优化多少个参考点。 |
| `dynamics_a` | 简化一阶动力学中的状态保留系数，必须在 `(0,1)`。越大表示系统响应越慢。 |
| `tracking_weight` | 跟随 policy reference 的代价权重。 |
| `command_delta_weight` | 抑制连续命令突变的代价权重。 |
| `max_velocity` / `max_acceleration` / `position_limit_abs` | MPC 输出之后使用的同类执行限制。 |

### `viewer`：是否实时发布

| 字段 | 含义 |
|---|---|
| `enabled` | 是否把状态、相机、plan 和事件发布给 ManiMux Viewer。mock 默认关闭。 |
| `robot_adapter` | Viewer 采用哪套机器人几何和关节映射；`enabled: false` 时不生效。 |

### `recording`：保存 episode

| 字段 | 含义 |
|---|---|
| `enabled` | 当前必须为 `true`。真机运行强制保留 episode、事件和命令 lineage；写 `false` 会在配置校验阶段失败。 |
| `video_fps` | 每路相机保存 MP4 的目标帧率；`0` 表示关闭。它不改变 policy 或 Viewer 的相机频率。 |
| `video_codec` | OpenCV 四字符编码，默认 `mp4v`。 |
| `video_queue_size` | 异步视频队列容量。队列满时丢视频 bundle，不阻塞控制环。 |

### mock 中没有显式写出的常用字段

这些字段使用 schema 默认值，真实模型配置中经常显式声明：

| 字段 | 默认值 | 含义 |
|---|---:|---|
| `policy.startup_timeout_s` | `30.0` | 等待模型 worker 完成初始化的最长时间。 |
| `policy.trajectory_duration_s` | `null` | 若设置，则用 `duration / (horizon_steps-1)` 覆盖 `action_dt_s`。 |
| `policy.options` | `{}` | 具体模型插件自己的地址、相机映射、stats、group order 等参数。 |
| `robot.config` / `robot.options` | `null` / `{}` | RobotDriver 的外部配置路径和本体专用参数。 |
| `sensor.options` | `{}` | SensorDriver 专用的 endpoint、camera names 等参数。 |
| `execution.runtime` | `manimux` | 选择共享 Runtime 内的 strategy；支持内置 `manimux`/`act_temporal_ensemble`/`rtc`/`aac`/`paint`/`autohorizon`/`dvac`、entry point 或 `module:factory`。 |
| `execution.inference_schedule` | `deadline` | 仅默认 strategy 使用；`deadline` 允许旧请求到期后提交新请求，`single_inflight` 始终只保留一个未完成请求。 |
| `execution.rtc` | defaults | 仅 RTC strategy 使用的 delay、最小执行步数和 guidance 参数。 |
| `execution.temporal_ensemble` | defaults | 仅 ACT strategy 使用的官方指数权重和查询间隔。 |
| `execution.aac` | defaults | 仅 AAC strategy 使用的多样本数、motion floor 和候选选择参数。 |
| `execution.paint` | defaults | 仅 PAINT strategy 使用的执行窗口、延迟先验和滚动窗口。 |
| `execution.dvac` | defaults | 仅 DVAC strategy 使用的 denoising tail、滚动阈值和执行长度边界。 |
| `viewer.policy_label` | `""` | Viewer 与 Recorder 中显示的模型名称。 |
| `viewer.camera_hz` | `5.0` | 向 Viewer 发布相机图像的最高频率；不改变 policy 读取相机的频率。 |
| `recording.video_fps` | `0.0` | 默认不编码视频；正式实验 config 可显式启用。 |
