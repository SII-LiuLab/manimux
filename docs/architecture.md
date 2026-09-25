# ManiMux V1 功能架构

> 当前重组的目录边界与迁移状态见 [代码组织](code-organization.md)。下文保留 V1
> 运行机制说明，其中旧配置类、插件加载和目录名称不代表重组后的目标布局。

状态：V1 runtime contract

日期：2026-08-21

范围：单工作站、本地机器人、本地模型、本地数据

## 1. V1 要验证什么

ManiMux V1 只验证以下闭环是否可靠：

```text
robot state + cameras
        -> local policy inference（异步）
        -> time-indexed action chunk
        -> dual-arm control（连续、安全）
        -> local episode record + Universal Viewer
```

成功标准不是“功能很多”，而是：

1. 模型推理卡顿时 controller 仍按固定频率运行；
2. 新 action chunk 能按时间对齐并原子替换旧 chunk；
3. stale/invalid response 永远不会发给真机；
4. 不改核心 runtime 即可替换 robot driver 或 policy adapter；
5. Smooth 或本地 MPC 能把异步 chunk 变成连续受约束的 command；
6. 一次 episode 可以在本地查看、评分和 replay。

## 2. V1 部署结构

V1 不实现 Control Plane、Inference Gateway 或 Data Service。一个 CLI launcher 启动三个顶层运行单元：

```text
manimux run --config manimux/configs/experiments/my_task/my_model/my_run.yaml
│
├── edge-agent
│   ├── RobotBase / SensorBase
│   ├── Observation Builder
│   ├── InferenceStrategy + Action Timeline + Executor + Safety
│   ├── local Recorder
│   └── built-in Viewer publisher/control client
│
├── policy-worker
│   └── one local model + one PolicyAdapter
│
└── manimux-viewer
```

三个单元都运行在同一工作站。Recorder 可在 edge-agent 内使用独立子进程隔离磁盘 I/O，但不是一个需要部署或管理的服务。

### 2.1 Launcher

Launcher 只做：

- 读取并严格校验一个 YAML；
- 创建 `run_id` 和本地 run 目录；
- 启动 policy-worker、Viewer 和 edge-agent；
- 等待 ready，处理 Ctrl-C，按反序关闭；
- 将 resolved config、代码版本和启动信息写入 `run.json`。

它不做服务 registry、lease、调度、promotion 或后台 daemon。配置中的插件名称只由
进程内 factory、Python entry point 或显式 `module:factory` 解析。

### 2.2 Edge Agent

Edge Agent 是 V1 的核心，唯一允许触达机器人 driver：

```text
Robot/Sensors
     │
     ▼
Latest State Buffers ──► Observation Builder ──► policy request queue
     │                                                │
     │                                                ▼
     │                                        policy response queue
     │                                                │
     ▼                                                ▼
Safety/Watchdog ◄── Smooth/MPC Executor ◄── Action Timeline ◄── Validator/Adapter
     │
     ▼
RobotBase.send_command()

all streams ── best-effort copy ──► local Recorder / Viewer
```

Edge Agent 内部可以有多个 thread/process，但 controller loop 不能等待其他组件。
普通异步 chunk、ACT Temporal Ensembling 和 RTC 共用这一条控制循环；它们只替换
`InferenceStrategy`，不会复制 Robot、Sensor、Timeline、Executor、Safety、Recorder、
Viewer 或退出清理逻辑。

### 2.3 Policy Worker

一个 run 启动一个 worker，加载一个本地模型。它只需要：

```python
reset(context) -> None
infer(observation) -> RawActionChunk
close() -> None
```

请求队列和响应队列都有固定上限。默认规则：

- 同一时刻只计算一个 request；
- 等待区最多保留一个最新 observation；
- 尚未开始的旧 request 被新 request 替换；
- 已开始的 inference 可以完成，但 edge 根据 `request_seq` 和 deadline 决定是否接受；
- worker crash 后 session 失效，edge 进入 hold/fault，不自动沿用模型内部 cache。

这就是 V1 所需的全部“admission”：一个有界 latest-wins queue。它不需要 auth、配额、路由或 Gateway。

### 2.4 Built-in Universal Viewer

Viewer 直接位于 `manimux/viewer/`，不再依赖另一个 checkout：

- policy/runtime 发布通用 `PolicyPlan`、`RobotSnapshot` 和 `RuntimeEvent`；
- robot adapter 负责关节拆分、FK、模型和场景，不把 Viewer 写死为 YAM；
- YAM 是首个内置 adapter，模型资源随 `manimux` 一起发布；
- edge 可读取 pause/resume/home/step/finish，MolmoAct 集成当前采用 observe 模式；
- Viewer 缺席或断开时默认 pause；
- Viewer 命令只是 intent，edge safety state 决定是否接受；
- Viewer/Recorder 都是 best-effort 旁路，不得阻塞 control loop。

## 3. V1 必要接口

V1 不建立多个 registry/manifest 层，只保留三个插件接口和一个 run config。

### 3.1 RobotBase

```python
# Public methods of manimux.embodiments.robot.RobotBase.
# Assemblies inherit the existing base; no separate robot Protocol is needed.
class RobotBase(ABC):
    def connect(self) -> None: ...
    def get_state(self) -> RobotState: ...
    def send_command(self, command: RobotCommand) -> None: ...
    def home(self) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...
```

Robot state/command 使用 named group，例如 `left_arm`、`right_arm`、`left_gripper`、`right_gripper`。双臂仍作为一个 robot instance 连接和停止。

### 3.2 SensorBase

```python
# manimux.embodiments.sensor.SensorBase
class SensorBase(ABC):
    def start(self) -> None: ...
    def read(self) -> SensorFrame | dict[str, SensorFrame]: ...
    def close(self) -> None: ...
```

每个 frame 必须包含 capture monotonic timestamp 和递增 sequence。大图像进入固定 shared-memory slot，队列只传 descriptor。

### 3.3 PolicyAdapter

PolicyAdapter 隔离模型与机器人组合的特殊语义：

```python
class PolicyAdapter(ABC):
    def build_observation(self, snapshot: ObservationSnapshot) -> object: ...
    def prepare_request(self, request: InferenceRequest) -> InferenceRequest: ...
    def decode_action(self, raw: object, context: ActionContext) -> ActionChunk: ...
    def validate(self, robot: dict, policy: dict) -> None: ...
```

它负责：

- observation resize/order；
- 模型服务的观测字段与整机状态映射（checkpoint normalization 归 XPolicyLab）；
- joint/group order；
- relative/absolute 与 joint/EE action 解释；
- RTC condition 从 canonical joint 空间到模型 native action 空间的编码；
- gripper 语义；
- 输出 shape、finite value 和基础 continuity 校验。

每个 run 只选择一个明确 adapter。V1 不尝试自动匹配任意模型和机器人。

#### XPolicy 的两层 adapter

XPolicy 接入时有两个职责不同的 adapter：

```text
YAM state/cameras
    -> ManiMux JointAdapter
    -> XPolicy standard observation
    -> XPolicy policy/Pi_05/model.py
    -> OpenPI native input
```

- ManiMux 的 `JointAdapter` 负责 YAM group/camera 与 XPolicy wire format 的互转；
- XPolicy 的 `policy/<name>/model.py` 负责 checkpoint、模型预处理、normalization 和
  模型原生 action 解码；
- ManiMux RTC 根据真实执行进度生成 condition 和 soft mask；支持 RTC 的 XPolicy 模型
  adapter 把它注入模型采样过程，不支持时明确失败；
- XPolicy `HELLO_ACK` 声明 `default/rtc/aac` sampling capability，PolicyWorker 在机器人连接前
  将它交给 Runtime 做兼容检查；
- `XPolicyLab/` submodule 只固定源码版本。模型仍在独立环境运行，ManiMux 通过
  WebSocket 发出一次完整推理请求，不把模型依赖装进 runtime 环境。

### 3.4 InferenceStrategy

Runtime 只有一条控制循环。InferenceStrategy 只负责三个决定：

```python
build_submission(...) -> InferenceRequest | None
prepare_chunk(...) -> ActionChunk
commit_settings(...) -> anchor + blend_steps
on_plan_accepted(...) -> strategy state
```

- `DefaultChunkStrategy` 在 Timeline 剩余时间低于阈值时请求普通 chunk；
- `ACTTemporalEnsembleStrategy` 按 policy step 周期请求 chunk，并按官方 ACT 公式合并
  同一绝对时刻的重叠预测；
- `RtcInferenceStrategy` 根据 `H/s/d` 选择推理时机，并构造 condition 与 soft mask；
- `AacInferenceStrategy` 等待上一段自适应 chunk 结束，再请求 XPolicy 的多样本 hook；
- 所有 strategy 的结果都必须先成为 canonical `joint_position` ActionChunk，才能进入 Timeline；
- 新算法应新增 strategy，而不是复制一份 EdgeRuntime。
- strategy 与其它插件一样支持内置名、Python entry point 和 `module:factory`；第三方算法
  不需要修改 `build_runtime()`。

## 4. 一个配置文件

V1 使用一个严格校验的 YAML，不使用服务注册中心或 artifact manifest。`driver`、
`worker` 和 `adapter` 是轻量插件选择字段：

```yaml
run:
  task: fold_cloth
  output_dir: ./data
  max_control_steps: 500

robot:
  type: yam
  config: ../../embodiment/robot/yam_dual.yaml
  control_hz: 100

sensors:
- name: overhead
  driver: realsense
  serial: '...'
- name: left_wrist
  driver: realsense
  serial: '...'

policy:
  worker: xpolicylab_ws
  adapter:
    type: manimux.policy_adapter.joint:JointAdapter
  action_dt_s: 0.05
  timeout_s: 1.0


viewer:
  enabled: true
  robot: my_dual_arm

recording:
  enabled: true
  video_codec: h264
inference:
  refill_threshold_s: 0.4
  commit_lead_s: 0.02
  max_plan_age_s: 1.0
  underrun_hold_s: 0.5
executor:
  type: smooth
  smooth:
    cutoff_hz: 8.0
    max_velocity: ['...']
    max_acceleration: ['...']
  mpc:
    horizon_control_steps: 15
    control_dt_s: 0.01
    max_velocity: ['...']
    max_acceleration: ['...']
```

启动时保存 resolved config。配置 hash 进入每个 episode metadata；不另外建立 ModelArtifactManifest、RobotManifest 或 Embodiment Registry。

## 5. 异步 inference 与 action timeline

### 5.1 最小消息字段

V1 Python 进程间使用 typed dataclass + bounded multiprocessing queue。消息至少包含：

```text
InferenceRequest:
  session_id, request_seq, observation_time_ns, deadline_ns, observation

InferenceResponse:
  session_id, request_seq, finished_time_ns, inference_ms, raw_action

ActionChunk:
  plan_id, request_seq, action_space, dt_ns, groups{name -> [T, D]}
```

不在 V1 提前引入 protobuf/gRPC。接入 Embodied.cpp 等 C++ backend 时，再为同一逻辑类型增加 Unix socket + protobuf codec。

### 5.2 正常流程

```text
t0  edge 对齐最新 state/camera，发送 request N
t1  当前 chunk N-1 继续执行
t2  response N 返回
    - 检查 session/request/deadline/shape/finite
    - adapter 转成 canonical ActionChunk
    - 根据 observation age 丢弃已过时前缀
    - 与当前 command 做短 continuity blend/限速
t3  在 now + commit_lead 后原子替换未来 timeline
t4  controller 在每个 tick 采样 timeline
```

### 5.3 Timeline 与 Executor V1 能力

Timeline 只实现：

- 一个 active plan；
- 按 monotonic time 采样；
- stale prefix trimming；
- 双臂 group 原子 commit；
- 短 blend window；
- buffer underrun 时 hold，超过 TTL 后 fault。

Timeline 输出 reference horizon，再交给每个 run 静态选择的 executor：

```python
class Executor(Protocol):
    def reset(self, state: RobotState) -> None: ...
    def step(
        self, now_ns: int, state: RobotState, reference: ActionHorizon
    ) -> RobotCommand: ...
```

#### SmoothExecutor

- 对 reference 做固定频率 resample/interpolation；
- 使用 causal low-pass/EMA 或等价简单滤波；
- 执行 position、velocity、acceleration limit；
- 输出当前 tick 的 `optimized_action`。

#### MPCExecutor

- 使用当前 measured state 和 timeline future reference 做有限时域跟踪；
- V1 只做 joint-position space local MPC；relative EE policy 需由 PolicyAdapter/IK 转成 joint reference；
- 约束 position、velocity、acceleration，并保持双臂 command 同 tick 输出；
- solver 超时或失败时记录有界 reason 并 hold/fault，不静默切换 executor。

V1 不实现 learned speed selection、whole-body nonlinear MPC、多 plan arbitration 或运行时 executor 自动切换。

MPC solver 作为可选依赖并 lazy import；选择 `executor: smooth` 时不安装、不初始化 MPC 依赖。配置只校验当前选中的 executor block。

### 5.4 接受新 chunk 的条件

新 chunk 必须同时满足：

1. session 相同；
2. `request_seq` 比当前已接受 request 新；
3. 未超过 deadline/max plan age；
4. action group、shape、unit 和 dt 与当前 run config 相符；
5. action 全为 finite value；
6. 左右臂具有同一个 plan id 和时间网格；
7. continuity/limits 检查通过。

失败时丢弃整个双臂 chunk，并写一个有界 reason event。

## 6. V1 Safety 与状态

只保留五个状态：

```text
DISCONNECTED -> IDLE -> RUNNING <-> PAUSED
        any active state ----------> FAULT
```

- 启动连接成功后进入 `PAUSED` 或 `IDLE`，不会自动执行；
- start/resume 需要新鲜 state、camera 和可接受 plan；
- pause 停止消费未来 policy plan并 hold；
- home 是一个受限命令，不单独增加复杂状态；
- driver/state watchdog、plan TTL、joint/workspace limits 触发 hold/fault；
- FAULT 需要人工确认，不自动恢复；
- 任一臂 fault 默认使整个双臂 robot fault。

实体急停和厂商安全控制器不属于 ManiMux 软件状态机。

## 7. 本地 Recorder

### 7.1 文件布局

V1 不使用数据库、MCAP、WAL 或对象存储。目录本身就是 catalog：

```text
data/
  <run_id>/
    run.json
    <episode_id>.partial/
      meta.json
      data.zarr
      events.jsonl
      videos/
        overhead.mp4
        left_wrist.mp4
      result.json
```

正常结束后 flush 所有 writer，并将 `<episode_id>.partial` 原子重命名为 `<episode_id>`。崩溃遗留的 `.partial` 不删除，Viewer/repair tool 可以显示为 incomplete。

### 7.2 必记数据

`data.zarr` 至少包含：

- robot state 和 timestamp；
- raw model action 和 request metadata；
- scheduled/timeline action；
- optimized action，以及 `executor_kind = smooth | mpc`；
- command sent；
- 每路 camera frame timestamp；
- step、chunk index 和 inference latency。

`events.jsonl` 至少记录：

- episode start/end；
- plan accepted/rejected/replaced；
- timeout、stale response、underrun；
- pause/resume/home/finish；
- safety clamp/fault；
- recorder/video error。

V1 action lineage 只有五个必需阶段：

```text
raw_model_action -> scheduled_action -> optimized_action -> command_sent -> measured_state
```

`scheduled_action` 对应 executor 输入，`optimized_action` 对应 Smooth/MPC 输出。若以后需要 solver 内部诊断，再追加 named arrays，不改变基础五阶段。

### 7.3 结果与查询

`result.json` 保存：

```text
success
terminal_reason
steps
wall_time_s
operator_note
evaluator_version
```

本地 run 列表通过扫描 `data/*/run.json` 和 episode 目录生成。只有真实出现查询性能问题后才引入 SQLite。

## 8. V1 代码结构

```text
repository/
├── manimux/
│   ├── configs/               # experiments / embodiment / policy / inference / executor
│   ├── embodiments/           # robot / arm / end_effector / sensor
│   ├── policy_adapter/        # observation/action conversion
│   ├── policies/              # model clients and worker processes
│   ├── servers/               # camera service and XPolicyLab launchers
│   ├── runtime/               # strategies, timeline, executors, safety
│   ├── kinematics/
│   ├── viewer/replay_data/   # Offline trajectory readers
│   ├── viewer/
│   ├── recording/
│   ├── evaluation/
│   └── cli.py / session.py / types.py / clock.py
├── tests/
├── docs/
├── scripts/                   # maintenance, installation and data tools
└── XPolicyLab/                # independent model repository
```

首版保持一个 Python package。不要先拆 `apps/`、`packages/`、generated contracts 或多个 deployable service。

## 9. 实现顺序

### Milestone 0：纯 mock 闭环

- config + CLI；
- mock dual-arm robot/sensors；
- fake policy worker；
- timeline + fake clock；
- SmoothExecutor 与 MPCExecutor contract/fake tests；
- local Zarr/JSONL recorder；
- Universal Viewer bridge。

验收：无 GPU、无真机运行 100 个 chunk，能 pause/resume、记录并 replay。

### Milestone 1：首个真机 + Python model

- 首个 dual-arm RobotBase；
- RealSense SensorBase；
- 一个真实 PolicyAdapter；
- watchdog、limits、partial episode recovery；
- Smooth 和 joint-space local MPC 真机低速验证；
- 手工 success/failure 标注。

验收：分别终止 policy-worker、Viewer 和 recorder writer，机器人行为可预测且 controller 不被阻塞。

### Milestone 2：第二组 adapter

- 第二个模型或第二种机器人；
- 验证核心 timeline/recorder 不修改；
- 增加本地 episode comparison/replay 工具；
- 真正需要时再增加 Embodied.cpp 的 protobuf adapter。

验收：通过配置和 adapter 切换两组 robot × model 组合，而不是复制 edge runtime。

## 10. V1 验收场景

1. Fake model 每 200 ms 返回 1 秒双臂 chunk，controller 100 Hz 连续运行；
2. 注入 50–500 ms inference jitter，controller 不阻塞；
3. response 乱序、重复、超时、维度错误或含 NaN 时，整个 chunk 被拒绝；
4. 左右臂 plan 始终同版本原子替换；
5. Smooth 模式满足 continuity/velocity/acceleration limits；
6. MPC 模式跟踪同一 timeline，solver failure 触发明确 hold/fault；
7. worker crash 后先消费仍有效 timeline，再 hold，TTL 后 fault；
8. Viewer crash 后进入预期 pause/hold，不影响 controller tick；
9. recorder 写失败不影响控制，并留下 incomplete episode/event；
10. episode 可本地 replay，raw/scheduled/optimized/command/measured 时间轴可对齐；
11. 第二个 PolicyAdapter 不修改 timeline/executor/controller；
12. 所有测试除 hardware suite 外均不依赖真机、GPU、网络或云服务。

## 11. 明确延期

以下能力只有出现实际需求后再设计：

- auth、Inference Gateway、remote/cloud worker；
- run-manager、registry、lease、scheduler；
- SQLite/PostgreSQL、MCAP、对象存储和上传；
- protobuf/gRPC 通用协议；
- Prometheus、OpenTelemetry 和 fleet dashboard；
- learned speed selection、whole-body nonlinear MPC、复杂 time parameterization 和 collision prediction；
- model artifact promotion、shadow/canary；
- Universal Viewer v2；
- LeRobot/RLDS 等批量 exporter。

V1 的稳定核心只有四个：typed adapter、bounded async inference、atomic action timeline、local episode format。
