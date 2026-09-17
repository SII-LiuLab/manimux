# UMI & 天机 Replay Pipeline

> 四段独立验证：**数据 → 离线 → 起始构型 → 实机**。
> 姊妹文档：[`快速上手-PICO与天机Teleop.md`](./快速上手-PICO与天机Teleop.md)（遥操作），
> [`control.md`](./control.md)（命令参考）。

| | 路径 |
|---|---|
| Teleop 代码 | `~/Desktop/project/teleop` |
| 机器人 SDK | `~/Downloads/TJ_FX_ROBOT_CONTRL_SDK` |
| 夹爪 SDK | `~/Desktop/project/TacCap-Gripper/python`（`TACCAP_SDK` 可覆盖） |
| 录制端 | `~/project/xense/xense-taccap-lerobot` |
| 示例数据 | `~/Downloads/replay_test` |

**运行前注意**

1. **起始构型决定安全，不是速度决定安全** —— 见 §3。2026-08-25 那次险些撞机就是
   起点余量不够，而当时所有速度指标都在额度内。
2. **这套栈没有自碰撞模型** —— 安全闸只查关节限位、关节速率、跟踪误差。唯一能提前
   看见"轨迹朝本体去"的东西是 §4.2 的下发前预演。
3. **夹爪走独立 USB，与机械臂完全解耦** —— 控制器不通电也能单独验夹爪。
4. **指尖偏移当前还没标定** —— 录制的是 leader 夹爪的两指中点，而 FK 给的是法兰，
   现在整条链路跟随的是法兰。纯平移看不出来，一旦腕部旋转就不等价。见 §1.3。

---

# 0. 这条管线是什么

把手持 UMI 采集装置（TacCap 主爪 + Pico4 定位器）录的 demo，在天机双臂上回放。

**复用整条遥操作链路。** `drivers/umi_source.py` 把录制帧伪装成 `XRFrame`
（`Retargeter` 只调用 `.pose()` 和 `.button()` 两个方法），所以低通 → 离合 →
笛卡尔限速 → IK backoff → 安全闸原样复用，一行安全逻辑都没改。回放和遥操作走的
是同一段代码 —— 否则回放验证的就不是真正会跑的东西。

**相对映射，不是绝对。** 录制位姿的原点是 VR 应用启动瞬间的头显位置，每次重启
VR 都会漂。所以第 0 帧锁定机械臂的**实测位姿**，之后只跟随相对运动。手臂从当前
位置起步，不会跳到某个录制的绝对坐标，也没有什么需要跨会话重新标定的东西。

**代价：相对映射保留位移，不保留余量。** 这是 §3 存在的全部原因。

---

# 1. 数据

### 1.1 检查数据集

```bash
python3 drivers/umi_source.py ~/Downloads/replay_test
```

**通过**：打印出帧数、时长、每只手的位置包络和夹爪范围。示例数据应为：

```
robot_type bi_taccap_gripper  codebase v3.0  fps 30
299 帧 = 9.93 秒
left  位置包络 [ 19.6 152.8  22.4] mm  速度 均值57 / p95 236 / 峰值463 mm/s
      夹爪 0.045..0.652 (0=闭合)  起始 0.480
right 位置包络 [35.5 95.6 29.6] mm  速度 均值50 / p95 164 / 峰值338 mm/s
      夹爪 0.041..0.629 (0=闭合)  起始 0.456
```

数学自检（不需要数据）：

```bash
python3 drivers/umi_source.py --self-test
```

### 1.2 数据格式速查

LeRobot v3.0，`action` / `observation.state` 各 20 维（左右各 10）：

| 字段 | 含义 |
|---|---|
| `{side}_tcp.x/y/z` | 位置，**米** |
| `{side}_tcp.r1..r3` | 旋转矩阵**第一列** |
| `{side}_tcp.r4..r6` | 第二列（6D 旋转表示） |
| `{side}_gripper.pos` | 归一化开口，**0=闭合，1=张开** |

外加 6 路视频：`{left,right}_wrist`(480×640) + `{left,right}_tactile_{left,right}`(400×700)。
回放不用视频。

三件要知道的事：

1. **`action[t]` 与 `observation.state[t+1]` 逐位完全相同**（20 维平均绝对差精确
   为 0）。采集装置是被动记录，`send_action()` 是 no-op，所谓 action 就是下一帧的
   实测位姿。
2. **位姿在 Pico VR 世界系**：X 前 / Y 左 / Z 上，重力对齐，原点 = VR 应用启动时
   的头显位置。
3. **夹爪 0=闭合**，和 TacCap 从爪的 `set_target()` 同一约定，所以录制值直通，
   中间没有任何转换。

### 1.3 指尖对齐（当前**未标定**）

录制的 `{side}_tcp.*` 是 **leader 夹爪的两指中点**（xense-taccap-lerobot 的
`ee_transform.py`，两侧都由 CAD 量出，2026-08-02 在 Rerun 里确认过标记点确实落在
两指中间）。而这套栈的 FK 返回的是**法兰**（`ccs_m6_40.MvKDCfg` 的 DH 链到法兰面
为止，没有任何地方调 `set_tool()`）。

**把指尖轨迹喂给瞄准法兰的 IK，结果是法兰忠实地走了人的指尖轨迹，而机器人自己的
指尖走在往外偏 |t| 的一条弧线上。** 纯平移时看不出来 —— 相对映射会把常数偏移抵消
掉，这也是之前回放看着一切正常的原因；一旦腕部旋转就不成立，而这正是操作 demo 里
最关键的那些帧（本段 demo 峰值 106°/s）。

实测（臂A，`SCALE=0.5`，`replay.py --umi` 离线）：

| 工具偏移 | 被钳位帧 | 笛卡尔限速介入 |
|---|---|---|
| 无（法兰） | 15 / 2484 | 7 帧 |
| `--tip 0,0,180` | **123 / 2484** | 65 帧 |

同一段数据、同一份预算，钳位多了 8 倍 —— 差的就是那条没人做过的弧线。

**怎么补上**（需要机器人，只读不下发）：

```bash
# 终端 1：零力拖动
set-state A release --tool umi
# 终端 2：夹爪完全闭合，把两指中点抵在一个固定尖点上，
#         保持这个点不动、把手腕摆到尽量不同的姿态，每摆好一次敲回车
tip-calib --arm A
```

它会打印可直接粘贴进 `configs/tool/umi.yaml` 的两行（`kine_offset` +
`kine_offset_source`）。质量门槛：≥4 个点、姿态张角 ≥60°、残差 ≤2mm ——
**姿态张角不够时残差小是骗人的**，那种情况下沿视线方向根本没有约束。

标定前所有 UMI 入口都会打印警告说明它判的是法兰轨迹；也可以先用 `--tip x,y,z`
临时试一个值。原理与"为什么只需要平移、不需要工具姿态"的证明见
`docs/algos.md#tool_framepy`。

坐标系映射 `config.AXIS_MAP_UMI` **不是新标定** —— 它是已有的 `AXIS_MAP` 与
`PICO_TO_WORLD_R` 复合的结果（`AXIS_MAP @ Gᵀ`，数值验证差值 0.0、det +1）。
所以它继承 `AXIS_MAP` 的可信度：A 臂有实测支撑，**B 臂那组仍是推导未验证**。
B 臂结果不对时先怀疑 `AXIS_MAP['B']`。推导过程见 `docs/drivers.md#umi_sourcepy`。

---

# 2. 离线验证（不需要机器人）

这是唯一完全不碰硬件的一步 —— `RobotConnection` 总会真连控制器，`--dry-run`
只是不下发。**别跳过这步**：坐标系映射错了，在这里表现为 IK 拒解率，而不是
表现为手臂往错误方向走。

```bash
python3 replay.py --umi ~/Downloads/replay_test --hand left
python3 replay.py --umi ~/Downloads/replay_test --hand right
```

**通过**：两只手都是

```
IK 成功    : 2484/2484 = 100.00%
安全闸放行 : 2484/2484 = 100.00%
P2 验收: ✅ 通过
```

常用参数：`--scale`（运动缩放）、`--vel-ratio`（速度档）、`--episode`。

**速度预算**：这段 demo 贴着上限跑。峰值 463 mm/s（左手）对 `VEL_RATIO=55` 时的
446 mm/s 额度，旋转 106°/s 对 107°/s。安全闸会钳位，钳位就是它的职责：

| `VEL_RATIO` | 左臂钳位帧 | 右臂钳位帧 |
|---|---|---|
| 55（默认） | 8 | 13 |
| 60 | 6 | 2 |
| 65 | 0 | 0 |

（`replay.py` 离线、从台面构型起步的数字。）`--vel-ratio 65` 两臂全清零，但速度档
要小步验证着往上提，别一次跳 —— 见 `docs/config.md#speed-budget`。

### 本体可行性过滤（把这台机器人做不到的段剔掉）

上面那两条命令回答"整段能不能跟"，这条回答"**哪几段**能跟、剩下的为什么不能"：

```bash
python3 scripts/umi_filter.py ~/Downloads/replay_test              # 双臂
python3 scripts/umi_filter.py ~/Downloads/replay_test --sweep --out mask.json
```

它把 episode 按控制周期逐帧过一遍**真实管线**（`Retargeter` → `ArmIK` →
安全闸，和 `ArmChannel` 装的是同一批对象），给每一帧打标签，再切成可用段和剔除段。
完全离线，不连机器人。

| 标签 | 含义 | 降速能救吗 |
|---|---|---|
| `ok` / `slow` | 跟得住（`slow` = 被钳位但滞后仍在 5mm 内） | — |
| `speed` | 超速度预算够久，指尖已经掉队 | ✅ |
| `ik:*` | IK 无解 / 分支跳变（backoff、轴投影都救不回来） | ❌ |
| `gate:*` | 安全闸拦下（关节限位等） | ❌ |
| `keepout:*` | 越 `UMI_REPLAY_KEEPOUT`（本体余量） | ❌，换起点 |

**判定是有前提的，不是数据本身的属性**：起始构型（默认
`config.UMI_START_JOINTS`，`--start home` / `--start-joints` 可换）、`--speed`、
`--scale`、`--vel-ratio`，还有 §1.3 的指尖偏移。这些都写进 `--out` 的 JSON 里，
否则那份 mask 不可复现。

示例数据、从 `UMI_START_JOINTS` 起、`--scale 1.0`：

| | 臂A | 臂B |
|---|---|---|
| 逐帧可跟随 | 2484/2484 | 2484/2484 |
| 峰值关节速率 | **1.42× 额度** | **0.86× 额度** |
| 指尖滞后峰值 | 1.8 mm | 1.3 mm |
| 最近限位 | 27.4° | 24.9° |

这两个倍率独立复现了 §4.3 的实机数字（A 需要约 127°/s 对 89°/s 额度 = 1.42×，
实跑被钳 10 帧；B 峰值 76.6°/s = 0.86×，实跑零钳位）—— 离线过滤器和实机对"哪条臂
是瓶颈、超了多少"给出同一个答案。

换成从 `HOME_JOINTS` 起，同一段数据臂A 只剩 8.8% 的帧，全部因 `keepout:Z` 被剔
—— 这就是 §3.1 那次险些撞机，以过滤器判据的形式出现，而不是以操作员拍急停的形式。

**`--sweep`：是做不到，还是只是太快？**

```
速度     臂A 可跟随 / 峰值速率  臂B 可跟随 / 峰值速率
1.00   100.00%  1.42×  100.00%  0.86×
0.70   100.00%  1.00×  100.00%  0.60×
0.50   100.00%  0.72×  100.00%  0.43×
```

**用它切出来的段回放**（下标是**录制帧**，和数据集同一套编号）：

```bash
python3 replay.py --umi ~/Downloads/replay_test --hand left --frames 33:52
python3 scripts/umi_replay.py ~/Downloads/replay_test --frames 33:52 \
    --goto-start --arms A --speed 0.3
```

双臂数据取**交集**：一条臂卡住的那几帧，另一条臂的帧也不能用了 —— 那个瞬间的
录制 action 已经不再描述机器人真正做了什么。

设计取舍（为什么短暂拒解要桥接、为什么过短的段要丢、为什么每段还要从起点复检一遍）
见 `docs/algos.md#embodimentpy`。

---

# 3. 起始构型（这条管线最关键的一步）

### 3.1 为什么

**余量是 demo 的属性，不是机器人的属性。** 相对映射复现的是示教者的位移；示教者
双手张开在身前起始、内侧空间充足，而机械臂有多少余量完全取决于它从哪个构型起步，
回放对此一无所知。

这段 demo 是"双手内收"动作，映射到基座系就是往 **−Z** 扫（两臂基座系互为镜像，
但"内侧"对两臂都是 −Z）：

| 臂 | 内收扫掠 | HOME 时 TCP Z | 从 HOME 起最近本体 |
|---|---|---|---|
| A | **152.8 mm** | +174.5 | **+22.2 mm** ← 2026-08-25 险些撞机 |
| B | **95.6 mm** | +174.5 | +79.3 mm |

注意：从 HOME 起步时 IK **2484/2484 全通过**、关节限位全满足、速率在额度内。
安全闸没有任何理由报警 —— 它看不见本体。

> **HOME 本身没问题。** 实测 w=304.4、σmin=0.733、限位余量 30°（瓶颈是 J2 的
> −120 下限，不是腕关节）。它**不是** UMI 用 `wrist2=+90` 避开的那个腕部对齐奇异
> —— 加 J6=+30 几乎不改变数值。HOME 缺的只有余量。

### 3.2 方案 A：固定起始构型（可复现，推荐日常用）

对应 UMI `bimanual_umi_env.py` 的 `j_init`：

```bash
python3 scripts/umi_replay.py ~/Downloads/replay_test --goto-start --arms A --speed 0.3
```

`--goto-start` 先以 6°/s 走到 `config.UMI_START_JOINTS`，再预演、确认、回放。

```python
UMI_START_JOINTS = {
    'A': [ 70.0, -80.0, -110.0, -70.0, 0.0, 20.0, 0.0],
    'B': [-70.0, -80.0,  110.0, -70.0, 0.0, 20.0, 0.0],
}
```

**刻意不从 HOME 推导。** HOME 是停放位（折叠收拢、TCP 离中心线仅 174.5mm），
这是一个"工作准备位"：前伸外展，离 HOME 的 TCP 219mm。**每个关节都是整十度** ——
这些数字要在终端里读出来再敲回去，`88.23` 这种值本身就是抄错的隐患。

来源：在可达空间上做 **10° 分辨率网格搜索**（330750 个臂部构型 × 整十度腕部组合），
按可操作度排序，约束为限位余量 ≥25° 且 TCP 落在目标框内。

| | A HOME | **A 起点** | B HOME | **B 起点** |
|---|---|---|---|---|
| TCP | 427.0/305.0/174.5 | 486.7/293.9/**384.4** | 427.0/−305.0/174.5 | 486.7/−293.9/**384.4** |
| 离 HOME | — | **219 mm** | — | **219 mm** |
| 可操作度 w | 304.4 | **312.8** | 303.7 | **312.8** |
| σmin | 0.733 | **0.782** | 0.731 | **0.782** |
| 起始限位余量 | 30.0° | **40.0°** | 30.0° | **38.0°** |
| 全管线跟随 | 2484/2484 | 2484/2484 | 2484/2484 | 2484/2484 |
| 最近本体 | 22.2 mm | **232.1 mm** | 79.3 mm | **289.2 mm** |

**每一项都比 HOME 好，不只是余量。**

B 是 A 的镜像（取反 J1/J3/J5/J7，保持 J2/J4/J6）—— 该规则用 `HOME_JOINTS` 和
`HOME_WAYPOINTS` 两组已知构型验证过，且能精确复现 Y 取反的 TCP。但 B 仍单独跑了
一遍完整管线，没有因为"镜像所以对"就跳过。

> 基座系竖直方向由 `robot.ini` 的重力向量定死：A 臂 `GravityY=+9.81`、
> B 臂 `GravityY=−9.81`，即 **A 的 +Y 向下、B 的 −Y 向下**（两臂镜像）。

**换数据要重算。** 扫掠量属于 demo。§3.4 给判据。

### 3.3 方案 B：release 手动摆位（UMI 的原生做法）

UMI 部署时并不靠 `j_init` 对齐数据 —— `eval_real.py` 是人用 SpaceMouse 把臂开到
合适起点，按 C 交给策略、S 停。我们的等价物是**零力拖动**：

```bash
# 终端 1：零力自由拖动，手把臂摆到起点
set-state A release --tool umi

# 终端 2：读关节角（纯只读，可与终端 1 并行）
get-current-pos --arms A
```

选 `release` 不选 `drag`：`release` 是 SDK 原生 `STATE_COOP_RELEASE`，**全轴**零力
且控制器自己做重力补偿，无需调 K/D；`drag` 的笛卡尔模式是**单方向**的，而且还背着
两个未解症状（某关节仍下垂、腕关节偏硬，见 `docs/scripts.md#set_statepy`）。

**`--tool umi` 不能省。** 不注册夹爪质量的话，它就是控制器重力模型外的未建模负载，
直接表现为下垂 —— 这是查了很久才定位的根因。`configs/tool/umi.yaml` 里 A(0.759kg)、
B(0.739kg) 两臂都已标定完（2026-08-23）。

**摆完不必写进 config。** 预演是从**实测位姿**算的，直接跑就行：

```bash
python3 scripts/umi_replay.py ~/Downloads/replay_test --arms A --speed 0.3
```

这就是 UMI "jog → 交接" 的循环，而且预演替代了操作员对余量的目测判断。

⚠️ **B 臂 release 有前科**：第一次 B 臂辨识（0.725kg）在 release 验证时严重下垂，
关节 4 漂到 −130°。现用值是第三次采集（0.739kg）。B 臂第一次进 release 手别离太远。

⚠️ 退出 release 后手臂失能、可能下坠。这件事是自愈的（预演读当下的实测位姿），
但别摆完放着不管太久。

### 3.4 怎么判断一个起点够不够

看预演打印的 **Z 最小值**。判据：

```
起点 TCP 的 Z  ≥  该臂的内收扫掠量  +  你要的本体余量
A: ≥ 152.8 + margin        B: ≥ 95.6 + margin
```

`config.UMI_REPLAY_KEEPOUT` 目前设了 Z 下限 100 mm，能拦下 HOME 起步（22.2 越界）、
放行上面那组起点（152.2）。

> ⚠️ **这个 100 mm 是 PROVISIONAL，不是量出来的。** 它来自 2026-08-25 那一次观察到的
> 接近，不是测量的本体包络。X/Y 我留空未设，同样是不想给假保护。**用真实测量值替掉它。**

---

# 4. 实机

### 4.1 dry-run（连接，但不下发）

```bash
python3 scripts/umi_replay.py ~/Downloads/replay_test --dry-run
```

`--dry-run` 对硬件完全无副作用：`prepare()` 跳过模式切换（`arm_driver.py:128`）、
`send_joint_commands` 提前返回（`:284`）、`disable()` 是 no-op（`:216`）、归位跳过。
只有 `state()` / `subscribe()` 读操作。

**通过**：两臂都 `2484/2484 帧`、`保护闭锁 0 次`、`下发 0 次`。

> dry-run 里跟踪误差恒为 0 是**故意的**：什么都不发，手臂不动，`q_meas` 停在原地而
> `q_cmd` 一直走，算出来的"跟踪误差"衡量的是 dry-run 本身而不是轨迹，0.5 秒内必然
> 触发。所以这条路径按完美伺服跟踪处理，和 `replay.py` 离线循环同一个假设。

### 4.2 下发前预演（自动，别跳过）

任何真实下发之前，脚本会用**真实的 `ArmChannel`**（只是不发送）把整条轨迹跑一遍，
打印 TCP 在基座系的包络，对照 keep-out 判断，再要求交互确认：

```
=== 下发前预演（未发送任何指令）===
臂A  起点 TCP [ +426.9  +304.6  +304.5] mm   跟随 2484/2484 帧   峰值关节速度 89.1°/s (额度 89)
      X   +412.0 ..  +431.1 mm   (行程  19.1 mm)
      Y   +297.5 ..  +317.6 mm   (行程  20.1 mm)
      Z   +152.2 ..  +304.5 mm   (行程 152.3 mm)
```

用真实 `ArmChannel` 而不是重写一遍管线，是因为**重写的预演可能与随后真跑的管线不一致**。

`--no-preview` / `--yes` 可跳过。别跳。

### 4.3 真实下发

```bash
# 单臂慢速（第一次务必如此）
python3 scripts/umi_replay.py ~/Downloads/replay_test --goto-start --arms A --speed 0.3

# 双臂
python3 scripts/umi_replay.py ~/Downloads/replay_test --goto-start --vel-ratio 65
```

**速度阶梯实测（2026-08-25，新起始构型，`VEL_RATIO=55`，`--scale 1.0`）**

臂 A（左手，录制位移 152.8mm）：

| 速度 | 帧数 | 跟踪误差峰值 | 最大帧增量 | 峰值关节速度 | 钳位 | 滞后峰值 |
|---|---|---|---|---|---|---|
| 0.3x | 8278/8278 | 1.96° | 0.156° | 39.0°/s | 0 | — |
| 0.5x | 4967/4967 | 2.94° | 0.258° | 64.6°/s | 0 | — |
| 0.7x | 3548/3548 | 3.30° | 0.356° | 89.1°/s | 1 | — |
| 1.0x | 2484/2484 | 4.20° | 0.357° | 89.1 / 89 | 10 | 1.8 mm |

臂 B（右手，录制位移 95.6mm）：

| 速度 | 帧数 | 跟踪误差峰值 | 最大帧增量 | 峰值关节速度 | 钳位 | 滞后峰值 |
|---|---|---|---|---|---|---|
| 0.3x | 8278/8278 | 1.32° | 0.091° | 22.8°/s | 0 | 0.4 mm |
| 0.5x | 4967/4967 | 2.07° | 0.153° | 38.1°/s | 0 | 0.7 mm |
| 1.0x | 2484/2484 | 3.83° | 0.306° | 76.6 / 89 | **0** | 1.3 mm |

**双臂同跑 1.0x**（`umi_replay.py DIR --speed 1.0`）：

| | A | B | 循环 |
|---|---|---|---|
| 帧数 | 2484/2484 | 2484/2484 | 2485 次，下发 2484 |
| 跟踪误差峰值 | 4.21° / 5.0 | 3.87° / 5.0 | — |
| 钳位 / 滞后峰值 | 10 / 1.8 mm | 0 / 1.3 mm | — |
| 保护闭锁 | 0 | 0 | 超时 **0** 次，抖动 **0.0 ms** |

全部零保护闭锁。TCP 最近本体：A 232.1mm、B 289.2mm。

四条可复用的经验：

1. **双臂同跑几乎不付出代价。** 250Hz 环每拍跑两条臂的 IK 仍是零超时、抖动 0.0ms，
   每臂指标与单臂跑相差 ≤0.04°。控制环有余量。
2. **跟踪误差随速度次线性增长**（A：1.96 → 2.94 → 3.30 → 4.20）。这是真实伺服跟踪
   误差，不是 dry-run 里那个恒为 0 的假值。1.0x 时仍有 16% 余量。
3. **1.0x 的瓶颈是关节速率而非跟踪误差，且只发生在 A 臂。** 左手录制位移 152.8mm、
   右手只有 95.6mm，所以 A 原速需要约 127°/s（额度 89）而 B 只要 76.6°/s。**B 在
   1.0x 是完全忠实的回放，A 有 10 帧被钳位**（2484 帧的 0.4%，滞后峰值 1.8mm）。
   要让 A 也完全忠实需要提 `--vel-ratio` —— 那是**另一个变量**，改的是控制器自身
   速度上限，按 `docs/config.md#speed-budget` 的规矩小步验证着提，别和回放速度一起调。
4. **两臂不会互撞**：两个 TCP 全程都在各自基座外侧（A 的 Z ≥ 232mm、B 的 Z ≥ 289mm），
   从更外侧往内收，不是朝对方去。注意这个判断给不出两爪之间的**绝对距离** ——
   两个基座的间距不在 `robot.ini` / `config.py` / URDF 里的任何地方。

> 报告里的「笛卡尔滞后」取全程**峰值**。早期版本打印 `rt.cart_lag_mm` 末帧瞬时值，
> 轨迹收敛后恒为 0.0mm，看着像没有滞后 —— 那是 bug，已修。

### 4.4 加夹爪

```bash
python3 scripts/umi_replay.py ~/Downloads/replay_test --goto-start --gripper
```

录制的夹爪值直通 `set_target()`，无转换（两端同为 0=闭合）。

### 4.5 常用参数

| 参数 | 含义 |
|---|---|
| `--speed 0.3` | **纯时间缩放**：慢 3 倍，路径完全相同。>1.0 会按同比例抬高笛卡尔速度 |
| `--scale 0.5` | 缩放运动本身。1.0（默认）复现原 demo |
| `--arms A` | 只跑一条臂 |
| `--goto-start` | 先走到 `UMI_START_JOINTS`（UMI 的 `--init_joints`） |
| `--vel-ratio 65` | 速度档，可消除本 demo 的钳位 |
| `--frames 33:52` | 只回放这一段（**录制帧**下标，`scripts/umi_filter.py` 会打印） |
| `--tip x,y,z` | 临时指定指尖偏移；默认读 `configs/tool/umi.yaml`（§1.3） |
| `--viewer` | 镜像到 universal_viewer |
| `--park start` | **默认**：收尾停在 `UMI_START_JOINTS`，下次可直接重跑 |
| `--park home` | 收尾停回 `HOME_JOINTS` —— 注意这条轨迹**没有余量从 HOME 起步** |
| `--no-home` | 结束完全不移动，停在轨迹终点 |

### 4.6 中止条件

- 安全闸 sustained 故障 → **录制回放没有离合可松开，故障闩锁无法清除**，脚本直接
  中止并报第几帧。这是刻意的：faulted 的回放不是成功的回放，不该跑完还打印健康报告。
- 夹爪保护（力矩 >0.30 N·m / 观测过期 / 无效）→ 中止。
- Ctrl+C → `before_disable` 里先停夹爪、再归位、再下伺服。

---

# 5. 夹爪

### 5.1 它不在机械臂的 CAN 上

TacCap 从爪是**每侧一条独立 USB 串口**（CH343 → MCU → FDCAN → 电机，
`/dev/ttyACM*` @ 3 Mbps），和 Marvin 的末端 CAN 透传毫无关系。所以控制器不通电也能
单独验夹爪，控制器重启也不影响它。

`drivers/umi_gripper.py` 与 `drivers/gripper.py`（OmniGripper/DM4310）**零代码共享**。
三个原因，按"错了有多难发现"排序，见 `docs/drivers.md#umi_gripperpy`；最要紧的一条：
**力矩量级差 25 倍**（0.30 N·m vs 8.0），照抄旧常量等于没有保护。

### 5.2 单独验证（不碰机械臂）

```bash
python3 drivers/umi_gripper.py --scan            # 列出 SN / role / side
python3 drivers/umi_gripper.py --arm A --num 3   # 3 次开合 + 实时遥测
```

`--scan` 先跑：它同时告诉你装的到底是 **follower（带电机）还是被动钳口** —— 整套
`ControlLoop` 方案只在 `role=Follower` 下成立。把 SN 填进 `config.UMI_GRIPPER_SN`
**写死侧别**，别靠自动判：USB 枚举顺序变了会静默把左右标反。

### 5.3 约定

**0 = 闭合，1 = 张开**，和 `drivers/gripper.py` 相反。这是刻意的：SDK 和录制数据
都用这个约定，所以主路径零转换，转换负担落在不常用的旧路径上。

---

# 6. 收工

```bash
ps aux | grep -E "umi_replay|run_teleop|MarvinPlatform" | grep -v grep   # 应无输出
get-current-pos                                                          # 确认停位
```

**收尾停在 `UMI_START_JOINTS`，不是 HOME**（`--park` 可改）。这是刻意的：HOME 正是
这条轨迹余量不够的那个位置（§3.1），停回去等于让下一次运行没法从原地起步 —— 不加
`--goto-start` 直接重跑会被预演以 `Z 轴越界` 拦下。停在起始构型则可以直接重跑。

`core.homing.home_arms` 因此接受 `targets_by_arm` 参数；不传时仍是 `config.HOME_JOINTS`，
`run_teleop.py` 的行为不变。

下伺服在**每一条退出路径**上都会执行：`ControlLoop.run()` 的 finally 管正常路径，
`main()` 的 finally 兜住所有提前 return（起始构型未到位、预演越界）。夹爪
`loop.stop()` → `motor.disable()` 同理。

---

# 7. 故障速查

| 症状 | 原因 / 处理 |
|---|---|
| `臂A 存在错误（状态 100 错误码 13）` | 急停闩锁。**先复位物理急停按钮**，脚本里的 `ensure_clear` 会带确认地清（`check_and_clear_errors` 只 fire 一次不确认，急停后不够用） |
| 预演报 `Z 轴越界` | 起点余量不够。`--goto-start`，或 release 手动摆到更外侧，或降 `--scale` |
| 跑到一半 `触发保护闭锁` | 见 §4.6。若伴随急停，跟踪误差多半是**结果**不是原因 —— 手臂冻结后指令继续发 0.5 秒就会累积到阈值 |
| `发现 0 个夹爪` | `lsusb \| grep 1a86` 查 CH343 是否枚举；`/dev/ttyACM*` 是否存在；用户是否在 `dialout` 组 |
| 夹爪 `未标定（flags=0x...）` | 归一化位置无物理意义，拒绝启动。跑上电自标定或厂商标定流程 |
| release 模式下垂 | `--tool umi` 忘了加；或该臂 `configs/tool/umi.yaml` 的质量/质心不准 |
| IK 拒解率高 | 坐标系映射或起始构型问题。先回 §2 离线定位，别在实机上试 |

---

# 附：一次完整流程的命令速查

```bash
# ---- 1. 数据 ----
python3 drivers/umi_source.py --self-test
python3 drivers/umi_source.py ~/Downloads/replay_test

# ---- 2. 离线（无机器人）----
python3 replay.py --umi ~/Downloads/replay_test --hand left
python3 replay.py --umi ~/Downloads/replay_test --hand right
python3 scripts/umi_filter.py ~/Downloads/replay_test --sweep    # 哪几段能跟
# 指尖偏移还没标（§1.3）：终端1 set-state A release --tool umi，终端2 tip-calib --arm A

# ---- 3. 夹爪（机械臂可不通电）----
python3 drivers/umi_gripper.py --scan
python3 drivers/umi_gripper.py --arm A --num 3

# ---- 4. 实机 ----
# 物理准备：手臂举空 → 周围清场 → 人手放 E-stop
python3 scripts/umi_replay.py ~/Downloads/replay_test --dry-run
python3 scripts/umi_replay.py ~/Downloads/replay_test --goto-start --arms A --speed 0.3
# 收尾停在 UMI_START_JOINTS，所以后续重跑不必再加 --goto-start：
python3 scripts/umi_replay.py ~/Downloads/replay_test --arms A --speed 0.3
python3 scripts/umi_replay.py ~/Downloads/replay_test --goto-start --gripper --vel-ratio 65

# 起点想手动摆（UMI 原生做法）：
#   终端1: set-state A release --tool umi
#   终端2: get-current-pos --arms A
#   然后不加 --goto-start 直接跑，预演会告诉你余量够不够

# ---- 5. 收工 ----
ps aux | grep -E "umi_replay|MarvinPlatform" | grep -v grep
get-current-pos
```
