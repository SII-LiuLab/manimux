# 天机速度限制的来源与当前差异

核查日期：2026-09-13。通过 `git ls-remote origin refs/heads/main` 确认
`SII-LiuLab/manimux` 远端 main 为 `9e313ebbe3ffea0b85de44e282ddfdc2a0f74765`。
本地天机分支核查版本为 `d424996`。两者的 `src/manimux/config.py`、
`runtime/safety.py` 和整个 `runtime/executors/` 目录没有代码差异。

## 结论

CalibWrist 的等比例关节限速来自 teleop 的 `algos.safety.SafetyGate`，并非模型
自带。远端 ManiMux 已有独立的逐关节速度/加速度削减和超限拒绝机制。
本次天机集成使用的是 ManiMux 现有机制，新增了天机配置中的数值。
上述核查版本尚没有可选的等比例削减模式；本地后续实现状态见本文末节。

| 部分 | 实现来源 | 当前作用 |
|---|---|---|
| 厂商速度与加速度档 | Marvin SDK `set_vel_acc` | 控制器接收 `velRatio` 和 `AccRatio` |
| teleop 的命令速率预算 | `teleop/config.py` | 从厂商速度能力、速度档和 0.9 余量计算 |
| teleop 的等比例步长削减 | `teleop/algos/safety.py` | 每臂七个关节使用同一个缩放比例 |
| CalibWrist 的发送前轨迹处理 | `deploy/tianji/real_run.py::_validator` | 直接调用上述 SafetyGate，并保存削减后的关节命令 |
| CalibWrist 的发送时步长检查 | `deploy/tianji/command_sink.py::TianjiCommandSink.send` | 超过速率 × 命令时间间隔 × 1.05 就停，不在这里削减 |
| ManiMux 的命令削减 | `runtime/executors/limits.py` | 对每个关节分别限制速度、加速度 |
| ManiMux 的拒绝检查 | `runtime/safety.py::SafetyGuard` | 对命令差分计算速度、加速度，越界报错 |
| 天机在 ManiMux 中的具体限值 | `configs/experiments/runtime/tianji_taccap.yaml` | 整机 runtime 明确声明的执行与安全参数 |

## teleop 和 CalibWrist

teleop 当前工作树的参数为：

```text
JOINT_VMAX_DEG_S = 180.0       # SDK 的 ccs_m6_40 PNVA 表
VEL_RATIO = 32               # 本地未提交；HEAD 中为 20
ACC_RATIO = 100              # 控制器加速度档，独立于 VEL_RATIO
CMD_RATE_MARGIN = 0.90
MAX_JOINT_RATE_DEG_S = 180 × 32% × 0.9 = 51.84 deg/s
MAX_STEP_DT_S = 4 / 250 = 0.016 s
```

`SafetyGate.step_budget(dt)` 计算 `51.84 × min(dt, 0.016)` 度的单步预算。
250 Hz 时正常一帧预算为 0.20736°；循环发生停顿时最多只允许兑现 16 ms，
避免按整个停顿时间形成大步长。每条手臂各算一个缩放因子，不是将双臂十四个
关节合并为一组。

`clamp_joint_step` 对七关节增量 `d` 找到最大绝对值 `m`；若超预算 `b`，返回
`q_previous + d × (b/m)`。该实现至少已存在于本地 teleop 历史的 2026-08-14
版本，后续 `58bd1a43` 补充了等比例行为的说明和验证。

CalibWrist 的 `real_run.py` 从 `algos.safety` 导入 SafetyGate，使用 teleop 的
`config.MAX_JOINT_RATE_DEG_S` 等参数。检查不是只判断真假：它把
`verdict.joints` 写回预计算的轨迹，后续确实发送削减后的命令。
微分 IK 路径还把 QP 的 `vmax` 压到同一命令速率预算。

`IK_MAX_STEP_DEG = 1.8` 是解析 IK 相对种子的分支跳变检查，和以上按时间计算的
速度限制是两项独立检查，不能用它代替速度限制。

teleop 和 CalibWrist 的这条 SafetyGate 没有对命令序列计算二阶差分、再按软件
加速度上限削减。它们仍向控制器设置 `ACC_RATIO`；因此「没有这一层软件加速度
限制」不表示控制器没有加速度控制。

## 远端 ManiMux 确实已有的代码

- [limit_velocity / limit_step（核实版本）](https://github.com/SII-LiuLab/manimux/blob/9e313ebbe3ffea0b85de44e282ddfdc2a0f74765/src/manimux/runtime/executors/limits.py)：先用 `np.clip` 逐元素限制速度，再限制相对上一次速度的变化。
- [DirectExecutor（核实版本）](https://github.com/SII-LiuLab/manimux/blob/9e313ebbe3ffea0b85de44e282ddfdc2a0f74765/src/manimux/runtime/executors/direct.py)：配置 `motion_limits` 后，调用上述函数限制发出的关节命令；夹爪用自己的限值单独处理。
- [SmoothExecutor（核实版本）](https://github.com/SII-LiuLab/manimux/blob/9e313ebbe3ffea0b85de44e282ddfdc2a0f74765/src/manimux/runtime/executors/smooth.py)：也使用同一套逐关节限速函数，另有跟踪和平滑逻辑。
- [SafetyGuard（核实版本）](https://github.com/SII-LiuLab/manimux/blob/9e313ebbe3ffea0b85de44e282ddfdc2a0f74765/src/manimux/runtime/safety.py)：逐关节检查命令速度与加速度，超限报错。

历史上，SafetyGuard 的这段速度/加速度拒绝检查来自 `a946b87`（2026-08-28）；
共享 motion_limits 配置以及 DirectExecutor 的接入来自 `9d14670`（2026-09-12）。
它们已在远端 main 中，早于今天的天机集成。

框架支持与本体配置启用是两回事。远端
[YAM common.yaml](https://github.com/SII-LiuLab/manimux/blob/9e313ebbe3ffea0b85de44e282ddfdc2a0f74765/configs/robots/yam/common.yaml)
的 `command_safety` 为 null，arm 的速度/加速度限值也为 null，只显式限制夹爪
闭合速度 1.0。不能因为默认 YAM 没启用手臂限值，就断言远端没有实现。

核查版本的 `MotionRateConfig` 已允许速度或加速度设为 null；但 `CommandSafetyConfig`
和 SafetyGuard 要求位置上下界、速度、加速度一起配置。要保留关节限位和速度
拒绝、同时关闭软件加速度拒绝，需要调整这一通用配置约束与对应检查。

## 今天本地新增的是什么

`3d5beff` 为天机配置了以下参数，未改共享限速算法：

| 参数 | 当前数值 | 来源 |
|---|---|---|
| 控制器速度档 | 32% | 当时本地 teleop 的未提交配置 |
| SafetyGuard 速度上限 | 57.6°/s = 1.005309649 rad/s | 180°/s × 32% |
| executor 速度上限 | 51.84°/s = 0.904778684 rad/s | 再乘 teleop 的 0.9 命令余量 |
| SafetyGuard 加速度上限 | J1/J2 450°/s²，其余 900°/s² | SDK PNVA 表；将它用于软件检查是这次集成增加的选择 |
| executor 加速度上限 | 405°/s² = 7.068583471 rad/s² | 最小 PNVA 加速度再乘 0.9；同样是本次集成选择 |
| 夹爪速度/加速度 | 3.0 / 12.0，闭合速度 1.0 | ManiMux 既有夹爪参数约定 |

所以：速度数值的计算沿用了 teleop，算法执行方式使用了 ManiMux；额外启用的
软件加速度限制并不与 teleop 的 SafetyGate 等价。

## 两种削减方式的实际差异

离线直接调用两份代码的函数，使用相同单位与相同的每步最大增量 2：

```text
期望增量：       [1,    2,   8]
teleop 输出：    [0.25, 0.5, 2]  # 同比例缩放，保留关节增量方向
ManiMux 输出：   [1,    2,   2]  # 逐关节截断
```

为单独比较速度限制，ManiMux 示例关闭了加速度限制。该检查只执行纯数值代码，
没有连接硬件。等比例缩放保留的是关节增量方向，不等于保证笛卡尔空间的完整
轨迹在有限步长下完全不变。

原会话提出的 `mode: per_joint | isotropic` 和 `max_step_dt_s` 在远端 main 与本地
`d424996` 中均未实现。上面的来源核查没有修改行为。

## 后续按用户要求实现的配置模式

来源核查后，用户明确要求将削减方式做成配置模式。本地新增：

- `motion_limits.arm.mode: per_joint | isotropic`，默认 per_joint；Direct 和 Smooth
  共用，夹爪独立。每条手臂分别缩放，不将两个手臂耦合。
- `max_step_dt_s` 可选，约束速度预算中的单步时间，主循环和 nominal control dt 不变。
- 软件加速度拒绝检查可选；仍保留配置完整性、位置和速度拒绝检查。
- 天机 common profile 改用 isotropic、16 ms 步长时间上限、关闭软件手臂加速度削减
  和 command_safety 加速度检查；速度数值和控制器 `acc_ratio: 100` 保留。

显式配置加速度限制时，isotropic 的两阶段分别整体缩放速度向量、速度变化向量；
此时不能再声称最终位置增量一定保持原目标方向。默认 per_joint 的计算路径保留。
Smooth 的滤波/制动、位置边界和夹爪配置继续独立生效。用法见
[配置说明](../configs/README.md#选择手臂命令的削减方式)。

验证：`test_executors.py`、`test_config.py`、`test_tianji_driver.py` 共 111 项通过。
另在 30、100、250 Hz 下，以相同关节目标分别运行 teleop 的原始
`clamp_joint_step` 和 DirectExecutor 的 isotropic 模式；600 条手臂命令的最大
差异为 `1.1102230246251565e-16 rad`，夹爪目标不参与缩放。此对照关闭软件
手臂加速度限制，仅验证削减算法；没有连接硬件，也不验证整个运行时的轨迹等价性。
