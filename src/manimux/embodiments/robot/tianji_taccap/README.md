# Tianji–TacCap 整机

整机名称 `tianji-taccap`，Python 类 `TianjiTaccapRobot`，实验配置类型
`robot.type: tianji_taccap`。机械臂组件仍叫 `TianjiArm`。

## 调用链

```text
实验 YAML → config.load_config()
          → EdgeRuntime
          → embodiments.robot.build_robot()
          → TianjiTaccapRobot.from_config()
          → RobotBase 管理 arm、end effector 和 sensor
```

注册入口按 `robot.type` 选择整机类，`robot.config` 相对实验 YAML 解析。
`RobotBase` 持有 `RobotModel` 并直接复用其 `kinematics`；子类不再覆盖一份独立运动学。
`RobotModel` 同文件保存组件模型和安装数据，支持不构造控制连接的离线 action decode。
旧 `robots` 导入和 `robot.driver` 配置保留兼容入口；YAM 实现和实验 YAML 未迁移。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `configs/embodiment/robot/tianji_taccap.yaml` | 组件、安装关系、控制组和显示资产 |
| `configs/experiments/runtime/tianji_taccap.yaml` | 原实验调度、平滑和执行约束 |
| `.local/tianji_taccap.yaml` | 私有设备、服务和路径绑定 |
| `configs/embodiment/arm/tianji_left.yaml`、`tianji_right.yaml` | 左右机械臂模型和控制参数 |
| `configs/embodiment/end_effector/taccap.yaml` | 末端执行器参数 |
| `configs/embodiment/sensor/taccap.yaml` | 相机采集参数 |
| `embodiments/robot/base.py` | 通用生命周期、组件协调、离线装配数据 |
| `embodiments/robot/tianji_taccap/robot.py` | 建立共享 Tianji 控制器及组件 |
| `embodiments/arm/tianji/arm.py` | 官方控制 SDK 连接、反馈、批量命令 |
| `embodiments/arm/tianji/kinematics.py` | 原有 Tianji 数值算法及官方法兰接口 |
| `kinematics/composed.py` | 通用 arm + end effector 的 TCP 变换组合 |
| `integrations/umi_dp_tianji/policy_plugin.py` | UMI 观测与动作格式、时间语义及 FK/IK 调用 |

SDK 和 assets 随所属组件存放。旧 `kinematics/tianji.py` 保留旧参数和 TCP 接口的兼容封装，
数值求解算法只有 arm 目录中的一份。`TianjiSDKKinematics` 和旧入口共同继承
`TianjiArmKinematics` 的原版求解规则。arm 不加载末端配置或保存 TCP 偏移；
新整机的末端装配由公共 `ComposedManipulatorKinematics` 负责。

## 坐标系与求解行为

每组 `[joint_1, ..., joint_7, gripper]`：关节 rad，夹爪 0 闭合、1 张开。
**FK 和 IK 的 TCP 位姿始终在对应机械臂基座系下，平移单位 m。**

```text
T_arm_tcp = FK_official(q_arm) × T_flange_tool × T_tool_tcp
T_arm_flange_target = T_arm_tcp_target × inverse(T_flange_tool × T_tool_tcp)
```

`root_frame`、机械臂的 `mount` / `MountedGroup.base_transform` 仅保存显示场景的安装关系，
不传入控制运动学。没有额外的 body 坐标系。改变显示安装位置不会改变 FK/IK 结果。
末端执行器的 `mount` 和自身 TCP 偏移参与控制计算，且只应用一次。

IK 保留原版官方 `ik + ik_nsp`、回代检查、关节限位和余量、实测 J6/J7 干涉约束及
相对 seed 的分支跳变处理。右臂 J6 的原有覆盖值位于 arm 配置。
装配层不再叠加另一套更严格的 TCP 接受阈值。失败不返回可执行的替代关节目标。

UMI 服务已把模型相对观测 TCP 的预测还原成各臂基座下的绝对目标。
ManiMux adapter 保持该约定，不添加公共坐标系转换。同进程使用整机的模型对象；
独立 action decode 子进程从相同配置加载离线模型，不获取硬件连接。
可选差分 IK 继续使用原 QP；装配层去掉末端偏移后把法兰目标交给原求解器。

## 离线构造和硬件生命周期

```python
from manimux.embodiments.robot.tianji_taccap import TianjiTaccapRobot

robot = TianjiTaccapRobot.from_config("configs/embodiment/robot/tianji_taccap.yaml")
# robot.fk(configuration) / robot.ik(targets, seed, fixed_coordinates=...)
# 此时未连接机械臂、夹爪或相机；无需填写全部硬件绑定。
```

`from_config()` 默认 `execute=False`、`end_effector_control=False`，与原实验的只读默认一致。
直接 Python `ip/arms` 构造保留此前 API 默认值。IP、串口和增益等设备参数在对应
`connect()` / `start()` 阶段使用，不再遍历整份配置汇总空字段报错。

`connect()` 读取机械臂和夹爪反馈；当前组件接口要求机械臂已禁用，首次执行命令时
才在实测关节处使能。左右臂共享一次目标批次，夹爪随后独立下发，跨设备不具有原子性。
配置中的关节限位、跟踪误差、反馈过期和控制器故障检查保留。
`stop()` 请求停止已拥有的组件，`close()` 失败后可重试。`home()` 尚未实现；没有后台轨迹规划。

相机启动独立于 `connect()`。新 UMI 实验仍订阅现有 camera server，未自动调用
`robot.start_sensors()`。订阅层把 `left_wrist/right_wrist` 映射为
`left_wrist_camera/right_wrist_camera`，保留 RGB、时间戳和帧序号。
光学外参为 null 表示未知；TacCap TCP 仍是原 CAD 固定近似，未新增标定或开合补偿。

## 实验入口与验证

实验入口位于 `configs/experiments/pass_ball/tianji_taccap_umi_dp.yaml`。
通过 `--local` 选择工位；policy 配置位于 `configs/policy/umi_dp/`。
仍使用既有 XPolicyLab UMI_DP 模型和 `xpolicylab_ws`，需用原启动脚本绑定 checkpoint
身份后才可运行。细节见 `docs/umi-dp-tianji-taccap-runbook.md`。

测试覆盖注册到 adapter、原算法数值回归、场景变换不影响控制、子进程解码一致性、
原时间/运动约束保持，以及 fake SDK 下的只读和执行分发。未启动真实硬件或模型服务。
本轮没有改 Viewer；临时 Viewer 接入仍待后续修正坐标语义，不能作为新整机显示入口使用。
交互服务不加载新整机尚未实现的回零/拖动恢复功能，旧恢复模块不在本轮迁移范围。
