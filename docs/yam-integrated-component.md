# YAM 臂爪一体组件

YAM 组件位于 `manimux/embodiments/arm/yam/`。它拥有自带夹爪的完整模型和
i2rt 会话，控制布局为六个弧度关节加一个归一化夹爪值：`0` 闭合，`1` 张开。

```text
manimux/
├── embodiments/
│   ├── arm/yam/
│   │   ├── arm.py            # SDK 生命周期、完整 7 维读写及模型入口
│   │   ├── kinematics.py     # FK/IK、完整 TCP 和显示坐标映射
│   │   ├── assets/           # 原模型和网格，保留授权说明
│   │   └── README.md         # i2rt 安装及控制约定
│   ├── robot/yam/
│   │   ├── robot.py          # 按配置装配、起始姿态及两阶段 Home
│   │   └── README.md
│   └── sensor/realsense/
│       ├── sensor.py         # 唯一 SDK 采集、后台缓存和资源释放
│       └── README.md
├── camera_server/            # 通用 ZMQ 相机服务与客户端
└── collection/yam/camera/
    └── realsense.py          # 只转换采集帧格式
```

```yaml
groups:
  left_arm: {arm: left_yam, end_effector: null}
  right_arm: {arm: right_yam, end_effector: null}
```

`null` 表示不额外挂接末端组件。装配层直接复用 `ManipulatorKinematicsBase`，
不会把 YAM 当成裸法兰，不会追加第二次 TCP 偏移，也不会拆分夹爪命令。
Tianji–TacCap 的独立工具继续走 `ComposedManipulatorKinematics`。

## 接口和坐标

- 完整状态、目标和 IK 结果每组均为 7 维；`ArmBase.num_joints` 在这里表示组件
  提交的坐标数，包含内置夹爪。组名来自装配 YAML。
- `YamManipulatorKinematics` 复用原 `YamKinematics` 求解器，TCP 为 `grasp_site`，
  位姿始终相对对应手臂基座。本体不定义公共 `root_frame` 或机械臂底座显示变换；
  Viewer 的 `groups.<name>.viewer_display_frame` 独立负责场景摆放。
- IK 使用 `fixed_coordinates={"gripper": opening}`，先将目标夹爪值交给原求解器，
  再返回完整配置；不收敛时返回无解结果。IK 可选依赖仍为原 i2rt/mink 环境。
- Viewer 的两个指尖关节使用米，按原显示映射 `opening * -0.04695` 展开。
  这个 8 维显示配置只交给 URDF，控制与 policy 始终使用 7 维。
- 配置按上述接口直接读取。新接入代码不增加配置、类型和维度的手写报错校验；
  SDK 的原生错误自然传播。原求解器与必要的线程退出行为保留；不再保留旧硬件转发层。

## 连接和执行

`RobotModel.from_config()`、`YamRobot.from_config()` 和 Viewer 模型加载均不打开 CAN。
`connect()` 才创建 i2rt 设备，直接调用 SDK 的初始化流程。反馈时间表示 SDK 读取
完成的主机时刻，不冒充电机采样时间。每个 CAN 通道由一个 controller 管理。

`execute` 对臂爪一体的整个目标生效。`move_to_start_on_connect` 仅在允许执行时
使用配置的分组起始姿态；默认不执行起始运动。Home 保持旧顺序：先保持夹爪开度
将所有臂归零，等待完成后再打开夹爪。stop 保持反馈位置；close 释放 SDK 线程和
CAN，是否退出前 Home 仍由 runtime 的 `home_on_close` 决定。

## 配置入口

- 组件：`manimux/configs/embodiment/arm/yam.yaml`
- 整机：`manimux/configs/embodiment/robot/yam_dual.yaml`
- 工位：`manimux/configs/local/yam.example.yaml`，实际绑定可复制到忽略的 `.local/`
- Pi05 示例：`manimux/configs/experiments/put_bottles/yam_pi05_joint.yaml`
- Viewer：`manimux/viewer/robots/yam/viewer.yaml`，入口支持 `manimux-viewer --robot yam`

Pi05 示例匹配已有 `manimux/configs/policy/pi05/yam/put-bottles/joint-step30000.yaml` 服务，
沿用原 30 Hz 动作间隔、50 步 horizon、100 Hz 控制环、RTC、平滑和运动限幅。
预期后端保留 checkpoint/stats 来源及模型身份，公共模板不包含本机绝对权重路径。
初始 `execute: false`；它是实际设备实验配置，不是 mock 机器人配置。

下面仅构造对象与离线模型，不连接设备或服务：

```python
from manimux.cli import load_config
from manimux.clock import SystemClock
from manimux.embodiments.robot import build_robot

config = load_config(
    "manimux/configs/experiments/put_bottles/yam_pi05_joint.yaml",
    local="manimux/configs/local/yam.example.yaml",
)
robot = build_robot(config["robot"], SystemClock())
```

YAM 采集与原有 63 份实验入口已切换为 `embodiments.robot.yam.YamRobot`。
旧 `robots/yam/`、`kinematics/yam.py`、独立 `hardware.py` / `model.py` 已移除。
原始 CAN 反馈旁路录制、锁探针及其配置/调用入口已删除，正常状态与动作记录保留。
依赖旧协议的历史 MolmoAct 直连 launcher 随旧驱动退役；这不代表其模型迁移完成。

RealSense 的 SDK 实现统一在 `embodiments/sensor/realsense/`，网络服务移到
`camera_server/`；采集侧只保留格式适配。安装、生命周期与参数见：

- [YAM 组件](../manimux/embodiments/arm/yam/README.md)
- [YAM 整机](../manimux/embodiments/robot/yam/README.md)
- [RealSense](../manimux/embodiments/sensor/realsense/README.md)

策略适配器仍保留各自已配置的运动学选项；一级 `policy_adapter/` 的统一收敛尚未实施。

## 离线验证

`tests/unit/test_yam_assembly.py` 覆盖一体模型直通、旧 FK/显示映射一致性、假 SDK 的
7 维完整提交、连接生命周期、Home 阶段顺序、Pi05 动作契约和分体装配回归。
真实求解器回环测试需要 i2rt/mink；它只计算运动学，不连接硬件。
这些检查不代表模型服务就绪或真机任务成功。

本次迁移检查：189 项相关离线测试通过；63 份既有配置的 policy、execution、
sensors、run、group_dims 和 control_hz 与修改前一致，全部整机可离线构造。
wheel 已构建并从独立目录加载 YAM 双臂模型、网格、Viewer 和相机服务入口。
没有连接机器人或相机，没有重新验证 Tianji。

额外运行的 `test_control_timing_diagnostics.py` 与 `test_collection_timing.py`
仍有 17 项失败。相同测试在修改前备份中有 18 项失败，本次没有新增失败，
CameraHub 移除旧帧的修复使其中 1 项通过。既有在线调频、计时配置和写入帧率
问题不属于本次本体迁移，未据此宣称全仓测试通过。
