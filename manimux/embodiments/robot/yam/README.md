# YAM 整机装配

`YamRobot(RobotBase)` 从 YAML 装配左右 YAM 组件和可选相机。每条 CAN 由一个
`YamController` 管理；整机层只协调命名组和运动阶段，不穿透控制器调用 SDK。

```text
manimux/configs/embodiment/arm/yam.yaml          单臂配置与 SDK 参数
manimux/configs/embodiment/sensor/realsense.yaml 相机组件采集参数
manimux/configs/embodiment/robot/yam_dual.yaml   组件、安装关系与控制组
manimux/configs/embodiment/robot/yam_control.yaml 采集和部署共享的控制参数
manimux/configs/local/yam.example.yaml          CAN、相机序列号、网络服务绑定
```

`left_arm` 和 `right_arm` 各为 7 维，内置夹爪随 arm 一起控制，末端字段为 `null`。
相机的 `mount: null` 表示没有光学外参；配置不猜测相机位姿。
硬件环境安装见 [YAM 组件](../../arm/yam/README.md)，相机见
[RealSense 组件](../../sensor/realsense/README.md)。

## 生命周期

构造、模型加载及 FK/IK 不打开设备。`connect()` 默认不执行起始运动。
只有 `execute: true` 且 `move_to_start_on_connect: true` 才按 `start_joints`
移动。起始姿态按组名给出，保留旧驱动逐臂执行的顺序。Home 则先并行归零两臂、
保持当前夹爪开度，待全部完成后再打开夹爪。`home_on_close` 仍由 runtime 处理。
`execute: false` 禁用轨迹提交与 Home；显式 stop/关闭清理仍可提交当前位置保持。

采集与推理都通过 `embodiments.robot.build_robot()` 构造这个整机，不再使用
`robots/yam/` 或 `yam_dual` 驱动。既有实验已改为 `type: yam`，保留动作周期、
控制频率、horizon、执行限幅和模型身份。采集 GUI 从自己的 station 读取设备
绑定并传给相同组件；主从按钮和录制逻辑继续在采集模块中。

## 相机所有权与配置示例

`manimux/configs/experiments/put_bottles/yam_pi05_joint.yaml` 是新的组合配置示例。
独立相机服务用它的 `camera_server.cameras` 选择装配中的相机，runtime 读取网络
传感器，不会同时调用整机的 `start_sensors()`。单进程使用整机传感器时应由调用方
显式管理 `start_sensors()` / `close_sensors()`。同一设备只由一个采集端持有。

只检查离线装配，不连接硬件：

```python
from manimux.cli import load_config
from manimux.clock import SystemClock
from manimux.embodiments.robot import build_robot

config = load_config("manimux/configs/experiments/put_bottles/yam_pi05_joint.yaml",
                     local="manimux/configs/local/yam.example.yaml")
robot = build_robot(config["robot"], SystemClock())
print({name: model.num_coordinates for name, model in robot.kinematics.models.items()})
```

原始 CAN 高频记录与 SDK 锁探针已移除。普通轨迹与相机录制保留。
离线测试覆盖 SDK 接口、命名组、Home 阶段、中断与配置契约；不代表真机任务成功。
