# YAM 臂爪一体组件

`arm.py` 中的 `YamArm` / `YamController` 直接调用安装的 i2rt，没有另一层
`YAMRobot` 包装，也不在此目录复制 SDK。`kinematics.py` 包含完整 TCP 接口、
原有 MuJoCo FK / i2rt-Mink IK 和显示坐标映射；模型文件放在 `assets/`。

## 环境

在仓库根目录操作。新环境可以这样创建；已有环境直接使用其解释器安装依赖：

```bash
uv venv --python 3.12 envs/yam/.venv
uv pip install --python envs/yam/.venv/bin/python -e '.[replay,realsense,xpolicylab]'
uv pip install --python envs/yam/.venv/bin/python \
  'git+https://github.com/i2rt-robotics/i2rt.git@5d47b358bafb30c65e397f2ece506550a0db4594'
```

当前 SDK 基线是 i2rt 1.1.2、上述提交。SDK 管理 CAN、电机控制循环、重力补偿和
夹爪归一化。此版本线性夹爪力限为 50 N，不提供 `gripper_force_limit` 构造参数。
硬件进程不需要安装学习模型；模型服务使用 XPolicyLab 自己的环境。
这些环境是独立 venv，不要用 `uv sync` 将其覆盖为根目录的依赖集合。

## 接口

- 每臂状态与目标是 `[joint1 … joint6, gripper]`：关节为弧度，夹爪 0 闭合、1 张开。
- `end_effector: null` 表示无额外挂接工具，仍保留内置夹爪。
- `load_model()` 和构造组件都不打开 CAN；`connect()` 才创建 i2rt 对象。
- 状态直接复制 SDK 反馈；时间戳是读取完成的主机单调时钟，不是电机采样时刻。
- `stop()` 提交当前反馈位置；`close()` 等待服务线程和电机循环退出后释放 CAN。
  线程未退出时保留设备引用，沿用已有的关闭失败处理。
- TCP 为 `grasp_site`，FK/IK 均在各臂基座坐标系中计算，位置单位为米。
- IK 固定 `fixed_coordinates={"gripper": opening}`，成功返回完整 7 维配置。
- 显示时夹爪映射为两个指尖关节，URDF 是 8 维；该展开不用于硬件命令。

组件配置是 `manimux/configs/embodiment/arm/yam.yaml`。CAN 通道由整机的
`component_hardware` 或 local 配置绑定。双臂组装见
[整机说明](../../robot/yam/README.md)。

Teleoperation and demonstration collection are maintained outside this repository.

本次删除原始 CAN 旁路记录和锁耗时探针。正常 SDK 状态读取、图像、实际关节及
提交动作的 episode 记录继续保留；历史原始反馈数据仍可离线读取。
