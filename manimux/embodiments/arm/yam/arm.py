"""YAM 组件直接使用安装的 i2rt；一次命令包含六个关节和内置夹爪。"""

import threading

from manimux.clock import SystemClock
from manimux.embodiments.arm.base import ArmBase, ArmController, ArmModel, ArmState

from .kinematics import YamManipulatorKinematics, visual_configuration


class YamController(ArmController):
    """每个 CAN 通道拥有一个 SDK 实例；构造只保存参数，connect 才打开设备。

    SDK 管理电机循环和夹爪归一化。这里仅适配 ManiMux 生命周期和状态接口。
    状态时间是 SDK 读取完成的主机时间，不能当作电机的采样时间。
    """

    def __init__(
        self, *, channel, clock=None, arm_type="yam", gripper_type="linear_4310", **hardware_options
    ):
        self.channel = channel
        self.clock = clock if clock is not None else SystemClock()
        self.arm_type = arm_type
        self.gripper_type = gripper_type
        self.hardware_options = hardware_options
        self.robot = None
        self.sequence = 0

    def connect(self):
        if self.robot is None:
            from i2rt.robots.get_robot import get_yam_robot
            from i2rt.robots.utils import ArmType, GripperType

            self.robot = get_yam_robot(
                channel=self.channel,
                arm_type=ArmType.from_string_name(self.arm_type),
                gripper_type=GripperType.from_string_name(self.gripper_type),
                **self.hardware_options,
            )

    def get_states(self):
        # 复制 SDK 当前反馈，不用最后一次发送的目标代替实测位置。
        joints = self.robot.get_joint_pos().copy()
        self.sequence += 1
        return {self.channel: ArmState(joints, self.clock.now_ns(), self.sequence)}

    def send_commands(self, targets):
        # SDK 可能就地裁剪传入数组；命令本身仍由上层拥有。
        self.robot.command_joint_pos(targets[self.channel].copy())

    def move_joints(self, target, *, time_interval_s):
        """起始姿态和 Home 沿用 SDK 插值；整机层负责协调各臂的阶段。"""
        self.robot.move_joints(target.copy(), time_interval_s=time_interval_s)

    def stop(self):
        if self.robot is not None:
            self.robot.command_joint_pos(self.robot.get_joint_pos().copy())

    def close(self):
        if self.robot is None:
            return
        native = self.robot
        # 已安装的 i2rt.close 只 join 服务线程。保持原退出顺序：两个循环
        # 都结束后才释放 CAN，避免电机循环在关闭的 socket 上继续读写。
        native._stop_event.set()
        native._server_thread.join(timeout=2.0)
        if native._server_thread.is_alive():
            raise RuntimeError("i2rt robot server thread did not stop; CAN left open")
        chain = native.motor_chain
        chain.running = False
        threads = [
            thread
            for thread in threading.enumerate()
            if getattr(getattr(thread, "_target", None), "__self__", None) is chain
        ]
        for thread in threads:
            thread.join(timeout=2.0)
        if any(thread.is_alive() for thread in threads):
            raise RuntimeError("i2rt motor control thread did not stop; CAN left open")
        chain.close()
        self.robot = None


class YamArm(ArmBase):
    """完整控制坐标为 7 维；end_effector: null 保留这个组件的内置夹爪。"""

    num_joints = 7

    def __init__(self, controller, *, kinematics):
        self._controller = controller
        self._kinematics = kinematics

    @classmethod
    def load_model(cls, **options):
        model = YamManipulatorKinematics(**options)
        return ArmModel(
            model,
            model.coordinates,
            model.solver._assets_root / "i2rt/robot_models/arm/yam/yam.urdf",
            None,  # 一体组件提供完整 TCP，不声明额外挂接工具的法兰。
            base_frame=model.base_frame,
            visual_mapping=visual_configuration,
        )

    @property
    def controller(self):
        return self._controller

    @property
    def channel(self):
        return self._controller.channel

    @property
    def kinematics(self):
        return self._kinematics
