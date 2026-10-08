"""YAM 组件直接使用安装的 i2rt；一次命令包含六个关节和内置夹爪。"""

import threading
from enum import Enum

import numpy as np

from manimux.clock import SystemClock
from manimux.embodiments.arm.base import ArmBase, ArmController, ArmModel, ArmState
from manimux.timing import stage

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
        with stage(f"yam.{self.channel}.sdk_get_joint_pos"):
            sdk_joints = self.robot.get_joint_pos()
        joints = sdk_joints.copy()
        self.sequence += 1
        return {self.channel: ArmState(joints, self.clock.now_ns(), self.sequence)}

    def send_commands(self, targets):
        # SDK 可能就地裁剪传入数组；命令本身仍由上层拥有。
        target = targets[self.channel].copy()
        with stage(f"yam.{self.channel}.sdk_command_joint_pos"):
            self.robot.command_joint_pos(target)

    def move_joints(self, target, *, time_interval_s):
        """起始姿态和 Home 沿用 SDK 插值；整机层负责协调各臂的阶段。"""
        self.robot.move_joints(target.copy(), time_interval_s=time_interval_s)

    def sent_command_snapshots(self):
        """Read cached CAN-send evidence; an unpatched SDK has no evidence."""
        getter = getattr(self.robot, "get_sent_command_snapshot", None)
        sample = getter() if callable(getter) else None
        return {} if sample is None else {self.channel: sample}

    def runtime_metadata(self):
        """Read connected SDK settings and calibrated bounds without device I/O."""
        from manimux.recording.provenance import (
            installed_package_provenance,
            loaded_module_provenance,
        )

        result = {
            "sdk": installed_package_provenance("i2rt"),
            "connected": self.robot is not None,
            "channel": self.channel,
            "arm_type": self.arm_type,
            "gripper_type": self.gripper_type,
            "captured_monotonic_ns": self.clock.now_ns(),
            "effective": {},
            "unavailable": {},
        }
        if self.robot is None:
            result["unavailable"]["effective"] = "controller_not_connected"
            return result
        result["sdk"]["loaded_robot_module"] = loaded_module_provenance(type(self.robot).__module__)
        method = getattr(self.robot, "get_robot_info", None)
        if callable(method):
            try:
                info = method()
            except Exception as exc:
                info = {}
                result["unavailable"]["sdk_robot_info"] = type(exc).__name__
        else:
            info = {}
            result["unavailable"]["sdk_robot_info"] = "sdk_method_not_available"
        fields = {
            "kp": "kp",
            "kd": "kd",
            "grav_comp_kd": "grav_comp_kd",
            "joint_limits_rad": "joint_limits",
            "gripper_limits_raw_rad": "gripper_limits",
            "gripper_index": "gripper_index",
            "gripper_force_limit_n": "limit_gripper_effort",
            "gravity_comp_factor": "gravity_comp_factor",
            "use_coulomb_friction": "use_coulomb_friction",
            "coulomb_friction": "coulomb_friction",
            "enable_auto_recovery": "enable_auto_recovery",
        }
        for output, key in fields.items():
            result["effective"][output] = _metadata_value(info.get(key))
            if key not in info:
                result["unavailable"][output] = "sdk_field_not_available"
        chain = getattr(self.robot, "motor_chain", None)
        limiter = getattr(self.robot, "_gripper_force_limiter", None)
        attributes = {
            "use_gravity_comp": (self.robot, "use_gravity_comp"),
            "clip_motor_torque_nm": (self.robot, "_clip_motor_torque"),
            "motor_offset_rad": (chain, "motor_offset"),
            "motor_direction": (chain, "motor_direction"),
            "last_reported_communication_hz": (chain, "comm_freq"),
            "gripper_effort_average_window_s": (limiter, "average_torque_window"),
            "gripper_clog_effort_threshold_nm": (limiter, "clog_force_threshold"),
            "gripper_clog_speed_threshold": (limiter, "clog_speed_threshold"),
        }
        for output, (owner, key) in attributes.items():
            if owner is None or not hasattr(owner, key):
                result["effective"][output] = None
                result["unavailable"][output] = "sdk_field_not_available"
            else:
                result["effective"][output] = _metadata_value(getattr(owner, key))
        result["gripper_calibration"] = {
            "effective_closed_open_raw_rad": result["effective"]["gripper_limits_raw_rad"],
            "source": "connected_sdk_gripper_limits",
            "procedure": None,
            "procedure_reason": "sdk_does_not_report_calibration_or_override_origin",
        }
        return result

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


def _metadata_value(value):
    """Convert copied SDK values to JSON without serializing device objects."""
    if isinstance(value, Enum):
        return _metadata_value(value.value)
    if isinstance(value, np.ndarray):
        return _metadata_value(value.tolist())
    if isinstance(value, np.generic):
        return _metadata_value(value.item())
    if isinstance(value, (list, tuple)):
        return [_metadata_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return None


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
