"""用装配 YAML 创建 YAM 组件；构造、FK 和 Viewer 都不打开 CAN。"""

import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from manimux.clock import SystemClock
from manimux.embodiments.arm.yam import YamController
from manimux.embodiments.robot._interrupt import finish_move_before_interrupt
from manimux.embodiments.robot.base import RobotBase, RobotModel

log = logging.getLogger(__name__)


class YamRobot(RobotBase):
    """各组独立拥有一个臂爪一体 i2rt 会话，组名和安装关系来自配置。

    execute 控制完整 7 维目标，内置夹爪不受独立末端开关控制。
    connect 默认保持原位置；只有显式 move_to_start_on_connect 才移动。
    home 先保持夹爪开度归零双臂，再打开夹爪，沿用旧 YAM 的两阶段行为。
    close 负责释放连接；是否退出前 Home 仍由 runtime 决定。
    """

    def __init__(
        self,
        model,
        *,
        clock=None,
        hardware=None,
        component_hardware=None,
        execute=False,
        move_to_start_on_connect=False,
        start_joints=None,
        start_duration_s=5.0,
        home_duration_s=5.0,
        home_gripper_release_duration_s=1.0,
        home_on_close=False,
    ):
        clock = clock if clock is not None else SystemClock()
        control = {**model.hardware, **(hardware or {})}
        bound = {name: dict(component["hardware"]) for name, component in model.components.items()}
        for name, options in (component_hardware or {}).items():
            bound[name].update(options)
        arms = {}
        for name, group in model.groups.items():
            component = model.components[group.arm_name]
            options = dict(bound[group.arm_name])
            controller = YamController(channel=options.pop("channel", None), clock=clock, **options)
            arms[name] = component["class"](controller, kinematics=group.kinematics)
        sensors = {
            name: component["class"](
                name=name, clock=clock, **{**component["options"], **bound[name]}
            )
            for name, component in model.components.items()
            if component["type"] == "sensor"
        }
        super().__init__(
            arm_components=arms,
            models={name: group.kinematics for name, group in model.groups.items()},
            model=model,
            sensors=sensors,
            clock=clock,
            stale_timeout_s=control.get("stale_timeout_s", 0.2),
            execute=execute,
        )
        self._move_to_start = move_to_start_on_connect
        self._start_joints = start_joints
        self._start_duration = start_duration_s
        self._home_duration = home_duration_s
        self._release_duration = home_gripper_release_duration_s

    @classmethod
    def from_config(cls, path, **options):
        return cls(RobotModel.from_config(path), **options)

    def connect(self):
        if self._ready:
            return
        super().connect()
        if self._execute and self._move_to_start:
            self._move(
                self._start_joints,
                self._start_duration,
                "start position",
                interrupt=True,
                parallel=False,
            )

    def _move(self, targets, duration, label, *, interrupt=False, parallel=True):
        # 继续使用 SDK 的 move_joints，不在迁移中替换轨迹算法或运动参数。
        # 等待全部手臂返回后才进入下一阶段；反馈刷新同步旧包装的状态缓存。
        with finish_move_before_interrupt(label, log) as interrupted:
            if parallel:
                with ThreadPoolExecutor(max_workers=len(self.arm_components)) as pool:
                    futures = [
                        pool.submit(
                            arm.controller.move_joints,
                            np.asarray(targets[name]),
                            time_interval_s=duration,
                        )
                        for name, arm in self.arm_components.items()
                    ]
                    for future in futures:
                        future.result()
            else:
                for name, arm in self.arm_components.items():
                    arm.controller.move_joints(np.asarray(targets[name]), time_interval_s=duration)
        self.get_state()
        if interrupted and interrupt:
            raise KeyboardInterrupt

    def home(self):
        if not self._execute:
            return
        with self._lock:
            measured = self.get_state().groups
            targets = {name: np.r_[np.zeros(6), q[6]] for name, q in measured.items()}
            self._move(targets, self._home_duration, "zero home")
            for q in targets.values():
                q[6] = 1.0
            self._move(targets, self._release_duration, "home gripper release")
