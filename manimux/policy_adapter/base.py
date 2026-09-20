"""策略与整机之间的唯一适配接口；实现不拥有硬件连接。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from manimux.types import ActionChunk, ActionContext, InferenceRequest, ObservationSnapshot


class PolicyAdapter(ABC):
    """同一转换流程可被多个模型复用，具体实现由 policy.adapter.type 指定。

    policy.adapter 保存映射参数；policy 其余字段保存动作时间和客户端配置。
    kinematics 是已经装配的离线模型，传入它不会建立相机或机器人连接。
    """

    def __init__(self, robot: dict, policy: dict, *, kinematics=None):
        self.robot = robot
        self.policy = policy
        self.kinematics = kinematics

    def build_observation(self, snapshot: ObservationSnapshot) -> ObservationSnapshot:
        """无需改写观测的适配器直接保留原快照及时间戳。"""
        return snapshot

    def prepare_request(self, request: InferenceRequest) -> InferenceRequest:
        """专用适配器可添加位姿或历史信息，默认保持请求不变。"""
        return request

    @abstractmethod
    def decode_action(self, raw: object, context: ActionContext) -> ActionChunk:
        """将明确约定的服务端输出变成动作序列；基类不猜测动作格式。"""

    def validate(self, robot: dict, policy: dict) -> None:
        """保留专用适配器已有的契约检查入口；基类不添加检查。"""
        return None
