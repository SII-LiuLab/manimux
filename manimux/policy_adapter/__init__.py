"""从实验配置加载实际适配器实现，不维护旧名称或模型专属注册表。"""

from manimux.plugins import load_plugin
from manimux.policy_adapter.base import PolicyAdapter


def build_policy_adapter(robot: dict, policy: dict, *, kinematics=None) -> PolicyAdapter:
    adapter_class = load_plugin(
        policy["adapter"]["type"], group="manimux.policy_adapter", builtins={}
    )
    return adapter_class(robot, policy, kinematics=kinematics)
