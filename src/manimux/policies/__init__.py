from __future__ import annotations

from collections.abc import Callable

from manimux.plugins import load_plugin
from manimux.policies.base import PolicyAdapter, PolicyModel, action_interval
from manimux.policies.capabilities import PolicyCapabilities
from manimux.policies.fake import FakePolicyAdapter, FakePolicyModel

PolicyModelFactory = Callable[[dict], PolicyModel]
PolicyAdapterFactory = Callable[[dict, dict], PolicyAdapter]


def _fake_model_factory(config: dict) -> PolicyModel:
    return FakePolicyModel(
        action_dt_ns=int(action_interval(config) * 1_000_000_000),
        horizon_steps=config["horizon_steps"],
        delay_s=config["inference_delay_s"],
    )


def _identity_adapter_factory(
    _robot: dict,
    _policy: dict,
    *,
    kinematics=None,
) -> PolicyAdapter:
    return FakePolicyAdapter()


_MODEL_BUILTINS: dict[str, PolicyModelFactory | str] = {
    "fake": _fake_model_factory,
    "molmoact_http": "manimux.integrations.molmoact_yam.policy_plugin:build_model",
    "abc_http": "manimux.integrations.abc_yam.policy_plugin:build_model",
    "xpolicylab_ws": "manimux.integrations.xpolicylab.policy_plugin:build_model",
}
_ADAPTER_BUILTINS: dict[str, PolicyAdapterFactory | str] = {
    "identity": _identity_adapter_factory,
    "molmoact_yam": "manimux.integrations.molmoact_yam.policy_plugin:build_adapter",
    "abc_yam": "manimux.integrations.abc_yam.policy_plugin:build_adapter",
    "xr1_yam": "manimux.integrations.xr1_yam.policy_plugin:build_adapter",
    "sapolicy_yam": "manimux.integrations.sapolicy_yam.policy_plugin:build_adapter",
    "lingbot_vla2_yam": "manimux.integrations.lingbot_vla2_yam.policy_plugin:build_adapter",
    "xpolicylab": "manimux.integrations.xpolicylab.policy_plugin:build_adapter",
    "openwam_yam": "manimux.integrations.openwam_yam.policy_plugin:build_adapter",
}


def build_policy_model(config: dict) -> PolicyModel:
    factory = load_plugin(
        config["worker"],
        group="manimux.policies.models",
        builtins=_MODEL_BUILTINS,
    )
    return factory(config)


def build_policy_adapter(robot: dict, policy: dict, *, kinematics=None) -> PolicyAdapter:
    factory = load_plugin(
        policy["adapter"],
        group="manimux.policies.adapters",
        builtins=_ADAPTER_BUILTINS,
    )
    if kinematics is not None:
        return factory(robot, policy, kinematics=kinematics)
    return factory(robot, policy)


from manimux.policies.decoder import ActionDecoderClient  # noqa: E402
from manimux.policies.worker import PolicyWorkerClient  # noqa: E402

__all__ = [
    "ActionDecoderClient",
    "FakePolicyAdapter",
    "FakePolicyModel",
    "PolicyAdapter",
    "PolicyCapabilities",
    "PolicyAdapterFactory",
    "PolicyModel",
    "PolicyModelFactory",
    "PolicyWorkerClient",
    "build_policy_adapter",
    "build_policy_model",
]
