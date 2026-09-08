from __future__ import annotations

from collections.abc import Callable

from manimux.config import PolicyConfig, RobotConfig
from manimux.plugins import load_plugin
from manimux.policies.base import PolicyAdapter, PolicyModel
from manimux.policies.capabilities import PolicyCapabilities
from manimux.policies.fake import FakePolicyAdapter, FakePolicyModel

PolicyModelFactory = Callable[[PolicyConfig], PolicyModel]
PolicyAdapterFactory = Callable[[RobotConfig, PolicyConfig], PolicyAdapter]


def _fake_model_factory(config: PolicyConfig) -> PolicyModel:
    return FakePolicyModel(
        action_dt_ns=int(config.effective_action_dt_s * 1_000_000_000),
        horizon_steps=config.horizon_steps,
        delay_s=config.inference_delay_s,
    )


def _identity_adapter_factory(
    _robot: RobotConfig,
    _policy: PolicyConfig,
) -> PolicyAdapter:
    return FakePolicyAdapter()


def _molmoact_http_factory(config: PolicyConfig) -> PolicyModel:
    from manimux.integrations.molmoact_yam.policy_plugin import build_model

    return build_model(config)


def _molmoact_yam_adapter_factory(
    robot: RobotConfig,
    policy: PolicyConfig,
) -> PolicyAdapter:
    from manimux.integrations.molmoact_yam.policy_plugin import build_adapter

    return build_adapter(robot, policy)


def _abc_http_factory(config: PolicyConfig) -> PolicyModel:
    from manimux.integrations.abc_yam.policy_plugin import build_model

    return build_model(config)


def _abc_yam_adapter_factory(
    robot: RobotConfig,
    policy: PolicyConfig,
) -> PolicyAdapter:
    from manimux.integrations.abc_yam.policy_plugin import build_adapter

    return build_adapter(robot, policy)


def _xr1_yam_adapter_factory(
    robot: RobotConfig,
    policy: PolicyConfig,
) -> PolicyAdapter:
    from manimux.integrations.xr1_yam.policy_plugin import build_adapter

    return build_adapter(robot, policy)


def _xpolicylab_ws_factory(config: PolicyConfig) -> PolicyModel:
    from manimux.integrations.xpolicylab.policy_plugin import build_model

    return build_model(config)


def _xpolicylab_adapter_factory(
    robot: RobotConfig,
    policy: PolicyConfig,
) -> PolicyAdapter:
    from manimux.integrations.xpolicylab.policy_plugin import build_adapter

    return build_adapter(robot, policy)


def _sapolicy_yam_adapter_factory(
    robot: RobotConfig,
    policy: PolicyConfig,
) -> PolicyAdapter:
    from manimux.integrations.sapolicy_yam.policy_plugin import build_adapter

    return build_adapter(robot, policy)


def _lingbot_vla2_yam_adapter_factory(
    robot: RobotConfig,
    policy: PolicyConfig,
) -> PolicyAdapter:
    from manimux.integrations.lingbot_vla2_yam.policy_plugin import build_adapter

    return build_adapter(robot, policy)


def _openwam_ws_factory(config: PolicyConfig) -> PolicyModel:
    from manimux.integrations.openwam_yam.policy_plugin import build_model

    return build_model(config)


def _openwam_yam_adapter_factory(
    robot: RobotConfig,
    policy: PolicyConfig,
) -> PolicyAdapter:
    from manimux.integrations.openwam_yam.policy_plugin import build_adapter

    return build_adapter(robot, policy)


_MODEL_BUILTINS: dict[str, PolicyModelFactory] = {
    "fake": _fake_model_factory,
    "molmoact_http": _molmoact_http_factory,
    "abc_http": _abc_http_factory,
    "xpolicylab_ws": _xpolicylab_ws_factory,
    "openwam_ws": _openwam_ws_factory,
}
_ADAPTER_BUILTINS: dict[str, PolicyAdapterFactory] = {
    "identity": _identity_adapter_factory,
    "molmoact_yam": _molmoact_yam_adapter_factory,
    "abc_yam": _abc_yam_adapter_factory,
    "xr1_yam": _xr1_yam_adapter_factory,
    "sapolicy_yam": _sapolicy_yam_adapter_factory,
    "lingbot_vla2_yam": _lingbot_vla2_yam_adapter_factory,
    "xpolicylab": _xpolicylab_adapter_factory,
    "openwam_yam": _openwam_yam_adapter_factory,
}


def build_policy_model(config: PolicyConfig) -> PolicyModel:
    factory = load_plugin(
        config.worker,
        group="manimux.policies.models",
        builtins=_MODEL_BUILTINS,
    )
    return factory(config)


def build_policy_adapter(robot: RobotConfig, policy: PolicyConfig) -> PolicyAdapter:
    factory = load_plugin(
        policy.adapter,
        group="manimux.policies.adapters",
        builtins=_ADAPTER_BUILTINS,
    )
    return factory(robot, policy)


from manimux.policies.worker import PolicyWorkerClient  # noqa: E402

__all__ = [
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
