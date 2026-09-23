from __future__ import annotations

from collections.abc import Callable

from manimux.plugins import load_plugin
from manimux.policies.base import PolicyModel, action_interval
from manimux.policies.capabilities import PolicyCapabilities
from manimux.policies.fake import FakePolicyAdapter, FakePolicyModel

PolicyModelFactory = Callable[[dict], PolicyModel]


def _fake_model_factory(config: dict) -> PolicyModel:
    return FakePolicyModel(
        action_dt_ns=int(action_interval(config) * 1_000_000_000),
        horizon_steps=config["horizon_policy_steps"],
        delay_s=config["inference_delay_s"],
    )


_MODEL_BUILTINS: dict[str, PolicyModelFactory | str] = {
    "fake": _fake_model_factory,
    "molmoact_http": "manimux.policies.molmoact:build_model",
    "abc_http": "manimux.policies.abc:build_model",
    "xpolicylab_ws": "manimux.policies.xpolicylab.client:build_model",
}


def build_policy_model(config: dict) -> PolicyModel:
    factory = load_plugin(
        config["worker"],
        group="manimux.policies.models",
        builtins=_MODEL_BUILTINS,
    )
    return factory(config)


from manimux.policies.decoder import ActionDecoderClient  # noqa: E402
from manimux.policies.worker import PolicyWorkerClient  # noqa: E402

__all__ = [
    "ActionDecoderClient",
    "FakePolicyAdapter",
    "FakePolicyModel",
    "PolicyCapabilities",
    "PolicyModel",
    "PolicyModelFactory",
    "PolicyWorkerClient",
    "build_policy_model",
]
