"""Real-time chunking runtime (arXiv:2506.07339)."""

from manimux.runtime.rtc.mask import inpainting_condition, soft_mask
from manimux.runtime.rtc.request import RtcInferenceRequest
from manimux.runtime.rtc.strategy import RtcInferenceStrategy


def __getattr__(name: str):
    if name == "RtcRuntime":
        from manimux.runtime.rtc.runtime import RtcRuntime

        return RtcRuntime
    raise AttributeError(name)

__all__ = [
    "RtcInferenceRequest",
    "RtcInferenceStrategy",
    "RtcRuntime",
    "inpainting_condition",
    "soft_mask",
]
