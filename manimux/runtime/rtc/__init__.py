"""Real-time chunking runtime (arXiv:2506.07339)."""

from manimux.runtime.rtc.mask import inpainting_condition, soft_mask
from manimux.runtime.rtc.request import RtcInferenceRequest
from manimux.runtime.rtc.strategy import RtcInferenceStrategy

__all__ = [
    "RtcInferenceRequest",
    "RtcInferenceStrategy",
    "inpainting_condition",
    "soft_mask",
]
