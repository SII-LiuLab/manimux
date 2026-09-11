"""Shared execution runtime with replaceable inference strategies.

``execution.runtime`` selects the default latest-chunk strategy, ACT temporal
ensembling, Physical Intelligence real-time chunking, or a strategy plugin.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from manimux.config import ManiMuxConfig
from manimux.runtime.inference import build_inference_strategy

if TYPE_CHECKING:
    from manimux.runtime.edge import EdgeRuntime, RunResult


def __getattr__(name: str):
    # Workers unpickle runtime request types. Do not make the first inference
    # import all executors/viewer code and count that cost as model latency.
    if name in {"EdgeRuntime", "RunResult"}:
        from manimux.runtime import edge

        return getattr(edge, name)
    raise AttributeError(name)


def build_runtime(
    config: ManiMuxConfig,
    run_dir: Path,
    *,
    launch_mode: str = "run",
) -> EdgeRuntime:
    from manimux.runtime.edge import EdgeRuntime

    strategy = build_inference_strategy(config)
    if strategy.name != "rtc":
        return EdgeRuntime(config, run_dir, strategy=strategy, launch_mode=launch_mode)
    from manimux.runtime.rtc import RtcInferenceStrategy, RtcRuntime

    if not isinstance(strategy, RtcInferenceStrategy):
        raise TypeError("built-in RTC factory returned the wrong strategy type")
    return RtcRuntime(config, run_dir, strategy=strategy, launch_mode=launch_mode)


__all__ = ["EdgeRuntime", "RunResult", "build_runtime"]
