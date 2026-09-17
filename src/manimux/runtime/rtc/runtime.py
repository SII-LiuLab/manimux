"""Real-time chunking as an inference strategy over the shared control runtime."""

from __future__ import annotations

from pathlib import Path

from manimux.config import ManiMuxConfig
from manimux.runtime.edge import EdgeRuntime
from manimux.runtime.rtc.strategy import RtcInferenceStrategy


class RtcRuntime(EdgeRuntime):
    """Compatibility entry point selecting RTC inside the shared runtime."""

    def __init__(
        self,
        config: ManiMuxConfig,
        run_dir: Path,
        *,
        strategy: RtcInferenceStrategy | None = None,
        launch_mode: str = "embedded",
    ) -> None:
        strategy = strategy or RtcInferenceStrategy(config)
        super().__init__(config, run_dir, strategy=strategy, launch_mode=launch_mode)
        self._rtc_strategy = strategy

    def _execution_horizon(self, horizon: int, delay: int) -> int:
        return self._rtc_strategy.execution_horizon(horizon, delay)
