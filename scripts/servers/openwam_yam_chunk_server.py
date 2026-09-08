#!/usr/bin/env python3
"""Run the official OpenWAM engine with a full-chunk ManiMux wire boundary.

This script is executed by OpenWAM's isolated environment. It reuses the
official checkpoint loader, observation preprocessor, inference engine, CLI,
and WebSocket server. Only ``PolicyServer.predict`` is specialized so one
request returns the complete generated action chunk instead of keeping the
chunk in OpenWAM's process and returning one row at a time.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import cast

import numpy as np

OPENWAM_ROOT = Path(os.environ.get("OPENWAM_ROOT", "/home/ubuntu/OpenWAM")).resolve()
if not (OPENWAM_ROOT / "openwam" / "deploy" / "server.py").is_file():
    raise RuntimeError(f"OPENWAM_ROOT is not an OpenWAM checkout: {OPENWAM_ROOT}")
sys.path.insert(0, str(OPENWAM_ROOT))
sys.path.insert(0, str(OPENWAM_ROOT / "third_party"))

import openwam.deploy.server as official_server  # noqa: E402

log = logging.getLogger("openwam.manimux_chunk_server")


class ManiMuxChunkPolicyServer(official_server.PolicyServer):  # type: ignore[misc]
    """Official server lifecycle with stateless full-chunk responses."""

    def _action_horizon(self) -> int:
        from omegaconf import OmegaConf

        # ``merge_deploy_cfg`` materializes the effective raw action window in
        # inference.num_frames. Prefer it so a deploy-side override cannot make
        # the advertised chunk contract disagree with engine.generate().
        num_frames = int(
            OmegaConf.select(
                self.cfg,
                "inference.num_frames",
                default=OmegaConf.select(self.cfg, "dataloader.num_frames", default=33),
            )
        )
        if num_frames <= 1:
            raise RuntimeError(f"OpenWAM inference.num_frames must exceed one, got {num_frames}")
        return num_frames - 1

    def _ckpt_contract(self) -> dict[str, object]:
        contract = cast(dict[str, object], super()._ckpt_contract())
        contract.update(
            {
                "transport_mode": "manimux_chunk",
                "action_horizon": self._action_horizon(),
            }
        )
        return contract

    def predict(self, obs: dict[str, object]) -> dict[str, object]:
        """Generate and return one complete physical-unit action chunk."""
        self._init_policy()
        started = time.monotonic()
        processed = self._obs_preprocessor.preprocess(obs)
        conditions: dict[str, object] = {"observation": processed}
        image = processed.get("image")
        if image is not None:
            conditions["first_frame_image"] = [image]
        if processed.get("prompt"):
            conditions["prompt"] = processed["prompt"]
        if processed.get("state") is not None:
            conditions["proprio"] = processed["state"]

        result = self.engine.generate(conditions)
        actions = result.get("actions")
        if hasattr(actions, "detach"):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions)
        expected = self._action_horizon()
        if actions.ndim != 2 or actions.shape[0] != expected or not np.isfinite(actions).all():
            raise RuntimeError(
                f"OpenWAM generated invalid action chunk {actions.shape}; expected ({expected}, D)"
            )

        binary_dims = (
            getattr(getattr(self.engine, "architecture", None), "binary_command_dims", ()) or ()
        )
        if binary_dims:
            actions = actions.copy()
            for dimension in binary_dims:
                if dimension >= actions.shape[1]:
                    raise RuntimeError(
                        f"binary command dimension {dimension} exceeds action width "
                        f"{actions.shape[1]}"
                    )
                actions[:, dimension] = np.where(actions[:, dimension] > 0.5, 1.0, -1.0)

        latency_ms = (time.monotonic() - started) * 1000.0
        self._request_count += 1
        self._total_latency += latency_ms
        log.info(
            "generated action chunk step=%d shape=%s latency_ms=%.2f",
            self._request_count,
            actions.shape,
            latency_ms,
        )
        return {
            "action": actions.tolist(),
            "step": self._request_count,
            "latency_ms": round(latency_ms, 2),
        }


def main() -> None:
    official_server.PolicyServer = ManiMuxChunkPolicyServer
    official_server.main()


if __name__ == "__main__":
    main()
