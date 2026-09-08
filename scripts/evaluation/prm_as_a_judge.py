#!/usr/bin/env python3
"""Run the PRM-as-a-Judge submodule with ManiMux runtime overrides.

The upstream Robo-Dopamine adapter currently fixes vLLM's GPU memory budget at
0.9.  ManiMux often evaluates recordings while another policy process remains
on the GPU, so ``PRM_GPU_MEMORY_UTILIZATION`` provides a local override without
carrying modifications inside the submodule.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRM_EVAL_ROOT = PROJECT_ROOT / "PRM-as-a-Judge" / "eval"


def _install_gpu_budget_override() -> None:
    raw_value = os.environ.get("PRM_GPU_MEMORY_UTILIZATION")
    if raw_value is None:
        return
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise SystemExit("PRM_GPU_MEMORY_UTILIZATION must be a number") from exc
    if not 0.0 < value <= 1.0:
        raise SystemExit("PRM_GPU_MEMORY_UTILIZATION must be in (0, 1]")

    from prm_judge.prm import dopamine_inference

    upstream_llm = dopamine_inference.LLM

    def configured_llm(*args: Any, **kwargs: Any) -> Any:
        kwargs["gpu_memory_utilization"] = value
        return upstream_llm(*args, **kwargs)

    dopamine_inference.LLM = configured_llm


def main() -> None:
    if not (PRM_EVAL_ROOT / "prm_judge" / "cli.py").is_file():
        raise SystemExit(
            "PRM-as-a-Judge submodule is missing; run "
            "git submodule update --init PRM-as-a-Judge"
        )
    sys.path.insert(0, str(PRM_EVAL_ROOT))
    _install_gpu_budget_override()

    from prm_judge.cli import main as upstream_main

    upstream_main()


if __name__ == "__main__":
    main()
