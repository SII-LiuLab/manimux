#!/usr/bin/env python3
"""Offline proof that the XPolicyLab XR-1 path applies RTC inside the Euler sampler."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
XPOLICY_XR1_ROOT = (
    REPO_ROOT / "XPolicyLab/policy/Xiaomi_Robotics_1/xiaomi_robotics_1/xr1"
)


def _check_sampler(model_class: type[Any]) -> dict[str, object]:
    class Dummy:
        num_steps = 5

        @staticmethod
        def dit_forward(sample: torch.Tensor, timestep: torch.Tensor, **_: object) -> torch.Tensor:
            del timestep
            return torch.zeros_like(sample)

    dummy = Dummy()
    dummy._rtc_guidance_scale = types.MethodType(model_class._rtc_guidance_scale, dummy)
    dummy._generate_pi_rtc = types.MethodType(model_class._generate_pi_rtc, dummy)

    noise = torch.tensor([[[0.5, -0.5], [1.0, -1.0]]], dtype=torch.float32)
    target = torch.tensor([[[0.2, 0.3], [-0.4, 0.1]]], dtype=torch.float32)
    dummy._rtc_condition = target
    dummy._rtc_weights = torch.ones((1, 2, 1), dtype=torch.float32)
    dummy._rtc_beta = 5.0
    guided = model_class._generate(dummy, noise, {})
    if not torch.allclose(guided, target, atol=1e-6, rtol=0.0):
        raise AssertionError("conditioned sampler did not pull the sample to its target")

    dummy._rtc_weights = torch.zeros((1, 2, 1), dtype=torch.float32)
    unweighted = model_class._generate(dummy, noise, {})
    if not torch.equal(unweighted, noise):
        raise AssertionError("zero RTC weights changed the sampler output")

    return {
        "status": "ok",
        "denoise_steps": dummy.num_steps,
        "conditioned_inside_generate": True,
        "zero_weight_is_noop": True,
    }


def main() -> int:
    sys.path.insert(0, str(XPOLICY_XR1_ROOT))
    from mibot.models.VLA.xr1 import xr1 as XPolicyXR1

    print(
        json.dumps(
            {
                "xpolicy": _check_sampler(XPolicyXR1),
                "model_constructed": False,
                "gpu_used": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
