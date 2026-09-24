from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


def test_xr1_sampler_check_runs_without_model_or_gpu() -> None:
    python = Path("envs/xr1/.venv/bin/python")
    if not python.is_file():
        pytest.skip("XR-1 environment is not installed")
    result = subprocess.run(
        [str(python), "scripts/validation/check_xr1_rtc_sampler.py"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["model_constructed"] is False
    assert payload["gpu_used"] is False
    assert payload["native"]["conditioned_inside_generate"] is True
    assert payload["xpolicy"]["conditioned_inside_generate"] is True
    assert payload["xpolicy"]["padding_excluded_when_masked"] is True
    assert payload["xpolicy"]["unmasked_default_unchanged"] is True
