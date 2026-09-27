#!/usr/bin/env python3
"""Compatibility entry point for the XPolicyLab XR1 YAM converter."""

from pathlib import Path
import runpy

if __name__ == "__main__":
    converter = (
        Path(__file__).resolve().parents[2]
        / "XPolicyLab/policy/Xiaomi_Robotics_1/prepare_yam_dataset.py"
    )
    runpy.run_path(str(converter), run_name="__main__")
