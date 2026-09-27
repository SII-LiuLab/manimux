#!/usr/bin/env python3
"""Compatibility entry for XPolicyLab's recorded-YAM LeRobot converter."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    entry = Path(__file__).resolve().parents[2] / "XPolicyLab/utils/yam_to_lerobot.py"
    runpy.run_path(str(entry), run_name="__main__")
