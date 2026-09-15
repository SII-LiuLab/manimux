#!/usr/bin/env python3
"""Lightweight entry point; training orchestration lives inside XPolicyLab."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if __name__ == "__main__":
    from XPolicyLab.training.launch import main
    main()
