"""Validate XPolicyLab AutoHorizon settings through its public sampling API."""

import sys
from pathlib import Path

# The peer framework is a source checkout, not part of the ManiMux wheel.
source_parent = Path(__file__).resolve().parents[3] / "third_party"
if (source_parent / "XPolicyLab").is_dir():
    sys.path.insert(0, str(source_parent))

from XPolicyLab.utils.autohorizon_selection import validate_autohorizon  # noqa: E402

__all__ = ["validate_autohorizon"]
