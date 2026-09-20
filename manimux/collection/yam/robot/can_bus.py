"""Bring the CAN buses up.

Run at go-live (first Start Teleop) on the hardware path and re-exposed as the GUI
"Reset CAN" action. Reuses i2rt's vendored reset_all_can.sh.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "third_party" / "i2rt" / "scripts" / "reset_all_can.sh"
)


def list_can_interfaces() -> list[str]:
    """Every CAN interface present on this machine, udev-named or raw (``can_left``,
    ``can0``, ...). Feeds the GUI's channel pickers."""
    net = Path("/sys/class/net")
    if not net.is_dir():
        return []
    return sorted(p.name for p in net.iterdir() if p.name.startswith("can"))


def check_can_up(channels: list[str], timeout_s: float = 5.0) -> list[str]:
    """Return the subset of ``channels`` that are NOT present-and-up."""
    down: list[str] = []
    for ch in channels:
        try:
            r = subprocess.run(
                ["ip", "link", "show", ch],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except (subprocess.SubprocessError, OSError):
            down.append(ch)
            continue
        out = r.stdout or ""
        if r.returncode != 0 or ("state UP" not in out and "state UNKNOWN" not in out):
            down.append(ch)
    return down


def reset_can_buses(timeout_s: float = 30.0) -> tuple[bool, str]:
    raise RuntimeError("CAN reset is disabled in ManiMux collection")
