from __future__ import annotations

import re
from pathlib import Path

import manimux.embodiments.sensor.realsense as realsense_pkg


def test_camera_stack_does_not_import_any_policy_integration() -> None:
    """The camera service is shared by every policy, so it must not depend on one."""
    root = Path(realsense_pkg.__file__).resolve().parent.parent  # embodiments/sensor
    for path in sorted(root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        assert "manimux.integrations" not in source, path
        assert "molmoact" not in source.lower(), path


def test_realsense_capture_stays_one_explicit_stream_pair() -> None:
    """Capture is one explicit depth+color pair — no fallback or crop machinery."""
    source = (Path(realsense_pkg.__file__).resolve().parent / "camera.py").read_text()
    assert "_CAPTURE_FALLBACK" not in source
    assert "_center_crop" not in source

    streams = re.findall(
        r"enable_stream\(\s*rs\.stream\.(depth|color),\s*self\._width,"
        r"\s*self\._height,\s*rs\.format\.\w+,\s*self\._fps,?\s*\)",
        source,
    )
    assert len(streams) == 2
    assert set(streams) == {"depth", "color"}


def test_standalone_camera_config_matches_the_yam_devices() -> None:
    """The shared camera config must not drift from the embodiment config it replaces."""
    import yaml

    repo = Path(__file__).resolve().parents[2]
    standalone = yaml.safe_load((repo / "configs/cameras.yaml").read_text())
    embedded = yaml.safe_load((repo / "configs/robots/yam_left.yaml").read_text())
    assert standalone["sensors"]["cameras"] == embedded["sensors"]["cameras"]
