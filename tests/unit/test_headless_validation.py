from __future__ import annotations

from pathlib import Path

import pytest

from manimux.config import load_config
from scripts.validation.run_headless import run_headless


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("viewer", "enabled", True, "use manimux serve"),
        ("robot", "driver", "yam_dual", "supports only mock_dual_arm"),
        ("sensor", "driver", "camera_server", "supports only mock_camera"),
    ],
)
def test_headless_rejects_device_or_viewer_configs_before_runtime_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    config = load_config("configs/mock.yaml")
    config.run.output_dir = tmp_path
    target = config.sensors[0] if section == "sensor" else getattr(config, section)
    setattr(target, field, value)
    monkeypatch.setattr("manimux.cli._load_config", lambda *_args: config)

    def unexpected_runtime(*_args, **_kwargs):
        pytest.fail("invalid headless config created a runtime")

    monkeypatch.setattr("manimux.runtime.build_runtime", unexpected_runtime)

    with pytest.raises(ValueError, match=message):
        run_headless(Path("configs/mock.yaml"))

    assert not list(tmp_path.iterdir())
