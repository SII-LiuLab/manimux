from pathlib import Path

import pytest
from pydantic import ValidationError

from manimux.viewer.config import load_viewer_config
from manimux.viewer.dashboard import _parser
from manimux.viewer.robots.yam import YamAdapter


def test_default_camera_roles_and_existing_yam_aliases() -> None:
    cfg = load_viewer_config()
    assert cfg.camera_mode == "policy"
    assert cfg.cameras.model_dump() == {"top": "top", "left": "left", "right": "right"}
    assert cfg.cameras.resolve(
        ["front_camera", "left_camera", "right_camera"], YamAdapter().camera_slot,
    ) == {"top": "front_camera", "left": "left_camera", "right": "right_camera"}


def test_viewer_cli_config_overrides_only_external_camera(tmp_path: Path) -> None:
    path = tmp_path / "viewer.yaml"
    path.write_text("camera_mode: manual\ncameras:\n  top: gemini305\n", encoding="utf-8")
    args = _parser().parse_args(["--robot", "yam", "--config", str(path)])
    cfg = load_viewer_config(args.config)
    assert cfg.camera_mode == "manual"
    assert cfg.cameras.resolve(
        ["front_camera", "gemini305", "left_camera", "right_camera"],
        YamAdapter().camera_slot,
    ) == {"top": "gemini305", "left": "left_camera", "right": "right_camera"}


def test_missing_explicit_source_does_not_display_another_camera() -> None:
    cfg = load_viewer_config(Path("configs/viewer/yam-gemini305.yaml"))
    assert cfg.cameras.resolve(
        ["front_camera", "gemini335", "left_camera", "right_camera"], YamAdapter().camera_slot,
    ) == {"left": "left_camera", "right": "right_camera"}


@pytest.mark.parametrize(
    "payload",
    [
        "camera_map: {top: front_camera}",
        "cameras: {cam_head: front_camera}",
        "cameras: {top: ''}",
        "cameras: {top: '  '}",
        "cameras: {top: null}",
        "camera_mode: guess",
    ],
)
def test_invalid_viewer_config_fails_before_startup(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "viewer.yaml"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValidationError):
        load_viewer_config(path)
