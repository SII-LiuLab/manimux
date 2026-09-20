from __future__ import annotations

from pathlib import Path

import manimux.embodiments as robots_pkg


def test_robot_layer_does_not_import_any_policy_integration() -> None:
    """Every policy drives the same body, so the body must not depend on one."""
    root = Path(robots_pkg.__file__).resolve().parent
    for path in sorted(root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        assert "manimux.integrations" not in source, path
        assert "molmoact" not in source.lower(), path


def test_every_run_config_shares_one_yam_body() -> None:
    """The YAM config is embodiment state; it must not live under a policy."""
    import yaml

    repo = Path(robots_pkg.__file__).resolve().parents[2]
    seen = set()
    for config_path in sorted((repo / "manimux/configs").rglob("*.yaml")):
        config = yaml.safe_load(config_path.read_text())
        if not isinstance(config, dict) or config.get("robot", {}).get("type") != "yam":
            continue
        robot = config["robot"]
        if "config" in robot:
            seen.add((config_path.parent / robot["config"]).resolve())
            assert "right_config" not in robot.get("options", {})
    assert seen == {(repo / "manimux/configs/embodiment/robot/yam_dual.yaml").resolve()}
