"""Configuration selects conversion behavior independently of the model client."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from manimux.cli import load_config
from manimux.policy_adapter import build_policy_adapter
from manimux.types import ActionContext


def test_same_policy_can_select_distinct_action_formats(tmp_path):
    """A service's wire contract, not its model name, determines the adapter."""
    fixture = yaml.safe_load(Path("tests/fixtures/runtime.yaml").read_text())
    fixture["robot"]["group_dims"] = {"arm": 3}
    fixture["policy"].update(horizon_policy_steps=3, action_dt_s=0.1)
    rows = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]])
    context = ActionContext(request_seq=7, observation_time_ns=100, created_time_ns=200)
    chunks = []
    for implementation, payload in [
        ("manimux.policy_adapter.joint:JointAdapter", {"format": "joint", "actions": {"arm": rows}}),
        ("manimux.policy_adapter.abc_yam:AbcYamAdapter", rows),
    ]:
        recipe = deepcopy(fixture)
        recipe["policy"]["adapter"] = {
            "type": implementation,
            "group_order": ["arm"],
            "group_prefixes": {"arm": ""},
            "gripper_dofs": 1,
            "camera_map": {},
        }
        path = tmp_path / "experiment.yaml"
        path.write_text(yaml.safe_dump(recipe))
        config = load_config(path)
        assert config["policy"]["worker"] == fixture["policy"]["worker"]
        adapter = build_policy_adapter(config["robot"], config["policy"])
        chunks.append(adapter.decode_action(payload, context))

    for chunk in chunks:
        np.testing.assert_array_equal(chunk.groups["arm"], rows)
        assert chunk.action_space == "joint_position"
        assert chunk.dt_ns == 100_000_000
        assert chunk.request_seq == 7
        assert chunk.observation_time_ns == 100
        assert chunk.created_time_ns == 200


def test_algorithm_and_executor_presets_are_independent_and_relative(tmp_path, monkeypatch):
    recipe = yaml.safe_load(Path("tests/fixtures/runtime.yaml").read_text())
    (tmp_path / "inference.yaml").write_text(
        "commit_lead_s: 0.01\nrtc:\n  min_execute_policy_steps: 8\n"
    )
    (tmp_path / "executor.yaml").write_text("smooth:\n  cutoff_hz: 6.0\n")
    recipe["inference"] = {"algorithm": "rtc", "config": "inference.yaml"}
    recipe["executor"] = {"type": "smooth", "config": "executor.yaml"}
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(recipe))
    monkeypatch.chdir(tmp_path.parent)

    rtc = load_config(path)
    recipe["inference"] = {"algorithm": "manimux", "inference_schedule": "single_inflight"}
    path.write_text(yaml.safe_dump(recipe))
    ordinary = load_config(path)

    assert rtc["inference"]["rtc"]["min_execute_policy_steps"] == 8
    assert rtc["inference"]["commit_lead_s"] == 0.01
    assert rtc["executor"]["smooth"]["cutoff_hz"] == 6.0
    assert ordinary["executor"] == rtc["executor"]
    assert ordinary["policy"] == rtc["policy"]
    assert "execution" not in rtc
    assert "config" not in rtc["inference"]
    assert "config" not in rtc["executor"]
