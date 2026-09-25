from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

from manimux.viewer.dashboard import PolicyViewer


@pytest.fixture
def viewer(tmp_path):
    viewer = PolicyViewer.__new__(PolicyViewer)
    viewer.lock = threading.RLock()
    viewer.experiment_mode = True
    viewer.evaluation_complete = False
    viewer.episode_active = True
    viewer.episode_finalized = False
    viewer.current_episode_dir = tmp_path
    viewer.service_ready = False
    viewer.service_id = "test-session"
    viewer.launch_mode = "serve"
    viewer.preparing_rollout = False
    viewer.task_result = SimpleNamespace(value="unlabeled")
    viewer.smoothness_score = SimpleNamespace(value="3")
    viewer.reviewer_id = SimpleNamespace(value="operator")
    viewer.operator_note = SimpleNamespace(value="")
    viewer.failure_tag_inputs = {}
    for name in (
        "save_evaluation_btn", "skip_evaluation_btn", "prepare_normal_btn",
        "prepare_experiment_btn", "layout_id", "task", "status", "evaluation_status",
        "executor_info", "episode_path", "policy_name", "runtime_name", "rollout_setup_status",
    ):
        setattr(viewer, name, SimpleNamespace(disabled=True, visible=True, value="", content=""))
    for name in (
        "new_rollout_folder", "policy_control_folder", "recovery_folder",
        "evaluation_folder", "overlay_folder", "run_folder",
    ):
        setattr(viewer, name, SimpleNamespace(visible=False))
    viewer.camera_view = SimpleNamespace(clear_images=lambda: None)
    viewer._set_policy_controls_enabled = lambda enabled: None
    viewer._update_recovery = lambda metadata: None
    viewer._set_instruction = lambda instruction: None
    return viewer


def finish(viewer):
    viewer._update_event({"event": "episode_finished", "metadata": {}})


def ready(viewer):
    viewer._update_event({
        "event": "runtime_service_ready", "metadata": {"run_dir": "test-session"},
    })


@pytest.mark.parametrize("ready_before_skip", [False, True])
def test_experiment_can_skip_without_a_label_and_prepare_next(viewer, ready_before_skip):
    finish(viewer)
    assert viewer.evaluation_folder.visible
    assert not viewer.skip_evaluation_btn.disabled
    if ready_before_skip:
        ready(viewer)
    assert viewer.prepare_normal_btn.disabled

    viewer._skip_manual_evaluation()

    assert viewer.evaluation_complete
    assert not viewer.evaluation_folder.visible
    viewer.task_result.value = "success"
    viewer._save_manual_evaluation()
    assert not (viewer.current_episode_dir / "evaluation").exists()
    if not ready_before_skip:
        assert viewer.prepare_normal_btn.disabled
        ready(viewer)
    assert not viewer.prepare_normal_btn.disabled
    viewer._prepare_rollout(experiment_mode=False)
    assert viewer.new_rollout_requested
    assert not viewer.experiment_mode


def test_normal_rollout_never_requires_scoring(viewer):
    viewer.experiment_mode = False
    finish(viewer)
    ready(viewer)
    assert viewer.evaluation_complete
    assert not viewer.evaluation_folder.visible
    assert viewer.save_evaluation_btn.disabled
    assert viewer.skip_evaluation_btn.disabled
    assert not viewer.prepare_normal_btn.disabled
    assert not (viewer.current_episode_dir / "evaluation").exists()


def test_skip_during_rollout_does_not_release_evaluation_gate(viewer):
    viewer._skip_manual_evaluation()
    assert not viewer.evaluation_complete


def test_experiment_can_still_save_a_real_label(viewer):
    for name in ("meta.json", "result.json"):
        (viewer.current_episode_dir / name).write_text("{}")
    finish(viewer)
    ready(viewer)
    viewer.task_result.value = "failure"
    viewer.smoothness_score.value = "2"
    viewer._save_manual_evaluation()
    label = json.loads((viewer.current_episode_dir / "evaluation/human-label.json").read_text())
    assert label["task_result"] == "failure"
    assert label["smoothness_score"] == 2
    assert viewer.evaluation_complete
    assert not viewer.prepare_normal_btn.disabled
    assert viewer.skip_evaluation_btn.disabled


def test_skip_without_published_episode_path_does_not_block_next_rollout(viewer):
    viewer.current_episode_dir = None
    finish(viewer)
    ready(viewer)
    assert viewer.save_evaluation_btn.disabled
    assert not viewer.skip_evaluation_btn.disabled
    viewer._skip_manual_evaluation()
    assert not viewer.prepare_normal_btn.disabled
