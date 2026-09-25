import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from manimux.viewer import action_replay, dashboard


@pytest.fixture
def robot():
    return dashboard.load_robot_view(dashboard.load_viewer_config(robot="yam"))


@pytest.fixture
def actions(robot):
    return {g.name: np.tile(robot.initial_positions(g.name), (5, 1)) for g in robot.groups}


class Handle:
    def __init__(self, value=None):
        self.value = value
        self.content = ""

    def on_update(self, callback):
        self.callback = callback
        return callback


@pytest.fixture
def playback(monkeypatch, robot, actions):
    controls = {}

    def add(label, value=None, **options):
        handle = Handle(options.get("initial_value", value))
        controls[label] = handle
        return handle

    server = SimpleNamespace(
        gui=SimpleNamespace(add_checkbox=add, add_dropdown=add, add_slider=add, add_markdown=add),
        atomic=nullcontext,
        stop=lambda: None,
    )
    rendered = {}

    def build(viewer):
        viewer.robot_handles = {
            name: SimpleNamespace(update_cfg=lambda q, name=name: rendered.update({name: q}))
            for name in actions
        }

    monkeypatch.setattr(action_replay.ActionReplayViewer, "_build_scene", build)
    viewer = action_replay.ActionReplayViewer(actions, robot, action_dt_s=0.1, server=server)
    return viewer, rendered


def test_play_pause_speed_seek_and_stop_at_last_frame(playback, robot, actions):
    viewer, rendered = playback
    viewer.tick(now=viewer._last_time + 1)
    assert viewer.frame.value == 0
    viewer.play.value = True
    viewer.tick(now=viewer._last_time + 0.15)
    assert viewer.frame.value == 1
    viewer.play.value = False
    viewer.tick(now=viewer._last_time + 1)
    assert viewer.frame.value == 1
    viewer.seek(0)
    viewer.speed.value = "2"
    viewer.play.value = True
    viewer.tick(now=viewer._last_time + 0.11)
    assert viewer.frame.value == 2
    viewer.tick(now=viewer._last_time + 1)
    assert viewer.frame.value == 4
    assert viewer.play.value is False
    for name in actions:
        np.testing.assert_allclose(
            rendered[name], robot.visual_configuration(name, actions[name][-1])
        )
    viewer.play.value = True
    viewer.play.callback(None)
    viewer.tick(now=viewer._last_time)
    assert viewer.frame.value == 0


def test_playback_copies_inputs_and_server_slider_updates_do_not_seek(playback, actions):
    viewer, _ = playback
    actions["left_arm"][:] = 42
    assert not np.all(viewer.actions["left_arm"] == 42)
    viewer._position = 1.5
    viewer.frame.value = 1
    viewer.frame.callback(SimpleNamespace(client=None))
    assert viewer._position == 1.5
    viewer.frame.value = 3
    viewer.frame.callback(SimpleNamespace(client=object()))
    assert viewer._position == 3


def test_named_npz_loading_and_dimensions(tmp_path, robot, actions):
    path = tmp_path / "actions.npz"
    np.savez(path, **actions)
    loaded = action_replay.load_actions(path, robot)
    np.testing.assert_array_equal(loaded["left_arm"], actions["left_arm"])
    np.savez(path, left_arm=np.zeros((5, 6)))
    with pytest.raises(ValueError, match="expected 7"):
        action_replay.load_actions(path, robot)


@pytest.mark.parametrize("dt", [0, -1, float("nan"), float("inf")])
def test_invalid_timing_fails_before_server_creation(robot, actions, dt, monkeypatch):
    monkeypatch.setattr(
        action_replay.viser, "ViserServer", lambda **kwargs: pytest.fail("server opened")
    )
    with pytest.raises(ValueError, match="action_dt_s"):
        action_replay.ActionReplayViewer(actions, robot, action_dt_s=dt)


def test_replay_cli_never_constructs_live_policy_viewer(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "viewer",
            "--robot",
            "yam",
            "--replay-actions",
            str(tmp_path / "actions.npz"),
            "--action-dt-s",
            "0.05",
            "--port",
            "8087",
        ],
    )
    monkeypatch.setattr(
        dashboard, "PolicyViewer", lambda *a, **k: pytest.fail("live Viewer opened")
    )
    monkeypatch.setattr(action_replay, "serve_action_replay", lambda *a, **k: calls.append((a, k)))
    dashboard.main()
    assert calls[0][1]["action_dt_s"] == 0.05
    assert calls[0][1]["port"] == 8087


def test_real_yam_replay_scene_without_runtime_or_cameras(robot, actions):
    viewer = action_replay.ActionReplayViewer(actions, robot, action_dt_s=0.05, port=0)
    try:
        viewer.seek(4)
        assert viewer.frame.value == 4
        assert set(viewer.robot_handles) == set(actions)
        assert not hasattr(viewer, "receiver")
        assert not hasattr(viewer, "control_server")
    finally:
        viewer.close()


def test_live_viewer_keeps_plan_camera_state_and_control_workflow(robot, actions, tmp_path):
    from manimux.viewer.communication import PolicyPlan, RobotSnapshot

    app = dashboard.PolicyViewer(
        "127.0.0.1",
        0,
        "tcp://127.0.0.1:*",
        "tcp://127.0.0.1:*",
        robot,
        reference_root=tmp_path,
        viewer_config=robot.options,
    )
    try:
        app._update_event(
            {
                "event": "episode_started",
                "metadata": {
                    "control_mode": "managed",
                    "experiment_mode": False,
                    "camera_map": {"cam_head": "front_camera"},
                    "episode_dir": str(tmp_path),
                },
            }
        )
        app._update_plan(
            PolicyPlan(
                policy="test",
                instruction="test",
                groups=actions,
                action_dt=0.05,
                inference_ms=10,
                chunk_id=3,
                robot="yam",
            ).to_wire()
        )
        positions = {name: q[0] for name, q in actions.items()}
        app._update_state(
            RobotSnapshot(
                groups=positions,
                cameras={"front_camera": np.zeros((12, 16, 3), np.uint8)},
                step=1,
                max_steps=5,
                robot="yam",
                active_chunk_id=3,
            ).to_wire()
        )
        assert app.plan_chunk_id == 3
        assert app.camera_view._received
        for name, q in positions.items():
            np.testing.assert_allclose(app.last_joint_positions[name], q)
        app._finish_rollout(home=False)
        assert app.control_state()["finish_requested"]
        app._update_event({"event": "episode_finished", "metadata": {}})
        assert app.evaluation_complete
        assert not app.evaluation_folder.visible
    finally:
        app.close()
