from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from manimux.types import ActionChunk, ActionHorizon, RobotState, SensorFrame
from manimux.viewer.bridge import ViewerBridge
from manimux.viewer.chunk_timeline import ChunkTimelineView
from manimux.viewer.client import ViewerClient
from manimux.viewer.dashboard import (
    PolicyViewer,
    _camera_panel_html,
    _instruction_markdown,
    _prefill_task,
    _trajectory_colors,
)
from manimux.viewer.protocol import PolicyPlan, RobotSnapshot, RuntimeEvent
from manimux.viewer.robots import available_robot_adapters, load_robot_adapter
from manimux.viewer.robots.yam import DEFAULT_I2RT_ROOT, YamAdapter


@pytest.mark.parametrize("experiment_mode", [False, True])
def test_prepare_button_atomically_selects_rollout_mode(experiment_mode: bool) -> None:
    viewer = PolicyViewer.__new__(PolicyViewer)
    viewer.experiment_mode = not experiment_mode
    viewer.evaluation_saved = True
    viewer.service_ready = True
    viewer.new_rollout_requested = False
    viewer.preparing_rollout = False
    viewer.paused = False
    viewer.rollout_started = False
    viewer.home_requested = False
    viewer.finish_requested = False
    viewer.prepare_normal_btn = SimpleNamespace(disabled=False, visible=True)
    viewer.prepare_experiment_btn = SimpleNamespace(disabled=False, visible=True)
    viewer.layout_id = SimpleNamespace(disabled=False, value="layout-02")
    viewer.task = SimpleNamespace(disabled=False, value="fold the towel")
    viewer.status = SimpleNamespace(content="")
    viewer.rollout_setup_status = SimpleNamespace(content="")
    viewer.new_rollout_folder = SimpleNamespace(visible=True)
    viewer.policy_control_folder = SimpleNamespace(visible=False)
    viewer.recovery_folder = SimpleNamespace(visible=False)
    viewer.evaluation_folder = SimpleNamespace(visible=False)
    viewer.overlay_folder = SimpleNamespace(visible=False)
    viewer.run_folder = SimpleNamespace(visible=True)

    viewer._prepare_rollout(experiment_mode=experiment_mode)
    request = viewer.control_state()

    assert request["new_rollout_requested"] is True
    assert request["experiment_mode"] is experiment_mode
    assert request["task_command"] == "fold the towel"
    assert request["layout_id"] == "layout-02"
    assert viewer.prepare_normal_btn.disabled
    assert viewer.prepare_experiment_btn.disabled
    assert not viewer.prepare_normal_btn.visible
    assert not viewer.prepare_experiment_btn.visible
    assert viewer.task.disabled
    assert viewer.layout_id.disabled
    assert viewer.control_state()["new_rollout_requested"] is False


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("waiting", (True, False, False, False, False, False)),
        ("setup", (True, False, False, False, False, True)),
        ("preparing", (True, False, False, False, False, True)),
        ("control", (False, True, True, False, True, True)),
        ("evaluation", (False, False, False, True, False, True)),
        ("complete", (False, False, False, False, False, True)),
    ],
)
def test_viewer_stage_exposes_only_the_current_action_area(
    stage: str, expected: tuple[bool, bool, bool, bool, bool, bool]
) -> None:
    viewer = PolicyViewer.__new__(PolicyViewer)
    handles = [SimpleNamespace(visible=False) for _ in range(6)]
    (
        viewer.new_rollout_folder,
        viewer.policy_control_folder,
        viewer.recovery_folder,
        viewer.evaluation_folder,
        viewer.overlay_folder,
        viewer.run_folder,
    ) = handles

    viewer._set_stage(stage)  # type: ignore[arg-type]

    assert tuple(handle.visible for handle in handles) == expected


def test_camera_panel_is_screen_fixed_and_targets_stable_image_handles() -> None:
    html = _camera_panel_html()

    assert "position: fixed" in html
    assert "manimux-camera-anchor" in html
    assert "data:image/jpeg;base64" not in html
    assert "--manimux-camera-width: clamp(300px, 26vw, 460px)" in html
    assert "aspect-ratio: 16 / 9" in html
    assert "object-fit: cover" in html


def test_chunk_timeline_tracks_pending_rtc_overlap_and_execution() -> None:
    timeline = ChunkTimelineView()
    timeline.update(
        {
            "kind": "event",
            "event": "episode_started",
            "metadata": {"runtime": "rtc"},
        }
    )
    timeline.update(
        {
            "kind": "event",
            "event": "inference_submitted",
            "chunk_id": 1,
            "metadata": {
                "runtime": "rtc",
                "horizon_steps": 16,
                "conditioned": True,
                "conditioned_overlap_steps": 11,
                "frozen_steps": 4,
            },
        }
    )

    pending = timeline.lanes[0]
    assert pending.state == "pending"
    assert pending.overlap_steps == 11
    assert pending.frozen_steps == 4
    assert "sampling" in timeline.render_html()

    timeline.update(
        {
            "kind": "plan",
            "chunk_id": 1,
            "actions": [[0.0]] * 13,
            "inference_ms": 100.0,
            "metadata": {
                "runtime": "rtc",
                "raw_horizon_steps": 16,
                "trimmed_steps": 3,
                "conditioned": True,
                "conditioned_overlap_steps": 11,
                "frozen_steps": 4,
            },
        }
    )
    timeline.update(
        {
            "kind": "state",
            "active_chunk_id": 1,
            "chunk_index": 5,
        }
    )

    active = timeline.lanes[0]
    assert active.state == "active"
    assert active.cursor == 5
    assert active.trimmed_steps == 3
    rendered = timeline.render_html()
    assert "100 ms" in rendered


def test_chunk_timeline_marks_committed_closed_gripper_steps() -> None:
    timeline = ChunkTimelineView()
    timeline.update(
        {
            "kind": "plan",
            "chunk_id": 7,
            "actions": [[0.0]] * 4,
            "inference_ms": 90.0,
            "metadata": {
                "runtime": "rtc",
                "raw_horizon_steps": 6,
                "trimmed_steps": 2,
                "gripper_closed_steps": [False, True, True, False],
            },
        }
    )

    lane = timeline.lanes[0]
    assert lane.gripper_closed_steps == (False, False, False, True, True, False)
    rendered = timeline.render_html()
    assert "future gripper-closed" in rendered
    assert "gripper closing / closed" in rendered
    assert ".latency.gripper-closed" in rendered
    assert "background:#f59e0b; box-shadow:none" in rendered


def test_yam_gripper_marker_starts_when_closing_begins() -> None:
    adapter = YamAdapter.__new__(YamAdapter)
    actions = np.zeros((12, 7), dtype=np.float64)
    actions[:, 6] = [
        0.995,
        0.991,
        0.98,
        0.955,
        0.94,
        0.94,
        0.7,
        0.4,
        0.4,
        0.8,
        0.92,
        1.0,
    ]
    previous = np.zeros(7, dtype=np.float64)
    previous[6] = 1.0

    flags = adapter.gripper_closed_steps(
        {"left": actions},
        previous_positions={"left": previous},
    )

    assert flags.tolist() == [
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
    ]


def test_chunk_timeline_alternates_lanes_and_marks_superseded_tail() -> None:
    timeline = ChunkTimelineView()
    timeline.update(
        {
            "kind": "plan",
            "chunk_id": 1,
            "actions": [[0.0]] * 10,
            "inference_ms": 50.0,
            "metadata": {"runtime": "manimux", "raw_horizon_steps": 10},
        }
    )
    timeline.update(
        {
            "kind": "state",
            "active_chunk_id": 1,
            "chunk_index": 6,
        }
    )
    timeline.update(
        {
            "kind": "event",
            "event": "inference_submitted",
            "chunk_id": 2,
            "metadata": {"runtime": "manimux", "horizon_steps": 10},
        }
    )
    timeline.update(
        {
            "kind": "plan",
            "chunk_id": 2,
            "actions": [[0.0]] * 9,
            "inference_ms": 60.0,
            "metadata": {
                "runtime": "manimux",
                "raw_horizon_steps": 10,
                "trimmed_steps": 1,
                "previous_chunk_id": 1,
                "previous_chunk_index": 6,
                "superseded_steps": 4,
            },
        }
    )

    assert timeline.lanes[0].state == "retired"
    assert timeline.lanes[0].superseded_steps == 4
    assert timeline.lanes[1].state == "active"
    assert timeline.lanes[1].trimmed_steps == 1
    rendered = timeline.render_html()
    assert "60 ms" in rendered
    assert "RTC condition" not in rendered


def test_chunk_timeline_connects_rtc_condition_source_to_new_chunk() -> None:
    timeline = ChunkTimelineView()
    timeline.update(
        {
            "kind": "plan",
            "chunk_id": 18,
            "actions": [[0.0]] * 46,
            "inference_ms": 144.0,
            "metadata": {"runtime": "rtc", "raw_horizon_steps": 46},
        }
    )
    timeline.update(
        {
            "kind": "state",
            "active_chunk_id": 18,
            "chunk_index": 16,
        }
    )
    timeline.update(
        {
            "kind": "event",
            "event": "inference_submitted",
            "chunk_id": 19,
            "metadata": {
                "runtime": "rtc",
                "horizon_steps": 46,
                "active_chunk_id": 18,
                "active_chunk_index": 16,
                "conditioned": True,
                "conditioned_overlap_steps": 30,
                "frozen_steps": 4,
            },
        }
    )

    pending_html = timeline.render_html()
    assert "condition · 30 steps" in pending_html
    assert "manimux-chunk-cell condition-source" in pending_html
    assert 'manimux-chunk-condition-range source' in pending_html
    assert 'manimux-chunk-condition-range target' in pending_html
    assert "Conditioned prefix: 30 actions" in pending_html

    timeline.update(
        {
            "kind": "plan",
            "chunk_id": 19,
            "actions": [[0.0]] * 42,
            "inference_ms": 145.0,
            "metadata": {
                "runtime": "rtc",
                "raw_horizon_steps": 46,
                "trimmed_steps": 4,
                "previous_chunk_id": 18,
                "previous_chunk_index": 20,
                "superseded_steps": 26,
                "conditioned": True,
                "executed_steps": 22,
                "conditioned_overlap_steps": 30,
                "frozen_steps": 4,
            },
        }
    )

    rendered = timeline.render_html()
    assert timeline.lanes[0].state == "source"
    assert timeline.lanes[0].condition_from_index == 16
    assert "145 ms" in rendered
    assert "condition · 30 steps" in rendered
    assert "manimux-chunk-cell latency" in rendered
    assert "manimux-chunk-cell latency-trimmed" in rendered
    assert 'manimux-chunk-condition-range target' in rendered
    assert ">RTC link<" not in rendered
    assert ">removed<" not in rendered


def test_chunk_timeline_replaces_old_target_frame_when_lane_becomes_source() -> None:
    timeline = ChunkTimelineView()
    timeline.update(
        {
            "kind": "plan",
            "chunk_id": 18,
            "actions": [[0.0]] * 46,
            "metadata": {
                "runtime": "rtc",
                "raw_horizon_steps": 50,
                "trimmed_steps": 4,
                "previous_chunk_id": 17,
                "conditioned": True,
                "conditioned_overlap_steps": 30,
            },
        }
    )
    timeline.update(
        {
            "kind": "event",
            "event": "inference_submitted",
            "chunk_id": 19,
            "metadata": {
                "runtime": "rtc",
                "horizon_steps": 50,
                "active_chunk_id": 18,
                "executed_steps": 20,
                "conditioned": True,
                "conditioned_overlap_steps": 30,
                "frozen_steps": 5,
            },
        }
    )

    rendered = timeline.render_html()
    assert rendered.count('class="manimux-chunk-condition-range source"') == 1
    assert rendered.count('class="manimux-chunk-condition-range target"') == 1


def _service_ready_viewer() -> tuple[PolicyViewer, list[str]]:
    viewer = PolicyViewer.__new__(PolicyViewer)
    viewer.service_id = "/sessions/old"
    viewer.launch_mode = "serve"
    viewer.episode_active = True
    viewer.episode_finalized = False
    viewer.evaluation_saved = False
    viewer.experiment_mode = True
    viewer.current_episode_dir = None
    viewer.paused = False
    viewer.rollout_started = True
    viewer.home_requested = True
    viewer.finish_requested = True
    viewer.new_rollout_requested = True
    viewer.last_state_time = 1.0
    viewer.last_service_time = 1.0
    viewer.preparing_rollout = False
    viewer.service_ready = False
    viewer.policy_name = SimpleNamespace(value="old policy")
    viewer.runtime_name = SimpleNamespace(value="rtc")
    viewer.layout_id = SimpleNamespace(value="old-layout")
    viewer.rollout_setup_status = SimpleNamespace(content="")
    viewer.episode_path = SimpleNamespace(value="old episode")
    viewer.executor_info = SimpleNamespace(value="managed")
    viewer.evaluation_status = SimpleNamespace(content="")
    viewer.prepare_normal_btn = SimpleNamespace(visible=False)
    viewer.prepare_experiment_btn = SimpleNamespace(visible=False)
    viewer.status = SimpleNamespace(content="")
    viewer.task = SimpleNamespace(value="old task")
    stages: list[str] = []
    viewer._set_instruction = lambda _instruction: None  # type: ignore[method-assign]
    viewer._set_policy_controls_enabled = lambda _enabled: None  # type: ignore[method-assign]
    viewer._set_setup_controls_enabled = lambda _enabled: None  # type: ignore[method-assign]
    viewer._set_evaluation_enabled = lambda _enabled: None  # type: ignore[method-assign]
    viewer._update_prepare_enabled = lambda: None  # type: ignore[method-assign]
    viewer._set_stage = stages.append  # type: ignore[method-assign]
    return viewer, stages


def test_new_runtime_service_resets_an_unfinalized_rollout() -> None:
    viewer, stages = _service_ready_viewer()

    viewer._update_event(
        {
            "event": "runtime_service_ready",
            "metadata": {
                "run_dir": "/sessions/new",
                "task": "pick the ball",
                "runtime": "rtc",
                "policy_label": "Pi05",
                "default_layout_id": "default",
            },
        }
    )

    assert viewer.service_id == "/sessions/new"
    assert not viewer.episode_active
    assert not viewer.episode_finalized
    assert viewer.evaluation_saved
    assert viewer.current_episode_dir is None
    assert viewer.last_state_time == 0.0
    assert not viewer.rollout_started
    assert not viewer.home_requested
    assert not viewer.finish_requested
    assert not viewer.new_rollout_requested
    assert viewer.task.value == ""
    assert stages[-1] == "setup"
    assert "prepare a rollout" in viewer.status.content


def test_new_runtime_service_also_resets_a_finalized_unlabeled_rollout() -> None:
    viewer, stages = _service_ready_viewer()
    episode_dir = Path("/sessions/old/rollout-001")
    viewer.episode_active = False
    viewer.episode_finalized = True
    viewer.current_episode_dir = episode_dir

    viewer._update_event(
        {
            "event": "runtime_service_ready",
            "metadata": {
                "run_dir": "/sessions/new",
                "task": "pick the ball",
                "runtime": "rtc",
                "policy_label": "Pi05",
                "default_layout_id": "default",
            },
        }
    )

    assert viewer.current_episode_dir is None
    assert viewer.evaluation_saved
    assert stages[-1] == "setup"


def test_runtime_heartbeat_loss_fails_closed_without_deleting_episode_state() -> None:
    viewer, stages = _service_ready_viewer()
    episode_dir = Path("/sessions/old/rollout-001.partial")
    viewer.current_episode_dir = episode_dir
    viewer.service_ready = True

    viewer._mark_runtime_unavailable()

    assert not viewer.service_ready
    assert not viewer.episode_active
    assert viewer.paused
    assert viewer.current_episode_dir == episode_dir
    assert stages[-1] == "waiting"
    assert "Runtime unavailable" in viewer.status.content


def test_protocol_is_not_tied_to_yam_dimensions() -> None:
    actions = np.zeros((25, 6))
    plan = PolicyPlan("example", "task", actions, 1 / 30, 500, 2, robot="custom")
    wire = plan.to_wire()
    assert wire["actions"] == actions.tolist()
    assert wire["robot"] == "custom"
    assert wire["protocol_version"] == 1


def test_protocol_rejects_malformed_actions() -> None:
    with pytest.raises(ValueError, match="shape"):
        PolicyPlan("example", "task", np.zeros(6), 1 / 30, 1, 0)
    with pytest.raises(ValueError, match="finite"):
        PolicyPlan("example", "task", np.array([[np.nan]]), 1 / 30, 1, 0)


def test_snapshot_encodes_generic_robot_state() -> None:
    state = RobotSnapshot(
        np.zeros(4),
        {},
        step=3,
        max_steps=10,
        chunk_index=2,
        robot="custom",
        active_chunk_id=7,
    )
    wire = state.to_wire()
    assert wire["joint_positions"] == [0.0] * 4
    assert wire["robot"] == "custom"
    assert wire["active_chunk_id"] == 7
    assert wire["chunk_index"] == 2


def test_plan_can_start_inside_an_activated_async_chunk() -> None:
    plan = PolicyPlan(
        "example",
        "task",
        np.zeros((10, 6)),
        1 / 30,
        250,
        4,
        start_index=3,
    )
    assert plan.to_wire()["start_index"] == 3
    with pytest.raises(ValueError, match="start_index"):
        PolicyPlan(
            "example",
            "task",
            np.zeros((10, 6)),
            1 / 30,
            250,
            4,
            start_index=11,
        )


def test_runtime_event_is_policy_independent() -> None:
    event = RuntimeEvent(
        "inference_submitted",
        robot="custom",
        policy="example",
        step=12,
        chunk_id=3,
        metadata={"planned_switch_step": 19},
    ).to_wire()
    assert event["kind"] == "event"
    assert event["event"] == "inference_submitted"
    assert event["metadata"]["planned_switch_step"] == 19


class _RecordingPublisher:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.closed = False

    def publish(self, message: Any) -> None:
        self.messages.append(message.to_wire())

    def close(self) -> None:
        self.closed = True


def test_viewer_client_observes_without_owning_policy_execution() -> None:
    publisher = _RecordingPublisher()
    client = ViewerClient(
        robot="custom",
        policy="example",
        camera_hz=0,
        publisher=publisher,  # type: ignore[arg-type]
    )
    client.episode_started(instruction="task", max_steps=100)
    client.inference_submitted(step=5, chunk_id=2, planned_switch_step=12)
    client.plan_activated(
        actions=np.zeros((10, 6)),
        action_index=3,
        chunk_id=2,
        step=12,
        action_dt=1 / 30,
        inference_ms=400,
        instruction="task",
    )
    client.step_executed(
        joint_positions=np.zeros(6),
        cameras={},
        step=13,
        max_steps=100,
        action_index=4,
        chunk_id=2,
    )
    client.close()

    assert [message["kind"] for message in publisher.messages] == [
        "event",
        "event",
        "plan",
        "state",
    ]
    assert publisher.messages[2]["start_index"] == 3
    assert publisher.messages[3]["active_chunk_id"] == 2
    assert publisher.closed


def test_runtime_bridge_publishes_the_exact_committed_plan() -> None:
    publisher = _RecordingPublisher()
    bridge = ViewerBridge(
        enabled=False,
        robot_adapter="custom",
        group_order=["left", "right"],
        policy="molmoact_http",
        instruction="task",
    )
    bridge._enabled = True
    bridge._publisher = publisher
    bridge._policy_plan_type = PolicyPlan
    raw = ActionChunk(
        plan_id="plan-1",
        request_seq=1,
        observation_time_ns=0,
        created_time_ns=1,
        action_space="joint_position",
        dt_ns=10,
        groups={"left": np.full((4, 1), 9.0), "right": np.full((4, 1), -9.0)},
    )
    committed = ActionHorizon(
        start_time_ns=20,
        dt_ns=10,
        plan_id="plan-1",
        groups={"left": np.array([[1.0], [2.0]]), "right": np.array([[-1.0], [-2.0]])},
    )

    bridge.publish_plan(
        raw,
        250.0,
        committed=committed,
        metadata={"runtime": "rtc", "trimmed_steps": 2},
    )

    assert len(publisher.messages) == 1
    message = publisher.messages[0]
    assert message["policy"] == "molmoact_http"
    assert message["instruction"] == "task"
    assert message["actions"] == [[1.0, -1.0], [2.0, -2.0]]
    assert message["metadata"]["committed_start_time_ns"] == 20
    assert message["metadata"]["runtime"] == "rtc"
    assert message["metadata"]["trimmed_steps"] == 2


def test_runtime_bridge_publishes_managed_lifecycle_event() -> None:
    publisher = _RecordingPublisher()
    bridge = ViewerBridge(
        enabled=False,
        robot_adapter="custom",
        group_order=["arm"],
        policy="policy",
    )
    bridge._enabled = True
    bridge._publisher = publisher
    bridge._runtime_event_type = RuntimeEvent

    bridge.publish_event(
        "episode_started",
        chunk_id=7,
        metadata={"control_mode": "managed", "instruction": "task"},
    )

    assert publisher.messages[0]["metadata"]["control_mode"] == "managed"
    assert publisher.messages[0]["chunk_id"] == 7


def test_runtime_bridge_throttles_camera_encoding_off_the_control_rate() -> None:
    publisher = _RecordingPublisher()
    bridge = ViewerBridge(
        enabled=False,
        robot_adapter="custom",
        group_order=["arm"],
        camera_hz=1.0,
    )
    bridge._enabled = True
    bridge._publisher = publisher
    bridge._snapshot_type = RobotSnapshot
    bridge.set_state_metadata({"episode_active": True, "instruction": "task"})
    state = RobotState(groups={"arm": np.zeros(2)}, monotonic_ns=1, sequence=1)
    frame = SensorFrame(
        name="camera",
        data=np.zeros((4, 4, 3), dtype=np.uint8),
        capture_monotonic_ns=1,
        sequence=1,
    )

    bridge.publish_state(state, {"camera": frame}, step=0, max_steps=2)
    bridge.publish_state(state, {"camera": frame}, step=1, max_steps=2)

    assert set(publisher.messages[0]["cameras_jpeg"]) == {"camera"}
    assert publisher.messages[1]["cameras_jpeg"] == {}
    assert publisher.messages[0]["metadata"]["episode_active"] is True
    assert publisher.messages[1]["metadata"]["instruction"] == "task"


def test_yam_is_a_discovered_adapter() -> None:
    assert "yam" in available_robot_adapters()
    assert isinstance(load_robot_adapter("yam"), YamAdapter)


def test_yam_adapter_splits_single_and_dual_arm_vectors() -> None:
    adapter = YamAdapter()
    assert set(adapter.split_actions(np.zeros((2, 7)), "joint_position")) == {"left"}
    assert set(adapter.split_actions(np.zeros((2, 14)), "joint_position")) == {
        "left",
        "right",
    }
    with pytest.raises(ValueError, match="action_space"):
        adapter.split_actions(np.zeros((2, 7)), "cartesian_delta")
    with pytest.raises(ValueError, match="7 or 14"):
        adapter.split_joint_positions(np.zeros(8))


def test_yam_fk_and_assets_are_self_contained() -> None:
    assert (DEFAULT_I2RT_ROOT / "i2rt/robot_models/arm/yam/yam.urdf").is_file()
    assert (DEFAULT_I2RT_ROOT / "i2rt/robot_models/gripper/linear_4310/linear_4310.xml").is_file()
    adapter = YamAdapter()
    joints = np.array([0.0, 0.8, 1.2, -0.3, 0.0, 0.0, 0.5])
    poses = adapter.positions("left", np.stack([joints, joints]))
    assert poses.shape == (2, 3)
    assert np.isfinite(poses).all()


def test_trajectory_gradient_uses_deep_to_light_purple() -> None:
    colors = _trajectory_colors(9)
    assert colors.shape == (9, 3)
    assert colors.dtype == np.uint8
    assert colors[0].tolist() == [67, 20, 133]
    assert colors[-1].tolist() == [210, 150, 255]
    assert len(np.unique(colors, axis=0)) == 9


def test_the_task_prompt_is_rendered_for_the_always_visible_header() -> None:
    assert "pick up the red ball" in _instruction_markdown("  pick up the red ball  ")
    assert "waiting" in _instruction_markdown("   ")


def test_republished_instructions_do_not_erase_an_operator_mid_edit() -> None:
    """The runtime resends its instruction on every chunk; typing must survive."""

    assert _prefill_task("", "  fold the towel  ") == "fold the towel"
    assert _prefill_task("my own comm", "fold the towel") == "my own comm"
