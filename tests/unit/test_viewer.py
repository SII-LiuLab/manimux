from __future__ import annotations

import inspect
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import viser

from manimux.types import ActionChunk, ActionHorizon, RobotState, SensorFrame
from manimux.viewer.camera_panel import CameraPanel
from manimux.viewer.chunk_timeline import ChunkTimelineView
from manimux.viewer.communication import PolicyPlan, RobotSnapshot, RuntimeEvent
from manimux.viewer.dashboard import (
    PolicyViewer,
    _camera_panel_html,
    _configure_gui,
    _demo_sample,
    _instruction_markdown,
    _prefill_task,
    _trajectory_colors,
    load_robot_view,
    load_viewer_config,
)
from manimux.viewer.publisher import ViewerBridge, ViewerClient
from manimux.viewer.robots import available_robot_adapters, load_robot_adapter
from manimux.viewer.robots.yam import DEFAULT_I2RT_ROOT, YamAdapter
from manimux.viewer.top_overlay import TopViewOverlay


def _camera_config(camera_mode="policy", **options):
    return {
        "camera_mode": camera_mode,
        "cameras": [
            {
                "source": slot,
                "label": {"left": "left side", "right": "right side"}.get(slot, slot),
                "slot": slot,
            }
            for slot in ("top", "left", "right")
        ],
        **options,
    }


def _tianji_view():
    return load_robot_view(load_viewer_config())


def test_gui_uses_explicit_right_dock_when_supported() -> None:
    calls = []
    gui = SimpleNamespace(
        configure_theme=lambda **kwargs: calls.append(("theme", kwargs)),
        main_panel=SimpleNamespace(dock_right=lambda: calls.append(("dock_right", None))),
    )

    _configure_gui(gui)

    assert calls[0][0] == "theme"
    assert calls[0][1]["control_layout"] == "floating"
    assert calls[1] == ("dock_right", None)


def test_gui_falls_back_to_fixed_layout_for_older_viser() -> None:
    calls = []
    gui = SimpleNamespace(configure_theme=lambda **kwargs: calls.append(kwargs))

    _configure_gui(gui)

    assert calls[0]["control_layout"] == "fixed"


@pytest.mark.parametrize("experiment_mode", [False, True])
def test_prepare_button_atomically_selects_rollout_mode(experiment_mode: bool) -> None:
    viewer = PolicyViewer.__new__(PolicyViewer)
    viewer.lock = threading.RLock()
    viewer.experiment_mode = not experiment_mode
    viewer.evaluation_complete = True
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
    assert "manimux-left-overlay-root" in html
    assert "overflow-y: auto" in html
    assert "scrollbar-width: thin" in html
    assert "manimux-camera-anchor" in html
    assert ":has(.mantine-Paper-root .manimux-left-overlay-root)" in html
    assert "data:image/jpeg;base64" not in html
    assert "--manimux-camera-width: clamp(300px, 26vw, 460px)" in html
    assert "--manimux-camera-top: 16px" in html
    assert "aspect-ratio: 16 / 9" in html
    assert "object-fit: cover" in html
    assert "←" not in html
    assert "<small>" not in html
    assert "left side" in html
    assert "right side" in html


class _CameraHandle(SimpleNamespace):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def on_update(self, callback):
        self.change = callback
        return callback

    def on_click(self, callback):
        self.click = callback
        return callback


class _CameraGui:
    """Validate GUI calls against installed Viser without opening sockets."""

    def __getattr__(self, name):
        def add(*args, **kwargs):
            bound = inspect.signature(getattr(viser.GuiApi, name)).bind(None, *args, **kwargs)
            bound.apply_defaults()
            values = dict(bound.arguments)
            values.pop("self")
            value = values.get("initial_value")
            if value is None and "options" in values:
                value = values["options"][0]
            return _CameraHandle(value=value, **values)

        return add


def _camera_viewer(config=None, robot=None):
    viewer = PolicyViewer.__new__(PolicyViewer)
    viewer.robot = robot or YamAdapter()
    if not hasattr(viewer.robot, "validate_groups"):
        viewer.robot.validate_groups = lambda groups: {k: np.asarray(v) for k, v in groups.items()}
    viewer.episode_active = True
    viewer.paused = True
    viewer.observe_only = False
    viewer.progress = SimpleNamespace(value=0)
    viewer.status = SimpleNamespace(content="")
    viewer._update_group = lambda *_: None
    overlay_updates = []
    viewer.camera_view = CameraPanel(
        _CameraGui(),
        config or (viewer.robot.options if hasattr(viewer.robot, "options") else _camera_config()),
        viewer.robot.camera_slot,
        overlay_updates.append,
    )
    return viewer, overlay_updates


def _camera_state(camera_map=None, frames=None, *, robot="yam"):
    return RobotSnapshot(
        robot=robot,
        groups={
            name: np.zeros(8 if robot.startswith("tianji") else 7)
            for name in (
                ("left_arm", "right_arm") if robot.startswith("tianji") else ("left", "right")
            )
        },
        cameras=frames or {},
        step=0,
        max_steps=100,
        metadata={"camera_map": camera_map} if camera_map is not None else {},
    ).to_wire()


@pytest.mark.parametrize(
    "config_path",
    [
        "manimux/configs/experiments/put_bottles/pi05/yam_pi05_rtc_joint_step30000.yaml",
        "manimux/configs/experiments/put_bottles/pi05/yam_pi05_rtc_joint_ee_step30000.yaml",
        "manimux/configs/experiments/put_bottles/sapolicy/yam_sapolicy_mv51_top_rtc.yaml",
        "manimux/configs/experiments/put_bottles/sapolicy/yam_sapolicy_mv51_gemini305_rtc.yaml",
        "manimux/configs/experiments/put_bottles/sapolicy/yam_sapolicy_mv51_gemini335_rtc.yaml",
    ],
)
@pytest.mark.parametrize("reverse_order", [False, True])
def test_default_viewer_follows_pi05_and_sa_inputs_without_viewer_config(
    config_path,
    reverse_order,
):
    from manimux.cli import load_config

    camera_map = load_config(config_path)["policy"]["adapter"]["camera_map"]
    colors = [(210, 30, 40), (40, 210, 60), (60, 80, 210)]
    frames = {
        physical: np.full((12, 16, 3), color, dtype=np.uint8)
        for physical, color in zip(camera_map.values(), colors, strict=True)
    }
    # Non-input cameras must never overwrite the selected view.
    for name in ("front_camera", "gemini305", "gemini335"):
        frames.setdefault(name, np.full((12, 16, 3), 250, dtype=np.uint8))
    viewer, overlay = _camera_viewer()
    wire_map = dict(reversed(list(camera_map.items()))) if reverse_order else camera_map
    viewer._update_state(_camera_state(wire_map, frames))

    for i, (_name, _source) in enumerate(camera_map.items()):
        np.testing.assert_allclose(viewer.camera_view.images[i].image[0, 0], colors[i], atol=3)
    assert next(iter(camera_map)) in viewer.camera_view.panel.content
    assert "left side" in viewer.camera_view.panel.content
    assert "right side" in viewer.camera_view.panel.content
    assert "←" not in viewer.camera_view.panel.content
    assert not any(source in viewer.camera_view.panel.content for source in camera_map.values())
    assert "模型输入相机" in viewer.camera_view.details.content
    assert "未应用模型内部的缩放与裁剪" not in viewer.camera_view.details.content
    np.testing.assert_array_equal(overlay[-1], viewer.camera_view.images[0].image)


@pytest.mark.parametrize("view", ["top", "gemini305", "gemini335"])
def test_manual_preview_stays_separate_and_still_reports_policy_inputs(view):
    from manimux.cli import read_yaml

    old = read_yaml(Path(f"manimux/configs/viewer/yam-{view}.yaml"))
    cfg = _camera_config(camera_mode="manual")
    for camera in cfg["cameras"]:
        camera["source"] = old["cameras"][camera["slot"]]
    viewer, _ = _camera_viewer(cfg)
    physical_names = [c["source"] for c in cfg["cameras"]]
    frames = {name: np.full((12, 16, 3), 90, np.uint8) for name in physical_names}
    camera_map = {"cam_head": "another_camera"}
    frames["another_camera"] = np.full((12, 16, 3), 210, np.uint8)
    viewer._update_state(_camera_state(camera_map, frames))

    for handle in viewer.camera_view.images:
        np.testing.assert_allclose(handle.image, 90, atol=3)
    assert "手动预览" in viewer.camera_view.details.content
    assert "cam_head ← another_camera" in viewer.camera_view.details.content


@pytest.mark.parametrize("include_history", [False, True])
@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.parametrize("runtime_config", ["default", "rtc"])
def test_tianji_wrist_views_keep_spatial_slots_without_agent_view(
    include_history,
    reverse_order,
    runtime_config,
):
    from manimux.cli import load_config

    camera_map = load_config(
        f"manimux/configs/experiments/pass_ball/umi_dp/tianji_umi_dp_{runtime_config}.yaml",
    )["policy"]["adapter"]["camera_map"]
    if not include_history:
        camera_map = {
            name: source for name, source in camera_map.items() if not source.endswith("_prev")
        }
    if reverse_order:
        camera_map = dict(reversed(list(camera_map.items())))
    viewer, overlay = _camera_viewer(robot=_tianji_view())
    panel = viewer.camera_view
    handles = tuple(panel.images)
    frames = {
        "left_wrist": np.full((12, 16, 3), (210, 30, 40), np.uint8),
        "right_wrist": np.full((12, 16, 3), (40, 210, 60), np.uint8),
        "left_wrist_prev": np.full((12, 16, 3), 60, np.uint8),
        "right_wrist_prev": np.full((12, 16, 3), 90, np.uint8),
    }
    # The live bridge publishes only the two physical cameras, even when the
    # policy map also declares temporal inputs assembled for model inference.
    live_frames = {name: frame for name, frame in frames.items() if not name.endswith("_prev")}
    viewer._update_state(_camera_state(camera_map, live_frames, robot="tianji"))

    assert all(a is b for a, b in zip(handles, panel.images[:2], strict=True))
    assert all(handle.visible for handle in panel.images[:2])
    assert overlay == []
    assert "visibility: hidden" not in panel.panel.content
    assert "<strong>top</strong>" not in panel.panel.content
    np.testing.assert_allclose(panel.images[0].image[0, 0], (210, 30, 40), atol=3)
    np.testing.assert_allclose(panel.images[1].image[0, 0], (40, 210, 60), atol=3)
    assert not panel.extra_folder.visible
    if include_history:
        assert len(panel.images) == 2
        assert "cam_left_wrist_prev" in panel.details.content
        assert "cam_right_wrist_prev" in panel.details.content
        assert panel.details.content.count("时序输入 · 复用物理相机") == 2
        viewer._update_state(_camera_state(camera_map, frames, robot="tianji"))
        assert overlay == []
        assert len(panel.images) == 2


def test_camera_diagnostics_can_be_deferred_to_match_main_panel_order():
    panel = CameraPanel(
        _CameraGui(),
        _camera_config(),
        lambda source: source,
        lambda _image: None,
        defer_diagnostics=True,
    )

    assert panel.details is None
    assert panel.extra_folder is None
    panel.add_diagnostics(expand_by_default=True)
    assert panel.diagnostics_folder.label == "Images"
    assert panel.details is not None
    assert panel.extra_folder is not None


@pytest.mark.parametrize(
    ("robot", "sources"),
    [
        (YamAdapter(), ("front_camera", "left_camera", "right_camera")),
        (_tianji_view(), ("left_wrist", "right_wrist")),
    ],
)
@pytest.mark.parametrize("camera_mode", ["policy", "manual"])
def test_missing_metadata_uses_labeled_default_preview_and_robot_aliases(
    robot,
    sources,
    camera_mode,
):
    viewer, _ = _camera_viewer(
        {
            **(robot.options if hasattr(robot, "options") else _camera_config()),
            "camera_mode": camera_mode,
        },
        robot=robot,
    )
    frames = {name: np.full((12, 16, 3), 60 + i * 50, np.uint8) for i, name in enumerate(sources)}
    viewer._update_state(_camera_state(frames=frames, robot=robot.name))
    title = "尚未获取模型输入配置" if camera_mode == "policy" else "手动预览"
    assert title in viewer.camera_view.details.content
    if robot.name == "tianji-taccap":
        assert viewer.camera_view.panel.content.count("<strong>wrist</strong>") == 2
    else:
        assert "left side" in viewer.camera_view.panel.content
        assert "right side" in viewer.camera_view.panel.content
    for i, handle in enumerate(viewer.camera_view.images):
        expected = 60 + i * 50
        assert handle.visible
        np.testing.assert_allclose(handle.image, expected, atol=3)


def test_policy_switch_reuses_handles_clears_old_images_and_handles_arbitrary_input_names():
    viewer, overlay = _camera_viewer()
    panel = viewer.camera_view
    original_handles = tuple(panel.images)
    old_map = {"cam_head": "front_camera", "cam_left": "left_camera", "cam_right": "right_camera"}
    old_frames = {name: np.full((12, 16, 3), 100, np.uint8) for name in old_map.values()}
    viewer._update_state(_camera_state(old_map, old_frames))
    # Unknown sources retain config order around recognized spatial sources.
    new_map = {"observation.images.view_7": "gemini305", "other_view": "left_camera"}
    viewer._update_state(_camera_state(new_map))
    assert all(a is b for a, b in zip(original_handles, panel.images, strict=True))
    assert overlay[-1] is None
    assert panel.images[0].image.max() == 0
    assert panel.images[2].visible
    assert panel.images[2].image.max() == 0
    assert "cam_head" not in panel.details.content
    assert "observation.images.view_7" in panel.panel.content
    # A packet of unselected cameras must not restore the old external image.
    viewer._update_state(_camera_state(new_map, old_frames))
    assert panel.images[0].image.max() == 0
    frames = {"gemini305": np.full((12, 16, 3), 160, np.uint8)}
    viewer._update_state(_camera_state(new_map, frames))
    np.testing.assert_allclose(panel.images[0].image, 160, atol=3)
    assert panel.images[1].image.max() == 0


def test_camera_throttling_keeps_images_but_missing_selected_source_clears_them():
    viewer, overlay = _camera_viewer()
    camera_map = {"external": "gemini335"}
    viewer._update_state(
        _camera_state(
            camera_map,
            {"gemini335": np.full((12, 16, 3), 170, np.uint8)},
        )
    )
    first_image = viewer.camera_view.images[0].image
    first_html = viewer.camera_view.panel.content
    viewer._update_state(_camera_state(camera_map))
    assert viewer.camera_view.images[0].image is first_image
    assert viewer.camera_view.panel.content == first_html
    viewer._update_state(
        _camera_state(
            camera_map,
            {"front_camera": np.zeros((12, 16, 3), np.uint8)},
        )
    )
    assert viewer.camera_view.images[0].image.max() == 0
    assert overlay[-1] is None
    assert "等待图像" in viewer.camera_view.details.content


def test_more_than_three_inputs_are_displayed_and_hidden_after_switch():
    viewer, _ = _camera_viewer()
    camera_map = {f"view_{i}": f"physical_{i}" for i in range(5)}
    frames = {
        name: np.full((12, 16, 3), 40 + 30 * i, np.uint8)
        for i, name in enumerate(camera_map.values())
    }
    viewer._update_state(_camera_state(camera_map, frames))
    panel = viewer.camera_view
    assert len(panel.images) == 5
    assert panel.extra_folder.visible
    for i, handle in enumerate(panel.images):
        assert handle.visible
        np.testing.assert_allclose(handle.image, 40 + 30 * i, atol=3)
    assert panel.images[4].label.startswith("view_4 ← physical_4")
    viewer._update_state(_camera_state({"one_view": "physical_0"}, frames))
    assert not panel.extra_folder.visible
    assert [handle.visible for handle in panel.images] == [True, True, True, False, False]
    assert panel.images[1].image.max() == 0
    assert panel.images[2].image.max() == 0


def test_reset_same_camera_map_clears_frames_and_escaped_labels_are_safe():
    viewer, overlay = _camera_viewer()
    camera_map = {"<img src=x onerror=alert(1)>": 'camera"<x>'}
    frames = {next(iter(camera_map.values())): np.full((12, 16, 3), 180, np.uint8)}
    viewer._update_state(_camera_state(camera_map, frames))
    panel = viewer.camera_view
    assert "<img src=x" not in panel.panel.content
    assert "&lt;img" in panel.panel.content
    assert "&quot;&lt;x&gt;" in panel.details.content
    panel.set_policy_map(camera_map, reset=True)
    assert panel.images[0].image.max() == 0
    assert overlay[-1] is None


@pytest.mark.parametrize("invalid_map", [{"view": None}, {"": "camera"}, ["camera"]])
def test_invalid_policy_metadata_is_labeled_and_does_not_break_default_preview(invalid_map):
    viewer, _ = _camera_viewer()
    viewer._update_state(
        _camera_state(
            invalid_map,
            {"front_camera": np.full((12, 16, 3), 90, np.uint8)},
        )
    )
    assert "映射无效" in viewer.camera_view.details.content
    np.testing.assert_allclose(viewer.camera_view.images[0].image, 90, atol=3)


def test_clearing_live_image_does_not_keep_old_camera_under_reference(tmp_path):
    overlay = TopViewOverlay(_CameraGui(), tmp_path)
    handle = overlay.image
    overlay.update(np.full((12, 16, 3), 200, np.uint8))
    overlay.update(None)
    assert overlay._live is None
    assert overlay.image is handle
    assert overlay.image.image.max() == 0
    overlay.layouts.save("task", "01", np.full((12, 16, 3), 40, np.uint8))
    overlay.refresh.click(None)
    overlay.update(np.full((12, 16, 3), 200, np.uint8))
    overlay.update(None)
    np.testing.assert_array_equal(overlay.image.image, 40)
    assert "仅显示参考图" in overlay.status.content


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
            "groups": {"arm": [[0.0]] * 13},
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
    assert ">current</span>" in rendered


def test_chunk_timeline_aligns_grouped_gripper_steps_after_trim() -> None:
    timeline = ChunkTimelineView()
    timeline.update(
        {
            "kind": "plan",
            "chunk_id": 7,
            "groups": {"arm": [[0.0]] * 4},
            "inference_ms": 90.0,
            "metadata": {
                "runtime": "rtc",
                "raw_horizon_steps": 6,
                "trimmed_steps": 2,
                "gripper_closed_steps_by_group": {
                    "left": [False, True, True, False],
                    "right": [True, False, False, True],
                },
            },
        }
    )

    lane = timeline.lanes[0]
    assert lane.gripper_closed_steps_by_group == {
        "left": (False, False, False, True, True, False),
        "right": (False, False, True, False, False, True),
    }
    rendered = timeline.render_html()
    assert rendered.count("group-left closed") == 2
    assert rendered.count("group-right closed") == 2


def test_chunk_timeline_renders_left_and_right_grippers_independently() -> None:
    timeline = ChunkTimelineView()
    timeline.update(
        {
            "kind": "plan",
            "chunk_id": 8,
            "groups": {"arm": [[0.0]] * 4},
            "inference_ms": 80.0,
            "metadata": {
                "runtime": "manimux",
                "gripper_closed_steps_by_group": {
                    "left": [True, False, True, False],
                    "right": [False, True, True, False],
                },
            },
        }
    )

    lane = timeline.lanes[0]
    assert lane.gripper_closed_steps_by_group == {
        "left": (True, False, True, False),
        "right": (False, True, True, False),
    }
    rendered = timeline.render_html()
    assert rendered.count("group-left closed") == 2
    assert rendered.count("group-right closed") == 2
    assert "manimux-gripper-marker" not in rendered
    assert "manimux-gripper-strip upper" in rendered
    assert "manimux-gripper-strip lower" in rendered
    assert "height:6px" in rendered
    assert "padding:7px 0" in rendered
    assert rendered.index('<div class="manimux-gripper-strip upper"') < rendered.index(
        '<div class="manimux-chunk-cells">'
    )
    assert rendered.index('<div class="manimux-chunk-cells">') < rendered.index(
        '<div class="manimux-gripper-strip lower"'
    )
    assert "manimux-gripper-legend" in rendered
    assert "manimux-gripper-legend-icon" in rendered
    assert "left gripper state · action · right gripper state" in rendered
    assert "gripper open" not in rendered


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

    flags = adapter.gripper_closed_steps_by_group(
        {"left": actions},
        previous_positions={"left": previous},
    )["left"]

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


def test_yam_gripper_markers_preserve_left_and_right_state() -> None:
    adapter = YamAdapter.__new__(YamAdapter)
    left = np.zeros((3, 7), dtype=np.float64)
    right = np.zeros((3, 7), dtype=np.float64)
    left[:, 6] = [1.0, 0.9, 0.4]
    right[:, 6] = [1.0, 1.0, 1.0]
    previous = {
        "left": np.array([0.0] * 6 + [1.0]),
        "right": np.array([0.0] * 6 + [1.0]),
    }

    grouped = adapter.gripper_closed_steps_by_group(
        {"left": left, "right": right}, previous_positions=previous
    )

    assert grouped["left"].tolist() == [False, True, True]
    assert grouped["right"].tolist() == [False, False, False]


def test_chunk_timeline_alternates_lanes_and_marks_superseded_tail() -> None:
    timeline = ChunkTimelineView()
    timeline.update(
        {
            "kind": "plan",
            "chunk_id": 1,
            "groups": {"arm": [[0.0]] * 10},
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
            "metadata": {
                "runtime": "manimux",
                "horizon_steps": 10,
                "active_chunk_id": 1,
                "active_chunk_index": 6,
            },
        }
    )
    timeline.update(
        {
            "kind": "plan",
            "chunk_id": 2,
            "groups": {"arm": [[0.0]] * 9},
            "inference_ms": 60.0,
            "metadata": {
                "runtime": "manimux",
                "raw_horizon_steps": 10,
                "trimmed_steps": 1,
                "previous_chunk_id": 1,
                "previous_chunk_index": 8,
                "superseded_steps": 2,
                "active_chunk_index": 6,
            },
        }
    )

    assert timeline.lanes[0].state == "retired"
    assert timeline.lanes[0].superseded_steps == 2
    assert timeline.lanes[0].latency_from_index == 6
    assert timeline._cell_state(timeline.lanes[0], 5) == "executed"
    assert timeline._cell_state(timeline.lanes[0], 6) == "latency"
    assert timeline._cell_state(timeline.lanes[0], 7) == "latency"
    assert timeline._cell_state(timeline.lanes[0], 8) == "superseded"
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
            "groups": {"arm": [[0.0]] * 46},
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
    assert "manimux-chunk-condition-range source" in pending_html
    assert "manimux-chunk-condition-range target" in pending_html
    assert "Conditioned prefix: 30 actions" in pending_html

    timeline.update(
        {
            "kind": "plan",
            "chunk_id": 19,
            "groups": {"arm": [[0.0]] * 42},
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
    assert "manimux-chunk-condition-range target" in rendered
    assert ">RTC link<" not in rendered
    assert ">removed<" not in rendered


def test_chunk_timeline_does_not_double_count_rtc_trim_in_latency_range() -> None:
    timeline = ChunkTimelineView()
    timeline.update(
        {
            "kind": "plan",
            "chunk_id": 3,
            "groups": {"arm": [[0.0]] * 46},
            "metadata": {
                "runtime": "rtc",
                "raw_horizon_steps": 50,
                "trimmed_steps": 4,
            },
        }
    )
    timeline.update(
        {
            "kind": "event",
            "event": "inference_submitted",
            "chunk_id": 4,
            "metadata": {
                "runtime": "rtc",
                "horizon_steps": 50,
                "active_chunk_id": 3,
                "active_chunk_index": 16,
                "executed_steps": 20,
                "conditioned": True,
                "conditioned_overlap_steps": 30,
                "frozen_steps": 5,
            },
        }
    )
    timeline.update(
        {
            "kind": "plan",
            "chunk_id": 4,
            "groups": {"arm": [[0.0]] * 46},
            "metadata": {
                "runtime": "rtc",
                "raw_horizon_steps": 50,
                "trimmed_steps": 4,
                "previous_chunk_id": 3,
                "previous_chunk_index": 21,
                "conditioned": True,
                "executed_steps": 20,
                "conditioned_overlap_steps": 30,
                "frozen_steps": 5,
            },
        }
    )

    source = timeline.lanes[timeline._lane_by_chunk[3]]
    assert source.condition_from_index == 20
    assert source.latency_from_index == 20
    assert timeline._cell_state(source, 19) == "executed"
    assert [timeline._cell_state(source, index) for index in range(20, 25)] == ["latency"] * 5


def test_chunk_timeline_replaces_old_target_frame_when_lane_becomes_source() -> None:
    timeline = ChunkTimelineView()
    timeline.update(
        {
            "kind": "plan",
            "chunk_id": 18,
            "groups": {"arm": [[0.0]] * 46},
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
    viewer.evaluation_complete = False
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
    camera_updates = []
    viewer.camera_view = SimpleNamespace(
        set_policy_map=lambda raw, **kwargs: camera_updates.append((raw, kwargs)),
        updates=camera_updates,
    )
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
                "camera_map": {"cam_head": "gemini335"},
            },
        }
    )

    assert viewer.service_id == "/sessions/new"
    assert not viewer.episode_active
    assert not viewer.episode_finalized
    assert viewer.evaluation_complete
    assert viewer.current_episode_dir is None
    assert viewer.last_state_time == 0.0
    assert not viewer.rollout_started
    assert not viewer.home_requested
    assert not viewer.finish_requested
    assert not viewer.new_rollout_requested
    assert viewer.task.value == ""
    assert stages[-1] == "setup"
    assert "prepare a rollout" in viewer.status.content
    assert viewer.camera_view.updates == [({"cam_head": "gemini335"}, {"reset": True})]


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
    assert viewer.evaluation_complete
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
    assert viewer.camera_view.updates == [(None, {"reset": True})]


def test_tianji_clear_error_request_uses_recovery_ack_without_enabling_motion() -> None:
    viewer = PolicyViewer.__new__(PolicyViewer)
    viewer.lock = threading.RLock()
    viewer.robot = SimpleNamespace(name="tianji-taccap")
    viewer.service_ready = True
    viewer.episode_active = False
    viewer.preparing_rollout = False
    viewer.launch_mode = "serve"
    viewer.observe_only = False
    viewer.finish_btn = SimpleNamespace(disabled=True)
    viewer.home_btn = SimpleNamespace(disabled=True)
    viewer.drag_arm = SimpleNamespace(disabled=True)
    viewer.drag_btn = SimpleNamespace(disabled=True, label="Start drag")
    viewer.stop_drag_btn = SimpleNamespace(disabled=True, label="Exit drag")
    viewer.clear_error_btn = SimpleNamespace(disabled=True)
    viewer.recovery_status = SimpleNamespace(content="")
    viewer.recovery_details_folder = SimpleNamespace(visible=False)
    viewer.recovery_details = SimpleNamespace(content="")
    viewer._update_prepare_enabled = lambda: None  # type: ignore[method-assign]
    viewer._clear_recovery()
    viewer.last_rollout_error = "RuntimeError: left: controller fault 13, state 100"

    viewer._update_recovery(
        {
            "recovery": {
                "available": True,
                "actions": ["clear_error"],
                "busy": False,
                "state": "idle",
            }
        }
    )

    assert not viewer.clear_error_btn.disabled
    assert viewer.drag_btn.disabled
    assert viewer.home_btn.disabled
    viewer._request_recovery("clear_error")
    request_id = viewer.recovery_request_id
    assert viewer.recovery_request == "clear_error"
    assert not viewer.recovery_lease
    assert viewer.clear_error_btn.disabled

    viewer._update_recovery(
        {
            "recovery": {
                "available": True,
                "actions": ["clear_error"],
                "busy": False,
                "state": "cleared",
                "ack": request_id,
                "arm": "AB",
            }
        }
    )
    assert viewer.recovery_request == ""
    assert "no enable or motion command" in viewer.recovery_status.content


def test_protocol_is_not_tied_to_yam_dimensions() -> None:
    actions = np.zeros((25, 6))
    plan = PolicyPlan("example", "task", {"arm": actions}, 1 / 30, 500, 2, robot="custom")
    wire = plan.to_wire()
    assert wire["groups"] == {"arm": actions.tolist()}
    assert wire["robot"] == "custom"
    assert wire["protocol_version"] == 2


def test_protocol_rejects_malformed_actions() -> None:
    with pytest.raises(ValueError, match="1D|2D"):
        PolicyPlan("example", "task", {"arm": np.zeros(6)}, 1 / 30, 1, 0)
    with pytest.raises(ValueError, match="finite"):
        PolicyPlan("example", "task", {"arm": np.array([[np.nan]])}, 1 / 30, 1, 0)


def test_snapshot_encodes_generic_robot_state() -> None:
    state = RobotSnapshot(
        {"arm": np.zeros(4)},
        {},
        step=3,
        max_steps=10,
        chunk_index=2,
        robot="custom",
        active_chunk_id=7,
    )
    wire = state.to_wire()
    assert wire["groups"] == {"arm": [0.0] * 4}
    assert wire["robot"] == "custom"
    assert wire["active_chunk_id"] == 7
    assert wire["chunk_index"] == 2


def test_plan_can_start_inside_an_activated_async_chunk() -> None:
    plan = PolicyPlan(
        "example",
        "task",
        {"arm": np.zeros((10, 6))},
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
            {"arm": np.zeros((10, 6))},
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
        groups={"arm": np.zeros((10, 6))},
        action_index=3,
        chunk_id=2,
        step=12,
        action_dt=1 / 30,
        inference_ms=400,
        instruction="task",
    )
    client.step_executed(
        groups={"arm": np.zeros(6)},
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
        robot="custom",
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
    assert message["groups"] == {"left": [[1.0], [2.0]], "right": [[-1.0], [-2.0]]}
    assert message["metadata"]["committed_start_time_ns"] == 20
    assert message["metadata"]["runtime"] == "rtc"
    assert message["metadata"]["trimmed_steps"] == 2


def test_runtime_bridge_publishes_managed_lifecycle_event() -> None:
    publisher = _RecordingPublisher()
    bridge = ViewerBridge(
        enabled=False,
        robot="custom",
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
        robot="custom",
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
    assert publisher.messages[0]["groups"] == {"arm": [0.0, 0.0]}
    assert publisher.messages[0]["timestamp_ns"] == 1
    assert publisher.messages[0]["sequence"] == 1
    assert publisher.messages[0]["camera_metadata"] == {
        "camera": {"timestamp_ns": 1, "sequence": 1}
    }
    assert publisher.messages[1]["camera_metadata"] == {}
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


def test_tianji_view_uses_shared_model_geometry_and_preserves_scene():
    view = _tianji_view()
    assert view.name == "tianji-taccap"
    assert set(view.model.groups) == {"left_arm", "right_arm"}
    assert len(view.static_meshes) == 1
    (table,) = view.scene_boxes
    assert tuple(table.dimensions) == (0.8, 1.2, 0.04)
    assert tuple(table.position) == (0.65, 0, 0.68)
    for name, group in view.model.groups.items():
        q = view.initial_positions(name)
        np.testing.assert_allclose(view.pose(name, q), group.kinematics.fk(q))
        np.testing.assert_array_equal(
            view.visual_configuration(name, q), group.visual_configuration(q)
        )
        display = view.options["groups"][name]["viewer_display_frame"]
        np.testing.assert_allclose(view.group(name).base_position, display["xyz"])


def test_tianji_urdf_tcp_matches_model_and_mount_applied_once():
    import yourdfpy
    from scipy.spatial.transform import Rotation

    view = _tianji_view()
    for name, mounted in view.model.groups.items():
        q = view.initial_positions(name)
        urdf = yourdfpy.URDF.load(view.group(name).urdf_path)
        urdf.update_cfg(view.visual_configuration(name, q))
        suffix = "L" if name == "left_arm" else "R"
        local = urdf.get_transform("ee_tcp", f"Base_{suffix}")
        np.testing.assert_allclose(view.pose(name, q), local, atol=1e-4)
        group = view.group(name)
        world = np.eye(4)
        world[:3, :3] = Rotation.from_quat(
            np.asarray(group.base_orientation)[[1, 2, 3, 0]]
        ).as_matrix()
        world[:3, 3] = group.base_position
        display = view.options["groups"][name]["viewer_display_frame"]
        expected_world = np.eye(4)
        expected_world[:3, :3] = Rotation.from_euler("xyz", display["rpy"]).as_matrix()
        expected_world[:3, 3] = display["xyz"]
        np.testing.assert_allclose(
            world @ view.pose(name, q), expected_world @ local, atol=1e-4
        )


def test_tianji_named_group_shapes_are_not_guessed_from_array_width():
    view = _tianji_view()
    assert set(view.validate_groups({"right_arm": np.zeros(8)})) == {"right_arm"}
    with pytest.raises(ValueError):
        view.validate_groups({"right_arm": np.zeros(7)})
    with pytest.raises(ValueError):
        view.validate_groups({"left": np.zeros(8)})
    with pytest.raises(ValueError):
        view.validate_groups(np.zeros(16))
    with pytest.raises(ValueError):
        view.validate_groups(
            {"left_arm": np.zeros((2, 8)), "right_arm": np.zeros((3, 8))}, sequence=True
        )


def test_tianji_demo_and_gripper_markers_use_model_coordinate_names():
    view = _tianji_view()
    states, plans = _demo_sample(view, 1.0, 16)
    view.validate_groups(states)
    view.validate_groups(plans, sequence=True)
    for name, q in states.items():
        assert np.isfinite(view.pose(name, q)).all()
    plans["left_arm"][:, -1] = 0.4
    plans["right_arm"][:, -1] = 1.0
    flags = view.gripper_closed_steps_by_group(plans, previous_positions=states)
    assert flags["left_arm"].all()
    assert not flags["right_arm"].any()


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
