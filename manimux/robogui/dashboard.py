"""RoboGUI dashboard for live robot-policy inference and offline replay."""

from __future__ import annotations

import argparse
import json
import signal
import threading
import time
import uuid
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import viser

from manimux.evaluation import write_manual_evaluation
from manimux.evaluation.identity import rollout_identity, start_identity
from manimux.evaluation.rubric import count_result, evaluation_parameters, stage_option
from manimux.types import FloatArray, UInt8Array

from .camera_panel import CameraPanel
from .camera_panel import _camera_panel_html as _camera_panel_html
from .chunk_timeline import ChunkTimelineView
from .communication import ControlServer, PolicyPlan, RobotSnapshot, RoboGUIReceiver
from .reference_layouts import DEFAULT_LAYOUT_ROOT
from .score_choices import ScoreChoices
from .study_setup import reference_task, completed_repeats, next_repeat, unscored_experiments
from .robot_view import RobotGroup, RobotView, load_robogui_config, load_robot_view
from .top_overlay import TopViewOverlay

MAX_PLAN_HISTORY = 16
RoboGUIStage = Literal["waiting", "setup", "preparing", "control", "evaluation", "complete"]
_TRAJECTORY_COLOR_STOPS = np.asarray(
    [
        (67, 20, 133),
        (101, 31, 183),
        (139, 52, 230),
        (177, 92, 255),
        (210, 150, 255),
    ],
    dtype=np.float64,
)


def _trajectory_colors(point_count: int) -> UInt8Array:
    """Return deep-to-light purple from the current pose into the future."""

    if point_count <= 0:
        return np.empty((0, 3), dtype=np.uint8)
    progress = np.linspace(0.0, 1.0, point_count)
    stop_positions = np.linspace(0.0, 1.0, len(_TRAJECTORY_COLOR_STOPS))
    return np.stack(
        [
            np.interp(progress, stop_positions, _TRAJECTORY_COLOR_STOPS[:, channel])
            for channel in range(3)
        ],
        axis=1,
    ).astype(np.uint8)


def _instruction_markdown(instruction: str) -> str:
    """Render the live task prompt for the always-visible header."""

    text = instruction.strip()
    return f"### 📋 {text}" if text else "### 📋 _waiting for a task prompt_"


def _configure_gui(gui: Any) -> None:
    """Keep the operator panel on the right across Viser layout APIs."""

    main_panel = getattr(gui, "main_panel", None)
    gui.configure_theme(
        control_layout="floating" if main_panel is not None else "fixed",
        control_width="large",
        dark_mode=False,
        show_logo=False,
        show_share_button=False,
        brand_color=(70, 103, 190),
    )
    if main_panel is not None:
        main_panel.dock_right()


class PolicyRoboGUI:
    """Robot-independent dashboard consuming an offline RobotModel view."""

    def __init__(
        self,
        host: str,
        port: int,
        bridge_endpoint: str,
        control_endpoint: str,
        robot: RobotView,
        reference_root: Path = DEFAULT_LAYOUT_ROOT,
        robogui_config: dict | None = None,
        render_hz: float = 30.0,
    ) -> None:
        self.robot = robot
        self.robogui_config = robogui_config if robogui_config is not None else robot.options
        self.reference_root = reference_root
        self.server = viser.ViserServer(host=host, port=port, label="ManiMux RoboGUI")
        _configure_gui(self.server.gui)
        self.server.gui.set_panel_label("MANIMUX · ROBOGUI")
        self.lock = threading.RLock()
        self.running = True
        self.paused = True
        self.rollout_started = False
        self._warmup: dict[str, Any] | None = None
        self._warmup_preview_handles: list[Any] = []
        self._warmup_preview_chunk_id: int | None = None
        self._warmup_preview_closed = False
        self._episode_id = ""
        self.finish_requested = False
        self.home_requested = False
        self.finish_home: bool | None = None
        self._gui_stage: RoboGUIStage = "waiting"
        self._clear_recovery()
        self.new_rollout_requested = False
        self._rollout_request: dict[str, Any] = {}
        self._experiment_choices: list[dict[str, Any]] = []
        self._experiment_id = ""
        self._pending_experiment_id = ""
        self._policy_service_state: dict[str, Any] = {"status": "external"}
        self._start_identity: dict[str, Any] | None = None
        self.warmup_requested = False
        self.preparing_rollout = False
        self.service_ready = False
        self.experiment_mode = False
        self.experiment_template = None
        self.task_command = ""
        self._research_defaults: dict[str, Any] = {}
        self._reference_task = ""
        self._repeat_history: dict[tuple[str, str], set[int]] = {}
        self._study_root: Path | None = None
        self._unscored_experiments: dict[str, Path] = {}
        self.evaluation_complete = True
        self.evaluation_profile = evaluation_parameters()
        self.episode_active = False
        self.launch_mode = "unknown"
        self.service_id = ""
        self.last_service_time = 0.0
        self.last_state_time = 0.0
        self.tails: dict[str, deque[FloatArray]] = {
            group.name: deque(maxlen=300) for group in self.robot.groups
        }
        self.robot_handles: dict[str, Any] = {}
        self.plan_actions: dict[str, FloatArray] = {}
        self._plan_points: dict[str, FloatArray] = {}
        self._plan_cursor: int | None = None
        self.last_joint_positions: dict[str, FloatArray] = {}
        self.plan_chunk_id: int | None = None
        self.plan_start_index = 0
        self.plan_history_serial = 0
        self.show_plan_history = False
        self.current_plan_handles: dict[str, list[Any]] = {
            group.name: [] for group in self.robot.groups
        }
        self.plan_history_handles: dict[str, deque[Any]] = {
            group.name: deque() for group in self.robot.groups
        }
        self.chunk_timeline = ChunkTimelineView(
            {
                name: style.get("label", name)
                for name, style in self.robot.options.get("groups", {}).items()
                if style.get("aperture")
            }
        )
        self._last_chunk_timeline_render = 0.0
        self.observe_only = False
        self.current_episode_dir: Path | None = None
        self.episode_finalized = False
        self._build_scene()
        self._build_gui()
        self.camera_view.add_diagnostics(expand_by_default=True)

        def display_message(message: dict[str, Any]) -> None:
            # Batch link poses and overlays so clients do not see a partial update.
            with self.server.atomic():
                self.on_message(message)

        self.receiver = RoboGUIReceiver(bridge_endpoint, display_message, render_hz=render_hz)
        self.control_server = ControlServer(control_endpoint, self.control_state)

        @self.server.on_client_disconnect
        def _disconnect(_client: Any) -> None:
            if not self.server.get_clients() and self.recovery_lease:
                self._request_recovery("stop")

    @staticmethod
    def _root(group: RobotGroup) -> str:
        return f"/robot/{group.name}"

    def _build_scene(self) -> None:
        self.server.scene.set_up_direction("+z")
        # Adapters frame the view so the robot sits right of center, clear of the panels.
        view = self.robot.scene_view
        self.server.initial_camera.position = view.camera_position
        self.server.initial_camera.look_at = view.camera_look_at
        self.server.initial_camera.up = (0.0, 0.0, 1.0)
        self.server.initial_camera.fov = np.deg2rad(55.0)
        self.server.scene.add_grid(
            "/floor",
            width=view.grid_size[0],
            height=view.grid_size[1],
            cell_size=0.1,
            section_size=0.5,
            position=view.grid_position,
        )
        self.server.scene.add_frame("/world", axes_length=0.15, axes_radius=0.006)
        for box in self.robot.scene_boxes:
            self.server.scene.add_box(
                f"/environment/{box.name}",
                color=box.color,
                dimensions=box.dimensions,
                position=box.position,
            )
        for mesh in self.robot.static_meshes:
            mesh_root = f"/environment/{mesh.name}"
            self.server.scene.add_frame(
                mesh_root, show_axes=False, position=mesh.position, wxyz=mesh.orientation
            )
            try:
                from viser.extras import ViserUrdf

                ViserUrdf(self.server, mesh.urdf_path, root_node_name=mesh_root)
            # Scene context only; the arms and trajectories render without it.
            except Exception as exc:  # noqa: BLE001
                print(f"[robogui] {mesh.name} mesh unavailable ({exc})")
        for group in self.robot.groups:
            root = self._root(group)
            # Every group child (URDF, EE frame, plans, tails) inherits this placement.
            self.server.scene.add_frame(
                root,
                show_axes=False,
                position=group.base_position,
                wxyz=group.base_orientation,
            )
            self.server.scene.add_frame(f"{root}/current_ee", axes_length=0.08, axes_radius=0.004)
            if group.urdf_path is None:
                continue
            try:
                from viser.extras import ViserUrdf

                current = ViserUrdf(self.server, group.urdf_path, root_node_name=root)
                current.update_cfg(self.robot.initial_configuration(group.name))
                self.robot_handles[group.name] = current
            # A third-party visualizer may raise backend-specific exceptions;
            # URDF rendering is optional, so preserve trajectory-only operation.
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[robogui] {group.label} URDF unavailable ({exc}); "
                    "trajectory rendering remains enabled"
                )

    def _build_gui(self) -> None:
        self.top_overlay: TopViewOverlay | None = None
        self.camera_view = CameraPanel(
            self.server.gui,
            self.robogui_config,
            self.robot.camera_slot,
            lambda image: self.top_overlay.update(image) if self.top_overlay is not None else None,
            defer_diagnostics=True,
        )
        with self.camera_view.display_container:
            self.chunk_timeline_panel = self.server.gui.add_html(self.chunk_timeline.render_html())
        self.status = self.server.gui.add_markdown("🟠 **Waiting for policy executor**")
        self.instruction = self.server.gui.add_markdown(_instruction_markdown(""))
        self.new_rollout_folder = self.server.gui.add_folder(
            "① New rollout", expand_by_default=True
        )
        with self.new_rollout_folder:
            self.rollout_setup_status = self.server.gui.add_markdown(
                "⚪ Waiting for a ManiMux runtime service."
            )
            self.setup_summary = self.server.gui.add_markdown("Waiting for configuration.")
            self.experiment_policy = self.server.gui.add_dropdown(
                "Policy", ("default",), disabled=True, visible=False,
            )
            self.switch_policy_btn = self.server.gui.add_button(
                "Switch policy", color="blue", disabled=True, visible=False,
            )
            self.policy_service_status = self.server.gui.add_markdown("", visible=False)
            self.experiment_task = self.server.gui.add_dropdown(
                "Task", ("—",), disabled=True, visible=False,
            )
            self.experiment_config = self.server.gui.add_dropdown(
                "Method / Config", ("—",), disabled=True, visible=False,
            )
            self.initial_pose = self.server.gui.add_dropdown(
                "Initial pose", ("Default pose",), disabled=True, visible=False,
            )
            self.initial_pose_hint = self.server.gui.add_markdown(
                "Prepare moves to the selected pose. Selection alone does not move the robot.",
                visible=False,
            )
            self.layout = self.server.gui.add_dropdown("Layout", ("—",), disabled=True)
            self.layout_progress = self.server.gui.add_markdown("")
            self.prepare_normal_btn = self.server.gui.add_button(
                "Prepare Run", color="blue", disabled=True
            )
            self.prepare_experiment_btn = self.server.gui.add_button(
                "Prepare Experiment", color="green", disabled=True
            )
            self.unscored_experiment = self.server.gui.add_dropdown(
                "Unscored experiment", ("—",), disabled=True, visible=False
            )
            self.score_previous_btn = self.server.gui.add_button(
                "Score selected experiment", disabled=True, visible=False
            )
        self.policy_control_folder = self.server.gui.add_folder(
            "② Policy control", expand_by_default=True
        )
        with self.policy_control_folder:
            self.warmup_btn = self.server.gui.add_button(
                "Activate Warmup", color="blue", disabled=True, visible=False
            )
            self.warmup_status = self.server.gui.add_markdown("", visible=False)
            self.start_btn = self.server.gui.add_button(
                "Start rollout", color="blue", disabled=True
            )
            self.pause_btn = self.server.gui.add_button("Pause / Hold", color="gray", disabled=True)
            self.finish_btn = self.server.gui.add_button(
                "Finish rollout", color="red", disabled=True
            )
        self._build_recovery_gui()
        self.run_folder = self.server.gui.add_folder("Run details", expand_by_default=True)
        with self.run_folder:
            self.robot_name = self.server.gui.add_text("Robot", self.robot.label, disabled=True)
            self.recorded_layout = self.server.gui.add_text(
                "Recorded experiment", "—", disabled=True
            )
            self.policy_name = self.server.gui.add_text("Policy", "waiting", disabled=True)
            self.action_space = self.server.gui.add_text("Action space", "waiting", disabled=True)
            self.chunk_info = self.server.gui.add_text("Chunk", "—", disabled=True)
            self.executor_info = self.server.gui.add_text("Executor", "waiting", disabled=True)
            self.latency = self.server.gui.add_text("Inference", "—", disabled=True)
            self.display_timing = self.server.gui.add_text(
                "RoboGUI queue / draw", "—", disabled=True
            )
            self.progress = self.server.gui.add_number("Step", 0, disabled=True)
            self.runtime_name = self.server.gui.add_text("Runtime", "waiting", disabled=True)
            self.episode_path = self.server.gui.add_text("Episode", "waiting", disabled=True)
        self.evaluation_folder = self.server.gui.add_folder(
            "③ Post-rollout evaluation", expand_by_default=True
        )
        with self.evaluation_folder:
            self.evaluation_status = self.server.gui.add_markdown(
                "⚪ Finish the rollout before saving an evaluation."
            )
            self.evaluation_rule = self.server.gui.add_text(
                "Scoring", "General", disabled=True
            )
            self.task_result = ScoreChoices(
                self.server.gui, self.evaluation_folder,
                "Task result",
                ("unlabeled", "Completed", "Not completed"),
                disabled=True, lock=self.lock,
            )
            self.completed_count = ScoreChoices(
                self.server.gui, self.evaluation_folder,
                "Completed items", ("unlabeled",),
                disabled=True, visible=False, lock=self.lock,
            )
            self.completion_score = self.server.gui.add_markdown(
                "Select a completed count.", visible=False
            )
            self.completion_time = self.server.gui.add_text(
                "Completion seconds from Start (optional)", "", disabled=True,
                hint=("Successful trials only. Time when the task criterion was met, "
                      "not Finish. Blank = unknown."),
            )
            self.invalid_trial = self.server.gui.add_checkbox(
                "Invalid trial (exclude from results)", False, disabled=True
            )
            self.operator_note = self.server.gui.add_text(
                "Operator note", "", multiline=True, disabled=True
            )
            with self.server.gui.add_folder("Failure details (optional)", expand_by_default=False):
                self.reviewer_id = self.server.gui.add_text("Reviewer", "operator", disabled=True)
                self.failure_tag_inputs = {
                    "replay_backtrack": self.server.gui.add_checkbox(
                        "Replay / backtrack", False, disabled=True
                    ),
                    "hold_stall": self.server.gui.add_checkbox("Hold / stall", False, disabled=True),
                    "collision": self.server.gui.add_checkbox("Collision", False, disabled=True),
                    "drop_spill": self.server.gui.add_checkbox("Drop / spill", False, disabled=True),
                    "perception": self.server.gui.add_checkbox(
                        "Perception error", False, disabled=True
                    ),
                    "policy_semantics": self.server.gui.add_checkbox(
                        "Policy / task error", False, disabled=True
                    ),
                    "safety_stop": self.server.gui.add_checkbox("Safety stop", False, disabled=True),
                    "other": self.server.gui.add_checkbox("Other", False, disabled=True),
                }
            self.save_evaluation_btn = self.server.gui.add_button(
                "Save evaluation", color="blue", disabled=True
            )
            self.skip_evaluation_btn = self.server.gui.add_button(
                "Skip evaluation", disabled=True
            )
        self.overlay_folder = self.server.gui.add_folder(
            "Overlay controls", expand_by_default=False
        )
        with self.overlay_folder:
            self.show_plan = self.server.gui.add_checkbox("Predicted EE trajectory", True)
            self.show_tail = self.server.gui.add_checkbox("Achieved EE trail", True)
            self.show_frames = self.server.gui.add_checkbox("EE coordinate frames", True)
            self.show_history_btn = self.server.gui.add_button(
                "Show trajectory history", color="gray"
            )
            self.hide_history_btn = self.server.gui.add_button(
                "Current trajectory only", color="blue", visible=False
            )
            self.clear_history_btn = self.server.gui.add_button("Clear trajectory history")
            self.clear_btn = self.server.gui.add_button("Clear trails")
        if any(
            camera.get("slot", camera.get("source")) == "top"
            for camera in self.robogui_config.get("cameras", [])
        ):
            self.top_overlay = TopViewOverlay(
                self.server.gui,
                self.reference_root,
                display_container=self.camera_view.display_container,
            )
        self._set_setup_controls_enabled(False)
        self._set_stage("waiting")

        @self.warmup_btn.on_click
        def _warmup(_event: Any) -> None:
            with self.lock:
                self._activate_warmup()

        @self.start_btn.on_click
        def _start(_event: Any) -> None:
            with self.lock:
                self._start_rollout()

        @self.experiment_policy.on_update
        def _select_policy(event: Any) -> None:
            if event.client is not None:
                with self.lock:
                    self._update_prepare_enabled()

        @self.switch_policy_btn.on_click
        def _switch_policy(_event: Any) -> None:
            with self.lock:
                self._request_policy()

        @self.experiment_task.on_update
        def _select_task(event: Any) -> None:
            if event.client is None:
                return
            with self.lock:
                candidates = [c for c in self._experiment_choices
                              if c["task"] == self.experiment_task.value
                              and c.get("policy", "default") == self.experiment_policy.value]
                current = next((c for c in self._experiment_choices
                                if c["id"] == self._experiment_id), {})
                matching = [c for c in candidates if c["label"] == current.get("label")]
                candidates = matching or candidates
                pose_id = current.get("initial_pose", {}).get("id", "default")
                choice = next((c for c in candidates
                               if c.get("initial_pose", {}).get("id", "default") == pose_id),
                              candidates[0] if candidates else None)
                if choice is not None:
                    self._request_experiment(choice["id"])

        @self.experiment_config.on_update
        def _select_config(event: Any) -> None:
            if event.client is None:
                return
            with self.lock:
                candidates = [c for c in self._experiment_choices
                              if c["task"] == self.experiment_task.value
                              and c.get("policy", "default") == self.experiment_policy.value
                              and c["label"] == self.experiment_config.value]
                choice = next((c for c in candidates
                               if c.get("initial_pose", {}).get("label", "Default pose")
                               == self.initial_pose.value),
                              candidates[0] if candidates else None)
                if choice is not None:
                    self._request_experiment(choice["id"])

        @self.initial_pose.on_update
        def _select_initial_pose(event: Any) -> None:
            if event.client is None:
                return
            with self.lock:
                choice = next((c for c in self._experiment_choices
                               if c["task"] == self.experiment_task.value
                               and c.get("policy", "default") == self.experiment_policy.value
                               and c["label"] == self.experiment_config.value
                               and c.get("initial_pose", {}).get("label", "Default pose")
                               == self.initial_pose.value), None)
                if choice is not None:
                    self._request_experiment(choice["id"])

        @self.prepare_normal_btn.on_click
        def _prepare_run(_event: Any) -> None:
            self._prepare_rollout(experiment_mode=False)

        @self.prepare_experiment_btn.on_click
        def _prepare_experiment(_event: Any) -> None:
            self._prepare_rollout(experiment_mode=True)

        @self.layout.on_update
        def _layout(_event: Any) -> None:
            with self.lock:
                if not self.layout.disabled:
                    self._refresh_layout()

        @self.score_previous_btn.on_click
        def _score_previous(_event: Any) -> None:
            with self.lock:
                self._open_unscored_experiment()

        @self.pause_btn.on_click
        def _pause(_event: Any) -> None:
            with self.lock:
                if self.pause_btn.disabled:
                    return
                self.paused = True
                self._set_policy_controls_enabled(True)
                self.status.content = self._connected_status()

        @self.home_btn.on_click
        def _home(_event: Any) -> None:
            with self.lock:
                if self.home_btn.disabled:
                    return
                if not self.episode_active and "home" in self.recovery_actions:
                    self._request_recovery("home")
                    return
                self.paused = True
                self.home_requested = True
                self.status.content = "🟡 **Returning home · PAUSED**"

        @self.finish_btn.on_click
        def _finish(_event: Any) -> None:
            self._finish_rollout()

        @self.drag_btn.on_click
        def _drag(_event: Any) -> None:
            self._request_recovery(f"drag:{self.drag_arm.value}")

        @self.release_btn.on_click
        def _release(event: Any) -> None:
            self._show_release_options(event)

        @self.stop_drag_btn.on_click
        def _stop_drag(_event: Any) -> None:
            self._request_recovery("stop")

        @self.clear_error_btn.on_click
        def _clear_error(_event: Any) -> None:
            self._request_recovery("clear_error")

        @self.clear_btn.on_click
        def _clear(_event: Any) -> None:
            self._clear_achieved_tails()

        @self.show_history_btn.on_click
        def _show_history(_event: Any) -> None:
            self.show_plan_history = True
            self.show_history_btn.visible = False
            self.hide_history_btn.visible = True
            self._refresh_plan_visibility()

        @self.hide_history_btn.on_click
        def _hide_history(_event: Any) -> None:
            self.show_plan_history = False
            self.show_history_btn.visible = True
            self.hide_history_btn.visible = False
            self._refresh_plan_visibility()

        @self.clear_history_btn.on_click
        def _clear_history(_event: Any) -> None:
            self._clear_plan_history()

        @self.save_evaluation_btn.on_click
        def _save_evaluation(_event: Any) -> None:
            with self.lock:
                self._save_manual_evaluation()

        @self.skip_evaluation_btn.on_click
        def _skip_evaluation(_event: Any) -> None:
            self._skip_manual_evaluation()

        @self.task_result.on_update
        def _task_result(_event: Any) -> None:
            with self.lock:
                self._refresh_completion_score()

        @self.completed_count.on_update
        def _completed_count(_event: Any) -> None:
            with self.lock:
                self._refresh_completion_score()

        @self.invalid_trial.on_update
        def _invalid_trial(_event: Any) -> None:
            with self.lock:
                self._set_evaluation_enabled(
                    self.episode_finalized and self.experiment_mode
                    and not self.evaluation_complete
                )
                self._refresh_completion_score()

        @self.show_plan.on_update
        def _show_plan(_event: Any) -> None:
            self._refresh_plan_visibility()

    def _build_recovery_gui(self) -> None:
        self.recovery_folder = self.server.gui.add_folder(
            "Advanced recovery", expand_by_default=False, visible=False,
        )
        with self.recovery_folder:
            self.recovery_status = self.server.gui.add_markdown(
                "⚪ Waiting for runtime recovery capabilities.", visible=False,
            )
            self.clear_error_btn = self.server.gui.add_button(
                "Clear controller error", color="red", disabled=True, visible=False,
            )
            self.drag_arm = self.server.gui.add_dropdown(
                "Recovery arms", ("—",), initial_value="—", disabled=True, visible=False,
            )
            self.drag_btn = self.server.gui.add_button(
                "Start drag", color="blue", disabled=True, visible=False,
            )
            self.release_btn = self.server.gui.add_button(
                "Release", color="blue", disabled=True, visible=False,
            )
            self.stop_drag_btn = self.server.gui.add_button(
                "Exit drag", color="gray", disabled=True, visible=False,
            )
            self.home_btn = self.server.gui.add_button(
                "Return Home (keep rollout open)", color="gray", disabled=True
            )
            self.recovery_details_folder = self.server.gui.add_folder(
                "Last error", expand_by_default=False, visible=False
            )
            with self.recovery_details_folder:
                self.recovery_details = self.server.gui.add_markdown("")

    def _show_release_options(self, event: Any) -> None:
        with self.lock:
            if self.release_btn.disabled or not self.recovery_selections:
                return
            service_id = self.service_id
            gui = event.client.gui if event.client is not None else self.server.gui
            with gui.add_modal("Release") as modal:
                arms = gui.add_dropdown(
                    "Arms", self.recovery_selections, initial_value=self.drag_arm.value
                )
                gui.add_markdown(
                    "Hand-guide in the robot's release mode. "
                    "Support the selected arm before starting."
                )
                for action, label in (
                    ("release", "Continuous release"), ("release_5s", "Release 5s")
                ):
                    button = gui.add_button(
                        label, color="blue", disabled=not self._supports_hand_guiding(action)
                    )

                    @button.on_click
                    def _choose(_event: Any, action: str = action) -> None:
                        with self.lock:
                            if self.service_id == service_id:
                                self._request_recovery(f"{action}:{arms.value}")
                            modal.close()

                cancel = gui.add_button("Cancel", color="gray")

                @cancel.on_click
                def _cancel(_event: Any) -> None:
                    modal.close()

    def _finish_rollout(self, *, home: bool | None = None) -> None:
        with self.lock:
            if not self.episode_active or self.finish_btn.disabled:
                return
            self.paused = True
            self.finish_requested = True
            self._clear_warmup_preview()
            # Idle Home stays on Return Home. Finish homes only through the
            # experiment's home_on_close when the assembly does not declare home.
            # Switching to hand-guiding explicitly passes home=False in either case.
            self.finish_home = False if "home" in self.recovery_actions else home
            self.service_ready = False
            self._set_policy_controls_enabled(False)
            self._update_recovery_controls()
            self.status.content = "🟠 **Finishing rollout and saving episode**"

    def _activate_warmup(self) -> None:
        if self.warmup_btn.disabled:
            return
        self.warmup_requested = True
        self.warmup_btn.disabled = True
        self.warmup_status.content = "**Warmup activation requested**"

    def _freeze_start_selection(self) -> dict[str, Any]:
        selection: dict[str, Any] = {"layout_id": "", "reference_layout": None}
        if self.experiment_mode:
            selection = self._selected_layout(freeze=True)
            selection["repeat_id"] = self._suggest_repeat(selection)
        identity = start_identity(self._rollout_request, selection)
        self._show_recorded_identity(identity)
        return selection

    def _start_rollout(self) -> None:
        if self.start_btn.disabled:
            return
        if not self.rollout_started:
            try:
                self._start_identity = self._freeze_start_selection()
            except (OSError, TypeError, ValueError) as exc:
                self.status.content = f"🔴 Start was not submitted: {exc}"
                self._set_layout_controls_enabled(True)
                return
        self.paused = False
        self.rollout_started = True
        self._clear_warmup_preview()
        self._set_policy_controls_enabled(True)
        self._set_stage("control")
        self.status.content = self._connected_status()

    def _connected_status(self) -> str:
        if self.observe_only:
            return "🔵 **Connected · OBSERVE ONLY**"
        phase = self._warmup.get("phase") if self._warmup is not None else None
        if phase == "idle":
            return "🟡 **Prepared · adjust layout, then Activate Warmup**"
        if phase == "error":
            return "🔴 **Warmup / Start error · Pause then Start to retry, or Finish**"
        if phase == "resetting":
            return "🟠 **Starting rollout · waiting for backend reset**"
        if phase == "draining" or (phase == "warming" and not self.paused):
            return "🟠 **Starting rollout · waiting for warmup inference to finish**"
        if phase == "warming":
            return "🟡 **Warmup running · Start when ready**"
        return (
            "🟡 **Connected · PAUSED**"
            if self.paused else "🟢 **Connected · RUNNING**"
        )

    def _update_warmup(self, metadata: dict[str, Any] | None) -> None:
        """Keep controls concise; chunks and latency belong to the left panel."""

        self._warmup = dict(metadata) if metadata is not None else None
        self.warmup_status.visible = self._warmup is not None
        self.warmup_btn.visible = self._warmup is not None
        if self._warmup is None:
            self.warmup_requested = False
            self.warmup_btn.disabled = True
        # Idle precedes explicit activation; it must not close this episode's
        # preview gate. Start/Finish still permanently reject late previews.
        idle = self._warmup is not None and self._warmup.get("phase") == "idle"
        if not idle and not self._warmup_preview_allowed():
            self._clear_warmup_preview()
        if self._warmup is None:
            self.warmup_status.content = ""
            return
        phase = str(self._warmup.get("phase", "warming"))
        title = {
            "idle": "Warmup inactive · click Activate Warmup",
            "warming": "Warmup running · Start when ready",
            "draining": "Starting · finishing in-flight warmup",
            "resetting": "Starting · waiting for backend reset",
            "complete": "Warmup complete",
            "error": "Warmup / Start error",
        }.get(phase, "Warmup")
        calibrated = bool(self._warmup.get("latency_calibrated", False))
        if phase == "complete" and not calibrated:
            title = "Warmup ended · no latency calibration samples"
        self.warmup_status.content = f"**{title}**"
        error = str(self._warmup.get("error") or "").strip()
        if error:
            self.warmup_status.content += f"\n\nError: {error}"
        if self.episode_active and not self.finish_requested:
            self._set_policy_controls_enabled(not self.finish_btn.disabled)

    def _warmup_preview_allowed(self, metadata: dict[str, Any] | None = None) -> bool:
        if not (
            self.episode_active
            and self.paused
            and not self.rollout_started
            and not self.finish_requested
            and not self.finish_btn.disabled
            and not self._warmup_preview_closed
            and self._warmup is not None
            and self._warmup.get("phase") == "warming"
        ):
            return False
        if metadata is None:
            return True
        return bool(
            self._episode_id
            and str(metadata.get("episode_id", "")) == self._episode_id
            and str(metadata.get("run_dir", "")) == self.service_id
        )

    def _clear_warmup_plan(self) -> None:
        for handle in self._warmup_preview_handles:
            handle.remove()
        self._warmup_preview_handles.clear()
        self._warmup_preview_chunk_id = None

    def _clear_warmup_preview(self) -> None:
        if self._warmup_preview_closed and not self._warmup_preview_handles:
            return
        self._warmup_preview_closed = True
        self._clear_warmup_plan()
        timeline = getattr(self, "chunk_timeline", None)
        if timeline is not None:
            timeline.clear_warmup_preview()
            self._refresh_chunk_timeline(force=True)

    def _clear_recovery(self) -> None:
        self.recovery_available = False
        self.recovery_actions: set[str] = set()
        self.hand_guiding_actions: set[str] = set()
        self.recovery_task = ""
        self.recovery_selections: tuple[str, ...] = ()
        self.recovery_busy = False
        self.recovery_state = "idle"
        self.recovery_request = ""
        self.recovery_request_id = ""
        self.recovery_lease = False
        self.recovery_error = ""
        self.recovery_arm = ""
        self.last_rollout_error = ""
        self.last_failure_id = ""

    def _can_stop_and_drag(self, action: str = "drag") -> bool:
        return (
            self.episode_active
            and self.launch_mode == "serve"
            and self.recovery_available
            and self._supports_hand_guiding(action)
            and not self.observe_only
            and not self.finish_btn.disabled
        )

    def _recovery_pending(self) -> bool:
        return bool(getattr(self, "recovery_request", "") or getattr(self, "recovery_busy", False))

    def _request_recovery(self, action: str) -> None:
        with self.lock:
            kind, _, selection = action.partition(":")
            hand_guiding = kind in self.hand_guiding_actions
            if action == "stop":
                # Cancel queued hand-guiding even while runtime cleanup is in flight
                # or the service heartbeat has disappeared.
                if not self._recovery_pending() and not self.recovery_lease:
                    return
            else:
                if not self.recovery_available or self._recovery_pending():
                    return
                if kind not in self.recovery_actions:
                    return
                if (
                    kind in {"drag", "release", "release_5s"}
                    and not self._supports_hand_guiding(kind)
                ):
                    return
                if hand_guiding and selection not in self.recovery_selections:
                    return
                if self.episode_active:
                    if not hand_guiding or not self._can_stop_and_drag(kind):
                        return
                    self._finish_rollout(home=False)
                elif not self.service_ready or self.preparing_rollout:
                    return
            self.recovery_request = action
            self.recovery_request_id = uuid.uuid4().hex
            self.recovery_lease = hand_guiding
            self.recovery_status.content = (
                "🟠 **Waiting for recovery control**"
                if action != "stop" else "🟠 **Stopping recovery**"
            )
            self._update_prepare_enabled()
            self._update_recovery_controls()

    def _supports_hand_guiding(self, action: str) -> bool:
        return (
            action in self.recovery_actions
            and action in self.hand_guiding_actions
            and bool(self.recovery_selections)
        )

    def _sync_recovery_presentation(self) -> None:
        """Show recovery controls from the assembly declaration, not the robot name."""
        if not hasattr(self, "recovery_folder"):
            return
        declared = bool(self.recovery_actions)
        has_clear = "clear_error" in self.recovery_actions
        has_drag = self._supports_hand_guiding("drag")
        has_release = any(
            self._supports_hand_guiding(action) for action in ("release", "release_5s")
        )
        has_hand_guiding = has_drag or has_release
        self.recovery_folder.label = "Manual recovery" if declared else "Advanced recovery"
        self.recovery_folder.visible = self._gui_stage == "control" or declared
        self.clear_error_btn.visible = has_clear
        self.drag_arm.visible = has_hand_guiding
        self.drag_btn.visible = has_drag
        self.release_btn.visible = has_release
        self.stop_drag_btn.visible = has_hand_guiding
        self.recovery_status.visible = declared
        if hasattr(self, "finish_btn"):
            self.finish_btn.label = "Finish rollout"
        if has_hand_guiding and tuple(self.drag_arm.options) != self.recovery_selections:
            selected = self.drag_arm.value
            self.drag_arm.options = self.recovery_selections
            self.drag_arm.value = (
                selected if selected in self.recovery_selections else self.recovery_selections[-1]
            )

    def _recovery_choices(self) -> str:
        arms = " / ".join(self.recovery_selections)
        choices = [
            f"{label} {arms}"
            for action, label in (
                ("drag", "Drag"), ("release", "Release"), ("release_5s", "Release 5s")
            )
            if self._supports_hand_guiding(action)
        ]
        if "home" in self.recovery_actions:
            choices.append("Return Home")
        if "clear_error" in self.recovery_actions:
            choices.append("Clear controller error")
        return ", ".join(choices)

    def _update_recovery_controls(self) -> None:
        if not hasattr(self, "home_btn"):
            return
        idle = self.service_ready and not self.episode_active and not self.preparing_rollout
        allowed = idle and self.recovery_available and not self._recovery_pending()
        can_clear = allowed and "clear_error" in self.recovery_actions
        can_drag = (allowed and self._supports_hand_guiding("drag")) or (
            self._can_stop_and_drag() and not self._recovery_pending()
        )
        can_release = any(
            ((allowed and self._supports_hand_guiding(action)) or self._can_stop_and_drag(action))
            and not self._recovery_pending()
            for action in ("release", "release_5s")
        )
        self.clear_error_btn.disabled = not can_clear
        self.drag_arm.disabled = not (can_drag or can_release)
        self.drag_btn.disabled = not can_drag
        self.drag_btn.label = "Stop rollout & drag" if self.episode_active else "Start drag"
        if hasattr(self, "release_btn"):
            self.release_btn.disabled = not can_release
            self.release_btn.label = "Stop rollout & release" if self.episode_active else "Release"
        self.stop_drag_btn.disabled = not (
            self.recovery_lease or self.recovery_state in {"starting", "active"}
        )
        mode = "release" if (
            self.recovery_task.startswith("release") or self.recovery_request.startswith("release")
        ) else "drag"
        self.stop_drag_btn.label = (
            f"Cancel {mode}"
            if (self.recovery_lease and self.recovery_state == "idle")
            else f"Exit {mode}"
        )
        if not self.episode_active:
            self.home_btn.disabled = not (allowed and "home" in self.recovery_actions)
        self._render_recovery_status()

    def _render_recovery_status(self) -> None:
        error = self.recovery_error
        arm = self.recovery_arm
        task = self.recovery_task or self.recovery_request.partition(":")[0]
        mode = "release" if task.startswith("release") else "drag"
        title = "Release 5s" if task == "release_5s" else mode.title()
        if self.recovery_request == "stop":
            text = "🟠 **Stopping recovery · waiting for servo-off**"
        elif self.recovery_request == "clear_error":
            text = "🟠 **Clearing controller error** · no arm will be enabled."
        elif self.recovery_lease and self.episode_active:
            text = f"🟠 **Stopping rollout without homing · waiting to enter {mode}**"
        elif self.recovery_request:
            text = "🟠 **Waiting for recovery control**"
        elif self.recovery_state == "active":
            ending = "Stops automatically after 5 seconds." if task == "release_5s" else (
                f"Adjust by hand, then Exit {mode}."
            )
            text = f"🟢 **{title} {arm}** · {ending}"
        elif self.recovery_state == "starting":
            text = f"🟠 **Starting {mode} {arm}** · checking the controller"
        elif self.recovery_state == "stopping":
            text = f"🟠 **Exiting {mode} · waiting for servo-off**"
        elif self.recovery_state == "homing":
            text = "🟡 **Returning home**"
        elif self.recovery_state == "clearing":
            text = "🟠 **Clearing controller error** · sending the controller command."
        elif self.recovery_state == "cleared":
            text = (
                f"🟢 **Controller {arm} clear-error command accepted** · no enable or motion "
                "command was sent; verify the workspace, then Prepare rollout."
            )
        elif error:
            text = "🔴 **Recovery failed** · check Last error before retrying."
        elif self.preparing_rollout:
            text = "🟠 **Preparing robot** · recovery is available after preparation stops."
        elif self.episode_active:
            text = (
                "🟡 Stop rollout & drag/release ends this rollout without homing, "
                "then enters the selected recovery mode."
                if any(self._can_stop_and_drag(action) for action in self.hand_guiding_actions)
                else "🟠 Waiting for the rollout to release the robot."
            )
        elif not self.service_ready:
            text = "⚪ Waiting for the runtime service to release the robot."
        elif "clear_error" in self.recovery_actions and (
            "controller fault" in self.last_rollout_error.lower()
            or "emergency stop" in self.last_rollout_error.lower()
        ):
            text = (
                "🔴 **Controller fault is latched** · release the physical E-stop if active, "
                "then press Clear controller error."
            )
        elif "clear_error" in self.recovery_actions and not self.recovery_available:
            text = "⚪ Controller error recovery requires an executing runtime service."
        elif not self.recovery_available:
            text = "⚪ Recovery requires an executing runtime service."
        elif "emergency stop" in self.last_rollout_error.lower():
            text = (
                "🔴 **Rollout stopped by E-stop** · release the physical E-stop, "
                f"then choose {self._recovery_choices()} when ready."
            )
        elif self.last_rollout_error:
            text = self._interrupted_recovery_text()
        else:
            text = self._idle_recovery_text()
        self.recovery_status.content = text
        if hasattr(self, "recovery_details"):
            detail = error or self.last_rollout_error
            self.recovery_details_folder.visible = bool(detail)
            self.recovery_details.content = detail

    def _interrupted_recovery_text(self) -> str:
        choices = self._recovery_choices()
        if choices:
            return f"🟡 **Rollout interrupted** · choose {choices} when ready."
        return "🟡 **Rollout interrupted** · recovery is available."

    def _idle_recovery_text(self) -> str:
        choices = self._recovery_choices()
        if choices:
            return f"⚪ Choose {choices}."
        return "⚪ Recovery is available."

    def _update_recovery(self, metadata: dict[str, Any]) -> None:
        if not hasattr(self, "home_btn"):
            return
        recovery = metadata.get("recovery") or {}
        self.recovery_available = bool(recovery.get("available", False))
        actions = recovery.get("actions")
        self.recovery_actions = (
            {str(action) for action in actions}
            if isinstance(actions, (list, tuple, set))
            else set()
        )
        selections = recovery.get("selections")
        self.recovery_selections = (
            tuple(str(item) for item in selections)
            if isinstance(selections, (list, tuple))
            else ()
        )
        self.hand_guiding_actions = set(recovery.get("hand_guiding_actions", ("drag",)))
        self.recovery_task = str(recovery.get("task", ""))
        self._sync_recovery_presentation()
        self.recovery_busy = bool(recovery.get("busy", False))
        self.recovery_state = str(recovery.get("state", "idle"))
        if recovery.get("ack") == self.recovery_request_id:
            self.recovery_request = ""
        if not self._recovery_pending():
            self.recovery_lease = False
        self.recovery_error = str(recovery.get("error", ""))
        self.recovery_arm = str(recovery.get("arm", ""))
        self._update_recovery_controls()

    def _accept_rollout_failure(self, metadata: dict[str, Any]) -> bool:
        """Recover from the durable idle heartbeat if the failure event was lost."""
        error = str(metadata.get("last_error") or metadata.get("error") or "")
        failure_id = str(metadata.get("last_failure_id") or error)
        if not error or failure_id == getattr(self, "last_failure_id", ""):
            return False
        self.last_failure_id = failure_id
        self.last_rollout_error = error
        self._clear_warmup_preview()
        self.episode_active = False
        self.episode_finalized = False
        self.preparing_rollout = False
        self.service_ready = True
        self.paused = True
        self.rollout_started = False
        self.home_requested = False
        self.finish_requested = False
        self.finish_home = None
        self.new_rollout_requested = False
        self._rollout_request = {}
        self.evaluation_complete = True
        self._set_policy_controls_enabled(False)
        self._set_evaluation_enabled(False)
        self.executor_info.value = "rollout interrupted"
        self.evaluation_status.content = "⚪ Interrupted rollout; recovery is available."
        return True

    def _idle_status(self, last_error: str) -> None:
        if self._recovery_pending():
            self.status.content = "🟡 **Manual recovery · rollout controls locked**"
        elif "emergency stop" in last_error.lower() or "controller fault" in last_error.lower():
            self.status.content = "🔴 **Controller fault · clear error is available**"
        elif last_error:
            self.status.content = "🔴 **Rollout interrupted · service idle**"
        elif self.evaluation_complete:
            self.status.content = "🟡 **Runtime service ready · ready for next Prepare**"

    def _set_instruction(self, instruction: str) -> None:
        self.instruction.content = _instruction_markdown(instruction)
        self.task_command = instruction

    def _set_stage(self, stage: RoboGUIStage) -> None:
        """Expose only the controls that are actionable in the current stage."""

        self._gui_stage = stage
        self.new_rollout_folder.visible = stage in {"waiting", "setup", "preparing"} or (
            stage == "control" and not self.rollout_started
        )
        self.policy_control_folder.visible = stage == "control"
        self.evaluation_folder.visible = stage == "evaluation"
        self.overlay_folder.visible = stage == "control"
        self.run_folder.visible = stage not in {"waiting"}
        self._sync_recovery_presentation()
        if hasattr(self, "home_btn"):
            self.home_btn.label = (
                "Return Home (keep rollout open)" if stage == "control" else "Return Home"
            )
            self._update_recovery_controls()

    def _configure_evaluation(self, profile: dict | None) -> None:
        """Use the rollout's frozen rubric, never its editable instruction text."""

        self.evaluation_profile = evaluation_parameters(profile)
        counted = self.evaluation_profile["kind"] == "count"
        self.evaluation_rule.value = (
            f"Task · {self.evaluation_profile['label']}"
            if self.evaluation_profile.get("task_id") else "General"
        )
        self._stage_ids = {
            f"{o['label']} · {o['score']:g}": o["id"]
            for o in self.evaluation_profile.get("options", [])
        }
        self.task_result.value = "unlabeled"
        self.task_result.display_labels = (
            {f"{o['label']} · {o['score']:g}": f"{o['score']:g} · {o['label']}"
             for o in self.evaluation_profile["options"]}
            if self.evaluation_profile["kind"] == "stages"
            else {"Completed": "1 · Completed", "Not completed": "0 · Not completed"}
        )
        self.task_result.options = (
            ("unlabeled", *self._stage_ids)
            if self.evaluation_profile["kind"] == "stages"
            else ("unlabeled", "Completed", "Not completed")
        )
        self.completed_count.value = "unlabeled"
        self.completion_time.value = ""
        self.completed_count.display_labels = (
            {str(value): f"{value / self.evaluation_profile['target_count']:g} · "
             f"{self.evaluation_profile['count_label']}: {value}/{self.evaluation_profile['target_count']}"
             for value in range(self.evaluation_profile["target_count"] + 1)}
            if counted else {}
        )
        self.completed_count.options = (
            ("unlabeled",) + tuple(
                str(value) for value in range(self.evaluation_profile["target_count"] + 1)
            )
            if counted else ("unlabeled",)
        )
        self.completed_count.label = self.evaluation_profile.get(
            "count_label", "Completed items"
        )
        self.task_result.visible = not counted
        self.completed_count.visible = counted
        self.completion_score.visible = counted or bool(self._stage_ids)
        self._refresh_completion_score()

    def _refresh_completion_score(self) -> None:
        if self.evaluation_profile["kind"] == "binary":
            return
        if self.invalid_trial.value:
            self.completion_score.content = "Invalid trial · excluded from results."
            return
        if self.evaluation_profile["kind"] == "stages":
            selected = self._stage_ids.get(self.task_result.value)
            if selected is None:
                self.completion_score.content = "Select a result."
                return
            option = stage_option(self.evaluation_profile, selected)
            status = "Completed" if option["success"] else "Not completed"
            self.completion_score.content = f"**Score {option['score']:g}** · {status}"
            return
        if self.completed_count.value == "unlabeled":
            self.completion_score.content = "Select a completed count."
            return
        completed = int(self.completed_count.value)
        result = count_result(self.evaluation_profile, completed)
        target = self.evaluation_profile["target_count"]
        status = "Completed" if result == "success" else "Not completed"
        self.completion_score.content = (
            f"**{completed}/{target} · {100 * completed / target:.1f}%** · {status}"
        )

    def _set_evaluation_enabled(self, enabled: bool) -> None:
        for handle in (
            self.invalid_trial,
            self.reviewer_id,
            self.operator_note,
            self.completion_time,
            *self.failure_tag_inputs.values(),
        ):
            handle.disabled = not enabled
        score_enabled = enabled and not self.invalid_trial.value
        self.completion_time.disabled = not score_enabled
        counted = self.evaluation_profile["kind"] == "count"
        self.task_result.disabled = not score_enabled or counted
        self.completed_count.disabled = not score_enabled or not counted
        self.save_evaluation_btn.disabled = not enabled
        self.skip_evaluation_btn.disabled = (
            self.episode_active or not self.experiment_mode or self.evaluation_complete
        )

    def _set_experiment_mode(self, enabled: bool) -> None:
        self.experiment_mode = enabled
        self.rollout_setup_status.content = (
            "🟢 **Study rollout** · save or skip evaluation after Finish."
            if enabled
            else "🔵 **Free rollout** · no human label is required."
        )

    def _set_setup_controls_enabled(self, enabled: bool) -> None:
        allowed = (enabled and self.evaluation_complete and not self._recovery_pending()
                   and not self._pending_experiment_id)
        can_prepare = allowed and not self.preparing_rollout and not self.episode_active
        self.experiment_policy.disabled = not can_prepare
        retry = self._policy_service_state.get("status") == "failed"
        changed_policy = self._policy_selection_changed()
        self.switch_policy_btn.label = (
            "Retry policy" if retry and not changed_policy else "Switch policy"
        )
        self.switch_policy_btn.disabled = not can_prepare or not (changed_policy or retry)
        can_prepare = can_prepare and not self._policy_selection_changed()
        self.experiment_task.disabled = not can_prepare
        self.experiment_config.disabled = not can_prepare
        self.initial_pose.disabled = not can_prepare
        model_ready = self._policy_service_state.get("status") in {"ready", "external"}
        self.prepare_normal_btn.disabled = not can_prepare or not model_ready
        self.prepare_experiment_btn.disabled = not can_prepare or not model_ready
        self.unscored_experiment.disabled = not can_prepare or not self._unscored_experiments
        self.score_previous_btn.disabled = self.unscored_experiment.disabled
        self._set_layout_controls_enabled(allowed or (
            self.episode_active and not self.rollout_started and not self.finish_btn.disabled
        ))

    def _set_layout_controls_enabled(self, enabled: bool) -> None:
        changed = self.layout.disabled == enabled
        self.layout.disabled = not enabled
        if self.top_overlay is not None:
            self.top_overlay.set_selection_enabled(enabled)
            if enabled and changed:
                self._refresh_layout()

    def _configure_research(self, metadata: dict[str, Any], *, reset_selection: bool = True) -> None:
        """Task and study settings come from YAML, never duplicate UI inputs."""
        self.experiment_template = metadata.get("experiment_template")
        self._research_defaults = dict(metadata.get("research_defaults", {}))
        task = str(metadata.get("task", ""))
        self.task_command = task
        self.setup_summary.content = " · ".join(str(metadata.get(k, "")) for k in ("policy_label", "executor") if metadata.get(k))
        library = self.top_overlay.layouts if self.top_overlay is not None else None
        self._reference_task = reference_task(task, library.tasks() if library else ())
        template = self.experiment_template or {}
        ids = template.get("layout_ids") or (
            library.slots(self._reference_task) if library and self._reference_task else ()
        )
        selected_layout = self.layout.value
        self.layout.options = tuple(ids) or ("—",)
        self.layout.value = (
            selected_layout if not reset_selection and selected_layout in self.layout.options
            else self.layout.options[0]
        )
        self._refresh_layout_visibility()
        run_dir = metadata.get("run_dir")
        self._study_root = Path(run_dir).parent if run_dir else None
        self._refresh_repeat_history()
        self._refresh_layout()

    def _configure_experiment_choices(self, metadata: dict[str, Any]) -> None:
        self._experiment_choices = list(metadata.get("experiment_choices", []))
        self._experiment_id = str(metadata.get("experiment_id", ""))
        visible = bool(self._experiment_choices)
        self.experiment_task.visible = self.experiment_config.visible = visible
        policies = tuple(dict.fromkeys(c.get("policy", "default") for c in self._experiment_choices))
        self.experiment_policy.visible = len(policies) > 1
        self.switch_policy_btn.visible = (
            len(policies) > 1 or self._policy_service_state.get("status") != "external"
        )
        if not visible:
            self.initial_pose.visible = self.initial_pose_hint.visible = False
            return
        choice = next(c for c in self._experiment_choices if c["id"] == self._experiment_id)
        policy = choice.get("policy", "default")
        self.experiment_policy.options = policies
        self.experiment_policy.value = policy
        choices = [c for c in self._experiment_choices if c.get("policy", "default") == policy]
        self.experiment_task.options = tuple(dict.fromkeys(c["task"] for c in choices))
        self.experiment_task.value = choice["task"]
        self.experiment_config.options = tuple(dict.fromkeys(
            c["label"] for c in choices if c["task"] == choice["task"]
        ))
        self.experiment_config.value = choice["label"]
        self.initial_pose.options = tuple(dict.fromkeys(
            c.get("initial_pose", {}).get("label", "Default pose") for c in choices
            if c["task"] == choice["task"] and c["label"] == choice["label"]
        ))
        self.initial_pose.value = choice.get("initial_pose", {}).get("label", "Default pose")
        self.initial_pose.visible = len(self.initial_pose.options) > 1
        self.initial_pose_hint.visible = self.initial_pose.visible

    def _policy_selection_changed(self) -> bool:
        current = next(
            (c for c in self._experiment_choices if c["id"] == self._experiment_id), None
        )
        return (
            current is not None
            and self.experiment_policy.value != current.get("policy", "default")
        )

    def _request_policy(self) -> None:
        candidates = [c for c in self._experiment_choices
                      if c.get("policy", "default") == self.experiment_policy.value]
        if not candidates:
            return
        matching_task = [c for c in candidates if c["task"] == self.experiment_task.value]
        candidates = matching_task or candidates
        matching_method = [c for c in candidates if c["label"] == self.experiment_config.value]
        candidates = matching_method or candidates
        choice = next((c for c in candidates
                       if c.get("initial_pose", {}).get("label", "Default pose")
                       == self.initial_pose.value), candidates[0])
        self._request_experiment(choice["id"])

    def _request_experiment(self, choice_id: str) -> None:
        if (not self.service_ready or not self.evaluation_complete or self.preparing_rollout
                or self.episode_active or self._recovery_pending() or self._pending_experiment_id):
            return
        if (choice_id == self._experiment_id
                and self._policy_service_state.get("status") != "failed"):
            return
        self._pending_experiment_id = choice_id
        self._update_prepare_enabled()
        self.rollout_setup_status.content = "🟠 Switching experiment configuration…"

    def _refresh_layout_visibility(self) -> None:
        show_layout = not (self.preparing_rollout or self.episode_active) or self.experiment_mode
        self.layout.visible = show_layout and self.layout.options != ("—",)
        self.layout_progress.visible = self.layout.visible

    def _setup_hint(self) -> str:
        return "Run: no scoring. Experiment: choose a layout, then Prepare Experiment."

    def _refresh_repeat_history(self) -> None:
        self._repeat_history = completed_repeats(
            self._study_root, self.task_command, self._research_defaults
        )
        self._unscored_experiments = unscored_experiments(self._study_root, self.task_command)
        selected = self.unscored_experiment.value
        self.unscored_experiment.options = tuple(self._unscored_experiments) or ("—",)
        self.unscored_experiment.value = (
            selected if selected in self._unscored_experiments else self.unscored_experiment.options[0]
        )
        self.unscored_experiment.visible = bool(self._unscored_experiments)
        self.score_previous_btn.visible = bool(self._unscored_experiments)

    def _open_unscored_experiment(self) -> None:
        if (not self.service_ready or self.episode_active or self.preparing_rollout
                or not self.evaluation_complete or self._recovery_pending()):
            return
        selected = self.unscored_experiment.value
        self._refresh_repeat_history()
        episode = self._unscored_experiments.get(selected)
        if episode is None:
            self.rollout_setup_status.content = "This experiment is no longer awaiting evaluation."
            self._update_prepare_enabled()
            return
        try:
            meta = json.loads((episode / "meta.json").read_text())
            self._configure_evaluation(meta.get("evaluation"))
        except (OSError, TypeError, ValueError) as exc:
            self.rollout_setup_status.content = f"Could not open evaluation: {exc}"
            return
        self._set_experiment_mode(True)
        self._reset_evaluation(str(episode))
        self.episode_finalized = True
        self._show_recorded_identity(meta)
        self._set_evaluation_enabled(True)
        self._update_prepare_enabled()
        self._set_stage("evaluation")
        self.evaluation_status.content = (
            f"**Layout {meta.get('layout_id') or '—'} · Repeat {meta.get('repeat_id') or '—'}**"
            f" · {episode.parent.name}/{episode.name}\n\n"
            "Using this experiment's saved scoring rules."
        )
        self.status.content = "🟡 **Reviewing an unscored experiment · save or skip evaluation**"

    def _selected_layout(self, *, freeze: bool = False) -> dict[str, Any]:
        slot = "" if self.layout.value == "—" else self.layout.value
        selection: dict[str, Any] = {"layout_id": slot, "reference_layout": None}
        template = self.experiment_template or {}
        if self.top_overlay is not None and self._reference_task and slot:
            self.top_overlay.select(self._reference_task, slot)
            try:
                if freeze:
                    selection = self.top_overlay.freeze_selection()
                else:
                    selection = self.top_overlay.layouts.snapshot(self._reference_task, slot)[1]
            except (OSError, ValueError):
                if template.get("require_reference"):
                    raise
        if template.get("require_reference") and selection["reference_layout"] is None:
            raise ValueError(f"No saved Top reference for {self.task_command} / {slot}.")
        return selection

    def _suggest_repeat(self, selection: dict[str, Any]) -> int:
        digest = (selection.get("reference_layout") or {}).get("sha256", "")
        return next_repeat(
            self._repeat_history.get((selection["layout_id"], digest), set()),
            (self.experiment_template or {}).get("repeats"),
        )

    def _refresh_layout(self) -> None:
        if not self.layout.visible:
            return
        try:
            selection = self._selected_layout()
            repeat = self._suggest_repeat(selection)
            limit = (self.experiment_template or {}).get("repeats")
            suffix = f"/{limit}" if limit else ""
            self.layout_progress.content = f"**Layout {selection['layout_id'] or '—'} · Repeat {repeat}{suffix}**"
        except (OSError, ValueError) as exc:
            self.layout_progress.content = str(exc)

    def _show_recorded_identity(self, identity: dict[str, Any]) -> None:
        parts = [str(identity.get(key) or "") for key in
                 ("experiment_name", "condition", "layout_id")]
        if identity.get("repeat_id") is not None:
            parts.append(f"repeat {identity['repeat_id']}")
        self.recorded_layout.value = " · ".join(part for part in parts if part) or "Free rollout"

    def _prepare_rollout(self, *, experiment_mode: bool) -> None:
        with self.lock:
            if (self._recovery_pending() or not self.service_ready or not self.evaluation_complete
                    or self.preparing_rollout or self.episode_active or self._pending_experiment_id
                    or self._policy_selection_changed()
                    or self._policy_service_state.get("status") not in {"ready", "external"}):
                return
            try:
                selection = self._selected_layout(freeze=True) if experiment_mode else {}
                repeat_id = self._suggest_repeat(selection) if experiment_mode else None
                identity = rollout_identity({
                    "experiment_mode": experiment_mode,
                    "experiment_template": self.experiment_template,
                    **self._research_defaults,
                    "repeat_id": repeat_id,
                    **selection,
                })
            except (OSError, TypeError, ValueError) as exc:
                self.rollout_setup_status.content = f"🔴 Prepare was not submitted: {exc}"
                self._update_prepare_enabled()
                return
            self._rollout_request = {"task_command": self.task_command.strip(),
                                     "experiment_id": self._experiment_id, **identity}
            self._show_recorded_identity(identity)
            self._set_experiment_mode(experiment_mode)
            self._start_identity = None
            self.new_rollout_requested = True
            self.preparing_rollout = True
            self._refresh_layout_visibility()
            self.service_ready = False
            self.paused = True
            self._update_warmup(None)
            self._set_setup_controls_enabled(False)
            self.prepare_normal_btn.visible = False
            self.prepare_experiment_btn.visible = False
            self._set_stage("preparing")
            kind = "study" if experiment_mode else "free"
            self.status.content = f"🟠 **Preparing a new {kind} rollout**"

    def _set_policy_controls_enabled(self, enabled: bool) -> None:
        allowed = enabled and not self.observe_only
        phase = self._warmup.get("phase") if self._warmup is not None else None
        formal_started = self.rollout_started and phase in {None, "complete"}
        self.start_btn.label = "Resume rollout" if formal_started else "Start rollout"
        self.start_btn.disabled = (
            not allowed or not self.paused or phase in {"idle", "draining", "resetting"}
        )
        self.warmup_btn.disabled = (
            not allowed or not self.paused or phase != "idle" or self.warmup_requested
        )
        self._set_layout_controls_enabled(
            allowed and self.episode_active and not self.rollout_started
        )
        self.pause_btn.disabled = not allowed or self.paused
        self.home_btn.disabled = (
            not allowed
            or not self.paused
            or (
                "home" in self.recovery_actions
                and not getattr(self, "active_home_available", False)
            )
        )
        self.finish_btn.disabled = not allowed
        self._update_recovery_controls()

    def _update_prepare_enabled(self) -> None:
        self._set_setup_controls_enabled(self.service_ready)

    def _reset_evaluation(self, episode_dir: str) -> None:
        self.current_episode_dir = Path(episode_dir).expanduser() if episode_dir else None
        self.episode_finalized = False
        self.evaluation_complete = not self.experiment_mode
        self.episode_path.value = episode_dir or "not published"
        self.task_result.value = "unlabeled"
        self.completed_count.value = "unlabeled"
        self.completion_time.value = ""
        self.invalid_trial.value = False
        self._refresh_completion_score()
        self.reviewer_id.value = "operator"
        self.operator_note.value = ""
        for handle in self.failure_tag_inputs.values():
            handle.value = False
        self._set_evaluation_enabled(False)
        self.evaluation_status.content = "⚪ Finish the rollout before saving an evaluation."

    def _reset_for_new_service(self) -> None:
        """Make a new ``manimux serve`` instance a hard workflow boundary."""

        # A task belongs to one runtime service; reconnects must not retain it.
        self.task_command = ""
        self.paused = True
        self.rollout_started = False
        self.home_requested = False
        self._clear_recovery()
        self.finish_requested = False
        self.new_rollout_requested = False
        self._rollout_request = {}
        self.preparing_rollout = False
        self._pending_experiment_id = ""
        self.episode_active = False
        self.current_episode_dir = None
        self._episode_id = ""
        self.episode_finalized = False
        self.evaluation_complete = True
        self._configure_evaluation(None)
        self._update_warmup(None)
        self.last_state_time = 0.0
        self.executor_info.value = "service idle"
        self.recorded_layout.value = "—"
        self.evaluation_status.content = "⚪ No completed rollout is awaiting evaluation."
        self._set_policy_controls_enabled(False)
        self._set_evaluation_enabled(False)

    def _mark_runtime_unavailable(self) -> None:
        """Fail closed when either the idle service or active executor disappears."""

        if not self.service_ready and not self.episode_active:
            return
        self.paused = True
        self.rollout_started = False
        self.home_requested = False
        self.finish_requested = False
        self.new_rollout_requested = False
        self._rollout_request = {}
        self.preparing_rollout = False
        self.service_ready = False
        self.episode_active = False
        self._clear_warmup_preview()
        self.recovery_lease = False
        self._update_recovery_controls()
        self.camera_view.set_policy_map(None, reset=True)
        self._set_policy_controls_enabled(False)
        self._set_setup_controls_enabled(False)
        self._set_stage("waiting")
        self.rollout_setup_status.content = "⚪ Waiting for a ManiMux runtime service."
        self.status.content = "🟠 **Runtime unavailable · waiting for service**"

    def _save_manual_evaluation(self) -> None:
        if self.evaluation_complete or not self.experiment_mode:
            return
        if not self.episode_finalized or self.current_episode_dir is None:
            self.evaluation_status.content = "🔴 Episode is not finalized."
            return
        try:
            completed_count = None
            stage_id = None
            if self.invalid_trial.value:
                result = "invalid"
            elif self.evaluation_profile["kind"] == "stages":
                stage_id = self._stage_ids.get(self.task_result.value)
                if stage_id is None:
                    self.evaluation_status.content = "🔴 Select a result."
                    return
                option = stage_option(self.evaluation_profile, stage_id)
                result = "success" if option["success"] else "failure"
            elif self.evaluation_profile["kind"] == "count":
                if self.completed_count.value == "unlabeled":
                    self.evaluation_status.content = "🔴 Select a completed count."
                    return
                completed_count = int(self.completed_count.value)
                result = count_result(self.evaluation_profile, completed_count)
            else:
                result = {
                    "Completed": "success", "Not completed": "failure",
                }.get(str(self.task_result.value))
                if result is None:
                    self.evaluation_status.content = "🔴 Select Completed or Not completed."
                    return
            target = write_manual_evaluation(
                self.current_episode_dir,
                task_result=cast(Any, result),
                completed_count=completed_count,
                stage_id=stage_id,
                completion_time_s=(
                    float(self.completion_time.value)
                    if result == "success" and self.completion_time.value.strip()
                    else None
                ),
                failure_tags=[
                    name for name, handle in self.failure_tag_inputs.items() if handle.value
                ],
                operator_note=self.operator_note.value,
                reviewer_id=self.reviewer_id.value,
            )
        except (OSError, TypeError, ValueError) as exc:
            self.evaluation_status.content = f"🔴 Evaluation was not saved: {exc}"
            return
        self.evaluation_status.content = f"🟢 Saved `{target}`"
        self.evaluation_complete = True
        self._refresh_repeat_history()
        self._refresh_layout()
        self._set_evaluation_enabled(False)
        self._update_prepare_enabled()
        self._set_stage("setup" if self.service_ready else "complete")

    def _skip_manual_evaluation(self) -> None:
        with self.lock:
            if self.episode_active or not self.experiment_mode or self.evaluation_complete:
                return
            self.evaluation_complete = True
            self._refresh_repeat_history()
            self._refresh_layout()
            self.evaluation_status.content = "⚪ Evaluation skipped; no human label was saved."
            self._set_evaluation_enabled(False)
            self._update_prepare_enabled()
            self._set_stage("setup" if self.service_ready else "complete")
            self.status.content = "⚪ **Rollout finished · evaluation skipped**"

    def control_state(self) -> dict[str, Any]:
        requested_at = time.monotonic()
        with self.lock:
            locked_at = time.monotonic()
            state = {
                "paused": self.paused,
                "episode_id": self._episode_id,
                "control_run_dir": self.service_id,
                "warmup_requested": self.warmup_requested,
                "start_identity": deepcopy(self._start_identity),
                "home_requested": self.home_requested,
                "finish_requested": self.finish_requested,
                "finish_home": getattr(self, "finish_home", None),
                "recovery_request": getattr(self, "recovery_request", ""),
                "recovery_request_id": getattr(self, "recovery_request_id", ""),
                "recovery_service_id": getattr(self, "service_id", ""),
                "recovery_lease": getattr(self, "recovery_lease", False),
                "new_rollout_requested": self.new_rollout_requested,
                "select_experiment_id": self._pending_experiment_id,
                "evaluation_complete": self.evaluation_complete,
                **deepcopy(self._rollout_request),
            }
            self.home_requested = False
            self.finish_requested = False
            self.finish_home = None
            self.new_rollout_requested = False
            state["control_timing"] = {
                "lock_wait_ms": (locked_at - requested_at) * 1000,
                "state_read_ms": (time.monotonic() - locked_at) * 1000,
            }
            return state

    def _matches_selected_robot(self, message: dict[str, Any]) -> bool:
        robot_name = str(message.get("robot", ""))
        if not robot_name or robot_name == self.robot.name:
            return True
        self.status.content = (
            f"🔴 **Message targets robot `{robot_name}`; robogui uses `{self.robot.name}`**"
        )
        return False

    def on_message(self, message: dict[str, Any]) -> None:
        with self.lock:
            display_started = time.monotonic_ns()
            if not self._matches_selected_robot(message):
                return
            try:
                kind = message.get("kind")
                if kind == "plan":
                    metadata = message.get("metadata") or {}
                    if metadata.get("warmup_preview") and not self._warmup_preview_allowed(
                        metadata
                    ):
                        return
                    self._update_plan(message)
                elif kind == "state":
                    self._update_state(message)
                elif kind == "event":
                    self._update_event(message)
                timeline = getattr(self, "chunk_timeline", None)
                if timeline is not None:
                    timeline.update(message)
                    self._refresh_chunk_timeline(force=kind != "state")
                timing = getattr(self, "display_timing", None)
                received = message.get("_robogui_received_ns")
                if kind == "state" and timing is not None and received is not None:
                    queue_ms = max(0, display_started - received) / 1e6
                    draw_ms = (time.monotonic_ns() - display_started) / 1e6
                    timing.value = f"{queue_ms:.1f} / {draw_ms:.1f} ms"
            except (KeyError, TypeError, ValueError) as exc:
                self.status.content = (
                    f"🔴 **Rejected malformed {message.get('kind')} message: {exc}**"
                )

    def _refresh_chunk_timeline(self, *, force: bool = False) -> None:
        timeline = getattr(self, "chunk_timeline", None)
        panel = getattr(self, "chunk_timeline_panel", None)
        if timeline is None or panel is None:
            return
        now = time.monotonic()
        last_render = float(getattr(self, "_last_chunk_timeline_render", 0.0))
        if not force and now - last_render < 0.08:
            return
        panel.content = timeline.render_html()
        self._last_chunk_timeline_render = now

    def _update_plan(self, message: dict[str, Any]) -> None:
        action_space = str(message.get("action_space", "joint_position"))
        pose_plan = action_space == "absolute_tcp_pose"
        if pose_plan:
            from manimux.policy_adapter.pose_output import validate_pose_rows
            grouped_actions = {name: validate_pose_rows(rows) for name, rows in message["groups"].items()}
            if set(grouped_actions) != {g.name for g in self.robot.groups}:
                raise ValueError("Pose plan groups do not match the robot")
        elif action_space == "joint_position":
            grouped_actions = self.robot.validate_groups(message.get("groups"), sequence=True)
        else:
            raise ValueError(f"Unsupported plan space: {action_space}")
        horizon = len(next(iter(grouped_actions.values())))
        metadata = dict(message.get("metadata") or {})
        gripper_by_group = ({} if pose_plan else self.robot.gripper_closed_steps_by_group(
            grouped_actions,
            previous_positions=getattr(self, "last_joint_positions", {}),
        ))
        metadata["gripper_closed_steps_by_group"] = {
            group_name: flags.tolist() for group_name, flags in gripper_by_group.items()
        }
        message["metadata"] = metadata
        start_index = int(message.get("start_index", 0))
        if start_index < 0 or start_index > horizon:
            raise ValueError("plan start_index is outside the action horizon")
        if metadata.get("warmup_preview"):
            self._update_warmup_plan(message, grouped_actions, start_index=start_index)
            return
        chunk_id = int(message.get("chunk_id", 0))
        if self.plan_chunk_id is not None and chunk_id != self.plan_chunk_id:
            self._archive_current_plan()
        self.plan_actions = {
            group_name: np.asarray(group_actions, dtype=np.float64)
            for group_name, group_actions in grouped_actions.items()
        }
        self._plan_points = {
            name: (actions[:, :3].copy() if pose_plan else self.robot.positions(name, actions))
            for name, actions in self.plan_actions.items()
        }
        self._plan_cursor = start_index
        self.plan_chunk_id = chunk_id
        self.plan_start_index = start_index
        self.policy_name.value = str(message.get("policy", "unknown"))
        self._set_instruction(str(message.get("instruction", "")))
        self.action_space.value = action_space
        self.latency.value = f"{float(message.get('inference_ms', 0.0)):.0f} ms"
        self.chunk_info.value = (
            f"#{message.get('chunk_id', 0)} · {horizon} actions · "
            f"{float(message.get('action_dt', 0.0)):.3f}s"
        )
        for group in self.robot.groups:
            group_actions = grouped_actions.get(group.name)
            if group_actions is None:
                self._draw_plan(group, np.empty((0, 0)))
            else:
                self._draw_plan(group, group_actions[start_index:],
                                points=self._plan_points[group.name][start_index:])

    def _update_warmup_plan(
        self, message: dict[str, Any], grouped_actions: dict[str, FloatArray], *, start_index: int
    ) -> None:
        """Render one discarded inference without touching formal or measured motion."""

        metadata = message["metadata"]
        if not self._warmup_preview_allowed(metadata):
            return
        chunk_id = int(message["chunk_id"])
        if chunk_id >= 0:
            raise ValueError("Warmup preview requires a negative chunk_id")
        if self._warmup_preview_chunk_id is not None and chunk_id >= self._warmup_preview_chunk_id:
            return
        trajectories = {
            group.name: (grouped_actions[group.name][start_index:, :3].copy()
                         if message.get("action_space") == "absolute_tcp_pose" else
                         self.robot.positions(group.name, grouped_actions[group.name][start_index:]))
            for group in self.robot.groups
            if group.name in grouped_actions and len(grouped_actions[group.name][start_index:]) > 0
        }
        self._clear_warmup_plan()
        self._warmup_preview_chunk_id = chunk_id
        visible = bool(self.show_plan.value)
        color = (14, 165, 233)
        branch = str(metadata.get("warmup_branch", "ordinary"))
        for group in self.robot.groups:
            points = trajectories.get(group.name)
            if points is None:
                continue
            # FK stays local to the group root, just like the formal plan overlay.
            root = f"{self._root(group)}/warmup_preview"
            if len(points) > 1:
                segments = np.stack((points[:-1], points[1:]), axis=1)
                self._warmup_preview_handles.append(self.server.scene.add_line_segments(
                    f"{root}/path", segments,
                    np.broadcast_to(np.asarray(color, dtype=np.uint8), segments.shape).copy(),
                    line_width=7, visible=visible,
                ))
            self._warmup_preview_handles.append(self.server.scene.add_icosphere(
                f"{root}/end", radius=0.012, color=color,
                position=points[-1], visible=visible,
            ))
            self._warmup_preview_handles.append(self.server.scene.add_label(
                f"{root}/label", f"Warmup preview · {branch} · no execution",
                position=points[-1] + np.asarray((0.0, 0.0, 0.04)), visible=visible,
            ))

    def _draw_plan(
        self, group: RobotGroup, actions: FloatArray, *, points: FloatArray | None = None,
    ) -> None:
        root = f"{self._root(group)}/predicted_ee"
        if len(actions) < 2:
            for handle in self.current_plan_handles[group.name]:
                handle.visible = False
            self.current_plan_handles[group.name] = []
            return
        # This line lives below the group root, which already carries the arm's
        # base_position. Keep FK output in that local frame to avoid applying
        # the left/right base offset twice.
        if points is None:
            points = self.robot.positions(group.name, actions)
        segments = np.stack((points[:-1], points[1:]), axis=1)
        point_colors = _trajectory_colors(len(points))
        segment_colors = np.stack((point_colors[:-1], point_colors[1:]), axis=1)
        visible = bool(self.show_plan.value)
        path = self.server.scene.add_line_segments(
            f"{root}/path",
            segments,
            segment_colors,
            line_width=10,
            visible=visible,
        )
        start = self.server.scene.add_icosphere(
            f"{root}/start",
            radius=0.012,
            color=cast(tuple[int, int, int], tuple(int(value) for value in point_colors[0])),
            position=points[0],
            visible=visible,
        )
        end = self.server.scene.add_icosphere(
            f"{root}/end",
            radius=0.015,
            color=cast(tuple[int, int, int], tuple(int(value) for value in point_colors[-1])),
            position=points[-1],
            visible=visible,
        )
        self.current_plan_handles[group.name] = [path, start, end]

    def _archive_current_plan(self) -> None:
        """Keep the previous activated chunk as a dim trajectory overlay."""

        self.plan_history_serial += 1
        for group in self.robot.groups:
            actions = self.plan_actions.get(group.name)
            if actions is None:
                continue
            actions = actions[self.plan_start_index :]
            if len(actions) < 2:
                continue
            cached = self._plan_points.get(group.name)
            points = (cached[self.plan_start_index:] if cached is not None
                      else self.robot.positions(group.name, actions))
            point_colors = _trajectory_colors(len(points)).astype(np.float64)
            muted_colors = (0.55 * point_colors + 0.45 * 190.0).astype(np.uint8)
            history_colors = np.stack((muted_colors[:-1], muted_colors[1:]), axis=1)
            handle = self.server.scene.add_line_segments(
                f"{self._root(group)}/predicted_history/{self.plan_history_serial}",
                np.stack((points[:-1], points[1:]), axis=1),
                history_colors,
                line_width=3.5,
                visible=bool(self.show_plan.value) and self.show_plan_history,
            )
            history = self.plan_history_handles[group.name]
            history.append(handle)
            while len(history) > MAX_PLAN_HISTORY:
                history.popleft().remove()

    def _clear_plan_history(self) -> None:
        for history in self.plan_history_handles.values():
            while history:
                history.popleft().remove()

    def _refresh_plan_visibility(self) -> None:
        show_current = bool(self.show_plan.value)
        for handles in self.current_plan_handles.values():
            for handle in handles:
                handle.visible = show_current
        for handle in self._warmup_preview_handles:
            handle.visible = show_current
        show_history = show_current and self.show_plan_history
        for history in self.plan_history_handles.values():
            for handle in history:
                handle.visible = show_history

    def _reset_plan_overlay(self) -> None:
        self._clear_warmup_plan()
        for handles in self.current_plan_handles.values():
            for handle in handles:
                handle.visible = False
        self.current_plan_handles = {group.name: [] for group in self.robot.groups}
        self._clear_plan_history()
        self.plan_actions = {}
        self._plan_points = {}
        self._plan_cursor = None
        self.plan_chunk_id = None
        self.plan_start_index = 0

    def _clear_achieved_tails(self) -> None:
        for group in self.robot.groups:
            self.tails[group.name].clear()
            self._draw_tail(group, np.empty((0, 3)))

    def _update_state(self, message: dict[str, Any]) -> None:
        metadata = message.get("metadata") or {}
        if "camera_map" in metadata:
            self.camera_view.set_policy_map(metadata["camera_map"])
        if not self.episode_active and bool(metadata.get("episode_active", False)):
            self._update_event({"event": "episode_started", "metadata": metadata})
        if "warmup" in metadata:
            self._update_warmup(metadata["warmup"])
        grouped_positions = self.robot.validate_groups(message.get("groups"))
        self.last_joint_positions = {
            group_name: np.asarray(configuration, dtype=np.float64).copy()
            for group_name, configuration in grouped_positions.items()
        }
        self.last_state_time = time.time()
        self.progress.value = int(message.get("step", 0))
        if not message.get("connected", True):
            self.status.content = "🟠 **Executor disconnected**"
        else:
            self.status.content = self._connected_status()
        active_chunk_id = message.get("active_chunk_id")
        action_index = int(message.get("chunk_index", 0))
        if (
            active_chunk_id is not None
            and self.plan_chunk_id is not None
            and int(active_chunk_id) == self.plan_chunk_id
            and action_index != getattr(self, "_plan_cursor", None)
        ):
            for group in self.robot.groups:
                actions = self.plan_actions.get(group.name)
                self._draw_plan(
                    group,
                    actions[action_index:] if actions is not None else np.empty((0, 0)),
                    points=(self._plan_points[group.name][action_index:]
                            if group.name in self._plan_points else None),
                )
            self._plan_cursor = action_index
        for group_name, configuration in grouped_positions.items():
            self._update_group(self.robot.group(group_name), configuration)
        self.camera_view.update_images(message.get("cameras_jpeg", {}))

    def _update_event(self, message: dict[str, Any]) -> None:
        event = str(message.get("event", "unknown"))
        metadata = message.get("metadata") or {}
        if event == "episode_started":
            incoming_episode_id = str(metadata.get("episode_id", ""))
            incoming_service_id = str(metadata.get("run_dir", ""))
            if incoming_service_id != self.service_id or self.launch_mode == "unknown":
                self._configure_research({
                    **metadata, "task": metadata.get("instruction", ""),
                    "default_experiment_mode": metadata.get("experiment_mode", False),
                    "research_defaults": {k: metadata.get(k, "") for k in ("experiment_name", "condition", "notes")},
                })
            if (incoming_service_id, incoming_episode_id) != (self.service_id, self._episode_id):
                self._warmup_preview_closed = False
            self.camera_view.set_policy_map(metadata.get("camera_map"), reset=True)
            self._reset_plan_overlay()
            self._clear_achieved_tails()
            self.episode_active = True
            self._episode_id = incoming_episode_id
            self.launch_mode = str(metadata.get("launch_mode", "run"))
            if incoming_service_id:
                self.service_id = incoming_service_id
            self.observe_only = metadata.get("control_mode", "observe") == "observe"
            self.active_home_available = bool(metadata.get("home_available", False))
            self.paused = True
            self.rollout_started = bool(metadata.get("identity_frozen", False))
            self._start_identity = None
            self.warmup_requested = False
            self._rollout_request = {
                **{key: metadata.get(key, "") for key in
                   ("experiment_name", "condition", "notes")},
                **{key: metadata.get(key) for key in
                   ("layout_id", "repeat_id", "reference_layout")},
                "experiment_template": metadata.get("experiment_template"),
                "experiment_mode": bool(metadata.get("experiment_mode", False)),
            }
            # A free rollout has no template snapshot; keep the configured study for next time.
            if metadata.get("experiment_template") is not None:
                self.experiment_template = metadata["experiment_template"]
            if metadata.get("layout_id"):
                slot = metadata["layout_id"]
                if slot not in self.layout.options:
                    self.layout.options = (*self.layout.options, slot)
                self.layout.value = slot
            reference = metadata.get("reference_layout")
            if reference:
                self._reference_task = reference["task"]
            self._refresh_repeat_history()
            self.service_ready = False
            self.preparing_rollout = False
            self.last_rollout_error = ""
            self.recovery_available = bool(metadata.get("recovery_available", False))
            self._set_experiment_mode(bool(metadata.get("experiment_mode", False)))
            self._refresh_layout_visibility()
            self._configure_evaluation(metadata.get("evaluation"))
            self.rollout_setup_status.content = (
                "🟢 **Study rollout ready** · adjust layout before Start; "
                "save or skip evaluation after Finish."
                if self.experiment_mode
                else "🔵 **Free rollout ready** · Start when ready; "
                "Finish saves the episode."
            )
            self.prepare_normal_btn.visible = False
            self.prepare_experiment_btn.visible = False
            self._show_recorded_identity(metadata)
            if self.top_overlay is not None and metadata.get("reference_layout") is not None:
                self.top_overlay.restore_selection({
                    "layout_id": metadata["layout_id"],
                    "reference_layout": metadata["reference_layout"],
                })
            self._set_setup_controls_enabled(False)
            self._set_policy_controls_enabled(True)
            self._set_stage("control")
            self._set_instruction(str(metadata.get("instruction", self.task_command)))
            self.policy_name.value = str(metadata.get("policy_label", "waiting"))
            self.runtime_name.value = str(metadata.get("runtime", "waiting"))
            self._reset_evaluation(str(metadata.get("episode_dir", "")))
            self._update_warmup(metadata.get("warmup"))
            self.executor_info.value = "observe only" if self.observe_only else "managed"
            self.status.content = self._connected_status()
        elif event == "inference_submitted":
            planned = metadata.get("planned_switch_step")
            suffix = f" → switch {planned}" if planned is not None else ""
            self.executor_info.value = f"chunk #{message.get('chunk_id')} pending{suffix}"
        elif event == "episode_finished":
            self._clear_warmup_preview()
            self.camera_view.clear_images()
            self.episode_active = False
            self.service_ready = False
            self.launch_mode = str(metadata.get("launch_mode", self.launch_mode))
            self._set_experiment_mode(bool(metadata.get("experiment_mode", self.experiment_mode)))
            self._configure_evaluation(metadata.get("evaluation", self.evaluation_profile))
            self.executor_info.value = str(metadata.get("reason", "finished"))
            episode_dir = str(metadata.get("episode_dir", ""))
            if episode_dir:
                incoming_episode = Path(episode_dir).expanduser()
                if incoming_episode != self.current_episode_dir:
                    self._reset_evaluation(episode_dir)
                self.current_episode_dir = incoming_episode
                self.episode_path.value = episode_dir
            self.episode_finalized = self.current_episode_dir is not None
            self.evaluation_complete = not self.experiment_mode
            self.paused = True
            self.rollout_started = False
            self._set_policy_controls_enabled(False)
            self._set_evaluation_enabled(self.episode_finalized and self.experiment_mode)
            self._set_stage("evaluation" if self.experiment_mode else "complete")
            if not self.episode_finalized:
                self.evaluation_status.content = "🔴 Runtime did not publish a rollout path."
                self.status.content = "🔴 **Rollout finished without a saved path**"
            elif self.experiment_mode:
                self.evaluation_status.content = (
                    "🟡 Save a task result, or click Skip evaluation."
                )
                self.status.content = (
                    "⚪ **Rollout finished · save or skip evaluation**"
                    if self.launch_mode == "serve"
                    else "⚪ **One-shot run finished · save or skip evaluation**"
                )
            else:
                self.evaluation_status.content = (
                    "⚪ Experiment mode was OFF; no human reward is required."
                )
                self.status.content = (
                    "⚪ **Rollout finished · waiting for runtime service**"
                    if self.launch_mode == "serve"
                    else "⚪ **One-shot run finished · use `manimux serve` for UI rollouts**"
                )
        elif event == "runtime_service_ready":
            self._policy_service_state = dict(
                metadata.get("policy_service", {"status": "external"})
            )
            model_status = self._policy_service_state.get("status", "external")
            self.policy_service_status.visible = model_status != "external"
            descriptions = {
                "idle": "Waiting to start the selected policy.",
                "stopping": "Stopping the previous model and releasing its memory…",
                "starting": "Loading the selected model…",
                "ready": "Model server is listening. Prepare will verify identity and capabilities.",
                "failed": "Model startup failed. Fix the issue and Retry policy, or select another.",
            }
            self.policy_service_status.content = descriptions.get(model_status, "")
            if self._policy_service_state.get("error"):
                self.policy_service_status.content += "\n\n" + self._policy_service_state["error"]
            if self._policy_service_state.get("log_path"):
                self.policy_service_status.content += (
                    "\n\nLog: `" + self._policy_service_state["log_path"] + "`"
                )
            self.last_service_time = time.time()
            incoming_service_id = str(metadata.get("run_dir", ""))
            new_service = bool(incoming_service_id and incoming_service_id != self.service_id)
            selection_discovered = (str(metadata.get("experiment_id", "")) != self._experiment_id
                                    and not self._pending_experiment_id)
            first_service_announcement = self.launch_mode != "serve" or new_service or selection_discovered
            if new_service:
                self._reset_for_new_service()
            if first_service_announcement:
                self._configure_experiment_choices(metadata)
                self._configure_research(metadata)
            elif "experiment_template" in metadata and metadata["experiment_template"] != self.experiment_template:
                # Reconnecting during a free run only gives us its unscored identity.
                # Restore the service's study rules when idle, preserving the selected layout.
                self._configure_research(metadata, reset_selection=False)
            if new_service or first_service_announcement or "camera_map" in metadata:
                self.camera_view.set_policy_map(
                    metadata.get("camera_map"),
                    reset=new_service or first_service_announcement,
                )
            if incoming_service_id:
                self.service_id = incoming_service_id
            self.launch_mode = "serve"
            self.policy_name.value = str(metadata.get("policy_label", "waiting"))
            self.runtime_name.value = str(metadata.get("runtime", "waiting"))
            self._set_instruction(str(metadata.get("task", self.task_command)))
            last_error = str(metadata.get("last_error", ""))
            new_failure = self._accept_rollout_failure(metadata)
            if self.preparing_rollout and not new_failure:
                return
            self.preparing_rollout = False
            self.service_ready = True
            self._update_recovery(metadata)
            self.prepare_normal_btn.visible = True
            self.prepare_experiment_btn.visible = True
            self._refresh_layout_visibility()
            self.rollout_setup_status.content = self._setup_hint()
            if self.current_episode_dir is None or self.evaluation_complete:
                self.episode_path.value = str(metadata.get("last_episode_dir", "")) or "ready"
                self._configure_evaluation(metadata.get("evaluation"))
            self._update_prepare_enabled()
            self._set_stage("setup" if self.evaluation_complete else "evaluation")
            self._idle_status(last_error)
            if metadata.get("selection_error"):
                self._pending_experiment_id = ""
                self._configure_experiment_choices(metadata)
                self._update_prepare_enabled()
                self.rollout_setup_status.content = f"🔴 {metadata['selection_error']}"
            elif self._pending_experiment_id:
                self.rollout_setup_status.content = "🟠 Switching experiment configuration…"
        elif event == "episode_failed":
            if metadata.get("run_dir") and metadata["run_dir"] != self.service_id:
                return
            if not self._accept_rollout_failure(metadata):
                return
            self.camera_view.clear_images()
            self._update_recovery(metadata)
            self.prepare_normal_btn.visible = True
            self.prepare_experiment_btn.visible = True
            self._refresh_layout_visibility()
            self._set_policy_controls_enabled(False)
            self._update_prepare_enabled()
            self._set_stage("setup")
            self._idle_status(self.last_rollout_error)

    def _update_group(self, group: RobotGroup, configuration: FloatArray) -> None:
        robot_handle = self.robot_handles.get(group.name)
        if robot_handle is not None:
            robot_handle.update_cfg(self.robot.visual_configuration(group.name, configuration))
        transform = self.robot.pose(group.name, configuration)
        # current_ee is also a child of the translated group root.
        position = transform[:3, 3]
        from scipy.spatial.transform import Rotation

        xyzw = Rotation.from_matrix(transform[:3, :3]).as_quat()
        self.server.scene.add_frame(
            f"{self._root(group)}/current_ee",
            axes_length=0.08,
            axes_radius=0.004,
            position=position,
            wxyz=(xyzw[3], *xyzw[:3]),
            visible=bool(self.show_frames.value),
        )
        self.tails[group.name].append(position)
        self._draw_tail(group, np.asarray(self.tails[group.name]))

    def _draw_tail(self, group: RobotGroup, points: FloatArray) -> None:
        name = f"{self._root(group)}/achieved_tail"
        if len(points) < 2:
            self.server.scene.add_line_segments(
                name,
                np.empty((0, 2, 3)),
                group.trail_color,
                visible=False,
            )
            return
        self.server.scene.add_line_segments(
            name,
            np.stack((points[:-1], points[1:]), axis=1),
            group.trail_color,
            line_width=3,
            visible=bool(self.show_tail.value),
        )

    def close(self) -> None:
        self.running = False
        self.receiver.close()
        self.control_server.close()
        self.server.stop()


def _demo_sample(robot: RobotView, elapsed_s: float, horizon: int):
    states, plans = {}, {}
    for name, style in robot.options.get("groups", {}).items():
        q = robot.initial_positions(name).copy()
        demo = style.get("demo", {})
        amplitude = np.asarray(demo.get("amplitude", np.zeros_like(q)), dtype=float)
        phase = elapsed_s + float(demo.get("phase", 0))
        q += amplitude * np.sin(phase)
        states[name] = q
        plans[name] = np.stack(
            [
                robot.initial_positions(name) + amplitude * np.sin(phase + i / 30)
                for i in range(horizon)
            ]
        )
    return states, plans


def _demo(robogui: PolicyRoboGUI) -> None:
    elapsed_s = 0.0
    chunk_id = 0
    next_plan_s = 0.0
    while robogui.running:
        joints, actions = _demo_sample(robogui.robot, elapsed_s, horizon=25)
        if elapsed_s >= next_plan_s:
            robogui.on_message(
                PolicyPlan(
                    policy="Synthetic demo",
                    instruction="Inspect a predicted action chunk",
                    groups=actions,
                    action_dt=1 / 30,
                    inference_ms=824,
                    chunk_id=chunk_id,
                    robot=robogui.robot.name,
                ).to_wire()
            )
            chunk_id += 1
            next_plan_s = elapsed_s + 2.0
        height, width = 180, 320
        x = np.broadcast_to(np.linspace(0, 1, width)[None, :], (height, width))
        y = np.broadcast_to(np.linspace(0, 1, height)[:, None], (height, width))
        camera = np.stack((x, y, np.full_like(x, 0.3)), axis=-1)
        robogui.on_message(
            RobotSnapshot(
                groups=joints,
                cameras={"overview": (camera * 255).astype(np.uint8)},
                step=int(elapsed_s * 30),
                max_steps=1000,
                robot=robogui.robot.name,
            ).to_wire()
        )
        elapsed_s += 0.05
        time.sleep(0.05)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="RoboGUI YAML; defaults to following policy inputs, supports manual camera preview",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8086)
    parser.add_argument("--render-hz", type=float, default=30.0,
                        help="Live display refresh cap; independent of robot control Hz")
    parser.add_argument("--bridge-endpoint", default="tcp://127.0.0.1:5568")
    parser.add_argument("--control-endpoint", default="tcp://127.0.0.1:5569")
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_LAYOUT_ROOT)
    parser.add_argument(
        "--robot",
        default="tianji",
        help="RoboGUI preset containing robogui.yaml (yam, tianji, piper or aloha)",
    )
    parser.add_argument(
        "--list-robots",
        action="store_true",
        help="list body folders containing robogui.yaml and exit",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true", help="show synthetic data without hardware")
    mode.add_argument("--replay-actions", type=Path, help="offline NPZ of named joint trajectories")
    parser.add_argument("--action-dt-s", type=float, help="seconds between replay action points")
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if not np.isfinite(args.render_hz) or args.render_hz <= 0:
        parser.error("--render-hz must be finite and positive")
    if args.replay_actions is not None:
        if args.action_dt_s is None or not np.isfinite(args.action_dt_s) or args.action_dt_s <= 0:
            parser.error("--replay-actions requires a finite, positive --action-dt-s")
    elif args.action_dt_s is not None:
        parser.error("--action-dt-s requires --replay-actions")
    if args.list_robots:
        print(
            "\n".join(
                sorted(
                    p.parent.name for p in (Path(__file__).parent / "robots").glob("*/robogui.yaml")
                )
            )
        )
        return
    robogui_config = load_robogui_config(args.config, robot=args.robot)
    robot = load_robot_view(robogui_config)
    if args.replay_actions is not None:
        from .action_replay import serve_action_replay

        serve_action_replay(
            args.replay_actions, robot, action_dt_s=args.action_dt_s,
            host=args.host, port=args.port,
        )
        return
    robogui = PolicyRoboGUI(
        args.host,
        args.port,
        args.bridge_endpoint,
        args.control_endpoint,
        robot,
        reference_root=args.reference_root,
        robogui_config=robogui_config,
        render_hz=args.render_hz,
    )
    if args.demo:
        threading.Thread(target=_demo, args=(robogui,), daemon=True).start()
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    print(f"Robot model: {robot.name} ({robot.label})")
    print(f"RoboGUI camera mode: {robogui_config['camera_mode']}")
    display_host = {"0.0.0.0": "127.0.0.1", "localhost": "127.0.0.1", "::": "::1"}.get(
        args.host, args.host
    )
    if ":" in display_host:
        display_host = f"[{display_host}]"
    print(f"Open http://{display_host}:{args.port}")
    try:
        while not stop.wait(0.25):
            now = time.time()
            if (
                robogui.episode_active
                and robogui.last_state_time
                and now - robogui.last_state_time > 2
            ) or (
                robogui.service_ready
                and robogui.last_service_time
                and now - robogui.last_service_time > 3
            ):
                robogui._mark_runtime_unavailable()
    finally:
        robogui.close()


if __name__ == "__main__":
    main()
