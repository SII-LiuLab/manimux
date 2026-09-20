"""Universal Viser dashboard for live robot-policy inference."""

from __future__ import annotations

import argparse
import signal
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import viser

from manimux.evaluation import write_manual_evaluation
from manimux.types import FloatArray, UInt8Array

from .camera_panel import CameraPanel
from .camera_panel import _camera_panel_html as _camera_panel_html
from .chunk_timeline import ChunkTimelineView
from .communication import ControlServer, PolicyPlan, RobotSnapshot, ViewerReceiver
from .reference_layouts import DEFAULT_LAYOUT_ROOT
from .robot_view import RobotGroup, RobotView
from .top_overlay import TopViewOverlay

MAX_PLAN_HISTORY = 16
ViewerStage = Literal["waiting", "setup", "preparing", "control", "evaluation", "complete"]
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


def _prefill_task(current: str, incoming: str) -> str:
    """Seed the operator's editable command without overwriting their typing.

    The runtime republishes its instruction on every chunk, so assigning it each
    time would erase a command the operator is halfway through composing.
    """

    return incoming.strip() if not current.strip() else current


class PolicyViewer:
    """Robot-independent dashboard consuming an offline RobotModel view."""

    def __init__(
        self,
        host: str,
        port: int,
        bridge_endpoint: str,
        control_endpoint: str,
        robot: RobotView,
        reference_root: Path = DEFAULT_LAYOUT_ROOT,
        viewer_config: dict | None = None,
    ) -> None:
        self.robot = robot
        self.viewer_config = viewer_config if viewer_config is not None else robot.options
        self.reference_root = reference_root
        self.server = viser.ViserServer(host=host, port=port, label="Universal Policy Viewer")
        self.server.gui.configure_theme(
            control_layout="fixed",
            control_width="medium",
            dark_mode=False,
            show_logo=False,
            show_share_button=False,
            brand_color=(70, 103, 190),
        )
        self.server.gui.set_panel_label("UNIVERSAL · POLICY VIEWER")
        self.lock = threading.RLock()
        self.running = True
        self.paused = True
        self.rollout_started = False
        self.finish_requested = False
        self.home_requested = False
        self.finish_home: bool | None = None
        self._clear_recovery()
        self.new_rollout_requested = False
        self.preparing_rollout = False
        self.service_ready = False
        self.experiment_mode = False
        self.evaluation_saved = True
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
        self.receiver = ViewerReceiver(bridge_endpoint, self.on_message)
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
                print(f"[viewer] {mesh.name} mesh unavailable ({exc})")
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
                    f"[viewer] {group.label} URDF unavailable ({exc}); "
                    "trajectory rendering remains enabled"
                )

    def _build_gui(self) -> None:
        self.top_overlay: TopViewOverlay | None = None
        self.camera_view = CameraPanel(
            self.server.gui,
            self.viewer_config,
            self.robot.camera_slot,
            lambda image: self.top_overlay.update(image) if self.top_overlay is not None else None,
            defer_diagnostics=True,
        )
        with self.camera_view.display_container:
            self.chunk_timeline_panel = self.server.gui.add_html(self.chunk_timeline.render_html())
        self.status = self.server.gui.add_markdown("🟠 **Waiting for policy executor**")
        self.instruction = self.server.gui.add_markdown(_instruction_markdown(""))
        if self.robot.name in {"tianji", "tianji-taccap"}:
            self._build_recovery_gui()
        self.new_rollout_folder = self.server.gui.add_folder(
            "① New rollout", expand_by_default=True
        )
        with self.new_rollout_folder:
            self.rollout_setup_status = self.server.gui.add_markdown(
                "⚪ Waiting for a ManiMux runtime service."
            )
            self.task = self.server.gui.add_text("Task command", "", multiline=True, disabled=True)
            self.layout_id = self.server.gui.add_text(
                "Experiment layout / condition ID", "default", disabled=True
            )
            self.prepare_normal_btn = self.server.gui.add_button(
                "Prepare normal rollout", color="blue", disabled=True
            )
            self.prepare_experiment_btn = self.server.gui.add_button(
                "🧪 Prepare experiment rollout", color="green", disabled=True
            )
        self.policy_control_folder = self.server.gui.add_folder(
            "② Policy control", expand_by_default=True
        )
        with self.policy_control_folder:
            self.start_btn = self.server.gui.add_button(
                "Start rollout", color="blue", disabled=True
            )
            self.pause_btn = self.server.gui.add_button("Pause / Hold", color="gray", disabled=True)
            self.finish_btn = self.server.gui.add_button(
                "Finish rollout" if self.robot.name == "tianji-taccap" else "Finish & Home",
                color="red", disabled=True
            )
            if self.robot.name in {"tianji", "tianji-taccap"}:
                self.finish_no_home_btn = self.server.gui.add_button(
                    "Finish without homing", color="gray", disabled=True,
                    visible=self.robot.name != "tianji-taccap",
                )
        if self.robot.name not in {"tianji", "tianji-taccap"}:
            self._build_recovery_gui()
        self.run_folder = self.server.gui.add_folder("Run", expand_by_default=True)
        with self.run_folder:
            self.robot_name = self.server.gui.add_text("Robot", self.robot.label, disabled=True)
            self.policy_name = self.server.gui.add_text("Policy", "waiting", disabled=True)
            self.action_space = self.server.gui.add_text("Action space", "waiting", disabled=True)
            self.chunk_info = self.server.gui.add_text("Chunk", "—", disabled=True)
            self.executor_info = self.server.gui.add_text("Executor", "waiting", disabled=True)
            self.latency = self.server.gui.add_text("Inference", "—", disabled=True)
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
            self.task_result = self.server.gui.add_dropdown(
                "Task result",
                ("unlabeled", "success", "failure", "invalid"),
                initial_value="unlabeled",
                disabled=True,
            )
            self.smoothness_score = self.server.gui.add_dropdown(
                "Smoothness (1-5)",
                ("1", "2", "3", "4", "5"),
                initial_value="3",
                disabled=True,
            )
            self.reviewer_id = self.server.gui.add_text("Reviewer", "operator", disabled=True)
            self.operator_note = self.server.gui.add_text(
                "Operator note", "", multiline=True, disabled=True
            )
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
        self.overlay_folder = self.server.gui.add_folder(
            "Overlay controls", expand_by_default=True
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
            for camera in self.viewer_config.get("cameras", [])
        ):
            self.top_overlay = TopViewOverlay(
                self.server.gui,
                self.reference_root,
                display_container=self.camera_view.display_container,
            )
        self.camera_view.add_diagnostics(expand_by_default=True)
        self._set_stage("waiting")

        @self.start_btn.on_click
        def _start(_event: Any) -> None:
            with self.lock:
                if self.start_btn.disabled:
                    return
                self.paused = False
                self.rollout_started = True
                self._set_policy_controls_enabled(True)
                self.status.content = "🟢 **Connected · RUNNING**"

        @self.prepare_normal_btn.on_click
        def _prepare_normal(_event: Any) -> None:
            self._prepare_rollout(experiment_mode=False)

        @self.prepare_experiment_btn.on_click
        def _prepare_experiment(_event: Any) -> None:
            self._prepare_rollout(experiment_mode=True)

        @self.pause_btn.on_click
        def _pause(_event: Any) -> None:
            with self.lock:
                if self.pause_btn.disabled:
                    return
                self.paused = True
                self._set_policy_controls_enabled(True)
                self.status.content = "🟡 **Connected · PAUSED**"

        @self.home_btn.on_click
        def _home(_event: Any) -> None:
            with self.lock:
                if self.home_btn.disabled:
                    return
                if not self.episode_active and hasattr(self, "drag_btn"):
                    self._request_recovery("home")
                    return
                self.paused = True
                self.home_requested = True
                self.status.content = "🟡 **Returning home · PAUSED**"

        @self.finish_btn.on_click
        def _finish(_event: Any) -> None:
            self._finish_rollout(home=True)

        if hasattr(self, "drag_btn"):

            @self.finish_no_home_btn.on_click
            def _finish_no_home(_event: Any) -> None:
                self._finish_rollout(home=False)

            @self.drag_btn.on_click
            def _drag(_event: Any) -> None:
                self._request_recovery(f"drag:{self.drag_arm.value}")

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
            self._save_manual_evaluation()

        @self.show_plan.on_update
        def _show_plan(_event: Any) -> None:
            self._refresh_plan_visibility()

    def _build_recovery_gui(self) -> None:
        self.recovery_folder = self.server.gui.add_folder(
            "Manual recovery"
            if self.robot.name in {"tianji", "tianji-taccap"}
            else "Advanced recovery",
            expand_by_default=self.robot.name in {"tianji", "tianji-taccap"},
        )
        with self.recovery_folder:
            if self.robot.name in {"tianji", "tianji-taccap"}:
                self.recovery_status = self.server.gui.add_markdown(
                    "⚪ Waiting for a Tianji runtime service."
                )
                self.clear_error_btn = self.server.gui.add_button(
                    "Clear controller error", color="red", disabled=True
                )
                self.drag_arm = self.server.gui.add_dropdown(
                    "Drag arms", ("A", "B", "AB"), initial_value="AB", disabled=True
                )
                self.drag_btn = self.server.gui.add_button(
                    "Start drag", color="blue", disabled=True
                )
                self.stop_drag_btn = self.server.gui.add_button(
                    "Exit drag", color="gray", disabled=True
                )
            self.home_btn = self.server.gui.add_button(
                "Return Home (keep rollout open)", color="gray", disabled=True
            )
            if self.robot.name in {"tianji", "tianji-taccap"}:
                self.recovery_details_folder = self.server.gui.add_folder(
                    "Last error", expand_by_default=False, visible=False
                )
                with self.recovery_details_folder:
                    self.recovery_details = self.server.gui.add_markdown("")

    def _finish_rollout(self, *, home: bool) -> None:
        with self.lock:
            if not self.episode_active or self.finish_btn.disabled:
                return
            self.paused = True
            self.finish_requested = True
            # The new assembly has no Home trajectory. Its primary Finish
            # action must save/close cleanly instead of requesting robot.home().
            self.finish_home = (
                False if self.robot.name == "tianji-taccap"
                else home if self.robot.name == "tianji" else None
            )
            self.service_ready = False
            self._set_policy_controls_enabled(False)
            self._update_recovery_controls()
            self.status.content = "🟠 **Finishing rollout and saving episode**"

    def _clear_recovery(self) -> None:
        self.recovery_available = False
        self.recovery_actions: set[str] = set()
        self.recovery_busy = False
        self.recovery_state = "idle"
        self.recovery_request = ""
        self.recovery_request_id = ""
        self.recovery_lease = False
        self.recovery_error = ""
        self.recovery_arm = ""
        self.last_rollout_error = ""
        self.last_failure_id = ""

    def _can_stop_and_drag(self) -> bool:
        return (
            self.episode_active
            and self.launch_mode == "serve"
            and self.recovery_available
            and "drag" in self.recovery_actions
            and not self.observe_only
            and not self.finish_btn.disabled
        )

    def _recovery_pending(self) -> bool:
        return bool(getattr(self, "recovery_request", "") or getattr(self, "recovery_busy", False))

    def _request_recovery(self, action: str) -> None:
        with self.lock:
            if action == "stop":
                # Cancel a queued drag even while runtime cleanup is in flight
                # or the service heartbeat has disappeared.
                if not self._recovery_pending() and not self.recovery_lease:
                    return
            else:
                if not self.recovery_available or self._recovery_pending():
                    return
                if action.partition(":")[0] not in self.recovery_actions:
                    return
                if self.episode_active:
                    if not action.startswith("drag:") or not self._can_stop_and_drag():
                        return
                    self._finish_rollout(home=False)
                elif not self.service_ready or self.preparing_rollout:
                    return
            self.recovery_request = action
            self.recovery_request_id = uuid.uuid4().hex
            self.recovery_lease = action.startswith("drag:")
            self.recovery_status.content = (
                "🟠 **Waiting for recovery control**" if action != "stop" else "🟠 **Exiting drag**"
            )
            self._update_prepare_enabled()
            self._update_recovery_controls()

    def _update_recovery_controls(self) -> None:
        if not hasattr(self, "drag_btn"):
            return
        idle = self.service_ready and not self.episode_active and not self.preparing_rollout
        allowed = idle and self.recovery_available and not self._recovery_pending()
        can_clear = allowed and "clear_error" in self.recovery_actions
        can_drag = (allowed and "drag" in self.recovery_actions) or (
            self._can_stop_and_drag() and not self._recovery_pending()
        )
        self.clear_error_btn.disabled = not can_clear
        self.drag_arm.disabled = not can_drag
        self.drag_btn.disabled = not can_drag
        self.drag_btn.label = "Stop rollout & drag" if self.episode_active else "Start drag"
        self.stop_drag_btn.disabled = not (
            self.recovery_lease or self.recovery_state in {"starting", "active"}
        )
        self.stop_drag_btn.label = (
            "Cancel drag"
            if (self.recovery_lease and self.recovery_state == "idle")
            else "Exit drag"
        )
        if not self.episode_active:
            self.home_btn.disabled = not (allowed and "home" in self.recovery_actions)
        self._render_recovery_status()

    def _render_recovery_status(self) -> None:
        error = self.recovery_error
        arm = self.recovery_arm
        if self.recovery_request == "stop":
            text = "🟠 **Exiting / cancelling drag**"
        elif self.recovery_request == "clear_error":
            text = "🟠 **Clearing controller error** · no arm will be enabled."
        elif self.recovery_lease and self.episode_active:
            text = "🟠 **Stopping rollout without homing · waiting to enter drag**"
        elif self.recovery_request:
            text = "🟠 **Waiting for recovery control**"
        elif self.recovery_state == "active":
            text = f"🟢 **Drag {arm} · UMI** · adjust by hand, then Exit drag."
        elif self.recovery_state == "starting":
            text = f"🟠 **Starting drag {arm} · checking controller and UMI parameters**"
        elif self.recovery_state == "stopping":
            text = "🟠 **Exiting drag · waiting for servo-off**"
        elif self.recovery_state == "homing":
            text = "🟡 **Returning home**"
        elif self.recovery_state == "clearing":
            text = "🟠 **Clearing controller error** · sending the Marvin SDK command."
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
                "🟡 Stop rollout & drag ends this rollout without homing, then enters UMI drag."
                if self._can_stop_and_drag()
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
        elif self.robot.name == "tianji-taccap" and not self.recovery_available:
            text = "⚪ Controller error recovery requires an executing runtime service."
        elif not self.recovery_available:
            text = "⚪ Recovery requires an executing Tianji runtime service."
        elif "emergency stop" in self.last_rollout_error.lower():
            text = (
                "🔴 **Rollout stopped by E-stop** · release the physical E-stop, "
                "then Start drag or Return Home. Latched errors are checked and cleared first."
            )
        elif self.last_rollout_error:
            text = "🟡 **Rollout interrupted** · drag A / B / AB or Return Home when ready."
        elif {"drag", "home"} & self.recovery_actions:
            text = "⚪ Drag A / B / AB with UMI, or Return Home directly."
        else:
            text = "⚪ Clear controller error resets latched A/B faults without enabling motion."
        self.recovery_status.content = text
        if hasattr(self, "recovery_details"):
            detail = error or self.last_rollout_error
            self.recovery_details_folder.visible = bool(detail)
            self.recovery_details.content = detail

    def _update_recovery(self, metadata: dict[str, Any]) -> None:
        if not hasattr(self, "drag_btn"):
            return
        recovery = metadata.get("recovery") or {}
        self.recovery_available = bool(recovery.get("available", False))
        actions = recovery.get("actions")
        self.recovery_actions = (
            {str(action) for action in actions}
            if actions is not None
            else ({"drag", "home"} if self.recovery_available else set())
        )
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
        self.evaluation_saved = True
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
        elif self.evaluation_saved:
            self.status.content = "🟡 **Runtime service ready · prepare a rollout**"

    def _set_instruction(self, instruction: str) -> None:
        self.instruction.content = _instruction_markdown(instruction)
        self.task.value = _prefill_task(self.task.value, instruction)

    def _set_stage(self, stage: ViewerStage) -> None:
        """Expose only the controls that are actionable in the current stage."""

        self.new_rollout_folder.visible = stage in {"waiting", "setup", "preparing"}
        self.policy_control_folder.visible = stage == "control"
        self.recovery_folder.visible = stage == "control" or hasattr(self, "drag_btn")
        self.evaluation_folder.visible = stage == "evaluation"
        self.overlay_folder.visible = stage == "control"
        self.run_folder.visible = stage not in {"waiting"}
        if hasattr(self, "drag_btn"):
            self.home_btn.label = (
                "Return Home (keep rollout open)" if stage == "control" else "Return Home"
            )
            self._update_recovery_controls()

    def _set_evaluation_enabled(self, enabled: bool) -> None:
        for handle in (
            self.task_result,
            self.smoothness_score,
            self.reviewer_id,
            self.operator_note,
            *self.failure_tag_inputs.values(),
        ):
            handle.disabled = not enabled
        self.save_evaluation_btn.disabled = not enabled

    def _set_experiment_mode(self, enabled: bool) -> None:
        self.experiment_mode = enabled
        self.rollout_setup_status.content = (
            "🟢 **Experiment rollout** · a human label is required after Finish."
            if enabled
            else "🔵 **Normal rollout** · no human label is required."
        )

    def _set_setup_controls_enabled(self, enabled: bool) -> None:
        allowed = enabled and self.evaluation_saved and not self._recovery_pending()
        self.prepare_normal_btn.disabled = not allowed
        self.prepare_experiment_btn.disabled = not allowed
        self.layout_id.disabled = not allowed
        self.task.disabled = not allowed

    def _prepare_rollout(self, *, experiment_mode: bool) -> None:
        with self.lock:
            if self._recovery_pending() or not self.service_ready or not self.evaluation_saved:
                return
            self._set_experiment_mode(experiment_mode)
            self.new_rollout_requested = True
            self.preparing_rollout = True
            self.service_ready = False
            self.paused = True
            self._set_setup_controls_enabled(False)
            self.prepare_normal_btn.visible = False
            self.prepare_experiment_btn.visible = False
            self._set_stage("preparing")
            kind = "experiment" if experiment_mode else "normal"
            self.status.content = f"🟠 **Preparing a new {kind} rollout**"

    def _set_policy_controls_enabled(self, enabled: bool) -> None:
        allowed = enabled and not self.observe_only
        self.start_btn.label = "Resume rollout" if self.rollout_started else "Start rollout"
        self.start_btn.disabled = not allowed or not self.paused
        self.pause_btn.disabled = not allowed or self.paused
        self.home_btn.disabled = (
            not allowed or not self.paused or self.robot.name == "tianji-taccap"
        )
        self.finish_btn.disabled = not allowed
        if hasattr(self, "finish_no_home_btn"):
            self.finish_no_home_btn.disabled = not allowed
        self._update_recovery_controls()

    def _update_prepare_enabled(self) -> None:
        self._set_setup_controls_enabled(self.service_ready)

    def _reset_evaluation(self, episode_dir: str) -> None:
        self.current_episode_dir = Path(episode_dir).expanduser() if episode_dir else None
        self.episode_finalized = False
        self.evaluation_saved = not self.experiment_mode
        self.episode_path.value = episode_dir or "not published"
        self.task_result.value = "unlabeled"
        self.smoothness_score.value = "3"
        self.reviewer_id.value = "operator"
        self.operator_note.value = ""
        for handle in self.failure_tag_inputs.values():
            handle.value = False
        self._set_evaluation_enabled(False)
        self.evaluation_status.content = "⚪ Finish the rollout before saving an evaluation."

    def _reset_for_new_service(self) -> None:
        """Make a new ``manimux serve`` instance a hard workflow boundary."""

        # A task edit belongs to one runtime service.  Clear it at this hard
        # boundary so the following runtime_service_ready event can seed the
        # new service's task instead of preserving a stale command forever.
        self.task.value = ""
        self.paused = True
        self.rollout_started = False
        self.home_requested = False
        self._clear_recovery()
        self.finish_requested = False
        self.new_rollout_requested = False
        self.preparing_rollout = False
        self.episode_active = False
        self.current_episode_dir = None
        self.episode_finalized = False
        self.evaluation_saved = True
        self.last_state_time = 0.0
        self.executor_info.value = "service idle"
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
        self.preparing_rollout = False
        self.service_ready = False
        self.episode_active = False
        self.recovery_lease = False
        self._update_recovery_controls()
        self.camera_view.set_policy_map(None, reset=True)
        self._set_policy_controls_enabled(False)
        self._set_setup_controls_enabled(False)
        self._set_stage("waiting")
        self.rollout_setup_status.content = "⚪ Waiting for a ManiMux runtime service."
        self.status.content = "🟠 **Runtime unavailable · waiting for service**"

    def _save_manual_evaluation(self) -> None:
        if not self.episode_finalized or self.current_episode_dir is None:
            self.evaluation_status.content = "🔴 Episode is not finalized."
            return
        result = str(self.task_result.value)
        if result == "unlabeled":
            self.evaluation_status.content = "🔴 Select success, failure, or invalid."
            return
        try:
            target = write_manual_evaluation(
                self.current_episode_dir,
                task_result=cast(Any, result),
                smoothness_score=int(self.smoothness_score.value),
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
        self.evaluation_saved = True
        self._update_prepare_enabled()
        self._set_stage("setup" if self.service_ready else "complete")

    def control_state(self) -> dict[str, Any]:
        with self.lock:
            state = {
                "paused": self.paused,
                "home_requested": self.home_requested,
                "finish_requested": self.finish_requested,
                "finish_home": getattr(self, "finish_home", None),
                "recovery_request": getattr(self, "recovery_request", ""),
                "recovery_request_id": getattr(self, "recovery_request_id", ""),
                "recovery_service_id": getattr(self, "service_id", ""),
                "recovery_lease": getattr(self, "recovery_lease", False),
                "new_rollout_requested": self.new_rollout_requested,
                "task_command": self.task.value.strip(),
                "experiment_mode": self.experiment_mode,
                "layout_id": self.layout_id.value.strip(),
            }
            self.home_requested = False
            self.finish_requested = False
            self.finish_home = None
            self.new_rollout_requested = False
            return state

    def _matches_selected_robot(self, message: dict[str, Any]) -> bool:
        robot_name = str(message.get("robot", ""))
        if not robot_name or robot_name == self.robot.name:
            return True
        self.status.content = (
            f"🔴 **Message targets robot `{robot_name}`; viewer uses `{self.robot.name}`**"
        )
        return False

    def on_message(self, message: dict[str, Any]) -> None:
        with self.lock:
            if not self._matches_selected_robot(message):
                return
            try:
                kind = message.get("kind")
                if kind == "plan":
                    self._update_plan(message)
                elif kind == "state":
                    self._update_state(message)
                elif kind == "event":
                    self._update_event(message)
                timeline = getattr(self, "chunk_timeline", None)
                if timeline is not None:
                    timeline.update(message)
                    self._refresh_chunk_timeline(force=kind != "state")
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
        if action_space != "joint_position":
            raise ValueError("Viewer expects decoded joint_position plans")
        grouped_actions = self.robot.validate_groups(message.get("groups"), sequence=True)
        horizon = len(next(iter(grouped_actions.values())))
        metadata = dict(message.get("metadata") or {})
        gripper_by_group = self.robot.gripper_closed_steps_by_group(
            grouped_actions,
            previous_positions=getattr(self, "last_joint_positions", {}),
        )
        metadata["gripper_closed_steps_by_group"] = {
            group_name: flags.tolist() for group_name, flags in gripper_by_group.items()
        }
        message["metadata"] = metadata
        start_index = int(message.get("start_index", 0))
        if start_index < 0 or start_index > horizon:
            raise ValueError("plan start_index is outside the action horizon")
        chunk_id = int(message.get("chunk_id", 0))
        if self.plan_chunk_id is not None and chunk_id != self.plan_chunk_id:
            self._archive_current_plan()
        self.plan_actions = {
            group_name: np.asarray(group_actions, dtype=np.float64)
            for group_name, group_actions in grouped_actions.items()
        }
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
                self._draw_plan(group, group_actions[start_index:])

    def _draw_plan(self, group: RobotGroup, actions: FloatArray) -> None:
        root = f"{self._root(group)}/predicted_ee"
        if len(actions) < 2:
            for handle in self.current_plan_handles[group.name]:
                handle.visible = False
            self.current_plan_handles[group.name] = []
            return
        # This line lives below the group root, which already carries the arm's
        # base_position. Keep FK output in that local frame to avoid applying
        # the left/right base offset twice.
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
            points = self.robot.positions(group.name, actions)
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
        show_history = show_current and self.show_plan_history
        for history in self.plan_history_handles.values():
            for handle in history:
                handle.visible = show_history

    def _reset_plan_overlay(self) -> None:
        for handles in self.current_plan_handles.values():
            for handle in handles:
                handle.visible = False
        self.current_plan_handles = {group.name: [] for group in self.robot.groups}
        self._clear_plan_history()
        self.plan_actions = {}
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
        grouped_positions = self.robot.validate_groups(message.get("groups"))
        self.last_joint_positions = {
            group_name: np.asarray(configuration, dtype=np.float64).copy()
            for group_name, configuration in grouped_positions.items()
        }
        self.last_state_time = time.time()
        self.progress.value = int(message.get("step", 0))
        if not message.get("connected", True):
            self.status.content = "🟠 **Executor disconnected**"
        elif self.paused:
            self.status.content = (
                "🔵 **Connected · OBSERVE ONLY**"
                if self.observe_only
                else "🟡 **Connected · PAUSED**"
            )
        else:
            self.status.content = "🟢 **Connected · RUNNING**"
        active_chunk_id = message.get("active_chunk_id")
        action_index = int(message.get("chunk_index", 0))
        if (
            active_chunk_id is not None
            and self.plan_chunk_id is not None
            and int(active_chunk_id) == self.plan_chunk_id
        ):
            for group in self.robot.groups:
                actions = self.plan_actions.get(group.name)
                self._draw_plan(
                    group,
                    actions[action_index:] if actions is not None else np.empty((0, 0)),
                )
        for group_name, configuration in grouped_positions.items():
            self._update_group(self.robot.group(group_name), configuration)
        self.camera_view.update_images(message.get("cameras_jpeg", {}))

    def _update_event(self, message: dict[str, Any]) -> None:
        event = str(message.get("event", "unknown"))
        metadata = message.get("metadata") or {}
        if event == "episode_started":
            self.camera_view.set_policy_map(metadata.get("camera_map"), reset=True)
            self._reset_plan_overlay()
            self._clear_achieved_tails()
            self.episode_active = True
            self.launch_mode = str(metadata.get("launch_mode", "run"))
            incoming_service_id = str(metadata.get("run_dir", ""))
            if incoming_service_id:
                self.service_id = incoming_service_id
            self.observe_only = metadata.get("control_mode", "observe") == "observe"
            self.paused = True
            self.rollout_started = False
            self.service_ready = False
            self.preparing_rollout = False
            self.last_rollout_error = ""
            self.recovery_available = bool(metadata.get("recovery_available", False))
            self._set_experiment_mode(bool(metadata.get("experiment_mode", False)))
            self.rollout_setup_status.content = (
                "🟢 **Experiment rollout ready** · press Start rollout below; "
                "a label is required after Finish."
                if self.experiment_mode
                else "🔵 **Normal rollout ready** · press Start rollout below; "
                "Finish saves the episode."
            )
            self.prepare_normal_btn.visible = False
            self.prepare_experiment_btn.visible = False
            self.layout_id.value = str(metadata.get("layout_id", "")) or "default"
            self._set_setup_controls_enabled(False)
            self._set_policy_controls_enabled(True)
            self._set_stage("control")
            self._set_instruction(str(metadata.get("instruction", self.task.value)))
            self.policy_name.value = str(metadata.get("policy_label", "waiting"))
            self.runtime_name.value = str(metadata.get("runtime", "waiting"))
            self._reset_evaluation(str(metadata.get("episode_dir", "")))
            self.executor_info.value = "observe only" if self.observe_only else "managed"
            self.status.content = (
                "🔵 **Connected · OBSERVE ONLY**"
                if self.observe_only
                else "🟡 **Connected · PAUSED · press Start rollout**"
            )
        elif event == "inference_submitted":
            planned = metadata.get("planned_switch_step")
            suffix = f" → switch {planned}" if planned is not None else ""
            self.executor_info.value = f"chunk #{message.get('chunk_id')} pending{suffix}"
        elif event == "episode_finished":
            self.camera_view.clear_images()
            self.episode_active = False
            self.service_ready = False
            self.executor_info.value = str(metadata.get("reason", "finished"))
            episode_dir = str(metadata.get("episode_dir", ""))
            if episode_dir:
                self.current_episode_dir = Path(episode_dir).expanduser()
                self.episode_path.value = episode_dir
            self.episode_finalized = self.current_episode_dir is not None
            self.evaluation_saved = not self.experiment_mode
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
                    "🟡 Select the task result and smoothness score, then save."
                )
                self.status.content = (
                    "⚪ **Rollout finished · awaiting human reward**"
                    if self.launch_mode == "serve"
                    else "⚪ **One-shot run finished · awaiting human reward**"
                )
            else:
                self.evaluation_status.content = (
                    "⚪ Experiment mode was OFF; no human reward is required."
                )
                self.status.content = (
                    "⚪ **Rollout finished · preparing for the next rollout**"
                    if self.launch_mode == "serve"
                    else "⚪ **One-shot run finished · use `manimux serve` for UI rollouts**"
                )
        elif event == "runtime_service_ready":
            self.last_service_time = time.time()
            incoming_service_id = str(metadata.get("run_dir", ""))
            new_service = bool(incoming_service_id and incoming_service_id != self.service_id)
            first_service_announcement = self.launch_mode != "serve" or new_service
            if new_service:
                self._reset_for_new_service()
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
            self._set_instruction(str(metadata.get("task", self.task.value)))
            last_error = str(metadata.get("last_error", ""))
            new_failure = self._accept_rollout_failure(metadata)
            if self.preparing_rollout and not new_failure:
                return
            self.preparing_rollout = False
            self.service_ready = True
            self._update_recovery(metadata)
            self.prepare_normal_btn.visible = True
            self.prepare_experiment_btn.visible = True
            if first_service_announcement:
                self.layout_id.value = str(metadata.get("default_layout_id", "")) or "default"
            self.rollout_setup_status.content = (
                "Choose a normal rollout, or an experiment rollout that requires a label."
            )
            if self.current_episode_dir is None or self.evaluation_saved:
                self.episode_path.value = str(metadata.get("last_episode_dir", "")) or "ready"
            self._update_prepare_enabled()
            self._set_stage("setup" if self.evaluation_saved else "evaluation")
            self._idle_status(last_error)
        elif event == "episode_failed":
            if metadata.get("run_dir") and metadata["run_dir"] != self.service_id:
                return
            if not self._accept_rollout_failure(metadata):
                return
            self.camera_view.clear_images()
            self._update_recovery(metadata)
            self.prepare_normal_btn.visible = True
            self.prepare_experiment_btn.visible = True
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


def _demo(viewer: PolicyViewer) -> None:
    elapsed_s = 0.0
    chunk_id = 0
    next_plan_s = 0.0
    while viewer.running:
        joints, actions = _demo_sample(viewer.robot, elapsed_s, horizon=25)
        if elapsed_s >= next_plan_s:
            viewer.on_message(
                PolicyPlan(
                    policy="Synthetic demo",
                    instruction="Inspect a predicted action chunk",
                    groups=actions,
                    action_dt=1 / 30,
                    inference_ms=824,
                    chunk_id=chunk_id,
                    robot=viewer.robot.name,
                ).to_wire()
            )
            chunk_id += 1
            next_plan_s = elapsed_s + 2.0
        height, width = 180, 320
        x = np.broadcast_to(np.linspace(0, 1, width)[None, :], (height, width))
        y = np.broadcast_to(np.linspace(0, 1, height)[:, None], (height, width))
        camera = np.stack((x, y, np.full_like(x, 0.3)), axis=-1)
        viewer.on_message(
            RobotSnapshot(
                groups=joints,
                cameras={"overview": (camera * 255).astype(np.uint8)},
                step=int(elapsed_s * 30),
                max_steps=1000,
                robot=viewer.robot.name,
            ).to_wire()
        )
        elapsed_s += 0.05
        time.sleep(0.05)


def load_viewer_config(path: Path | None = None, *, robot="tianji") -> dict:
    from manimux.cli import read_yaml

    source = (
        path if path is not None else Path(__file__).parent / "robots" / robot / "viewer.yaml"
    ).resolve()
    config = read_yaml(source)
    if "model" not in config:
        raise ValueError("Viewer YAML must reference a RobotModel using model")
    model_path = (source.parent / config["model"]).resolve()
    if (
        not model_path.is_file()
        and source.is_relative_to(Path(__file__).parent / "robots")
        and "configs/embodiment/" in str(config["model"])
    ):
        # The wheel bundles the same repository configuration, not a second model definition.
        from importlib.resources import files

        suffix = str(config["model"]).split("configs/embodiment/", 1)[1]
        model_path = Path(str(files("manimux") / "configs" / "embodiment" / suffix))
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    config["model"] = model_path
    if config.get("camera_mode", "policy") not in {"policy", "manual"}:
        raise ValueError("camera_mode must be policy or manual")
    config.setdefault("camera_mode", "policy")
    cameras = config.setdefault("cameras", [])
    if not isinstance(cameras, list):
        raise ValueError("cameras must be an ordered list")
    slots = set()
    for camera in cameras:
        if (
            not isinstance(camera, dict)
            or not isinstance(camera.get("source"), str)
            or not camera["source"].strip()
        ):
            raise ValueError("each camera needs a source")
        camera.setdefault("label", camera["source"])
        camera.setdefault("slot", camera["source"])
        if any(
            not isinstance(camera[key], str) or not camera[key].strip() for key in ("label", "slot")
        ):
            raise ValueError("camera label and slot must be non-empty strings")
        if camera["slot"] in slots:
            raise ValueError("camera slots must be unique")
        slots.add(camera["slot"])
    return config


def load_robot_view(config: dict) -> RobotView:
    from manimux.embodiments.robot.base import RobotModel

    return RobotView(RobotModel.from_config(config["model"]), config)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="Viewer YAML; defaults to following policy inputs, supports manual camera preview",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8086)
    parser.add_argument("--bridge-endpoint", default="tcp://127.0.0.1:5568")
    parser.add_argument("--control-endpoint", default="tcp://127.0.0.1:5569")
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_LAYOUT_ROOT)
    parser.add_argument(
        "--robot",
        default="tianji",
        help="body folder containing viewer.yaml (currently tianji)",
    )
    parser.add_argument(
        "--list-robots",
        action="store_true",
        help="list body folders containing viewer.yaml and exit",
    )
    parser.add_argument("--demo", action="store_true", help="show synthetic data without hardware")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.list_robots:
        print(
            "\n".join(
                sorted(
                    p.parent.name for p in (Path(__file__).parent / "robots").glob("*/viewer.yaml")
                )
            )
        )
        return
    viewer_config = load_viewer_config(args.config, robot=args.robot)
    robot = load_robot_view(viewer_config)
    viewer = PolicyViewer(
        args.host,
        args.port,
        args.bridge_endpoint,
        args.control_endpoint,
        robot,
        reference_root=args.reference_root,
        viewer_config=viewer_config,
    )
    if args.demo:
        threading.Thread(target=_demo, args=(viewer,), daemon=True).start()
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    print(f"Robot model: {robot.name} ({robot.label})")
    print(f"Viewer camera mode: {viewer_config['camera_mode']}")
    print(f"Open http://localhost:{args.port}")
    try:
        while not stop.wait(0.25):
            now = time.time()
            if (
                viewer.episode_active
                and viewer.last_state_time
                and now - viewer.last_state_time > 2
            ) or (
                viewer.service_ready
                and viewer.last_service_time
                and now - viewer.last_service_time > 3
            ):
                viewer._mark_runtime_unavailable()
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
