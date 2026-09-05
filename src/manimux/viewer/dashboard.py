"""Universal Viser dashboard for live robot-policy inference."""

from __future__ import annotations

import argparse
import base64
import io
import signal
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import viser
from PIL import Image

from manimux.evaluation import write_manual_evaluation
from manimux.types import FloatArray, UInt8Array

from .chunk_timeline import ChunkTimelineView
from .protocol import PolicyPlan, RobotSnapshot
from .robots import available_robot_adapters, load_robot_adapter
from .robots.base import RobotAdapter, RobotGroup
from .transport import ControlServer, ViewerReceiver

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


def _camera_panel_html() -> str:
    """Place stable Viser image handles in a fixed screen-space camera panel."""

    return """
<style>
  :root {
    --manimux-camera-width: clamp(300px, 26vw, 460px);
    --manimux-camera-inner: calc(var(--manimux-camera-width) - 20px);
    --manimux-camera-top-height: clamp(157.5px, calc(14.625vw - 11.25px), 247.5px);
    --manimux-camera-small-width: clamp(136px, calc(13vw - 14px), 216px);
    --manimux-camera-height: clamp(262px, calc(21.9375vw + 8.875px), 397px);
  }
  .manimux-camera-panel {
    position: fixed; left: 16px; top: 64px;
    width: var(--manimux-camera-width);
    height: var(--manimux-camera-height);
    z-index: 4;
    padding: 10px; box-sizing: border-box; pointer-events: none;
    border: 1px solid rgba(128, 138, 156, 0.35); border-radius: 12px;
    background: rgba(18, 23, 32, 0.92); box-shadow: 0 8px 28px rgba(0,0,0,0.18);
  }
  .manimux-camera-label { position: fixed; z-index: 7;
    padding: 3px 9px; border-radius: 999px; color: white; background: rgba(8,12,18,0.78);
    font: 600 12px/1.4 system-ui, sans-serif; text-transform: uppercase; }
  .manimux-camera-label-top { left: 26px; top: 74px; }
  .manimux-camera-label-left { left: 26px;
    top: calc(82px + var(--manimux-camera-top-height)); }
  .manimux-camera-label-right {
    left: calc(34px + var(--manimux-camera-small-width));
    top: calc(82px + var(--manimux-camera-top-height)); }

  div:has(> .manimux-camera-anchor) + div,
  div:has(> .manimux-camera-anchor) + div + div,
  div:has(> .manimux-camera-anchor) + div + div + div {
    position: fixed; z-index: 5; margin: 0; padding: 0 !important;
    overflow: hidden; border-radius: 8px; background: #0e131c;
    pointer-events: auto;
  }
  div:has(> .manimux-camera-anchor) + div {
    left: 26px; top: 74px; width: var(--manimux-camera-inner);
    height: var(--manimux-camera-top-height);
  }
  div:has(> .manimux-camera-anchor) + div + div {
    left: 26px; top: calc(82px + var(--manimux-camera-top-height));
    width: var(--manimux-camera-small-width); aspect-ratio: 16 / 9;
  }
  div:has(> .manimux-camera-anchor) + div + div + div {
    left: calc(34px + var(--manimux-camera-small-width));
    top: calc(82px + var(--manimux-camera-top-height));
    width: var(--manimux-camera-small-width); aspect-ratio: 16 / 9;
  }
  div:has(> .manimux-camera-anchor) + div > div,
  div:has(> .manimux-camera-anchor) + div + div > div,
  div:has(> .manimux-camera-anchor) + div + div + div > div {
    width: 100%; height: 100%;
  }
  div:has(> .manimux-camera-anchor) + div img,
  div:has(> .manimux-camera-anchor) + div + div img,
  div:has(> .manimux-camera-anchor) + div + div + div img {
    width: 100%; height: 100% !important; max-width: none !important;
    object-fit: cover; display: block;
  }
  @media (max-width: 900px) {
    .manimux-camera-panel, .manimux-camera-label,
    div:has(> .manimux-camera-anchor) + div,
    div:has(> .manimux-camera-anchor) + div + div,
    div:has(> .manimux-camera-anchor) + div + div + div { display: none; }
  }
</style>
<div class="manimux-camera-anchor"></div>
<section class="manimux-camera-panel"></section>
<span class="manimux-camera-label manimux-camera-label-top">top</span>
<span class="manimux-camera-label manimux-camera-label-left">left</span>
<span class="manimux-camera-label manimux-camera-label-right">right</span>
"""


class PolicyViewer:
    """Robot-independent dashboard backed by one selected robot adapter."""

    def __init__(
        self,
        host: str,
        port: int,
        bridge_endpoint: str,
        control_endpoint: str,
        robot: RobotAdapter,
    ) -> None:
        self.robot = robot
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
        self.chunk_timeline = ChunkTimelineView()
        self._last_chunk_timeline_render = 0.0
        self.observe_only = False
        self.current_episode_dir: Path | None = None
        self.episode_finalized = False
        self._build_scene()
        self._build_gui()
        self.receiver = ViewerReceiver(bridge_endpoint, self.on_message)
        self.control_server = ControlServer(control_endpoint, self.control_state)

    @staticmethod
    def _root(group: RobotGroup) -> str:
        return f"/robot/{group.name}"

    def _build_scene(self) -> None:
        self.server.scene.set_up_direction("+z")
        self.server.initial_camera.position = (1.45, -1.8, 1.25)
        self.server.initial_camera.look_at = (0.25, 0.0, 0.28)
        self.server.initial_camera.up = (0.0, 0.0, 1.0)
        self.server.initial_camera.fov = np.deg2rad(55.0)
        self.server.scene.add_grid(
            "/floor",
            width=2.0,
            height=1.6,
            cell_size=0.1,
            section_size=0.5,
            position=(0.25, 0.0, -0.01),
        )
        self.server.scene.add_frame("/world", axes_length=0.15, axes_radius=0.006)
        for box in self.robot.scene_boxes:
            self.server.scene.add_box(
                f"/environment/{box.name}",
                color=box.color,
                dimensions=box.dimensions,
                position=box.position,
            )
        for group in self.robot.groups:
            root = self._root(group)
            self.server.scene.add_frame(root, show_axes=False, position=group.base_position)
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
        self.camera_panel = self.server.gui.add_html(_camera_panel_html())
        placeholder = np.zeros((90, 160, 3), dtype=np.uint8)
        with self.server.gui.add_folder(None):
            self.camera_images = {
                slot: self.server.gui.add_image(
                    placeholder,
                    label=None,
                    format="jpeg",
                    jpeg_quality=70,
                )
                for slot in ("top", "left", "right")
            }
        self.chunk_timeline_panel = self.server.gui.add_html(
            self.chunk_timeline.render_html()
        )
        self.status = self.server.gui.add_markdown("🟠 **Waiting for policy executor**")
        self.instruction = self.server.gui.add_markdown(_instruction_markdown(""))
        self.new_rollout_folder = self.server.gui.add_folder(
            "① New rollout", expand_by_default=True
        )
        with self.new_rollout_folder:
            self.rollout_setup_status = self.server.gui.add_markdown(
                "⚪ Waiting for a ManiMux runtime service."
            )
            self.task = self.server.gui.add_text(
                "Task command", "", multiline=True, disabled=True
            )
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
            self.pause_btn = self.server.gui.add_button(
                "Pause / Hold", color="gray", disabled=True
            )
            self.finish_btn = self.server.gui.add_button(
                "Finish & Home", color="red", disabled=True
            )
        self.recovery_folder = self.server.gui.add_folder(
            "Advanced recovery", expand_by_default=False
        )
        with self.recovery_folder:
            self.home_btn = self.server.gui.add_button(
                "Return Home (keep rollout open)", color="gray", disabled=True
            )
        self.run_folder = self.server.gui.add_folder("Live run", expand_by_default=True)
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
            self.reviewer_id = self.server.gui.add_text(
                "Reviewer", "operator", disabled=True
            )
            self.operator_note = self.server.gui.add_text(
                "Operator note", "", multiline=True, disabled=True
            )
            self.failure_tag_inputs = {
                "replay_backtrack": self.server.gui.add_checkbox(
                    "Replay / backtrack", False, disabled=True
                ),
                "hold_stall": self.server.gui.add_checkbox(
                    "Hold / stall", False, disabled=True
                ),
                "collision": self.server.gui.add_checkbox(
                    "Collision", False, disabled=True
                ),
                "drop_spill": self.server.gui.add_checkbox(
                    "Drop / spill", False, disabled=True
                ),
                "perception": self.server.gui.add_checkbox(
                    "Perception error", False, disabled=True
                ),
                "policy_semantics": self.server.gui.add_checkbox(
                    "Policy / task error", False, disabled=True
                ),
                "safety_stop": self.server.gui.add_checkbox(
                    "Safety stop", False, disabled=True
                ),
                "other": self.server.gui.add_checkbox("Other", False, disabled=True),
            }
            self.save_evaluation_btn = self.server.gui.add_button(
                "Save evaluation", color="blue", disabled=True
            )
        self.overlay_folder = self.server.gui.add_folder(
            "Trajectory overlays", expand_by_default=False
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
        self._set_stage("waiting")

        @self.start_btn.on_click
        def _start(_event: Any) -> None:
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
            self.paused = True
            self._set_policy_controls_enabled(True)
            self.status.content = "🟡 **Connected · PAUSED**"

        @self.home_btn.on_click
        def _home(_event: Any) -> None:
            self.paused = True
            self.home_requested = True
            self.status.content = "🟡 **Returning home · PAUSED**"

        @self.finish_btn.on_click
        def _finish(_event: Any) -> None:
            self.finish_requested = True
            self._set_policy_controls_enabled(False)
            self.status.content = "🟠 **Finishing rollout and saving episode**"

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

    def _set_instruction(self, instruction: str) -> None:
        self.instruction.content = _instruction_markdown(instruction)
        self.task.value = _prefill_task(self.task.value, instruction)

    def _set_stage(self, stage: ViewerStage) -> None:
        """Expose only the controls that are actionable in the current stage."""

        self.new_rollout_folder.visible = stage in {"waiting", "setup", "preparing"}
        self.policy_control_folder.visible = stage == "control"
        self.recovery_folder.visible = stage == "control"
        self.evaluation_folder.visible = stage == "evaluation"
        self.overlay_folder.visible = stage == "control"
        self.run_folder.visible = stage not in {"waiting"}

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
        allowed = enabled and self.evaluation_saved
        self.prepare_normal_btn.disabled = not allowed
        self.prepare_experiment_btn.disabled = not allowed
        self.layout_id.disabled = not allowed
        self.task.disabled = not allowed

    def _prepare_rollout(self, *, experiment_mode: bool) -> None:
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
        self.home_btn.disabled = not allowed or not self.paused
        self.finish_btn.disabled = not allowed

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
        state = {
            "paused": self.paused,
            "home_requested": self.home_requested,
            "finish_requested": self.finish_requested,
            "new_rollout_requested": self.new_rollout_requested,
            "task_command": self.task.value.strip(),
            "experiment_mode": self.experiment_mode,
            "layout_id": self.layout_id.value.strip(),
        }
        self.home_requested = False
        self.finish_requested = False
        self.new_rollout_requested = False
        return state

    @staticmethod
    def _image(payload: str) -> UInt8Array:
        raw = base64.b64decode(payload)
        return np.array(Image.open(io.BytesIO(raw)).convert("RGB"), copy=True)

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
        # ``joint_actions`` keeps old local publishers readable during migration.
        raw_actions = message.get("actions", message.get("joint_actions"))
        if raw_actions is None:
            raise ValueError("plan message is missing actions")
        actions = np.asarray(raw_actions, dtype=np.float64)
        action_space = str(message.get("action_space", "joint_position"))
        grouped_actions = self.robot.split_actions(actions, action_space)
        metadata = dict(message.get("metadata") or {})
        metadata["gripper_closed_steps"] = self.robot.gripper_closed_steps(
            grouped_actions,
            previous_positions=getattr(self, "last_joint_positions", {}),
        ).tolist()
        message["metadata"] = metadata
        start_index = int(message.get("start_index", 0))
        if start_index < 0 or start_index > len(actions):
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
            f"#{message.get('chunk_id', 0)} · {len(actions)} actions · "
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
        if not self.episode_active and bool(metadata.get("episode_active", False)):
            self._update_event({"event": "episode_started", "metadata": metadata})
        joint_positions = np.asarray(message.get("joint_positions", []), dtype=np.float64)
        grouped_positions = self.robot.split_joint_positions(joint_positions)
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
        for source_name, payload in message.get("cameras_jpeg", {}).items():
            slot = self.robot.camera_slot(source_name)
            if slot in self.camera_images:
                self.camera_images[slot].image = self._image(str(payload))

    def _update_event(self, message: dict[str, Any]) -> None:
        event = str(message.get("event", "unknown"))
        metadata = message.get("metadata") or {}
        if event == "episode_started":
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
            self.episode_active = False
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
            if incoming_service_id:
                self.service_id = incoming_service_id
            self.launch_mode = "serve"
            self.policy_name.value = str(metadata.get("policy_label", "waiting"))
            self.runtime_name.value = str(metadata.get("runtime", "waiting"))
            self._set_instruction(str(metadata.get("task", self.task.value)))
            last_error = str(metadata.get("last_error", ""))
            if self.preparing_rollout and not last_error:
                return
            self.preparing_rollout = False
            self.service_ready = True
            self.prepare_normal_btn.visible = True
            self.prepare_experiment_btn.visible = True
            if first_service_announcement:
                self.layout_id.value = (
                    str(metadata.get("default_layout_id", "")) or "default"
                )
            self.rollout_setup_status.content = (
                "Choose a normal rollout, or an experiment rollout that requires a label."
            )
            if self.current_episode_dir is None or self.evaluation_saved:
                self.episode_path.value = str(metadata.get("last_episode_dir", "")) or "ready"
            self._update_prepare_enabled()
            self._set_stage("setup" if self.evaluation_saved else "evaluation")
            if last_error:
                self.status.content = (
                    f"🔴 **Last rollout failed · service idle**  \\n`{last_error}`"
                )
            elif self.evaluation_saved:
                self.status.content = "🟡 **Runtime service ready · prepare a rollout**"
        elif event == "episode_failed":
            self.episode_active = False
            self.preparing_rollout = False
            self.service_ready = True
            self.evaluation_saved = True
            self.prepare_normal_btn.visible = True
            self.prepare_experiment_btn.visible = True
            self._set_policy_controls_enabled(False)
            self._update_prepare_enabled()
            self._set_stage("setup")
            error = str(metadata.get("error", "unknown startup error"))
            self.status.content = f"🔴 **Rollout failed before execution**  \\n`{error}`"

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


def _demo(viewer: PolicyViewer) -> None:
    elapsed_s = 0.0
    chunk_id = 0
    next_plan_s = 0.0
    while viewer.running:
        joints, actions = viewer.robot.demo_sample(elapsed_s, horizon=25)
        if elapsed_s >= next_plan_s:
            viewer.on_message(
                PolicyPlan(
                    policy="Synthetic demo",
                    instruction="Inspect a predicted action chunk",
                    actions=actions,
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
                joint_positions=joints,
                cameras={"overview": (camera * 255).astype(np.uint8)},
                step=int(elapsed_s * 30),
                max_steps=1000,
                robot=viewer.robot.name,
            ).to_wire()
        )
        elapsed_s += 0.05
        time.sleep(0.05)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8086)
    parser.add_argument("--bridge-endpoint", default="tcp://127.0.0.1:5568")
    parser.add_argument("--control-endpoint", default="tcp://127.0.0.1:5569")
    parser.add_argument(
        "--robot",
        default="yam",
        help="built-in name, installed entry point, or module:factory",
    )
    parser.add_argument(
        "--robot-model-root",
        type=Path,
        help="optional model/dependency root forwarded to the robot adapter",
    )
    parser.add_argument(
        "--list-robots",
        action="store_true",
        help="list bundled/discovered adapters and exit",
    )
    parser.add_argument("--demo", action="store_true", help="show synthetic data without hardware")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.list_robots:
        print("\n".join(available_robot_adapters()))
        return
    robot = load_robot_adapter(args.robot, args.robot_model_root)
    viewer = PolicyViewer(
        args.host,
        args.port,
        args.bridge_endpoint,
        args.control_endpoint,
        robot,
    )
    if args.demo:
        threading.Thread(target=_demo, args=(viewer,), daemon=True).start()
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    print(f"Robot adapter: {robot.name} ({robot.label})")
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
