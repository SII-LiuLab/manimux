"""CollectSession: the one stateful live object behind the Collect tab.

Owns the ControlLoop, the EpisodeRecorder, and the CameraHub, and implements the
continuous-loop + sync-gate model:

  * "Go live" (first Start Teleop): apply the edited config, bring the CAN buses
    up (hardware), build the devices, start the loop, and enable sync.
  * Once live, the loop runs continuously so the teaching-handle buttons stay
    readable. The **top button** and the GUI toggle both flip ``sync_enabled``
    (follower mirrors or not); the **second button** and the GUI toggle both
    start/stop recording. State is reported via ``status()`` so the GUI mirrors
    the hardware buttons and vice versa.

Hardware isn't touched until go-live, so the operator can edit the Station rail
first. The mock path has no CAN/buttons — sync/record are GUI-driven only.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from ..camera.worker import CameraWorker
from ..config import (
    ControllerConfig,
    FORMAT_OPTIONS,
    StationConfig,
    controller_channel_for,
    gello_variant,
    is_passive_gello,
    leader_joint_signs_for,
    robot_channel_for,
)
from ..data.recorder import EpisodeRecorder, task_slug
from ..data.schema import WRITE_COMPLETE_FLAG
from ..robot.can_bus import check_can_up
from ..runtime import build_arm_units, build_cameras_from_config
from ..teleop.loop import ControlLoop
from .camera_hub import CameraHub


_UNSET_TASK_NAMES = {"", "unknown", "unknow"}


def valid_task_name(name: str | None) -> str | None:
    """Return a cleaned operator task, or ``None`` for unset placeholders."""
    cleaned = (name or "").strip()
    return None if cleaned.lower() in _UNSET_TASK_NAMES else cleaned


class CollectSession:
    def __init__(self, cfg: StationConfig, mock: bool = False):
        self.mock = mock
        self.cfg = cfg
        # Restore the last task name so a GUI restart pre-fills the Task field with what
        # you were collecting, instead of the station yaml's default.
        last = self._load_last_task()
        if last:
            self.cfg.task_name = last
        self.hub = CameraHub()
        self.live = False
        # Live for autonomy: followers only, no leaders and no teleop loop. Set by a
        # deploy go-live and by _release_leaders(); teleop can't resume until a rebuild.
        self.followers_only = False
        self.powered_off = False
        # E-STOP latch at session level. The teleop ControlLoop has its own, but a
        # followers-only session has no loop to hold one, so the GUI would show
        # "idle" after an E-STOP. Cleared by go_live() and reset_session().
        self._estopped = False
        self.last_can_msg = ""
        # Serializes sync transitions so a hardware button and a GUI click can't
        # both run engage() at once.
        self._sync_lock = threading.Lock()
        # Cameras can be connected (preview) without teleop; robots/loop are built
        # at go-live. Nothing here touches hardware.
        self.units: list = []
        self.loop = self.recorder = None
        self.deploy_loop = None  # autonomous policy rollout (Deploy tab)
        self.deploy_recorder = None  # logs the rollout to a separate save path
        # Used by both the GUI Record button and the physical second button.
        self.record_eepose = True
        self._record_save_thread: threading.Thread | None = None
        self.record_path: str | None = None
        self.last_episode_path: str | None = None
        self.last_record_warning: str | None = None
        # A hardware record button cannot carry a browser form payload.  When it is
        # pressed without a real task name, publish a one-shot event through status()
        # so the GUI can ask the operator instead of silently recording as "unknown".
        self.record_task_required = False
        self.record_task_request_id = 0
        self.workers: list = []
        self.cameras_connected = False
        self._cam_cfg: dict = {}
        self._cam_sig: list = []

    # --- last-task persistence (survive a GUI restart) --------------------
    def _last_task_path(self) -> Path:
        return Path(self.cfg.save_root).parent / ".last_task"

    def _load_last_task(self) -> str | None:
        try:
            return self._last_task_path().read_text().strip() or None
        except OSError:
            return None

    def _save_last_task(self, name: str) -> None:
        name = (name or "").strip()
        if not name:
            return
        try:
            p = self._last_task_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(name)
        except OSError:
            pass

    # --- cameras (independent of teleop; feed previews) -------------------
    @staticmethod
    def _camera_sig(cfg: StationConfig) -> list:
        return [(c.name, c.type, c.role, c.serial, c.mode) for c in cfg.cameras]

    def connect_cameras(self, cfg: StationConfig) -> None:
        """Open the cameras and start their capture threads (feeding previews),
        independent of teleop/CAN. Re-opens only if the camera config changed, so
        a later go-live reuses already-previewing cameras. A device opens once."""
        if self.cameras_connected and self._camera_sig(cfg) == self._cam_sig:
            self.cfg = cfg
            return
        if self.live:
            # Reopening would stop the workers the running loop reads from (below).
            raise RuntimeError("stop teleop or the rollout before changing the camera set")
        if self.loop is not None:
            self.loop.stop()  # the loop shares these workers; stop before swapping
        self._disconnect_cameras()
        self.cfg = cfg
        drivers = build_cameras_from_config(cfg, mock=self.mock)
        self.workers = [CameraWorker(d, on_frame=self.hub.update) for d in drivers]
        try:
            for worker in self.workers:
                worker.start()
        except Exception:
            self._disconnect_cameras()
            raise
        self._cam_cfg = {c.name: c for c in cfg.cameras}
        self._cam_sig = self._camera_sig(cfg)
        self.cameras_connected = True

    def _disconnect_cameras(self) -> None:
        for w in self.workers:
            w.stop()
        self.workers = []
        self.cameras_connected = False

    # --- device stack -----------------------------------------------------
    @staticmethod
    def _needed_channels(cfg: StationConfig, followers_only: bool) -> list[str]:
        """The CAN buses go-live is about to open, mirroring ``build_arm_units``: one
        per follower, plus the controller bound to each follower unless followers-only.
        Iterating ``cfg.robot.controllers`` instead would also list a controller that
        isn't bound to any robot and so never gets opened."""
        robots = cfg.robot.robots or [cfg.robot.active_robot()]
        needed = [robot_channel_for(r) for r in robots]
        if not followers_only:
            needed += [controller_channel_for(cfg.robot.controller_for(r)) for r in robots]
        return needed

    def _build_teleop_stack(self, cfg: StationConfig) -> None:
        """Build the teleop loop and recorder over the already-built units and the
        already-connected cameras, and wire the button callbacks."""
        # Pass the running workers so the loop shares them (doesn't reopen devices).
        self.loop = ControlLoop(self.units, self.workers, cfg.control_hz)
        self.recorder = EpisodeRecorder(
            cfg.save_root, cfg, self.workers, arm_names=[u.name for u in self.units],
            backend=self.units[0].robot.backend
        )
        self.loop.attach_recorder(self.recorder)
        self.loop.on_sync_button = self.toggle_sync
        self.loop.on_record_button = self.toggle_record

    def go_live(self, cfg: StationConfig, followers_only: bool = False) -> None:
        """Apply ``cfg``, bring CAN up (hardware), build the robots over the connected
        cameras, and (unless ``followers_only``) start the teleop loop with sync on.

        ``followers_only`` is the autonomy bring-up: it opens one bus per follower
        instead of two, and deliberately builds no teleop loop -- ``enable_sync``'s
        ``engage_all()`` would ramp every follower to wherever its leader is sitting,
        which is a real arm motion nobody asked for right before a policy takes over.
        """
        self._teardown_units()
        if not self.mock:
            # Pre-flight: every bus we're about to open must be up before i2rt opens
            # it, else fail with a clear message instead of deep inside the driver.
            needed = self._needed_channels(cfg, followers_only)
            down = check_can_up(needed)
            if down:
                raise RuntimeError(
                    f"CAN interface(s) not up: {', '.join(down)}. "
                    f"Run scripts/setup_can_sudoers.sh once, check the arm is powered, "
                    f"and that udev names match (expected {needed})."
                )
        self.connect_cameras(cfg)  # reuse preview cameras if already up
        try:
            self.cfg = cfg
            self.units = build_arm_units(cfg, mock=self.mock, followers_only=followers_only)
            self.followers_only = followers_only
            if not followers_only:
                self._build_teleop_stack(cfg)
            self.live = True
            self.powered_off = False
            self._estopped = False  # re-armed: Start Teleop is a documented E-STOP recovery
            if not followers_only:
                self.loop.start()
                self.enable_sync()
        except Exception:
            # Release the half-built units on failure so a retry can rebuild. Cameras stay.
            self._teardown_units()
            self.live = False
            raise

    def _release_leaders(self) -> None:
        """Close the leader CAN buses while the followers stay energized, so an
        autonomous rollout runs on the follower buses only.

        The agents are kept, not dropped: E-STOP and Power Off Arms reach the hardware
        through ``self.units``, and a motorized lead arm keeps its bus (releasing it
        would drop gravity comp and the arm would sag). ``release_leader`` poisons the
        agent either way, so nothing can command a follower from a frozen leader reading.
        """
        for u in self.units:
            release = getattr(u.agent, "release_leader", None)
            if not callable(release):
                continue
            try:
                if not release():
                    logging.warning(
                        "%s leader is a motorized lead arm and keeps its CAN bus — it "
                        "needs gravity compensation to stay up. Only the passive-GELLO "
                        "leaders are released for autonomy.", u.name,
                    )
            except Exception:  # noqa: BLE001 — a stuck leader must not block the rollout
                logging.exception("failed to release the %s leader", u.name)
        self.followers_only = True

    def _teardown_units(self) -> None:
        if self.loop is not None:
            self.loop.stop()
            self.loop = None
        if self.recorder is not None and self.recorder.is_recording:
            self.recorder.abort()
        self.recorder = None
        for unit in self.units:
            release = getattr(unit.agent, "release_leader", None)
            if release is not None:
                release()
        if self.units:
            self.units[0].robot.backend.close()
        self.units = []
        self.followers_only = False

    def start_teleop(self, cfg: StationConfig | None = None) -> None:
        """Go live on the first call (applying ``cfg``); afterwards just re-enable
        sync. The physical top button toggles sync once live."""
        if self.live and self.followers_only:
            # No leaders and no teleop loop in this session — re-enabling sync would
            # silently do nothing. The leaders come back only on a rebuild.
            raise RuntimeError(
                "session is live for autonomy (follower buses only) — Stop the rollout "
                "and click Reset Session before starting teleop"
            )
        if not self.live:
            self.go_live(cfg or self.cfg)
        else:
            self.enable_sync()

    def stop_teleop(self) -> None:
        # Disable sync but keep the loop running so the buttons stay live.
        self.disable_sync()

    def enable_sync(self) -> None:
        with self._sync_lock:
            # Never (re)engage on an estopped loop — it's a latch; recovery is a
            # full rebuild via go_live(), not enable_sync().
            if self.loop is None or self.loop.sync_enabled or self.loop.estopped:
                return
            # Ease every follower to its leader before mirroring so it never snaps
            # (i2rt slow_move pattern); no-op on the mock.
            self.loop.engage_all()
            # If E-STOP fired during the ramp, engage_all() aborted — don't (re)enable
            # sync, or the loop would start mirroring right after a stop.
            if not self.loop.estopped:
                self.loop.sync_enabled = True

    def disable_sync(self) -> None:
        with self._sync_lock:
            if self.loop is not None:
                with self.loop._io_lock:
                    self.loop.sync_enabled = False
                    if self.units:
                        self.units[0].robot.backend.pause()

    def toggle_sync(self) -> None:
        if self.loop is not None and self.loop.sync_enabled:
            self.disable_sync()
        else:
            self.enable_sync()

    def zero_gello(self, side: str) -> dict:
        """Hardware-zero a passive-GELLO leader's joint encoders and gripper at its current
        pose (writes each encoder's EEPROM via i2rt ``reset_zero_position``). Hold the leader
        at the pose that should map to the follower's home, with the trigger released.

        Safe to run while live: sync is disabled first so the instantaneous
        reported-angle change can't jerk the follower, and the zero (being in
        hardware EEPROM) takes effect immediately for the running loop — no rebuild.
        Returns ``{ok, message, ...}`` so the GUI can show the outcome.
        """
        if self.mock:
            return {"ok": False, "message": "mock mode: no hardware to zero"}
        if side not in ("left", "right"):
            return {"ok": False, "message": f"unknown side {side!r} (expected left/right)"}

        sync_was_on = bool(self.loop and self.loop.sync_enabled)
        if sync_was_on:
            self.disable_sync()

        from ..robot.passive_gello import PassiveGelloLeader

        cfg = self.cfg
        # Use the leader this station has on that side, so its own bus and signs apply: a
        # mobile GELLO or an assigned channel isn't reachable via the passive_gello_* defaults.
        controller = next(
            (c for c in cfg.robot.controllers if is_passive_gello(c.type) and c.type.endswith(side)),
            ControllerConfig(type=f"passive_gello_{side}"),
        )
        channel = controller_channel_for(controller)
        n = cfg.robot.num_arm_joints
        grip = cfg.robot.leader_gripper
        gello_type, gello_side = gello_variant(controller.type)
        try:
            lead = PassiveGelloLeader(
                channel=channel,
                num_arm_joints=n,
                gripper_config=tuple(grip) if grip else (n, 0.7, 0.0),
                joint_signs=leader_joint_signs_for(controller, cfg.robot.leader_joint_signs),
                gello_type=gello_type,
                side=gello_side,
            )
        except Exception as e:  # noqa: BLE001 — surface a readable reason to the UI
            return {
                "ok": False,
                "message": f"{type(e).__name__}: {e}",
                "sync_disabled": sync_was_on,
            }
        try:
            devices = lead.hardware_zero()
        finally:
            lead.stop()
        return {
            "ok": True,
            "message": (
                f"zeroed {channel} encoders {devices} (arm joints + gripper) at current pose"
            ),
            "devices": devices,
            "sync_disabled": sync_was_on,
        }

    # --- recording --------------------------------------------------------
    def start_recording(
        self,
        task_name: str,
        include_eepose: bool | None = None,
        save_root: str | None = None,
        data_format: str | None = None,
    ) -> str:
        if self.recorder is None:
            raise RuntimeError("not live — Start Teleop before recording")
        task_name = valid_task_name(task_name)
        if task_name is None:
            raise ValueError("set a Task name before recording")
        self.cfg.task_name = task_name
        self.record_task_required = False
        if include_eepose is not None:
            self.record_eepose = bool(include_eepose)
        if save_root is not None:
            save_root = save_root.strip()
            if not save_root:
                raise ValueError("save_root cannot be empty")
            self.cfg.save_root = save_root
        if data_format is not None:
            if data_format not in FORMAT_OPTIONS:
                raise ValueError(f"unsupported data format {data_format!r}")
            self.cfg.data_format = data_format
        self._save_last_task(task_name)  # remember for the next GUI restart
        path = str(
            self.recorder.start(
                task_name,
                include_eepose=self.record_eepose,
                save_root=self.cfg.save_root,
                writer_name=self.cfg.data_format,
            ).resolve()
        )
        self.record_path = path
        self.last_record_warning = None
        return path

    def set_recording_task_name(self, task_name: str | None) -> None:
        """Synchronize the browser Task field into the hardware-button path."""
        self.cfg.task_name = (task_name or "").strip()
        if valid_task_name(self.cfg.task_name) is not None:
            self.record_task_required = False

    def stop_recording(self) -> dict:
        if self.recorder is None:
            raise RuntimeError("not recording — no active session")
        path = self.recorder.stop()
        if path is None:  # empty episode was discarded (e.g. phantom button tap)
            self.record_path = None
            return {"path": None, "frames": 0}
        saved_path = str(path.resolve())
        self.last_episode_path = saved_path
        self.record_path = None
        self.last_record_warning = self.recorder.last_stop_warning
        return {
            "path": saved_path,
            "frames": self.recorder.frame_count,
            "eepose": self.recorder.include_eepose,
            "warning": self.last_record_warning,
        }

    def toggle_record(self) -> None:
        """Second-button / GUI toggle. Uses the rail's Task name; refuses to start a
        nameless recording (an empty task poisons the dataset + training prompt)."""
        if self.recorder is None:
            return
        if self.recorder.is_recording:
            # This callback runs inside ControlLoop._step(). Video encoding and FK
            # must not stall the live robot loop, so only the hardware-button path
            # saves in a worker. The HTTP stop route remains synchronous and returns
            # the final path/warning to the GUI.
            if self._record_save_thread is None or not self._record_save_thread.is_alive():
                self._record_save_thread = threading.Thread(
                    target=self._stop_recording_from_button,
                    name="episode-save",
                    daemon=True,
                )
                self._record_save_thread.start()
        elif self.recorder.is_saving:
            print("[yam-abc] recording ignored — previous episode is still saving", flush=True)
        elif (task_name := valid_task_name(self.cfg.task_name)) is not None:
            self.start_recording(task_name)
        else:
            self.record_task_required = True
            self.record_task_request_id += 1
            print("[yam-abc] recording ignored — set a Task name in the Station rail first", flush=True)

    def _stop_recording_from_button(self) -> None:
        try:
            result = self.stop_recording()
            print(f"[yam-abc] episode saved: {result.get('path')}", flush=True)
            if result.get("warning"):
                print(f"[yam-abc] warning: {result['warning']}", flush=True)
        except Exception:  # noqa: BLE001 — surface save failure without killing teleop
            logging.exception("failed to save episode started/stopped by teaching-handle button")

    def wait_for_episode_save(self, timeout: float | None = None) -> bool:
        """Wait for a hardware-button save before shutdown/reset tears down the app."""
        thread = self._record_save_thread
        if thread is None or thread is threading.current_thread():
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()

    # --- safety -----------------------------------------------------------
    # --- autonomous deployment -------------------------------------------
    def start_deploy(self, *args, **kwargs) -> None:
        raise RuntimeError("Use manimux serve and Viewer for policy deployment")

    def stop_deploy(self) -> dict:
        """Stop the rollout and, if recording, save the episode. Returns the saved
        path + frame count (empty if not recording)."""
        if self.deploy_loop is not None:
            self.deploy_loop.stop()
            self.deploy_loop = None
        out: dict = {}
        if self.deploy_recorder is not None:
            if self.deploy_recorder.is_recording:
                path = self.deploy_recorder.stop()
                out = {"path": str(path), "frames": self.deploy_recorder.frame_count}
            self.deploy_recorder = None
        return out

    def set_deploy_prompt(self, prompt: str) -> None:
        if self.deploy_loop is not None:
            self.deploy_loop.set_prompt(prompt)

    def estop(self) -> None:
        self._estopped = True
        if self.deploy_loop is not None:
            self.deploy_loop.estop()
            self.deploy_loop = None
        if self.deploy_recorder is not None:
            self.deploy_recorder.abort()  # discard the partial rollout episode
            self.deploy_recorder = None
        if self.loop is not None:
            self.loop.estop()
        if self.recorder is not None:
            # Discard any half-recorded episode so E-STOP never leaves a partial,
            # uncompleted folder behind.
            self.recorder.abort()
        # Hard-stop every arm directly as well. A followers-only session that has already
        # stopped its rollout has no loop and no deploy_loop left to route this through,
        # and the arms are still energized holding the last commanded pose.
        for u in self.units:
            try:
                u.robot.stop()
            except Exception:  # noqa: BLE001 — one bad arm must not skip the others
                logging.exception("E-STOP: failed to stop the %s arm", u.name)
        # Leave the live state so E-STOP is recoverable. Setting live=False makes start_teleop() re-arm.
        self.live = False
        self.followers_only = False

    def power_off_arms(self) -> dict:
        raise RuntimeError("Motor power-off is not exposed by the ManiMux collection driver")

    def reset_session(self) -> dict:
        """Tear down loops/recorders/units, reset the CAN buses, and clear live/estop so
        the next Start Teleop rebuilds from scratch. Cameras are kept. Used to recover from
        E-STOP or a failed go-live without restarting the GUI."""
        for obj, meth in ((self.deploy_loop, "stop"), (self.deploy_recorder, "abort"),
                          (self.recorder, "abort")):
            if obj is not None:
                try:
                    getattr(obj, meth)()
                except Exception:  # noqa: BLE001
                    pass
        self.deploy_loop = self.deploy_recorder = None
        # Stop the teleop loop + release the robot units (drop i2rt handles) before the reset.
        self._teardown_units()
        can_msg = "CAN configuration left unchanged"
        self.live = False
        self._estopped = False
        return {"ok": True, "can": can_msg}

    # --- queries ----------------------------------------------------------
    def list_episodes(self, task_name: str | None = None) -> list[str]:
        """Completed episodes under the CURRENT task's folder
        (``save_root/<task_slug>/<ep>`` with a ``write_complete.flag``).

        Counts the recorder's task when one is active/known — the GUI form task
        only reaches ``self.cfg`` on Start Teleop, so counting cfg alone shows 0
        for episodes just recorded under a different (or post-restart) name."""
        task = task_name or getattr(self.recorder, "task_name", None) or self.cfg.task_name
        root = self.recorder.save_root if self.recorder is not None else Path(self.cfg.save_root)
        task_dir = root / task_slug(task)
        if not task_dir.exists():
            return []
        return sorted(str(p) for p in task_dir.iterdir() if (p / WRITE_COMPLETE_FLAG).exists())

    def camera_descriptors(self) -> list[dict]:
        """One row per video stream (a stereo cam yields two), enriched with the
        configured device specs so the GUI can render the config rail and build
        per-eye preview URLs without a second request."""
        out = []
        for cam in self.workers:
            c = self._cam_cfg.get(cam.name)
            for k in cam.image_keys():
                eye = None if k == "rgb" else k
                out.append(
                    {
                        "name": cam.name,
                        "role": cam.role,
                        "mode": cam.mode.value,
                        "eye": eye,
                        "type": (c.type if c else "mock"),
                        "width": (c.width if c else 0),
                        "height": (c.height if c else 0),
                        "fps": (c.fps if c else max(1, int(round(self.cfg.control_hz)))),
                    }
                )
        return out

    def status(self) -> dict:
        loop, rec, dep = self.loop, self.recorder, self.deploy_loop
        return {
            "live": self.live,
            # Live for autonomy: no leaders, no teleop loop. The GUI greys out the
            # teleop/record controls, which would otherwise 409 on click.
            "followers_only": self.followers_only,
            # "teleop_running" now means sync is enabled (follower mirroring).
            "teleop_running": bool(loop and loop.sync_enabled),
            "recording": bool(rec and rec.is_recording),
            "saving_episode": bool(rec and rec.is_saving),
            "record_eepose": self.record_eepose,
            "record_save_root": str(rec.save_root) if rec else self.cfg.save_root,
            "record_path": self.record_path,
            "last_episode_path": self.last_episode_path,
            "last_record_warning": self.last_record_warning,
            "record_task_required": self.record_task_required,
            "record_task_request_id": self.record_task_request_id,
            "estopped": self._estopped or bool(loop and loop.estopped),
            "powered_off": self.powered_off,
            "current_task": (rec.task_name if rec else None),
            "episodes_done": len(self.list_episodes()),
            # Live teaching-handle inputs for the GUI indicators.
            "buttons": (list(loop.buttons) if loop else [False, False]),
            "trigger": (round(loop.trigger, 3) if loop else 0.0),
            "cameras": self.camera_descriptors(),
            # Per-unit follower joint vector + leader command, for calibration/debug.
            # A rollout reports its own (same shape, no leader rows) — keyed on the loop
            # existing, not on `deploying`, which is still False while homing/ramping.
            "arms": (dep.joint_snapshot() if dep else (loop.joint_snapshot() if loop else {})),
            # Unit roster + most recent teleop-loop error, for the status badge.
            "units": [u.name for u in self.units],
            "last_error": (loop.last_error if loop else None),
            # Autonomous-deploy state for the Deploy tab.
            "deploying": bool(self.deploy_loop and self.deploy_loop.running),
            "deploy_hz": (round(self.deploy_loop.actual_hz, 1) if self.deploy_loop else 0.0),
            "deploy_error": (self.deploy_loop.last_error if self.deploy_loop else None),
            "deploy_recording": bool(self.deploy_recorder and self.deploy_recorder.is_recording),
            "deploy_frames": (self.deploy_recorder.frame_count if self.deploy_recorder else 0),
        }
