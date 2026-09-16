"""ControlLoop: the fixed-rate read -> act -> command -> record cycle.

Hardware-agnostic. Runs either synchronously (``run_for``, for the CLI) or in a
background thread (``start`` / ``stop``, for the GUI). An ``on_frame`` callback
feeds the latest camera frames to the GUI's CameraHub without coupling the loop
to it. ``estop()`` immediately stops commanding and hard-stops the robot.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from contextlib import ExitStack

from ..camera.interface import CameraDriver, CameraFrame
from ..camera.worker import CameraWorker
from ..config import StationConfig
from ..data.control_timing import ControlTimings, arm_scope, stage
from ..runtime import ArmUnit


class ControlLoop:
    def __init__(
        self,
        units: list[ArmUnit],
        cameras: list[CameraDriver],
        control_hz: float = 30.0,
        on_frame: Callable[[str, CameraFrame], None] | None = None,
    ):
        # One or more driven arms; global sync gates all of them together.
        # Every unit needs its leader: a followers-only (autonomy) unit has no agent,
        # and estop() would then raise partway through the units and leave the rest
        # of the arms uncommanded. Refuse at construction instead.
        leaderless = [u.name for u in units if u.agent is None]
        if leaderless:
            raise ValueError(
                f"teleop needs a leader per arm, but {leaderless} were built without one "
                f"(followers-only build). Reset Session and Start Teleop to rebuild them."
            )
        self.units = units
        self._feedback_backends = list(dict.fromkeys(
            backend for u in units
            if (backend := getattr(u.robot, "backend", None)) is not None
            and callable(getattr(backend, "feedback_cycle", None))
        ))
        self.cameras = cameras
        # Cameras may arrive as raw drivers (the loop owns capture — CLI path) or
        # as already-running CameraWorkers owned elsewhere (the GUI session, which
        # starts them for preview before teleop). A device can only be opened once,
        # so when workers are passed the loop shares them and does not start/stop.
        self._own_cameras = not (cameras and all(isinstance(c, CameraWorker) for c in cameras))
        self._workers = [
            c if isinstance(c, CameraWorker) else CameraWorker(c, on_frame=on_frame)
            for c in cameras
        ]
        self.dt = 1.0 / control_hz if control_hz > 0 else 0.0
        self._recorder = None
        self._running = False
        self._estopped = False
        self.last_error: str | None = None  # last crash reason, surfaced to the GUI
        self._estop_event = threading.Event()
        # Serializes all i2rt CAN access (the loop's own reads/commands and the
        # engage() ramp) so a GUI-thread engage can't overlap _step() on another
        # thread. See engage_all()/_step().
        self._io_lock = threading.Lock()
        # Frequency changes must not split read_inputs/act/record across periods.
        # RLock permits teaching-handle callbacks inside the current step.
        self.cycle_lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self.actual_hz = 0.0
        # Raw (un-EMA'd) last period + a missed-deadline counter, for diagnosing
        # rate problems that the smoothed actual_hz hides.
        self.last_period = 0.0
        self.overruns = 0
        self.timings = ControlTimings()
        # Continuous-loop model: the loop always reads the controller so its
        # buttons are live; ``sync_enabled`` gates whether the follower mirrors it
        # (the top button / GUI "Start Teleop" toggle this). See _step.
        self.sync_enabled = False
        # Latest controller inputs, for the status feed + GUI indicators. OR'd across units
        # (any controller drives sync/record); ``_last_inputs`` keeps each leader's own
        # reading for the per-arm feed.
        self.buttons: list[bool] = [False, False]
        # Latest per-unit follower observation + leader-derived command, cached each
        # tick so the GUI/debug can read all arms without touching the CAN buses.
        self._last_obs: dict = {}
        self._last_action: dict = {}
        self._last_controller_input: dict = {}
        self._last_inputs: dict[str, tuple[list[bool], float]] = {}
        # Per-unit (raw, cal) leader joint angles in rad, for the live readout. Only a passive
        # GELLO has raw encoders to report; None for other leaders.
        self._last_leader_angles: dict[str, tuple] = {}
        self.trigger: float = 0.0
        # Rising-edge callbacks for the two teaching-handle buttons.
        self.on_sync_button: Callable[[], None] | None = None
        self.on_record_button: Callable[[], None] | None = None
        self._btn_last: list[bool] = [False, False]

    # --- recording hook ----------------------------------------------------
    def attach_recorder(self, recorder) -> None:
        self._recorder = recorder
        setter = getattr(recorder, "set_camera_workers", None)
        if setter is not None:
            setter(self._workers)

    @property
    def estopped(self) -> bool:
        return self._estopped

    # --- button edge detection --------------------------------------------
    def _handle_button_edges(self, buttons: list[bool]) -> None:
        """Fire the sync/record callbacks on a rising edge (False->True) only, so a
        held button never retriggers. Callbacks run in the loop thread."""
        for i in range(min(2, len(buttons))):
            pressed = bool(buttons[i])
            if pressed and not self._btn_last[i]:
                cb = self.on_sync_button if i == 0 else self.on_record_button
                if cb is not None:
                    cb()
            self._btn_last[i] = pressed

    # --- camera worker lifecycle ------------------------------------------
    def _start_cameras(self) -> None:
        if self._own_cameras:
            for w in self._workers:
                w.start()

    def _stop_cameras(self) -> None:
        if self._own_cameras:
            for w in self._workers:
                w.stop()

    def engage_all(self) -> None:
        """Ease every follower to its leader before mirroring (mock no-op). Passes
        the E-STOP event so a stop aborts the ramp; bails between arms too. Holds
        _io_lock so the ramp can't overlap _step()'s CAN access on another thread."""
        with self.timings.cycle(self.dt, kind="alignment"), self._io_lock, ExitStack() as scopes:
            for backend in self._feedback_backends:
                scopes.enter_context(backend.feedback_cycle(reuse=False))
            for u in self.units:
                if self._estop_event.is_set():
                    return
                u.agent.engage(self._estop_event)

    # --- one iteration -----------------------------------------------------
    def _step(self) -> None:
        with self.timings.cycle(self.dt, sync_enabled=self.sync_enabled):
            with stage("cycle_lock_wait"):
                self.cycle_lock.acquire()
            try:
                with ExitStack() as scopes:
                    for backend in self._feedback_backends:
                        scopes.enter_context(backend.feedback_cycle())
                    self._step_once()
            finally:
                self.cycle_lock.release()

    def _step_once(self) -> None:
        with stage("camera_cache"):
            frames: dict[str, CameraFrame] = {}
            for w in self._workers:
                fr = w.read()  # latest cached frame; non-blocking (worker feeds preview)
                if fr is None:
                    continue
                frames[w.name] = fr
        # Read every controller so buttons stay live even when sync is off; global
        # sync/record fire on a rising edge of ANY controller's top/second button.
        obs: dict[str, dict] = {}
        top = second = False
        # Serialize CAN reads under _io_lock (engage_all() holds it during its ramp).
        # Released before the button callback, which may itself run engage_all(), so
        # there is no re-entrant deadlock.
        with stage("read_lock_wait"):
            self._io_lock.acquire()
        try:
            for u in self.units:
                with stage(f"{u.name}.follower_observation"):
                    obs[u.name] = u.robot.get_observations()
                with arm_scope(u.name), stage("leader_inputs"):
                    inp = u.agent.read_inputs()
                if inp is not None:
                    btns, trig = inp
                    top = top or (len(btns) > 0 and bool(btns[0]))
                    second = second or (len(btns) > 1 and bool(btns[1]))
                    self.trigger = trig
                    self._last_inputs[u.name] = ([bool(b) for b in btns[:2]], float(trig))
                raw = getattr(u.agent, "leader_raw", None)
                if callable(raw):
                    with stage(f"{u.name}.leader_raw"):
                        self._last_leader_angles[u.name] = raw()
        finally:
            self._io_lock.release()
        self.buttons = [top, second]
        with stage("button_callbacks"):
            self._handle_button_edges(self.buttons)
        self._last_obs = obs

        # A handle callback can pause or run an entire alignment ramp. Its old
        # observations must not reach bilateral feedback, commands or recording.
        if any(not backend.feedback_cycle_valid() for backend in self._feedback_backends):
            self._last_action = {}
            return

        if self._estopped or not self.sync_enabled:
            # Drop the cached action; a stale vector reads as a live command in the per-arm feed.
            self._last_action = {}
            return
        with stage("command_lock_wait"):
            self._io_lock.acquire()
        try:
            if self._estopped or not self.sync_enabled:
                return
            tick_timestamp_ns, tick_monotonic_ns = time.time_ns(), time.monotonic_ns()
            with stage("target_batch"), self.units[0].robot.backend.target_batch():
                actions = {}
                for u in self.units:
                    with arm_scope(u.name), stage("act"):
                        actions[u.name] = u.agent.act(obs[u.name])
            with stage("controller_snapshot"):
                controller_inputs = {
                    u.name: getter()
                    for u in self.units
                    if callable(getter := getattr(u.robot, "get_controller_input", None))
                }
        finally:
            self._io_lock.release()
        self._last_action = actions
        self._last_controller_input = {
            name: sample for name, sample in controller_inputs.items() if sample is not None
        }
        rec = self._recorder
        if rec is not None and rec.is_recording:
            rec.tick(actions, obs, frames, controller_inputs=controller_inputs,
                     tick_timestamp_ns=tick_timestamp_ns, tick_monotonic_ns=tick_monotonic_ns)

    def joint_snapshot(self) -> dict:
        """Per-unit view of the last tick's cache — no CAN reads. ``follower`` and
        ``leader_cmd`` are ``[arm..., gripper]``; ``leader_cmd`` is None while the loop
        isn't commanding (sync off / e-stop). ``leader_buttons``/``leader_grip`` are per
        leader, unlike the merged ``buttons``/``trigger`` that drive sync/record.
        ``leader_raw``/``leader_cal`` are encoder angles in rad before and after the
        leader's sign table (None for a leader with no raw readout); their gripper element
        is an angle, not the normalized ``leader_grip``."""
        def _vec(a):
            return None if a is None else [round(float(x), 4) for x in list(a)]

        out: dict = {}
        for u in self.units:
            o = self._last_obs.get(u.name) or {}
            jp, gp = o.get("joint_pos"), o.get("gripper_pos")
            foll = None
            if jp is not None:
                foll = _vec(jp) + (_vec(gp) or [])
            btns, grip = self._last_inputs.get(u.name, (None, None))
            raw, cal = self._last_leader_angles.get(u.name) or (None, None)
            out[u.name] = {
                "follower": foll,
                "leader_cmd": _vec(self._last_action.get(u.name)),
                "controller_cmd": _vec(
                    (self._last_controller_input.get(u.name) or {}).get("joint_pos")
                ),
                "leader_buttons": (list(btns) if btns is not None else None),
                "leader_grip": (round(grip, 4) if grip is not None else None),
                "leader_raw": _vec(raw),
                "leader_cal": _vec(cal),
            }
        return out

    def _sleep_to_rate(self, t0: float) -> None:
        # perf_counter (monotonic) so an NTP/clock step can't inject a missed or
        # doubled deadline.
        elapsed = time.perf_counter() - t0
        requested_sleep = 0.0
        actual_sleep_ns = 0
        if self.dt > 0 and elapsed >= self.dt:
            self.overruns += 1  # step blew the period budget — no sleep this tick
            logging.debug(
                "control loop overrun: step %.1f ms > period %.1f ms",
                elapsed * 1e3, self.dt * 1e3,
            )
        elif self.dt > 0:
            requested_sleep = self.dt - elapsed
            sleep_start = time.monotonic_ns()
            time.sleep(requested_sleep)
            actual_sleep_ns = time.monotonic_ns() - sleep_start
        total = time.perf_counter() - t0
        self.last_period = total  # raw, un-smoothed
        if total > 0:
            # EMA-smooth so the GUI readout doesn't jitter on single-frame spikes.
            inst = 1.0 / total
            self.actual_hz = inst if self.actual_hz == 0.0 else 0.8 * self.actual_hz + 0.2 * inst
        self.timings.record_pacing(
            work_s=elapsed, requested_s=requested_sleep, actual_ns=actual_sleep_ns,
            total_s=total, actual_hz=self.actual_hz,
        )

    # --- threaded mode (GUI) ----------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        self._start_cameras()  # warms up each camera; raises if one never produces
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while self._running:
            t0 = time.perf_counter()
            try:
                self._step()
            except Exception as e:  # noqa: BLE001
                # Record the failure with a hint for the common motor-comm case.
                msg = f"{type(e).__name__}: {e}"
                low = str(e).lower()
                if any(k in low for k in ("communicate", "motor", "can", "timeout", "bus")):
                    msg += (" — a motor stopped responding. Most common causes: power "
                            "brown-out / undervoltage (DM error 0x9), a loose CAN cable, or "
                            "the arm powered off. Check power + CAN, then Reset Session.")
                self.last_error = msg
                logging.exception("control loop step failed; e-stopping")
                try:
                    self.estop()
                finally:
                    self.timings.flush("loop_error")
                break
            self._sleep_to_rate(t0)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                raise RuntimeError("collection loop did not stop; ownership retained")
            self._thread = None
        self._stop_cameras()
        self.timings.flush("loop_stop")
        self.timings.wait_for_saves()

    def estop(self) -> None:
        """Safety stop: stop commanding immediately and hard-stop every arm. Sets
        the abort event first so an in-progress engage() ramp stops within one step
        instead of continuing toward the leader."""
        self._estop_event.set()
        self._estopped = True
        self.sync_enabled = False
        self.timings.request_save("estop")
        for u in self.units:
            try:
                u.agent.stop()
            finally:
                u.robot.stop()

    # --- synchronous mode (CLI) -------------------------------------------
    def run_for(
        self,
        seconds: float,
        record_task: str | None = None,
        save_root: str | None = None,
        station: StationConfig | None = None,
    ) -> str | None:
        from ..data.recorder import EpisodeRecorder

        rec = None
        if record_task:
            assert save_root and station, "recording needs save_root + station"
            rec = EpisodeRecorder(
                save_root, station, self.cameras, arm_names=[u.name for u in self.units]
            )
            rec.start(record_task)
            self.attach_recorder(rec)

        # Ease each follower to its leader before tracking (mock no-op), then
        # enable sync so the loop commands + records all arms.
        self._start_cameras()
        self.engage_all()
        self.sync_enabled = True
        end = time.perf_counter() + seconds
        try:
            while time.perf_counter() < end:
                t0 = time.perf_counter()
                self._step()
                self._sleep_to_rate(t0)
        finally:
            self._stop_cameras()

        if rec is not None:
            path = rec.stop()
            print(f"saved episode: {path}  ({rec.frame_count} frames)")
            return str(path)
        return None
