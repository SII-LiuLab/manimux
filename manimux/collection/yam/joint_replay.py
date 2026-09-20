"""Operator-started physical replay of saved controller commands at 60 Hz.

Without --execute this only validates files/configuration. Reuses collection's
executor, robot ownership lock, recorder and camera workers; never opens leaders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import select
import signal
import sys
import threading
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np

from .backend import CollectionBackend, load_backend_config
from .config import build_station_config
from .data.command_resampling import load_command_rate_replay

METHODS = {"original": "native", "10": "resample_0", "30": "resample_1"}


class ReplayBackend(CollectionBackend):
    def metadata(self):
        result = super().metadata()
        result["policy"] = "saved_command_replay"
        return result

    def hold_for_shutdown(self):
        with self._mutex:
            self._enabled = False
            return self.driver.hold_healthy_arms()

    def _stop_on_error(self, error):
        # Collection's default stop submits a bimanual cached pose. Replay must
        # never command a failed arm just to stop its healthy partner.
        with self._mutex:
            self._fault = str(error)
            self._halted = True
            try:
                self.hold_for_shutdown()
            except Exception as exc:
                self._fault += f"; hold failed: {exc}"


def held_gripper_targets(groups, frame, grippers):
    """Recorded gripper observations must never become replay actuator targets."""
    return {arm: np.r_[values[frame, :6], grippers[arm]] for arm, values in groups.items()}


def publish(backend, targets):
    with backend.target_batch():
        for arm, values in targets.items():
            backend.set_target(f"{arm}_arm", values)


def read_observations(backend):
    return {arm: backend.observation(f"{arm}_arm") for arm in ("left", "right")}


def prepare_start(backend, groups, grippers, duration_s, *, sleep=time.sleep, now=time.monotonic):
    """Explicit operator preparation; excluded from trial and never moves grippers."""
    observed = read_observations(backend)
    start = now()
    while True:
        tick = now()
        fraction = min(1.0, (tick - start) / duration_s)
        blend = fraction * fraction * (3 - 2 * fraction)
        targets = {
            arm: np.r_[o["joint_pos"] + blend * (groups[arm][0, :6] - o["joint_pos"]),
                       grippers[arm]] for arm, o in observed.items()
        }
        publish(backend, targets)
        if fraction >= 1:
            return
        sleep(max(0.0, 1 / 60 - (now() - tick)))


def execute_trajectory(backend, recorder, data, method, grippers, *, settle_s=1.0,
                       now_ns=time.monotonic_ns, unix_ns=time.time_ns, sleep=time.sleep,
                       journal=None):
    """Select by elapsed wall time, with at least one period between submissions.

    Pace from the previous publish return. A late observation must not squeeze
    two targets against adjacent grid deadlines. Work can lower achieved Hz;
    elapsed time still selects the source frame, so the trial stays at 1x speed.

    Observations precede this row's new command; they have their own timestamps.
    Compare native CAN feedback against the latest submitted target at its time,
    not against a command which had not yet been submitted.
    """
    groups = data.trajectories[METHODS[method]]
    period_ns = 1e9 / data.target_hz
    minimum_interval_ns = math.ceil(period_ns)
    start = now_ns()
    start_unix = unix_ns()
    duration_ns = int(round((data.time_s[-1] - data.time_s[0] + settle_s) * 1e9))
    deadline = start
    rows = []
    result = {} if journal is None else journal
    result.update(start_monotonic_ns=start, start_unix_ns=start_unix, ticks=rows,
                  minimum_submit_interval_ns=minimum_interval_ns,
                  pacing="one_period_after_previous_publish_return")
    while True:
        sleep(max(0.0, (deadline - now_ns()) / 1e9))
        obs = read_observations(backend)
        tick = now_ns()
        tick_unix = unix_ns()
        elapsed = tick - start
        if elapsed > duration_ns:
            break
        frame = min(int(elapsed / period_ns), len(data.time_s) - 1)
        targets = held_gripper_targets(groups, frame, grippers)
        publish(backend, targets)
        submitted = now_ns()
        inputs = {arm: backend.controller_input(f"{arm}_arm") for arm in targets}
        recorder.tick(targets, obs, {}, inputs, tick_timestamp_ns=tick_unix,
                      tick_monotonic_ns=tick)
        rows.append({"frame": frame, "elapsed_s": elapsed / 1e9,
                     "tick_monotonic_ns": tick, "tick_unix_ns": tick_unix,
                     "submit_return_monotonic_ns": submitted,
                     "command_unix_ns": int(inputs["left"]["timestamp_ns"]),
                     "late_ms": (tick - deadline) / 1e6})
        # A future grid boundary alone can be less than 1 ms away after a late
        # tick. Wait a whole period instead; select/skip source frames above.
        deadline = submitted + minimum_interval_ns
    return result


def wait_for_operator(message, backend, prompt=None):
    """A dead control thread must abort even while the operator has not pressed Enter."""
    backend.driver.check_health()
    if prompt is not None:
        prompt(message)
    else:
        print(message, end="", flush=True)
        while True:
            backend.driver.check_health()
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if ready:
                if not sys.stdin.readline():
                    raise EOFError("Replay needs an interactive terminal for Enter prompts")
                break
    backend.driver.check_health()


class HomeNotReached(RuntimeError):
    """The control loops are alive but the measured arms have not reached Home."""


@contextmanager
def protect_recovery():
    """Further interrupts must not unwind cleanup into an automatic torque release."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def interrupted(*_):
        print("退出处理中；不会因再次中断而自动释放力矩。", file=sys.stderr, flush=True)

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def wait_for_support(prompt=None, *, sleep=time.sleep):
    """A dead driver cannot use wait_for_operator's health-gated Enter prompt.

    Require a distinct token: leftover Enter, Ctrl-C and terminal EOF cannot
    authorize releasing torque. No motor is reconnected or homed here.
    """
    message = (
        "无法确认双臂安全回到 Home。已尝试保持仍正常的机械臂；断联关节无法保证保持。\n"
        "请先可靠支撑双臂，再输入 RELEASE 退出并释放力矩："
    )
    print(message, end="", flush=True)
    while True:
        try:
            answer = input() if prompt is None else prompt(message)
            if answer is not None and answer.strip() == "RELEASE":
                return
            print("只有输入 RELEASE 才会关闭驱动；Enter/Ctrl+C 不会释放力矩。", flush=True)
        except (KeyboardInterrupt, EOFError, OSError):
            # Keep ownership and motor threads alive even if stdin disappears.
            sleep(0.25)


class ReplayShutdown:
    """One owner of Home/physical-support permission before driver closure."""

    def __init__(self, backend, prompt=None):
        self.backend, self.prompt = backend, prompt
        self.begin_trial()

    def begin_trial(self):
        self.home_attempted = False
        self.home_result = None
        self.support_confirmed = False
        self.manifest = None
        self.episode = None

    def recover(self):
        with protect_recovery():
            try:
                holds = self.backend.hold_for_shutdown()
                self.backend.driver.check_health()
                if any(row["status"] != "hold_submitted" for row in holds.values()):
                    raise RuntimeError(f"Some arms could not accept hold: {holds}")
                if self.home_attempted:
                    raise RuntimeError("Home already attempted; automatic retry is disabled")
                self.home_attempted = True
                print("正在回 Home（夹爪保持；回程不计入实验录制）。", flush=True)
                self.home_result = home_after_replay(self.backend)
                print("Home 已通过实测关节检查。", flush=True)
                return self.home_result
            except BaseException as exc:
                self.home_result = {
                    "status": "failed" if self.home_attempted else "skipped_unhealthy",
                    "release_grippers": False, "error": f"{type(exc).__name__}: {exc}",
                }
                # Home itself may have failed mid-motion; stop only responsive
                # arms before awaiting physical support, never retry the move.
                try:
                    self.home_result["holds"] = self.backend.hold_for_shutdown()
                except Exception as hold_error:
                    self.home_result["hold_error"] = str(hold_error)
                print(self.home_result["error"], file=sys.stderr, flush=True)
                wait_for_support(self.prompt)
                self.support_confirmed = True
                self.home_result["support_confirmed"] = True
                return self.home_result

    def close(self):
        with protect_recovery():
            if not self.support_confirmed:
                verified = False
                if self.home_result and self.home_result["status"] == "completed":
                    try:
                        state = self.backend.driver.get_state()
                        verified = all(np.max(np.abs(q[:6])) <= 0.05
                                       for q in state.groups.values())
                    except Exception:
                        pass
                if not verified:
                    # Covers exceptions before a trial, failed saving, and a
                    # motor thread dying after Home while data was encoded.
                    self.recover()
            if self.manifest is not None and self.episode is not None and self.episode.exists():
                self.manifest["shutdown"] = {
                    "release_authorized": True, "support_confirmed": self.support_confirmed,
                    "home": self.home_result,
                }
                try:
                    (self.episode / "replay-experiment.json").write_text(
                        json.dumps(self.manifest, indent=2, allow_nan=False))
                except Exception as exc:
                    print(f"Failed to save shutdown result: {exc}", file=sys.stderr, flush=True)
            self.backend.close()


def home_after_replay(backend, *, timeout_s=2.0, tolerance_rad=0.05,
                      now=time.monotonic, sleep=time.sleep):
    """Use the driver's Home and verify measured joints before normal shutdown."""
    backend.driver.home(release_grippers=False)
    deadline = now() + timeout_s
    while True:
        state = backend.driver.get_state()
        error = max(float(np.max(np.abs(q[:6]))) for q in state.groups.values())
        if error <= tolerance_rad:
            # Home bypasses the collection executor; reset its state to fresh
            # feedback before another trial can publish a target.
            backend.pause()
            return {"status": "completed", "joint_target": "zero",
                    "release_grippers": False, "max_abs_joint_rad": error,
                    "tolerance_rad": tolerance_rad,
                    "feedback_monotonic_ns": state.monotonic_ns,
                    "feedback": {name: q.tolist() for name, q in state.groups.items()}}
        if now() >= deadline:
            backend.pause()
            raise HomeNotReached(
                f"Home 未到位：最大关节偏差 {np.rad2deg(error):.2f}°，"
                f"阈值 {np.rad2deg(tolerance_rad):.2f}°。"
            )
        sleep(0.02)


def run_trials(data, station, config, *, method, save_root, prepare_s, prompt=None):
    if data.source != "command":
        raise ValueError(
            "Physical replay requires saved controller commands, not achieved feedback"
        )
    from .camera.worker import CameraWorker
    from .data.recorder import EpisodeRecorder
    from .runtime import build_cameras_from_config

    backend = ReplayBackend(config, config_path=station.manimux_config)
    with ExitStack() as stack:
        # SDK imports / USB pipeline startup can block Python. Warm cameras
        # before starting timing-sensitive motor threads, as the collection GUI does.
        drivers = build_cameras_from_config(station)
        workers = [CameraWorker(driver) for driver in drivers]
        for worker in workers:
            stack.callback(worker.stop)
        for worker in workers:
            worker.start()
        backend.connect(start_thread=False)
        # Every exit after connection must pass Home verification or explicit
        # physical-support confirmation before camera/driver cleanup can proceed.
        shutdown = ReplayShutdown(backend, prompt)
        stack.callback(shutdown.close)
        observed = read_observations(backend)
        grippers = {arm: float(o["gripper_pos"][0]) for arm, o in observed.items()}
        recorder = EpisodeRecorder(
            save_root, station, workers, ["left", "right"], backend=backend,
        )
        methods = list(METHODS) if method == "all" else [method]
        for selected in methods:
            shutdown.begin_trial()
            groups = data.trajectories[METHODS[selected]]
            episode = None
            manifest = {"schema_version": 1, "method": selected, "source": "command",
                        "source_fields": ["controller-{arm}-joint.npy",
                                          "controller-{arm}-timestamp-ns.npy",
                                          "manimux-control.jsonl:command"],
                        "source_episode": str(data.episode), "source_origin_ns": data.origin_ns,
                        "source_interval_s": data.time_s[[0, -1]].tolist(),
                        "target_hz": data.target_hz, "gripper_mode": "hold_initial",
                        "gripper_targets": grippers, "status": "preparing",
                        "joint_sha256": {arm: hashlib.sha256(
                            np.ascontiguousarray(q[:, :6]).tobytes()).hexdigest()
                            for arm, q in groups.items()}}
            shutdown.manifest = manifest
            failure = None
            saved = None

            def failed(exc, manifest=manifest):
                nonlocal failure
                detail = f"{type(exc).__name__}: {exc}"
                if failure is None:
                    failure = exc
                    manifest.update(status="interrupted", error=detail)
                else:
                    manifest.setdefault("cleanup_errors", []).append(detail)
                print(detail, file=sys.stderr, flush=True)

            try:
                wait_for_operator(
                    f"\n{selected}: Enter 开始 {prepare_s:g} 秒移至公共起点（夹爪保持）；"
                    "Ctrl+C 停止并尝试回 Home。", backend, prompt)
                prepare_start(backend, groups, grippers, prepare_s)
                # Operator waiting must not leave an expiring leader target enabled.
                backend.pause()
                wait_for_operator(
                    f"{selected}: 起点过渡结束。Enter 开始回放并记录；Ctrl+C 停止并尝试回 Home。",
                    backend, prompt)
                manifest["initial_feedback"] = {
                    arm: {"joint_rad": o["joint_pos"].tolist(),
                          "timestamp_ns": int(o["feedback_timestamp_ns"]),
                          "max_start_error_deg": float(np.max(np.abs(np.rad2deg(
                              o["joint_pos"] - groups[arm][0, :6]))))}
                    for arm, o in read_observations(backend).items()
                }
                episode = recorder.start(f"command_replay_{selected}")
                shutdown.episode = episode
                manifest["status"] = "recording"
                (episode / "replay-experiment.json").write_text(json.dumps(manifest, indent=2))
                execute_trajectory(backend, recorder, data, selected, grippers, journal=manifest)
                manifest["status"] = "completed"
            except BaseException as exc:
                failed(exc)
            finally:
                with protect_recovery():
                    if recorder.is_recording:
                        try:
                            recorder.freeze()
                        except BaseException as exc:
                            failed(exc)
                    manifest["home"] = shutdown.recover()
                    if manifest["home"]["status"] != "completed" and failure is None:
                        failed(RuntimeError(manifest["home"]["error"]))
                    if recorder.is_saving:
                        try:
                            saved = recorder.stop()
                            if saved is not None:
                                meta = json.loads((saved / "metadata.json").read_text())
                                actual = meta["extra"]["timing"]["commands"]["left"]["actual_hz"]
                                rate = f"{actual:.2f}" if actual is not None else "不足两条记录"
                                print(f"Saved {selected}: {saved}\n实际下发平均: {rate} Hz",
                                      flush=True)
                        except BaseException as exc:
                            failed(exc)
                    # Preserve the original error and Home result even when the
                    # writer failed and could not finalize metadata/video.
                    if episode is not None and episode.exists():
                        try:
                            (episode / "replay-experiment.json").write_text(
                                json.dumps(manifest, indent=2, allow_nan=False))
                        except BaseException as exc:
                            failed(exc)
            if failure is not None:
                raise failure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--method", choices=(*METHODS, "all"), default="all")
    parser.add_argument(
        "--config", type=Path, default=Path("manimux/configs/collection/yam/station.yaml")
    )
    parser.add_argument(
        "--cameras", type=Path, default=Path("manimux/configs/collection/yam/cameras.yaml")
    )
    parser.add_argument("--camera", default="top")
    parser.add_argument("--save-root", type=Path, required=True)
    parser.add_argument("--prepare-s", type=float, default=5.0)
    parser.add_argument("--execute", action="store_true",
                        help="enable physical replay, gated by Enter")
    args = parser.parse_args(argv)
    if not math.isfinite(args.prepare_s) or args.prepare_s < 3:
        parser.error("--prepare-s must be finite and at least 3 seconds")
    data = load_command_rate_replay(args.episode)
    if set(data.native) != {"left", "right"} or any(
        not np.isfinite(q[:, :6]).all()
        for groups in data.trajectories.values() for q in groups.values()
    ):
        raise ValueError("Physical replay requires complete finite trajectories for both arms")
    station = build_station_config(args.config, args.cameras)
    station.collection_hz = station.control_hz = 60.0
    station.execution_mode = "synchronous"
    station.data_format = "default"
    station.cameras = [c for c in station.cameras if c.name == args.camera]
    if len(station.cameras) != 1:
        raise ValueError(f"Camera {args.camera!r} must exist exactly once in station config")
    config = load_backend_config(station)
    if config.executor.type != "direct":
        raise ValueError("Replay experiment requires the configured Direct executor")
    limits = config.executor.motion_limits
    if limits and any(getattr(limits.arm, field) is not None
                      for field in ("max_velocity", "max_acceleration", "max_step_dt_s")):
        raise ValueError("Shared arm motion limits would modify the comparison trajectories")
    save_root = args.save_root.expanduser().resolve()
    if save_root == data.episode or data.episode in save_root.parents:
        raise ValueError("Replay outputs must be outside the original episode")
    print(f"Source controller command: {data.episode}\n"
          f"{len(data.time_s)} frames @ 60 Hz comparison grid; "
          f"duration {data.time_s[-1] - data.time_s[0]:.3f} s + 1 s settling.\n"
          f"Methods: {args.method}; camera: {args.camera}; grippers: hold initial.\n"
          "Normal end / software error / Ctrl+C: freeze recording, Home if healthy, then save.\n"
          "Unhealthy driver / failed Home: await physical support and RELEASE before closing.\n"
          f"Save: {save_root}", flush=True)
    if not args.execute:
        print("Validated only; no hardware connected. Add --execute to run with Enter prompts.")
        return
    def interrupted(*_):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        run_trials(data, station, config, method=args.method, save_root=save_root,
                   prepare_s=args.prepare_s)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
