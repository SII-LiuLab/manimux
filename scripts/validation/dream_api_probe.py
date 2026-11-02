"""Exercise Dream API, the process worker, joint adapter and Smooth without hardware."""

import argparse
import json
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np

from manimux.cli import load_config
from manimux.embodiments.sensor import SensorBase
from manimux.policies.base import attach_sampling_stats
from manimux.policies.worker import PolicyWorkerClient
from manimux.policy_adapter import build_policy_adapter
from manimux.runtime.executors.smooth import SmoothExecutor
from manimux.runtime.timeline import ActionTimeline
from manimux.types import (
    ActionContext,
    InferenceRequest,
    ObservationSnapshot,
    RobotState,
    SensorFrame,
)
from scripts.validation.starvla_fixtures.robot import RobotDouble


class ProbeRobot(RobotDouble):
    """Synthetic plant implementing the current metadata contract."""

    @property
    def arm_components(self):
        return {}

    @property
    def end_effectors(self):
        return {}

    @property
    def recovery_actions(self):
        return ()

    def command_bounds(self):
        return {}

    def command_status(self):
        return {}

    def sent_command_snapshots(self):
        return {}

    def send_command(self, command):
        super().send_command(command)
        self.command_times.append(command.monotonic_ns)
        for name, values in command.groups.items():
            self.command_rows[name].append(values.copy())

    def close(self):
        super().close()
        if self.command_log is not None:
            np.savez(
                self.command_log,
                monotonic_ns=np.asarray(self.command_times, dtype=np.int64),
                **{name: np.asarray(rows) for name, rows in self.command_rows.items()},
            )

    def runtime_metadata(self):
        return {"kind": "synthetic_dream_api_probe", "hardware_used": False}


def build_probe_robot(config, clock):
    from manimux.embodiments.robot.base import RobotModel

    robot = ProbeRobot(config["group_dims"], clock)
    robot.model = RobotModel.from_config(config["config"])
    robot._kinematics = robot.model.kinematics
    robot._groups = {
        name: np.array(values, dtype=float, copy=True)
        for name, values in config["options"]["initial_groups"].items()
    }
    robot._target = {name: values.copy() for name, values in robot._groups.items()}
    robot.command_rows = {name: [] for name in robot._groups}
    robot.command_times = []
    robot.command_log = config["options"].get("command_log")
    return robot


class RecordedSensor(SensorBase):
    """Replay saved pixels on a simulation clock; never open a camera device."""

    def __init__(self, config, clock):
        self.name, self.clock = config["name"], clock
        with np.load(config["observation_npz"], allow_pickle=False) as archive:
            self.pixels = archive[f"image__{self.name}"].copy()
        self.sequence = 0

    def start(self):
        pass

    def read(self):
        self.sequence += 1
        return SensorFrame(self.name, self.pixels, self.clock.now_ns(), self.sequence)

    def close(self):
        pass


def build_recorded_sensor(config, clock):
    return RecordedSensor(config, clock)


def smooth_preview(config, chunk, state):
    """Replay complete rows on a simulated clock; do not imply live admission."""
    timeline = ActionTimeline(
        config["robot"]["group_dims"], action_start_mode="first_step_when_ready"
    )
    dt_ns = round(1e9 / config["robot"]["control_hz"])
    now = chunk.created_time_ns
    result = timeline.commit(
        chunk,
        now_ns=now,
        max_plan_age_ns=60_000_000_000,
        current_command=state.groups,
        blend_steps=0,
    )
    if not result.accepted:
        raise ValueError(f"Offline full-chunk replay rejected: {result.reason}")
    executor = SmoothExecutor(config["executor"]["smooth"], dt_ns / 1e9)
    executor.reset(state)
    output = {group: [] for group in state.groups}
    for tick in range(round((chunk.horizon_steps - 1) * chunk.dt_ns / dt_ns) + 1):
        at = now + tick * dt_ns
        reference = timeline.reference_horizon(
            now_ns=at, dt_ns=dt_ns, horizon_steps=executor.horizon_steps
        )
        if reference is None:
            break
        command = executor.step(at, state, reference)
        state = RobotState(command.groups, at, tick)
        for group, row in command.groups.items():
            output[group].append(row.copy())
    return {group: np.stack(rows) for group, rows in output.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Local Dream API + Smooth configuration")
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument(
        "--recorded-observation", type=Path,
        help="NPZ with image__<camera> RGB and state__<group>; replay, not live sensing",
    )
    parser.add_argument("--task-prompt", help="Instruction for this disconnected probe")
    parser.add_argument(
        "--runtime-seconds",
        type=float,
        default=0,
        help="Also exercise the actual Async runtime on a synthetic test plant (0..20 s)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.calls <= 10:
        parser.error("--calls must be between 1 and 10")
    if not np.isfinite(args.runtime_seconds) or not 0 <= args.runtime_seconds <= 20:
        parser.error("--runtime-seconds must be between 0 and 20")
    config = load_config(args.config)
    if args.task_prompt is not None:
        config["run"]["task_prompt"] = args.task_prompt
    if config["policy"]["worker"] != "dream_api" or config["executor"]["type"] != "smooth":
        parser.error("Select the Dream API + Smooth recipe")
    # Only offline RobotModel geometry is loaded by the adapter. No build_robot,
    # sensor constructor, runtime service, controller or actuator is instantiated.
    adapter = build_policy_adapter(config["robot"], config["policy"])
    worker = PolicyWorkerClient(config["policy"], "dream-api-probe")
    report = {
        "hardware_used": False,
        "input": "three synthetic RGB gradients, one frame per camera",
        "preview": "full-chunk Smooth replay on a simulated clock, not live admission",
        "calls": [],
    }
    arrays = {}
    recorded = None
    if args.recorded_observation:
        with np.load(args.recorded_observation, allow_pickle=False) as archive:
            recorded = {key: archive[key].copy() for key in archive.files}
        report["input"] = {
            "kind": "recorded observation replay; playback timestamps, no live feedback",
            "path": str(args.recorded_observation.resolve()),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        worker.start()
        report["backend"] = worker.capabilities.backend_metadata
        for index in range(args.calls):
            now = time.monotonic_ns()
            groups = {
                group: np.array([0.0, 0.5, 0.5, -0.35, 0.0, 0.0, 1.0])
                for group in config["robot"]["group_dims"]
            }
            if recorded is not None:
                groups = {group: recorded[f"state__{group}"] for group in groups}
            state = RobotState(groups, now, index)
            y, x = np.indices((176, 320))
            rgb = np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(np.uint8)
            frames = {
                name: SensorFrame(
                    name, recorded[f"image__{name}"] if recorded is not None else rgb.copy(),
                    now, index,
                )
                for name in config["policy"]["adapter"]["camera_map"].values()
            }
            request = adapter.prepare_request(
                InferenceRequest(
                    "dream-api-probe",
                    index,
                    now,
                    now + round(config["policy"]["timeout_s"] * 1e9),
                    adapter.build_observation(ObservationSnapshot(state, frames)),
                    config["run"]["task_prompt"],
                )
            )
            worker.submit_latest(request)
            ticks = 0
            response = None
            deadline = time.monotonic() + config["policy"]["options"]["request_timeout_s"] + 5
            while response is None:
                if time.monotonic() > deadline:
                    raise TimeoutError("Dream process worker did not finish")
                response = worker.poll()
                ticks += 1
                if response is None:
                    time.sleep(0.01)
            if response.error:
                raise RuntimeError(response.error)
            if (response.session_id, response.request_seq, response.observation_time_ns) != (
                request.session_id,
                request.request_seq,
                request.observation_time_ns,
            ):
                raise AssertionError("Worker lost request identity or observation timestamp")
            chunk = attach_sampling_stats(
                adapter.decode_action(
                    response.raw_action,
                    ActionContext(
                        index,
                        now,
                        response.finished_time_ns,
                        observation_state=state,
                    ),
                ),
                response.raw_action,
            )
            timeline = ActionTimeline(
                config["robot"]["group_dims"],
                action_start_mode=config["inference"]["action_start_mode"],
            )
            admission = timeline.commit(
                chunk,
                now_ns=response.finished_time_ns,
                max_plan_age_ns=round(config["inference"]["max_plan_age_s"] * 1e9),
                current_command=state.groups,
                blend_steps=0,
            )
            smoothed = smooth_preview(config, chunk, state)
            for group in groups:
                arrays[f"call_{index}_{group}_raw"] = chunk.groups[group]
                arrays[f"call_{index}_{group}_smooth"] = smoothed[group]
            row = {
                "index": index,
                "inference_ms": response.inference_ms,
                "main_loop_ticks_during_request": ticks,
                "shapes": {g: list(v.shape) for g, v in chunk.groups.items()},
                "dt_ns": chunk.dt_ns,
                "timing": chunk.metadata["sampling_stats"]["dream_api"],
                "live_clock_admission": asdict(admission),
                "smooth_preview_rows": {g: len(v) for g, v in smoothed.items()},
                "smooth_finite": all(np.isfinite(v).all() for v in smoothed.values()),
            }
            report["calls"].append(row)
            print(json.dumps(row), flush=True)
        worker.submit_reset("dream-api-probe-reset")
        deadline = time.monotonic() + config["policy"]["startup_timeout_s"]
        reset = None
        while reset is None and time.monotonic() < deadline:
            reset = worker.poll_reset()
            time.sleep(0.01)
        if reset is None or reset.error:
            raise RuntimeError(f"Reset failed: {reset}")
        report["reset_verified"] = True
        worker.close()
        if args.runtime_seconds:
            from manimux.runtime import build_runtime

            offline = deepcopy(config)
            offline["robot"]["type"] = "scripts.validation.dream_api_probe:build_probe_robot"
            offline["robot"]["options"] = {
                "offline_model": True,
                "initial_groups": groups,
                "move_to_start_on_connect": False,
                "home_on_close": False,
                "command_log": str(args.output.with_suffix(".commands.npz")),
            }
            offline["sensors"] = [
                {
                    "name": name,
                    "driver": "scripts.validation.starvla_fixtures.sensor:build_sensor",
                    "width": 320,
                    "height": 176,
                    "pattern": "spatial_gradient",
                }
                for name in config["policy"]["adapter"]["camera_map"].values()
            ]
            if recorded is not None:
                offline["sensors"] = [
                    {
                        "name": name,
                        "driver": "scripts.validation.dream_api_probe:build_recorded_sensor",
                        "observation_npz": str(args.recorded_observation.resolve()),
                    }
                    for name in config["policy"]["adapter"]["camera_map"].values()
                ]
            offline["robogui"]["enabled"] = False
            offline["recording"]["enabled"] = False
            offline["run"]["warmup_before_start"] = False
            offline["run"]["max_control_steps"] = round(
                args.runtime_seconds * config["robot"]["control_hz"]
            )
            runtime_dir = args.output.with_suffix(".runtime")
            runtime_dir.mkdir(exist_ok=True)
            result = build_runtime(offline, runtime_dir).run()
            report["runtime"] = asdict(result)
            with np.load(args.output.with_suffix(".commands.npz"), allow_pickle=False) as sent:
                report["command_sink"] = {
                    "kind": "disconnected synthetic robot; no actuator transport",
                    "count": len(sent["monotonic_ns"]),
                    "finite": all(np.isfinite(sent[group]).all() for group in groups),
                    "path": str(args.output.with_suffix(".commands.npz").resolve()),
                }
            print(json.dumps({"offline_runtime": report["runtime"]}, default=str), flush=True)
            if not result.run_completed:
                raise RuntimeError("Offline Async runtime did not complete")
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        worker.close()
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
        if arrays:
            np.savez(args.output.with_suffix(".npz"), **arrays)


if __name__ == "__main__":
    main()
