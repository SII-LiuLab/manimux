"""Evaluate real UMI_DP serving + Tianji IK + scheduling on a simulated plant.

Always substitutes robot/camera fixtures before constructing the runtime. It
never opens hardware. Synthetic or repeated RGB fixtures cannot establish task
success; this measures scheduling and command timing with the actual sampler.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import zarr

from manimux.config import ManiMuxConfig, load_config
from manimux.robots.mock import MockDualArmDriver
from manimux.runtime import build_runtime
from manimux.types import SensorFrame, copy_group_vector

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
MODULE = "scripts.validation.umi_dp_tianji_rtc_eval"


def fixture(path):
    if path:
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key].copy() for key in data.files}
    rng = np.random.default_rng(17)
    joints = np.radians([50, -40, -30, -100, -65, 0, 40])
    return {
        **{
            side + "_wrist": rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
            for side in ("left", "right")
        },
        **{side + "_arm": np.r_[joints, 0.8] for side in ("left", "right")},
    }


def build_plant(config, clock):
    plant = MockDualArmDriver(config.group_dims, clock)
    data = fixture(config.options.get("fixture"))
    groups = {name: np.asarray(data[name], dtype=np.float64) for name in config.group_dims}
    for name, dim in config.group_dims.items():
        if groups[name].shape != (dim,) or not np.isfinite(groups[name]).all():
            raise ValueError(f"Invalid fixture joint state for {name}")
    plant._groups = copy_group_vector(groups)
    plant._target = copy_group_vector(groups)
    return plant


class FixtureCameras:
    def __init__(self, config, clock):
        self.clock = clock
        data = fixture(config.options.get("fixture"))
        self.images = {name: data[name] for name in ("left_wrist", "right_wrist")}
        for image in self.images.values():
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise ValueError("Fixture images must be HxWx3 RGB uint8")
        self.period_ns = round(1e9 / config.fps)
        self.last_ns = -self.period_ns
        self.sequence = 0
        self.frames = {}

    def start(self):
        pass

    def read(self):
        now = self.clock.now_ns()
        if now - self.last_ns >= self.period_ns:
            self.last_ns = now
            self.sequence += 1
            self.frames = {
                name: SensorFrame(name, image, now, self.sequence)
                for name, image in self.images.items()
            }
        return self.frames

    def close(self):
        pass


def build_cameras(config, clock):
    return FixtureCameras(config, clock)


def percentiles(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return None
    return dict(
        zip(
            ("p50", "p95", "p99", "max"),
            map(float, np.percentile(values, [50, 95, 99, 100])),
            strict=True,
        )
    )


def summarize(episode, config):
    events = [json.loads(line) for line in (episode / "events.jsonl").read_text().splitlines()]
    accepted = [event for event in events if event["kind"] == "plan_accepted"]
    submissions = [event for event in events if event["kind"] == "inference_submitted"]
    conditioned = {e["request_seq"] for e in submissions if e.get("conditioned")}
    rejected = [
        event for event in events if event["kind"] in {"plan_rejected", "inference_rejected"}
    ]
    ticks = zarr.open(str(episode / "data.zarr"), mode="r")["ticks"]
    times = ticks["monotonic_ns"][:]
    command_step, tracking = [], []
    for name in config.robot.group_dims:
        commands = ticks[f"command/{name}"][:, :7]
        state = ticks[f"state/{name}"][:, :7]
        command_step.extend(np.max(np.abs(np.diff(commands, axis=0)), axis=1).tolist())
        tracking.extend(np.max(np.abs(commands - state), axis=1).tolist())
    boundaries = []
    for event in events:
        if event["kind"] == "plan_boundary" and event["previous_reference"]:
            boundaries.append(
                max(
                    float(
                        np.max(
                            np.abs(
                                np.asarray(event["committed_first"][name])[:7]
                                - np.asarray(event["previous_reference"][name])[:7]
                            )
                        )
                    )
                    for name in config.robot.group_dims
                )
            )
    return {
        "hardware_connected": False,
        "task_success": None,
        "observations": "fixed RGB fixtures; simulated capture times and joint feedback",
        "runtime": config.policy.options["history_strategy"],
        "ik_backend": config.policy.options["ik_backend"],
        "action_decoding": config.policy.action_decoding,
        "control_hz": config.robot.control_hz,
        "action_dt_s": config.policy.effective_action_dt_s,
        "horizon": config.policy.horizon_steps,
        "submitted": len(submissions),
        "conditioned_submissions": sum(bool(e.get("conditioned")) for e in submissions),
        "conditioned_accepted": sum(e["request_seq"] in conditioned for e in accepted),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "rejection_reasons": sorted({e.get("reason", "") for e in rejected}),
        "rtc_delay_infeasible": sum(e["kind"] == "rtc_delay_infeasible" for e in events),
        "tick_interval_ms": percentiles(np.diff(times) / 1e6),
        "decode_stage_ms": percentiles(
            [e["decode_stage_ms"] for e in accepted if "decode_stage_ms" in e]
        ),
        # The UMI chunk origin includes first_action_offset. Restore the measured
        # observation origin for the end-to-end latency, using recorded metadata.
        "observation_to_commit_ms": percentiles(
            [e["observation_to_commit_ms"] + e["first_action_offset_ns"] / 1e6 for e in accepted]
        ),
        "rtc_delay_ms": percentiles([e["rtc_delay_ms"] for e in accepted if "rtc_delay_ms" in e]),
        "joint_command_step_rad": percentiles(command_step),
        "joint_tracking_error_rad": percentiles(tracking),
        "reference_boundary_step_rad": percentiles(boundaries),
        "episode": str(episode),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Bound Tianji deployment config")
    parser.add_argument("--output", required=True, type=Path, help="New directory for this run")
    parser.add_argument("--server", help="Temporary evaluation model server WebSocket URL")
    parser.add_argument(
        "--fixture", type=Path, help="NPZ: wrist RGB arrays and two 8D joint states"
    )
    parser.add_argument("--strategy", choices=("rtc", "manimux"), default="rtc")
    parser.add_argument("--seconds", type=float, default=20)
    args = parser.parse_args()
    if args.seconds <= 0 or args.output.exists():
        parser.error("seconds must be positive and output must be a new directory")
    config = load_config(args.config)
    if (
        config.policy.worker != "xpolicylab_ws"
        or config.policy.adapter != "manimux.integrations.umi_dp_tianji.policy_plugin:build_adapter"
        or not config.policy.options.get("deployment_bound")
    ):
        parser.error("A bound UMI_DP Tianji deployment is required")
    data = config.model_dump(exclude_unset=True)
    fixture_path = None if args.fixture is None else str(args.fixture.resolve())
    data["robot"].update(driver=f"{MODULE}:build_plant", options={"fixture": fixture_path})
    data["sensors"] = [
        {
            "name": "fixture_wrists",
            "driver": f"{MODULE}:build_cameras",
            "fps": 30,
            "options": {"fixture": fixture_path},
        }
    ]
    data["viewer"]["enabled"] = False
    data["policy"]["action_decoding"] = "process"
    data["policy"]["options"]["history_strategy"] = args.strategy
    if args.server:
        data["policy"]["options"]["server"] = args.server
    execution = data["execution"]
    execution["runtime"] = "manimux.integrations.umi_dp_tianji.history:build_strategy"
    execution.pop("inference_schedule", None)
    execution.pop("refill_threshold_s", None)
    if args.strategy == "manimux":
        execution.update(inference_schedule="single_inflight", refill_threshold_s=0.25)
    data["run"]["max_steps"] = max(1, round(args.seconds * config.robot.control_hz))
    config = ManiMuxConfig.model_validate(data)
    args.output.mkdir(parents=True)
    runtime = build_runtime(config, args.output)
    error = None
    try:
        result = runtime.run()
        episode = result.episode_dir
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        episodes = list(args.output.glob("rollout-*"))
        if not episodes:
            raise
        episode = episodes[-1]
    report = summarize(episode, config)
    report["runtime_error"] = error
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if (
        error
        or not report["accepted"]
        or (args.strategy == "rtc" and not report["conditioned_accepted"])
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
