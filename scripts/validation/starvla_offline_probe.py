"""Check a paired StarVLA deployment with synthetic inputs and the real ManiMux path."""

import argparse
import json
import time
import traceback
from contextlib import closing
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from manimux.cli import load_config
from manimux.clock import SystemClock
from manimux.embodiments.robot import build_robot
from manimux.policies.starvla.client import StarVlaPolicyModel
from manimux.policy_adapter import build_policy_adapter
from manimux.types import (
    ActionContext,
    InferenceRequest,
    ObservationSnapshot,
    RobotState,
    SensorFrame,
)


def probe_sampling(client, config, snapshot, condition, report):
    """Exercise the actual worker request classes, including client AAC selection."""
    from manimux.runtime.aac import AacInferenceRequest
    from manimux.runtime.autohorizon import AutoHorizonInferenceRequest
    from manimux.runtime.paint import PaintInferenceRequest
    from manimux.runtime.rtc.request import RtcInferenceRequest

    if config["policy"]["options"]["action_format"] != "joint":
        raise ValueError("The flow sampler matrix currently requires joint actions")
    horizon = config["policy"]["horizon_policy_steps"]
    cases = [
        (
            "rtc",
            RtcInferenceRequest,
            dict(
                action_condition=condition,
                condition_weights=np.linspace(1, 0, horizon),
                rtc_beta=1.0,
            ),
        ),
        (
            "paint",
            PaintInferenceRequest,
            dict(paint_action_prefix=condition[:3], paint_delay_steps=3),
        ),
        (
            "aac",
            AacInferenceRequest,
            dict(
                aac_num_samples=4,
                aac_robot_config=str(
                    Path("manimux/configs/embodiment/robot/yam_dual.yaml").resolve()
                ),
                aac_ee_stats_path="scripts/validation/starvla_fixtures/aac_stats.json",
            ),
        ),
        ("autohorizon", AutoHorizonInferenceRequest, {}),
    ]
    report["sampler_calls"] = []
    for index, (mode, request_type, options) in enumerate(cases):
        entry = {"mode": mode}
        report["sampler_calls"].append(entry)
        try:
            now = time.monotonic_ns()
            request = request_type(
                "starvla-offline",
                index + 10,
                now,
                now + 180_000_000_000,
                snapshot,
                config["run"]["task"],
                **options,
            )
            started = time.perf_counter()
            raw = client.infer(request)
            entry["rpc_seconds"] = time.perf_counter() - started
            if not all(np.isfinite(rows).all() for rows in raw["actions"].values()):
                raise ValueError("Sampler returned non-finite actions")
            entry.update(
                status="passed", shapes={g: list(rows.shape) for g, rows in raw["actions"].items()}
            )
            if mode in raw:
                entry["metadata"] = raw[mode]
        except Exception as exc:
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        print(
            json.dumps({key: value for key, value in entry.items() if key != "metadata"}),
            flush=True,
        )
    if any(row["status"] != "passed" for row in report["sampler_calls"]):
        raise RuntimeError("At least one flow sampling mode failed; see sampler_calls")


def probe_observations(client, adapter, config, report, outputs, *, deterministic):
    """Check identity, repeated inputs, reset and changed RGB through the real client."""
    session = "starvla-offline"
    client.reset(session)
    capabilities = client.capabilities()
    report["backend"] = capabilities.backend_metadata
    report["sampling_modes"] = sorted(capabilities.sampling_modes)
    for index in range(4):
        if index == 2:
            client.reset(session)
        now = time.monotonic_ns()
        frames = {}
        for i, sensor in enumerate(config["sensors"]):
            y, x = np.indices((224, 224))
            rgb = np.stack((x, y, (x + y + i * 31) % 256), axis=-1).astype(np.uint8)
            if index == 3:
                rgb = 255 - rgb
            frames[sensor["name"]] = SensorFrame(sensor["name"], rgb, now, index)
        initial = config["robot"].get("options", {}).get("initial_groups", {})
        state = RobotState(
            {
                g: np.array(initial.get(g, np.zeros(d)), dtype=float, copy=True)
                for g, d in config["robot"]["group_dims"].items()
            },
            now,
            index,
        )
        request = InferenceRequest(
            session,
            index,
            now,
            now + 180_000_000_000,
            adapter.build_observation(ObservationSnapshot(state, frames)),
            config["run"]["task"],
        )
        request = adapter.prepare_request(request)
        started = time.perf_counter()
        raw = client.infer(request)
        elapsed = time.perf_counter() - started
        chunk = adapter.decode_action(
            raw, ActionContext(index, now, time.monotonic_ns(), measured_state=state)
        )
        rows = np.concatenate(list(chunk.groups.values()), axis=-1)
        if not np.isfinite(rows).all():
            raise ValueError("Non-finite robot targets")
        outputs.append(rows)
        report["calls"].append(
            dict(
                index=index,
                rpc_seconds=elapsed,
                wire_shape={g: list(a.shape) for g, a in raw["actions"].items()},
                joint_shape=list(rows.shape),
                dt_ns=chunk.dt_ns,
            )
        )
        print(json.dumps(report["calls"][-1]), flush=True)
    report["reset_completed"] = True
    report.update(
        repeat_max_abs_diff=float(np.max(np.abs(outputs[1] - outputs[0]))),
        reset_max_abs_diff=float(np.max(np.abs(outputs[2] - outputs[0]))),
        changed_images_max_abs_diff=float(np.max(np.abs(outputs[3] - outputs[0]))),
    )
    if deterministic:
        np.testing.assert_allclose(outputs[1], outputs[0], atol=1e-6, rtol=0)
        np.testing.assert_allclose(outputs[2], outputs[0], atol=1e-6, rtol=0)
    return ObservationSnapshot(state, frames), rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="Also run the real scheduler, executor and recorder on the test plant",
    )
    parser.add_argument(
        "--executor",
        choices=("direct", "smooth", "mpc"),
        help="Override the test plant executor for runtime validation",
    )
    parser.add_argument(
        "--sampling-matrix",
        action="store_true",
        help="Probe all five flow modes with their real ManiMux request classes",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Require identical predictions for repeated input and after reset (e.g. OFT)",
    )
    args = parser.parse_args()
    config = load_config(args.experiment, local=args.local)
    if args.executor:
        if not args.runtime:
            parser.error("--executor requires --runtime")
        config["executor"]["type"] = args.executor
    if config["robot"]["type"] != "scripts.validation.starvla_fixtures.robot:build_robot" or any(
        sensor["driver"] != "scripts.validation.starvla_fixtures.sensor:build_sensor"
        for sensor in config["sensors"]
    ):
        raise ValueError("This probe accepts only the offline test robot and synthetic sensors")
    robot = build_robot(config["robot"], SystemClock())
    adapter = build_policy_adapter(config["robot"], config["policy"], kinematics=robot.kinematics)
    client = StarVlaPolicyModel(config["policy"])
    report = dict(
        status="running",
        date_utc=datetime.now(UTC).isoformat(),
        experiment=str(args.experiment),
        validation_stage="native_server_manimux_adapter",
        hardware_used=False,
        robot_task_success_tested=False,
        geometry=(
            f"{robot.model.name} offline robot geometry"
            if robot.model is not None
            else "analytic Cartesian test plant"
            if robot.kinematics is not None
            else "joint-only test plant"
        ),
        executor=config["executor"]["type"],
        calls=[],
    )
    outputs = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with closing(client):
            snapshot, condition = probe_observations(
                client, adapter, config, report, outputs, deterministic=args.deterministic
            )
            if args.sampling_matrix:
                probe_sampling(client, config, snapshot, condition, report)
        if args.runtime:
            from manimux.runtime import build_runtime

            run_dir = args.output.with_suffix("")
            run_dir.mkdir(parents=True, exist_ok=True)
            result = build_runtime(config, run_dir).run()
            report["runtime"] = asdict(result)
            report["rollout_directory"] = str(run_dir)
            if not result.success or result.rejected_plans or result.accepted_plans < 1:
                raise RuntimeError(f"Offline runtime did not accept a clean plan: {result}")
        report["status"] = "passed"
    except Exception as exc:
        report.update(
            status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()
        )
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
        if outputs:
            np.savez(args.output.with_suffix(".npz"), decoded=np.stack(outputs))


if __name__ == "__main__":
    main()
