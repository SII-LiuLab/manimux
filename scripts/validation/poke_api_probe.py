"""Probe Poke through the process worker; never construct a hardware driver."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from manimux.cli import load_config
from manimux.kinematics.poses import pose_matrix
from manimux.policies.response_log import response_log_path
from manimux.policies.worker import PolicyWorkerClient
from manimux.policy_adapter import build_policy_adapter
from manimux.policy_adapter.poke import PokeRequest
from manimux.runtime.timeline import ActionTimeline
from manimux.types import ActionContext, ObservationSnapshot, RobotState, SensorFrame
from scripts.validation.dream_api_probe import smooth_preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--recorded-observation", type=Path, required=True)
    parser.add_argument(
        "--model-state",
        type=Path,
        help="Explicit model-frame JSON array; interface-only, skips FK/IK",
    )
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument("--input-provenance", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.calls <= 10:
        parser.error("--calls must be between 1 and 10")
    config = load_config(args.config)
    if config["policy"]["worker"] != "poke_api":
        parser.error("Select a poke_api configuration")
    with np.load(args.recorded_observation, allow_pickle=False) as archive:
        saved = {name: archive[name].copy() for name in archive.files}
    model_state = json.loads(args.model_state.read_text()) if args.model_state else None
    # Model-frame inputs exercise transport alone. Otherwise require real configured
    # transforms and run the ordinary offline adapter, admission and executor.
    adapter = (
        None if model_state is not None else build_policy_adapter(config["robot"], config["policy"], ik_options=config["execution"]["ik"])
    )
    report = {
        "hardware_used": False,
        "input": str(args.recorded_observation.resolve()),
        "input_timestamps": "recorded pixels/state replayed with current request timestamps",
        "mode": "interface_only" if adapter is None else "offline_adapter_and_executor",
        "model_state_source": str(args.model_state) if args.model_state else "configured FK",
        "calls": [],
    }
    if args.input_provenance:
        report["input_provenance"] = json.loads(args.input_provenance.read_text())
    arrays = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    worker = PolicyWorkerClient(config["policy"], "poke-api-probe")
    try:
        worker.start()
        report["backend"] = worker.capabilities.backend_metadata
        for index in range(args.calls):
            now = time.monotonic_ns()
            state = RobotState(
                {g: saved[f"state__{g}"] for g in config["robot"]["group_dims"]}, now, index
            )
            camera = config["policy"]["adapter"]["camera"]
            observation = ObservationSnapshot(
                state, {camera: SensorFrame(camera, saved[f"image__{camera}"], now, index)}
            )
            request = PokeRequest(
                "poke-api-probe",
                index,
                now,
                now + round(config["policy"]["timeout_s"] * 1e9),
                observation,
                config["run"]["task_prompt"],
                model_state or [],
            )
            if adapter is not None:
                request = adapter.prepare_request(request)
            worker.submit_latest(request)
            response = None
            deadline = time.monotonic() + config["policy"]["timeout_s"] + 5
            while response is None:
                if time.monotonic() > deadline:
                    raise TimeoutError("Poke worker did not finish")
                response = worker.poll()
                if response is None:
                    time.sleep(0.01)
            if response.error:
                raise RuntimeError(response.error)
            if (response.session_id, response.request_seq, response.observation_time_ns) != (
                request.session_id,
                index,
                now,
            ):
                raise AssertionError("Poke worker lost request association")
            raw = response.raw_action
            arrays[f"call_{index}_model_actions"] = raw["actions"]
            row = {
                "index": index,
                "shape": list(raw["actions"].shape),
                "timing": raw["sampling_stats"]["poke_api"],
                "response_path": str(response_log_path(
                    config["policy"]["options"].get("response_log_dir"),
                    request.session_id, index,
                )),
            }
            report["calls"].append(row)
            if adapter is not None:
                if adapter.decode_pose_only:
                    raise ValueError("This probe requires chunk IK, not pose-only decoding")
                ik_started = time.monotonic()
                chunk = adapter.decode_action(
                    raw,
                    ActionContext(
                        index,
                        now,
                        response.finished_time_ns,
                        ik_seed=state,
                        observation_state=state,
                    ),
                )
                row["ik_ms"] = (time.monotonic() - ik_started) * 1000
                model = adapter.kinematics.models[adapter.group]
                errors = []
                for action, joints in zip(
                    raw["actions"], chunk.groups[adapter.group][:len(raw["actions"])], strict=True,
                ):
                    target = (
                        adapter.base_from_model
                        @ pose_matrix(action[[0, 1, 2, 6, 3, 4, 5]])
                        @ adapter.model_tool_from_tcp
                    )
                    actual = model.fk(joints)
                    errors.append([
                        np.linalg.norm(target[:3, 3] - actual[:3, 3]),
                        Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).magnitude(),
                    ])
                ik = chunk.metadata["ik"][adapter.group]
                row["ik"] = ik
                row["ik_solved_rows"] = ik["solved_steps"]
                row["ik_held_rows"] = ik["failed_steps"]
                row["max_fk_position_error_m"] = float(np.max(errors, axis=0)[0])
                row["max_fk_rotation_error_rad"] = float(np.max(errors, axis=0)[1])
                for group, joints in chunk.groups.items():
                    arrays[f"call_{index}_{group}_joints"] = joints
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
                row["admission"] = asdict(admission)
                if not admission.accepted:
                    raise ValueError(f"Poke plan rejected: {admission.reason}")
                preview = smooth_preview(config, chunk, state)
                for group, commands in preview.items():
                    arrays[f"call_{index}_{group}_commands"] = commands
                row["commands_finite"] = all(np.isfinite(v).all() for v in preview.values())
                row["command_rows"] = {g: len(v) for g, v in preview.items()}
            print(json.dumps(row), flush=True)
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        worker.close()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if arrays:
            np.savez(args.output.with_suffix(".npz"), **arrays)


if __name__ == "__main__":
    main()
