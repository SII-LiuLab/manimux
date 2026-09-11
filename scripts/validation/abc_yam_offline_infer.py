"""Probe the actual ABC HTTP worker and YAM adapter using recorded observations.

Run with envs/yam/.venv/bin/python. Never constructs a robot or sends commands.
The model server must already be running; see docs/abc-yam-runbook.md.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import requests

from manimux.config import load_config
from manimux.policies import build_policy_adapter, build_policy_model
from manimux.types import (
    ActionContext,
    InferenceRequest,
    ObservationSnapshot,
    RobotState,
    SensorFrame,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 150, 300])
    parser.add_argument("--config", default="configs/abc/yam/infra/official-bottles-75k.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if config.policy.worker != "abc_http" or config.policy.adapter != "abc_yam":
        raise ValueError("this probe requires the ABC HTTP worker and YAM adapter")
    server = str(config.policy.options["server"]).rstrip("/").removesuffix("/act")
    response = requests.get(server + "/act", timeout=5)
    response.raise_for_status()
    health = response.json()
    if (health["state_dim"], health["action_dim"], health["chunk_length"]) != (14, 14, 30):
        raise ValueError(f"unexpected ABC model contract: {health}")

    joint = {
        arm: np.concatenate(
            [
                np.load(args.episode / f"{arm}-joint_pos.npy"),
                np.load(args.episode / f"{arm}-gripper_pos.npy").reshape(-1, 1),
            ],
            axis=1,
        )
        for arm in ("left", "right")
    }
    model = build_policy_model(config.policy)
    adapter = build_policy_adapter(config.robot, config.policy)
    adapter.validate(config.robot, config.policy)
    session = "abc-offline-probe"
    samples, arrays = [], {}
    try:
        model.reset(session)
        for seq, index in enumerate(args.indices):
            if index < 0 or any(index >= len(q) for q in joint.values()):
                raise ValueError(f"invalid recorded frame index: {index}")
            frames = {}
            for role, name in [
                ("top", "front_camera"),
                ("left", "left_camera"),
                ("right", "right_camera"),
            ]:
                video = cv2.VideoCapture(str(args.episode / f"{role}-images-rgb.mp4"))
                try:
                    video.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, bgr = video.read()
                    if not ok:
                        raise RuntimeError(f"cannot read {role} video frame {index}")
                finally:
                    video.release()
                # OpenCV decodes the saved RGB video into BGR; restore RGB once.
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frames[name] = SensorFrame(name, rgb, time.monotonic_ns(), seq)
            now = time.monotonic_ns()
            state = RobotState({f"{arm}_arm": q[index] for arm, q in joint.items()}, now, seq)
            snapshot = adapter.build_observation(ObservationSnapshot(state, frames))
            request = InferenceRequest(
                session,
                seq,
                now,
                now + int(config.policy.timeout_s * 1e9),
                snapshot,
                config.run.task,
            )
            start = time.perf_counter()
            raw = np.asarray(model.infer(request))
            elapsed_ms = (time.perf_counter() - start) * 1000
            chunk = adapter.decode_action(raw, ActionContext(seq, now, time.monotonic_ns()))
            assert raw.shape == (30, 14) and np.isfinite(raw).all()
            assert chunk.action_space == "joint_position"
            np.testing.assert_array_equal(chunk.groups["left_arm"], raw[:, :7])
            np.testing.assert_array_equal(chunk.groups["right_arm"], raw[:, 7:])
            grip = raw[:, [6, 13]]
            assert np.all((grip >= 0) & (grip <= 1))
            arrays[f"actions_{index}"] = raw
            samples.append(
                {
                    "frame": index,
                    "http_worker_ms": elapsed_ms,
                    "shape": list(raw.shape),
                    "gripper_min": float(grip.min()),
                    "gripper_max": float(grip.max()),
                }
            )
    finally:
        model.close()
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output / "actions.npz", **arrays)
    report = {
        "checkpoint": health,
        "episode": str(args.episode.resolve()),
        "config": args.config,
        "samples": samples,
        "hardware_commands_sent": False,
    }
    text = json.dumps(report, indent=2)
    (args.output / "report.json").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
