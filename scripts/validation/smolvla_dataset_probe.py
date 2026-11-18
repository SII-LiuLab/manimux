"""Run a LeRobotDataset sample through the real worker and joint decoder offline."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from manimux.policies import PolicyWorkerClient
from manimux.policy_adapter.joint import JointAdapter
from manimux.types import (
    ActionContext,
    InferenceRequest,
    ObservationSnapshot,
    RobotState,
    SensorFrame,
)


def wait_for(poll, timeout):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        result = poll()
        if result is not None:
            if result.error:
                raise RuntimeError(result.error)
            return result
        time.sleep(0.02)
    raise TimeoutError("policy worker did not respond")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--policy", type=Path, default=Path("integrations/lerobot/smolvla_base_offline.yaml")
    )
    parser.add_argument("--row", type=int, default=0)
    args = parser.parse_args()
    config = yaml.safe_load(args.policy.read_text())
    config["options"]["checkpoint"] = str(args.checkpoint.resolve(strict=True))
    row = LeRobotDataset("manimux/episode", root=args.dataset, video_backend="pyav")[args.row]
    group = config["options"]["group"]
    values = row["observation.state"].numpy()
    adapter = JointAdapter({"group_dims": {group: len(values)}}, config)
    worker = PolicyWorkerClient(config, "startup")
    try:
        worker.start()
        worker.submit_reset("probe")
        wait_for(worker.poll_reset, config["timeout_s"])
        now = time.monotonic_ns()
        frames = {}
        for feature, camera in config["adapter"]["camera_map"].items():
            rgb = row[feature].permute(1, 2, 0).mul(255).round().byte().numpy()
            frames[camera] = SensorFrame(camera, rgb, now, 0)
        obs = adapter.build_observation(
            ObservationSnapshot(RobotState({group: values}, now, 0), frames)
        )
        request = InferenceRequest(
            "probe", 0, now, now + int(config["timeout_s"] * 1e9), obs, row["task"]
        )
        worker.submit_latest(adapter.prepare_request(request))
        reply = wait_for(worker.poll, config["timeout_s"])
        chunk = adapter.decode_action(
            reply.raw_action, ActionContext(0, now, reply.finished_time_ns)
        )
        actions = chunk.groups[group]
        assert actions.shape == (config["horizon_policy_steps"], len(values))
        assert np.isfinite(actions).all()
        print(
            json.dumps(
                {
                    "backend": worker.capabilities.backend_metadata,
                    "row": args.row,
                    "shape": list(actions.shape),
                    "inference_ms": reply.inference_ms,
                }
            )
        )
    finally:
        worker.close()


if __name__ == "__main__":
    main()
