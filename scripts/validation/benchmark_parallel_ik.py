"""Compare serial and process-parallel full IK on saved SAPolicy predictions; no robot I/O."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from manimux.config import load_config
from manimux.integrations.sapolicy_yam.policy_plugin import SAPolicyYamAdapter
from manimux.policies.decoder import ActionDecoderClient
from manimux.types import ActionContext, InferenceResponse, RobotState


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recorded_diagnostics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    cfg = load_config("configs/sapolicy/yam/infra/manimux-direct-async.yaml")
    source = args.recorded_diagnostics
    predictions = np.load(source / "offline-ik-profile.npz")
    signals = np.load(source / "signals.npz")
    profile = json.loads((source / "offline-ik-profile.json").read_text())
    # Historical fixtures can have a shorter horizon than the live deployment.
    cfg.policy.horizon_steps = int(predictions["1_wire"].shape[0])
    adapter = SAPolicyYamAdapter(cfg.robot, cfg.policy)
    decoder = ActionDecoderClient(cfg.robot, cfg.policy, adapter)
    adapter.warmup_decode(None)
    rows = []
    seq = 0
    try:
        decoder.start()
        for index, entry in enumerate(profile, 1):
            tick = int(np.argmin(abs(signals["time_s"] - entry["seconds"])))
            groups = {name: signals[name + "_state"][tick].copy() for name in cfg.robot.group_dims}
            raw = predictions[f"{index}_wire"]
            serial_ms = []
            parallel_ms = []
            max_difference = 0.0
            for _ in range(args.repeats):
                seq += 1
                now = time.monotonic_ns()
                context = ActionContext(seq, now, now, measured_state=RobotState(groups, now, seq))
                before = time.perf_counter_ns()
                serial = adapter.decode_action(raw, context)
                serial_ms.append((time.perf_counter_ns() - before) / 1e6)
                response = InferenceResponse("offline", seq, now, 0.0, raw, observation_time_ns=now)
                before = time.perf_counter_ns()
                decoder.submit(response, context, time.monotonic_ns() + 10_000_000_000)
                result = None
                while result is None:
                    result = decoder.poll()
                    if result is None:
                        time.sleep(0.0005)
                parallel_ms.append((time.perf_counter_ns() - before) / 1e6)
                if result.error:
                    raise RuntimeError(result.error)
                for name in groups:
                    difference = np.max(np.abs(serial.groups[name] - result.chunk.groups[name]))
                    max_difference = max(max_difference, float(difference))
                    np.testing.assert_allclose(
                        serial.groups[name], result.chunk.groups[name], rtol=0, atol=1e-10
                    )
                    assert (
                        serial.metadata["ik"][name]["converged"]
                        == result.chunk.metadata["ik"][name]["converged"]
                    )
            rows.append(
                dict(
                    seconds=entry["seconds"],
                    serial_ms=serial_ms,
                    parallel_including_ipc_ms=parallel_ms,
                    serial_median_ms=float(np.median(serial_ms)),
                    parallel_median_ms=float(np.median(parallel_ms)),
                    max_joint_difference_rad=max_difference,
                    failed_steps={
                        name: serial.metadata["ik"][name]["failed_steps"] for name in groups
                    },
                )
            )
            print(json.dumps(rows[-1]), flush=True)
    finally:
        decoder.close()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "scope": "offline saved predictions; no hardware or model calls",
                "repeats": args.repeats,
                "samples": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
