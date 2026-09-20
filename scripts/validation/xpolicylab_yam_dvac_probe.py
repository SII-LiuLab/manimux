#!/usr/bin/env python3
"""Exercise Pi05 DVAC repeatedly in one hardware-free XPolicy session."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from xpolicylab_yam_forward_probe import _snapshot

from manimux.cli import load_config
from manimux.policies import build_policy_model
from manimux.policy_adapter import build_policy_adapter
from manimux.runtime.dvac import DvacInferenceRequest
from manimux.types import ActionContext


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--instruction", default="Pick the red block up.")
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()
    if args.requests < 2:
        raise ValueError("DVAC rolling probe requires at least two requests")
    if args.height <= 0 or args.width <= 0:
        raise ValueError("height and width must be positive")

    config_path = args.config.resolve()
    config = load_config(config_path)
    if config["policy"]["worker"] != "xpolicylab_ws":
        raise ValueError("DVAC probe requires policy.worker: xpolicylab_ws")
    if config["inference"]["algorithm"] != "dvac":
        raise ValueError("DVAC probe requires inference.algorithm: dvac")

    settings = config["inference"]["dvac"]
    maximum = settings["max_execution_steps"] or config["policy"]["horizon_steps"]
    group_order = list(config["policy"]["adapter"]["group_order"])
    expected_width = sum(int(config["robot"]["group_dims"][name]) for name in group_order)
    session_id = f"xpolicy-dvac-probe-{uuid.uuid4().hex[:8]}"
    model = build_policy_model(config["policy"])
    adapter = build_policy_adapter(config["robot"], config["policy"])
    reports: list[dict[str, object]] = []
    try:
        model.reset(session_id)
        for request_seq in range(1, args.requests + 1):
            snapshot = _snapshot(config_path, args.height, args.width)
            observation_time_ns = snapshot.state.monotonic_ns
            request = DvacInferenceRequest(
                session_id=session_id,
                request_seq=request_seq,
                observation_time_ns=observation_time_ns,
                deadline_ns=observation_time_ns
                + int(config["policy"]["timeout_s"] * 1_000_000_000),
                observation=snapshot,
                instruction=args.instruction,
                dvac_tail_steps=settings["tail_steps"],
                dvac_alpha=settings["alpha"],
                dvac_rolling_window_size=settings["rolling_window_size"],
                dvac_min_execution_steps=settings["min_execution_steps"],
                dvac_max_execution_steps=maximum,
            )
            started = time.perf_counter()
            raw = model.infer(request)
            round_trip_ms = (time.perf_counter() - started) * 1000.0
            metadata = raw.get("dvac") if isinstance(raw, Mapping) else None
            if not isinstance(metadata, Mapping):
                raise ValueError("DVAC reply is missing metadata")
            variance = np.asarray(metadata.get("variance"), dtype=np.float64)
            if (
                variance.shape != (config["policy"]["horizon_steps"],)
                or not np.isfinite(variance).all()
                or np.any(variance < 0)
            ):
                raise ValueError(
                    "DVAC variance must contain one finite non-negative value per action step"
                )
            chunk = adapter.decode_action(
                raw,
                ActionContext(
                    request_seq=request_seq,
                    observation_time_ns=observation_time_ns,
                    created_time_ns=time.monotonic_ns(),
                ),
            )
            packed = np.concatenate([chunk.groups[name] for name in group_order], axis=1)
            expected_shape = (config["policy"]["horizon_steps"], expected_width)
            if packed.shape != expected_shape or not np.isfinite(packed).all():
                raise ValueError(
                    f"DVAC adapter must return a finite {expected_shape} chunk, got {packed.shape}"
                )
            reports.append(
                {
                    "request_seq": request_seq,
                    "round_trip_ms": round(round_trip_ms, 1),
                    "execution_steps": metadata.get("execution_steps"),
                    "first_threshold_crossing": metadata.get("first_threshold_crossing"),
                    "threshold": metadata.get("threshold"),
                    "total_variance": metadata.get("total_variance"),
                    "rolling_states": metadata.get("rolling_states"),
                    "cold_start": metadata.get("cold_start"),
                    "tail_steps": metadata.get("tail_steps"),
                    "action_dim": metadata.get("action_dim"),
                    "variance_space": metadata.get("variance_space"),
                }
            )
    finally:
        model.close()

    if reports[0].get("cold_start") is not True:
        raise ValueError("the first DVAC request must report cold_start=true")
    if any(report.get("cold_start") is not False for report in reports[1:]):
        raise ValueError("later DVAC requests must reuse the prior rolling window")
    expected_rolling_states = [
        min(index, settings["rolling_window_size"]) for index in range(1, args.requests + 1)
    ]
    rolling_states = [report.get("rolling_states") for report in reports]
    if rolling_states != expected_rolling_states:
        raise ValueError(
            f"DVAC rolling states must be {expected_rolling_states}, got {rolling_states}"
        )

    print(
        json.dumps(
            {
                "status": "ok",
                "config": str(config_path),
                "server": config["policy"]["options"]["server"],
                "session_id": session_id,
                "requests": reports,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
