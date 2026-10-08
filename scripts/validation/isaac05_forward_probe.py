#!/usr/bin/env python3
"""Run one model-only Isaac 0.5 XPolicy request with synthetic LIBERO-shaped inputs."""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manimux.policies.xpolicylab.ws_client import XPolicyLabWsClient  # noqa: E402
from XPolicyLab.policy.Isaac_05.model import validate_deployment  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "manimux/configs/policy/isaac05/libero/base.yaml"


def _image(height: int, width: int, offset: int) -> np.ndarray:
    horizontal = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    vertical = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    red = np.broadcast_to(horizontal, (height, width))
    green = np.broadcast_to(vertical, (height, width))
    blue = ((red.astype(np.uint16) + green.astype(np.uint16) + offset) % 256).astype(
        np.uint8
    )
    return np.ascontiguousarray(np.stack([red, green, blue], axis=-1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--server", default="ws://127.0.0.1:8504")
    parser.add_argument("--instruction", default="Pick up the object.")
    parser.add_argument("--timeout-s", type=float, default=300.0)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report = validate_deployment(config)
    if report["status"] != "ready":
        raise RuntimeError("Isaac 0.5 deployment is not ready: " + "; ".join(report["errors"]))
    proprio_dim = int(report["proprio_dim"])
    action_dim = int(report["action_dim"])
    action_horizon = int(report["action_horizon"])
    observation = {
        "vision": {
            "primary": {"color": _image(256, 256, 17)},
            "wrist": {"color": _image(256, 256, 83)},
        },
        "state": {"observation.state": np.zeros(proprio_dim, dtype=np.float32)},
        "instruction": args.instruction,
        "additional_info": {"timestep": 0},
        "data_format_version": "v1.0",
        "env_idx": 0,
    }
    session = f"isaac05-probe-{uuid.uuid4().hex[:8]}"
    client = XPolicyLabWsClient(
        url=args.server,
        evaluation_id="isaac05-offline-probe",
        trial_id=session,
        request_timeout_s=args.timeout_s,
    )
    started = time.perf_counter()
    try:
        client.connect()
        client.reset()
        raw = client.infer(observation, sampling={"mode": "default"}, timeout_s=args.timeout_s)
    finally:
        client.close()
    if isinstance(raw, Mapping) and "actions" in raw:
        raw = raw["actions"]
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raise ValueError("Isaac actions must be a sequence of per-step mappings")
    rows = []
    for index, step in enumerate(raw):
        if not isinstance(step, Mapping) or "action" not in step:
            raise ValueError(f"Isaac action step {index} must contain an 'action' vector")
        rows.append(step["action"])
    actions = np.asarray(rows, dtype=np.float64)
    if actions.shape != (action_horizon, action_dim) or not np.isfinite(actions).all():
        raise ValueError(
            "Isaac actions must be finite with shape "
            f"{(action_horizon, action_dim)}, got {actions.shape}"
        )
    print(
        json.dumps(
            {
                "status": "ok",
                "server": args.server,
                "round_trip_ms": round((time.perf_counter() - started) * 1000.0, 1),
                "shape": list(actions.shape),
                "action_semantics": "checkpoint_native_ee",
                "minimum": float(actions.min()),
                "maximum": float(actions.max()),
                "first_action": actions[0].tolist(),
                "backend": client.backend_metadata,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
