#!/usr/bin/env python3
"""Run one model-only Isaac 0.5 XPolicy request with synthetic LIBERO-shaped inputs."""

from __future__ import annotations

import argparse
import json
import time
import uuid

import numpy as np

from manimux.integrations.isaac05 import (
    Isaac05BaseContract,
    build_wire_observation,
    decode_wire_actions,
)
from manimux.integrations.xpolicylab.ws_client import XPolicyLabWsClient


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
    parser.add_argument("--server", default="ws://127.0.0.1:8504")
    parser.add_argument("--instruction", default="Pick up the object.")
    parser.add_argument("--timeout-s", type=float, default=300.0)
    args = parser.parse_args()

    contract = Isaac05BaseContract()
    observation = build_wire_observation(
        primary_image=_image(256, 256, 17),
        wrist_image=_image(256, 256, 83),
        state=np.zeros(contract.proprio_dim, dtype=np.float32),
        instruction=args.instruction,
    )
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
    actions = decode_wire_actions(raw)
    print(
        json.dumps(
            {
                "status": "ok",
                "server": args.server,
                "round_trip_ms": round((time.perf_counter() - started) * 1000.0, 1),
                "shape": list(actions.shape),
                "action_semantics": contract.action_semantics,
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
