#!/usr/bin/env python3
"""Audit the local LingBot-VLA2 foundation checkpoint and official V2 source."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints/pretrained/lingbot-vla-v2-6b"
DEFAULT_SOURCE = REPO_ROOT / "XPolicyLab/policy/LingBot_VLA2/lingbot_vla_v2"
DEFAULT_SERVER_CONFIG = REPO_ROOT / "manimux/configs/policy/lingbot-vla2/yam/base.yaml"


def _tensor_shape(checkpoint: Path, tensor_name: str) -> list[int]:
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    shard = checkpoint / index["weight_map"][tensor_name]
    with shard.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    return list(header[tensor_name]["shape"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--server-config", type=Path, default=DEFAULT_SERVER_CONFIG)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    for path in (REPO_ROOT, REPO_ROOT / "XPolicyLab"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from XPolicyLab.policy.LingBot_VLA2.model import validate_deployment

    server_config = yaml.safe_load(args.server_config.read_text(encoding="utf-8"))
    deployment = validate_deployment(server_config)

    contract = {
        "checkpoint": str(checkpoint),
        "model_family": json.loads((checkpoint / "config.json").read_text()).get(
            "vlm_family"
        ),
        "state_projection_shape": _tensor_shape(checkpoint, "model.state_proj.weight"),
        "action_projection_shape": _tensor_shape(
            checkpoint, "model.action_out_proj.weight"
        ),
        "official_repository": "https://github.com/Robbyant/lingbot-vla-v2",
        "official_source_present": (
            args.source_root / "deploy/lingbot_vla_v2_policy.py"
        ).is_file(),
        "xpolicylab_adapter": "XPolicyLab/policy/LingBot_VLA2",
        "yam_norm_stats": deployment["norm_stats_path"],
        "yam_action_mapping": "absolute 12 arm joints + 2 grippers",
        "deployment_status": deployment["status"],
        "status": "ready" if deployment["status"] == "ready" else "blocked",
        "reason": (
            "The deployment paths are structurally ready for a base-checkpoint "
            "forward. Its YAM statistics are not matched post-training statistics, so "
            "task capability remains unverified."
            if deployment["status"] == "ready"
            else "; ".join(deployment["errors"])
        ),
    }
    print(json.dumps(contract, indent=2))
    return 0 if deployment["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
