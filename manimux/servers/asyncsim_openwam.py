"""Validate official assets and launch the existing XPolicyLab OpenWAM server."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from manimux.cli import read_experiment

REPO_ROOT = Path(__file__).resolve().parents[2]


def runtime_artifacts(expected):
    """Translate a verified HF recipe to XPolicyLab's file-only identity schema."""
    return {
        "checkpoint_file": expected["checkpoint_file"],
        "checkpoint_sha256": expected["checkpoint_sha256"],
        "training_config_sha256": expected["config_sha256"],
        "norm_stats_sha256": expected["norm_stats_sha256"],
    }


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def resolve_server(experiment, local, *, verify_weights=True):
    config = read_experiment(experiment, local=local)
    server = config["policy_server"]
    if (
        server["policy_name"],
        server["protocol"],
        server["env_cfg_type"],
        server["action_type"],
    ) != ("OpenWAM", "ws", "arx_x5", "ee"):
        raise ValueError("AsyncSim OpenWAM requires the XPolicyLab RoboDojo X5 EE recipe")
    if server.get("allow_dummy_policy") is not False:
        raise ValueError("Official OpenWAM deployment must load real weights")
    root = Path(server["model_dir"])
    expected = server["expected_artifacts"]
    manifest = json.loads((root / "download-manifest.json").read_text())
    if (manifest["repo_id"], manifest["revision"]) != (expected["repo_id"], expected["revision"]):
        raise ValueError("OpenWAM official checkpoint identity does not match the recipe")
    checks = {
        expected["checkpoint_file"]: expected["checkpoint_sha256"],
        "config.yaml": expected["config_sha256"],
        "normalization_stats.npy": expected["norm_stats_sha256"],
    }
    assets = {entry["path"]: entry for entry in manifest["files"]}
    for name, entry in assets.items():
        path = root / name
        if not entry.get("verified") or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"OpenWAM asset is missing or has the wrong size: {name}")
    for name, digest in checks.items():
        if assets[name]["sha256"] != digest:
            raise ValueError(f"OpenWAM manifest hash mismatch: {name}")
        if (verify_weights or name != expected["checkpoint_file"]) and _sha256(
            root / name
        ) != digest:
            raise ValueError(f"OpenWAM file hash mismatch: {name}")
    checkpoints = {path.name for path in root.glob("checkpoint_step_*.safetensors")}
    if checkpoints != {expected["checkpoint_file"]}:
        raise ValueError("OpenWAM directory must contain only the selected official checkpoint")
    if (
        server["action_horizon"] != config["policy"]["horizon_policy_steps"]
        or server["replan_steps"] != 32
    ):
        raise ValueError("OpenWAM deployment must preserve the official 32-step action horizon")
    server["ckpt_dir"] = str(root)
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    server = resolve_server(args.experiment, args.local)
    print(json.dumps(server, indent=2), flush=True)
    if args.check:
        return 0
    # resolve_server has already verified repo/revision and the actual file hashes.
    server["expected_artifacts"] = runtime_artifacts(server["expected_artifacts"])
    for path in (REPO_ROOT, REPO_ROOT / "XPolicyLab"):
        sys.path.insert(0, str(path))
    import setup_policy_server

    setup_policy_server.main(server)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
