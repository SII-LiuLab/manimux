#!/usr/bin/env python3
"""Export a resolved standalone XPolicyLab recipe; never load or start a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from manimux.cli import load_config, resolve_local_path  # noqa: E402
from manimux.runtime.chunktrust.trace import (  # noqa: E402
    TRACE_FORK_REVISION,
    TRACE_FORK_URL,
)


def export_backend(config_path: Path, *, local: Path | None, output: Path) -> dict:
    """Preserve actual resolved model options and write separate provenance."""
    config = load_config(config_path, local=resolve_local_path(config_path, local))
    if config["inference"]["algorithm"] not in {"chunktrust_ahs", "chunktrust_ahs_rtc"}:
        raise ValueError("select a ChunkTrust algorithm before exporting its backend recipe")
    if config["policy"]["worker"] != "xpolicylab_ws":
        raise ValueError("the ChunkTrust backend export requires xpolicylab_ws")
    if config["inference"]["algorithm"] == "chunktrust_ahs_rtc":
        raise ValueError(
            "the fixed public 53ac927 backend supports default trace only; "
            "no verified RTC+trace backend revision is available to export"
        )
    recipe = dict(config.get("policy_server") or {})
    if recipe.get("policy_name") != "Pi_05" or recipe.get("return_denoise_trace") is not True:
        raise ValueError("policy_server must select Pi_05 with return_denoise_trace: true")
    if "${" in str(recipe):
        raise ValueError("backend recipe contains unresolved paths; supply the station --local")
    for key in ("num_steps", "inference_seed", "action_horizon"):
        if type(recipe.get(key)) is not int:
            raise ValueError(f"policy_server.{key} must be an explicitly resolved integer")
    if recipe["num_steps"] <= 0 or not 0 <= recipe["inference_seed"] < 2**32:
        raise ValueError("invalid model num_steps or inference_seed")
    if recipe["action_horizon"] != config["policy"]["horizon_policy_steps"]:
        raise ValueError("model and runtime action horizons disagree")
    if "autohorizon" in recipe:
        raise ValueError("the native backend export cannot apply ManiMux AutoHorizon wrappers")
    output = output.expanduser().resolve()
    provenance_path = output.with_suffix(output.suffix + ".provenance.json")
    if output.exists() or provenance_path.exists():
        raise FileExistsError("refusing to overwrite an existing backend export or provenance")
    payload = yaml.safe_dump(recipe, sort_keys=False)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    provenance = {
        "schema_version": 1,
        "manimux_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "backend_source": {"url": TRACE_FORK_URL, "revision": TRACE_FORK_REVISION},
        "backend_source_status": "required_dependency_not_checked_out_or_loaded_by_exporter",
        "supported_trace_modes": ["default"],
        "config_sources_sha256": config.get("config_sources", {}),
        "runtime_algorithm": config["inference"]["algorithm"],
        "recipe_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "model_fields": {
            key: recipe[key]
            for key in (
                "policy_name",
                "num_steps",
                "inference_seed",
                "action_horizon",
                "return_denoise_trace",
            )
        },
        "validation": "configuration_only; no checkpoint, environment, GPU or hardware check",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload)
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    return {"recipe": str(output), "provenance": str(provenance_path), **provenance}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--local", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(export_backend(args.config, local=args.local, output=args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
