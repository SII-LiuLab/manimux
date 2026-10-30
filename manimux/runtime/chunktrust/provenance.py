"""Pinned provenance for the vendored, model-independent PI AHS numerical code."""

from __future__ import annotations

import hashlib
from pathlib import Path

UPSTREAM_URL = "https://github.com/hf618/ChunkTrust"
UPSTREAM_COMMIT = "a9fd57412b3a19a0728cd726b0558dd89395fb84"
UPSTREAM_PATH = "src/chunktrust/_pi_ahs.py"
UPSTREAM_SHA256 = "44ee901b6397c77405e073cc5909d18e869dd64f96aea7e2c17d94475481b8bd"
CORE_ID = "chunktrust-pi-ahs-a9fd574-v1"


def core_provenance() -> dict[str, object]:
    return {
        "core_id": CORE_ID,
        "upstream_url": UPSTREAM_URL,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_path": UPSTREAM_PATH,
        "sha256": UPSTREAM_SHA256,
        "licenses": ["chunktrust-MIT.txt", "chunktrust-openpi-Apache-2.0.txt"],
        "scope": "PI-family AHS only; no model, QHA, GR00T or StarVLA implementation",
    }


def verify_frozen_core() -> None:
    path = Path(__file__).with_name("_pi_ahs.py")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != UPSTREAM_SHA256:
        raise ValueError(
            f"ChunkTrust frozen core hash differs: expected {UPSTREAM_SHA256}, got {actual}"
        )
