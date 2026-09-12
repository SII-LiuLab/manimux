"""Launch the copied YAM collection interface with ManiMux execution."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import yaml

COLLECTORS = {"yam": "manimux.collection.yam.cli"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", "--station", dest="station", default="configs/collection/yam/station.yaml"
    )
    parser.add_argument("--cameras")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8043)
    args = parser.parse_args(argv)
    with Path(args.station).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    collector = config.get("collector") if isinstance(config, dict) else None
    if collector not in COLLECTORS:
        parser.error(f"config.collector must select one of {sorted(COLLECTORS)}")
    importlib.import_module(COLLECTORS[collector]).run_gui(args)


if __name__ == "__main__":
    main()
