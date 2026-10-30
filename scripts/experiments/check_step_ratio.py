"""Check a resolved experiment catalog without starting any services or hardware."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from manimux.collections import load_collection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-list", required=True, type=Path)
    parser.add_argument("--ratio", type=float, default=32 / 50)
    args = parser.parse_args()
    catalog = load_collection(args.config_list)
    for choice in catalog.choices.values():
        config = choice.config
        inference = config["inference"]
        algorithm = inference["algorithm"]
        horizon = config["policy"]["horizon_policy_steps"]
        expected = round(horizon * args.ratio)
        if algorithm in {"autohorizon", "chunktrust_ahs", "chunktrust_ahs_rtc"}:
            actual = None
        elif algorithm == "rtc":
            actual = inference["rtc"]["min_execute_policy_steps"]
        elif algorithm == "paint":
            actual = inference["paint"]["execution_policy_steps"]
        elif algorithm == "bid_backward":
            actual = inference["bid"]["replan_policy_steps"]
        elif algorithm == "serial":
            actual = inference["max_chunk_policy_steps"]
        elif algorithm == "async":
            if inference["request_trigger"] != "progress" or inference["max_chunk_policy_steps"] is not None:
                raise ValueError(f"{choice.id}: Async must use progress and retain the full horizon")
            actual = inference["chunk_policy_steps"]
        else:
            raise ValueError(f"{choice.id}: unsupported audit method {algorithm}")
        if actual is not None and actual != expected:
            raise ValueError(f"{choice.id}: expected K={expected}, got {actual}")
        if config["executor"]["type"] != "direct" or inference["blend_policy_steps"] != 0:
            raise ValueError(f"{choice.id}: expected Direct and blend=0")
        print(f"{choice.id}: H={horizon}, K={actual if actual is not None else 'adaptive'}, trigger={inference['request_trigger']}")
    print(f"PASS: {len(catalog.choices)} resolved choices; configuration checks only.")
    print("Async/RTC/PAINT source progress includes latency-trimmed rows; K is not a physical executed-row count.")


if __name__ == "__main__":
    main()
