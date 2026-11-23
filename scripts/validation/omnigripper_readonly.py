"""Read both OmniGripper feedback channels without connecting robot arms."""

import argparse
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from manimux.cli import load_config, resolve_local_path
from manimux.clock import SystemClock
from manimux.embodiments.robot import build_robot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--local", default=None)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--hz", type=float, default=100.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 <= args.duration <= 60 or not 0 < args.hz <= 100:
        raise ValueError("duration must be 0..60 seconds and hz in (0, 100]")
    config = load_config(args.config, local=resolve_local_path(args.config, args.local))
    if config["robot"].get("options", {}).get("execute", False) is not False:
        raise ValueError("read-only probe requires execute: false")
    robot = build_robot(config["robot"], SystemClock())
    def probe(tool):
        try:
            tool.connect()
            samples = []
            started = time.monotonic()
            deadline = started
            while True:
                state = tool.get_state()
                samples.append({"opening": state.opening, **tool.diagnostics["feedback"]})
                if time.monotonic() - started >= args.duration:
                    break
                deadline += 1 / args.hz
                time.sleep(max(0, deadline - time.monotonic()))
            return {**tool.diagnostics, "sample_count": len(samples),
                    "elapsed_s": time.monotonic() - started,
                    "opening": state.opening, "samples": samples}
        finally:
            tool.close()
    with ThreadPoolExecutor(max_workers=len(robot.end_effectors)) as pool:
        pending = {name: pool.submit(probe, tool) for name, tool in robot.end_effectors.items()}
        output = {name: result.result() for name, result in pending.items()}
    report = {"validation": "read-only grippers only; no arm connection", "groups": output}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    summary = {name: {k: v for k, v in data.items() if k != "samples"}
               for name, data in output.items()}
    print(json.dumps({"validation": report["validation"], "groups": summary}, indent=2))


if __name__ == "__main__":
    main()
