#!/usr/bin/env python3
"""Run one mock or Meshcat validation episode without a Viewer or physical devices."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import signal
import sys
from pathlib import Path


def run_headless(config_path: Path, executor: str | None = None) -> int:
    from manimux.cli import _create_run_dir, _load_config, _runtime_lock
    from manimux.runtime import build_runtime
    from manimux.runtime.lock import RuntimeLockError

    config = _load_config(config_path, executor)
    if config.viewer.enabled:
        raise ValueError(
            "headless validation requires viewer.enabled=false; "
            "use manimux serve for Viewer rollouts"
        )
    if config.robot.driver not in {"mock_dual_arm", "maniunicon_meshcat_dual_arm"}:
        raise ValueError(
            "headless validation supports only mock_dual_arm or maniunicon_meshcat_dual_arm"
        )
    if any(sensor.driver != "mock_camera" for sensor in config.sensors):
        raise ValueError("headless validation supports only mock_camera sensors")
    try:
        with _runtime_lock(config, config_path, mode="headless"):
            run_dir = _create_run_dir(config, config_path, mode="headless")
            result = build_runtime(config, run_dir, launch_mode="headless").run()
    except KeyboardInterrupt:
        print("validation interrupted; runtime shutdown and partial episode save completed")
        return 130
    except RuntimeLockError as exc:
        print(f"validation not started: {exc}")
        return 2
    status = "completed" if result.success else "FAULT"
    print(
        f"{status} {result.steps} steps; reason={result.terminal_reason}; "
        f"accepted={result.accepted_plans} rejected={result.rejected_plans}; "
        f"episode={result.episode_dir}"
    )
    return 0 if result.success else 2


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    from manimux.cli import _handle_termination

    mp.freeze_support()
    signal.signal(signal.SIGTERM, _handle_termination)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=repo_root / "configs/mock.yaml")
    parser.add_argument("--executor", choices=("direct", "smooth", "mpc"))
    args = parser.parse_args()
    return run_headless(args.config.expanduser().resolve(), args.executor)


if __name__ == "__main__":
    raise SystemExit(main())
