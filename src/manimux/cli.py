from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import signal
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from manimux.config import ManiMuxConfig, load_config
from manimux.runtime.lock import RuntimeInstanceLock, RuntimeLockError
from manimux.session import RuntimeSessionService


def _handle_termination(_signum: int, _frame: object) -> None:
    """Route SIGTERM through the same orderly shutdown path as Ctrl-C."""

    raise KeyboardInterrupt


def _git_sha(workdir: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=workdir,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_config(config_path: Path, executor: str | None = None) -> ManiMuxConfig:
    config = load_config(config_path)
    if executor is not None:
        if executor not in {"direct", "smooth", "mpc"}:
            raise ValueError(f"unsupported executor override {executor!r}")
        config.execution.executor = cast(Literal["smooth", "mpc"], executor)
    return config


def _create_run_dir(config: ManiMuxConfig, config_path: Path, *, mode: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"session-{timestamp}-{uuid.uuid4().hex[:8]}"
    run_dir = config.run.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    resolved_config = config_path.expanduser().resolve()
    config_sha256 = hashlib.sha256(resolved_config.read_bytes()).hexdigest()
    repository_root = Path(__file__).resolve().parents[2]
    with (run_dir / "session-manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "session_id": run_id,
                "mode": mode,
                "created_at": datetime.now(UTC).isoformat(),
                "git_sha": _git_sha(repository_root),
                "xpolicylab_git_sha": _git_sha(repository_root / "XPolicyLab"),
                "config_path": str(resolved_config),
                "config_sha256": config_sha256,
                "config": config.model_dump(mode="json"),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    return run_dir


def _serve(config_path: Path, executor: str | None = None) -> int:
    config = _load_config(config_path, executor)
    try:
        with _runtime_lock(config, config_path, mode="serve"):
            run_dir = _create_run_dir(config, config_path, mode="serve")
            RuntimeSessionService(config, run_dir).serve()
    except KeyboardInterrupt:
        print("runtime service stopped by operator")
        return 130
    except RuntimeLockError as exc:
        print(f"runtime service not started: {exc}")
        return 2
    return 0


def _runtime_lock(
    config: ManiMuxConfig,
    config_path: Path,
    *,
    mode: str,
) -> RuntimeInstanceLock:
    identity = config.viewer.robot_adapter.strip() or config.robot.driver
    return RuntimeInstanceLock(identity, mode=mode, config_path=config_path)


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--executor", choices=("direct", "smooth", "mpc"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="manimux")
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve_parser = subparsers.add_parser(
        "serve", help="keep one runtime service available for Viewer-controlled rollouts"
    )
    _add_runtime_arguments(serve_parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    mp.freeze_support()
    signal.signal(signal.SIGTERM, _handle_termination)
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        return _serve(args.config, args.executor)
    raise AssertionError(f"unhandled command {args.command}")
