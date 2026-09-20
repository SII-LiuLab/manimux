#!/usr/bin/env python3
"""Run one hardware-free SAPolicy episode through the ManiMux runtime.

Starts the dry-run XPolicyLab server (unless --no-server), then executes
``configs/sapolicy/yam/infra/mock.yaml``. This machine may not have mink/i2rt;
in that case IK holds the seed joints, which is valid for dry-run hold actions.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INFRA = REPO_ROOT / "configs/sapolicy/yam/infra/mock.yaml"
DEFAULT_SERVER = REPO_ROOT / "configs/sapolicy/yam/server/mock.yaml"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8500


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def _install_hold_ik_if_needed() -> str:
    try:
        import i2rt  # noqa: F401
        import mink  # noqa: F401
    except ImportError:
        from manimux.embodiments.arm.yam.kinematics import YamKinematics

        def _hold_ik(self, target_pose, init_joints, gripper):
            del self, target_pose, gripper
            return True, np.asarray(init_joints, dtype=np.float64).copy()

        YamKinematics.ik = _hold_ik  # type: ignore[method-assign]
        return "hold-seed (mink/i2rt unavailable)"
    return "yam mink"


def _start_server(config: Path) -> subprocess.Popen:
    env = os.environ.copy()
    pythonpath = os.pathsep.join(
        [str(REPO_ROOT), str(REPO_ROOT / "XPolicyLab"), env.get("PYTHONPATH", "")]
    )
    env["PYTHONPATH"] = pythonpath
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "scripts/servers/sapolicy_yam_server.py"), "--config", str(config)],
        cwd=str(REPO_ROOT),
        env=env,
    )


def _wait_for_server(proc: subprocess.Popen, host: str, port: int, timeout_s: float) -> None:
    # Avoid raw TCP probes against the WS port: they log InvalidMessage on the server.
    deadline = time.monotonic() + timeout_s
    time.sleep(0.4)
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"SAPolicy server exited with code {proc.returncode}")
        if _port_open(host, port):
            return
        time.sleep(0.2)
    raise TimeoutError(f"SAPolicy server did not listen on {host}:{port} within {timeout_s}s")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_INFRA)
    parser.add_argument("--server-config", type=Path, default=DEFAULT_SERVER)
    parser.add_argument("--no-server", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.path.insert(0, str(REPO_ROOT))

    ik_mode = _install_hold_ik_if_needed()
    print(f"[sapolicy-mock] IK mode: {ik_mode}", flush=True)

    server: subprocess.Popen | None = None
    if not args.no_server:
        if _port_open(DEFAULT_HOST, DEFAULT_PORT):
            print(f"[sapolicy-mock] reusing existing server on {DEFAULT_HOST}:{DEFAULT_PORT}", flush=True)
        else:
            server = _start_server(args.server_config.expanduser().resolve())
            _wait_for_server(server, DEFAULT_HOST, DEFAULT_PORT, timeout_s=20.0)

    from manimux.cli import _run

    try:
        return _run(args.config.expanduser().resolve())
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()


if __name__ == "__main__":
    raise SystemExit(main())
