"""Lifecycle of locally owned policy-server processes, independent of model code."""

from __future__ import annotations

import math
import os
import signal
import socket
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class PolicyServiceSpec:
    command: tuple[str, ...]
    cwd: Path
    env: dict[str, str | None]
    startup_timeout_s: float = 180.0
    stop_timeout_s: float = 30.0


def read_policy_services(raw: object, directory: Path) -> dict[str, PolicyServiceSpec]:
    if not isinstance(raw, dict):
        raise ValueError("policy_services must be a mapping keyed by policy group")
    result = {}
    for name, spec in raw.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(spec, dict):
            raise ValueError("Each policy service requires a name and a mapping")
        if set(spec) - {"command", "cwd", "env", "startup_timeout_s", "stop_timeout_s"}:
            raise ValueError(f"Unknown policy service fields for {name}")
        command = spec.get("command")
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(arg, str) or not arg for arg in command)
        ):
            raise ValueError(f"policy_services.{name}.command must be a nonempty argument list")
        cwd = spec.get("cwd", ".")
        if not isinstance(cwd, str):
            raise ValueError(f"policy_services.{name}.cwd must be a path")
        cwd = (directory / Path(cwd).expanduser()).resolve()
        if not cwd.is_dir():
            raise ValueError(f"Policy working directory does not exist: {cwd}")
        env = spec.get("env", {})
        if not isinstance(env, dict) or any(
            not isinstance(key, str)
            or not key
            or "=" in key
            or "\0" in key
            or (value is not None and (not isinstance(value, str) or "\0" in value))
            for key, value in env.items()
        ):
            raise ValueError(
                f"policy_services.{name}.env requires string keys and string/null values"
            )
        timeouts = {}
        for key, default in [("startup_timeout_s", 180.0), ("stop_timeout_s", 30.0)]:
            value = spec.get(key, default)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"policy_services.{name}.{key} must be positive and finite")
            timeouts[key] = float(value)
        result[name] = PolicyServiceSpec(tuple(command), cwd, dict(env), **timeouts)
    return result


def service_address(config: dict) -> tuple[str, int]:
    endpoint = config["policy"]["options"].get("server", "")
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"ws", "wss", "http", "https", "tcp"} or parsed.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise ValueError(
            "Managed policies require an explicit local policy.options.server endpoint"
        )
    port = parsed.port or {"ws": 80, "wss": 443, "http": 80, "https": 443}.get(parsed.scheme)
    if port is None:
        raise ValueError("Managed policy endpoint requires a port")
    return parsed.hostname, port


class ManagedPolicyService:
    """Pollable stop/start state machine. Never adopts or kills an external process."""

    def __init__(self, specs: dict[str, PolicyServiceSpec]):
        self.specs = specs
        self.process: subprocess.Popen | None = None
        self.active_policy = ""
        self.target = None
        self.status = "idle"
        self.error = ""
        self.log_path: Path | None = None
        self._deadline = 0.0
        self._after_stop = "idle"
        self._kill_sent = False

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    def metadata(self) -> dict:
        return dict(
            status=self.status,
            policy=self.target[0].policy if self.target else "",
            error=self.error,
            log_path=str(self.log_path or ""),
        )

    @staticmethod
    def _listening(address: tuple[str, int]) -> bool:
        try:
            with socket.create_connection(address, timeout=0.05):
                return True
        except OSError:
            return False

    def _signal(self, sig: int) -> None:
        if self.process is not None:
            with suppress(ProcessLookupError):
                os.killpg(self.process.pid, sig)

    def request(self, choice, run_dir: Path) -> None:
        same = self.target is not None and self.target[0].policy == choice.policy
        self.target = (choice, run_dir)
        if same and self.status in {"starting", "ready"}:
            return
        self.error = ""
        if self.process is not None:
            if self.status == "stopping":
                self._after_stop = "start"
            else:
                self._stop("start")
        else:
            self._start()

    def _stop(self, after: str) -> None:
        self._after_stop = after
        self.status = "stopping"
        self._kill_sent = False
        self._deadline = time.monotonic() + self.specs[self.active_policy].stop_timeout_s
        self._signal(signal.SIGTERM)

    def _start(self) -> None:
        choice, run_dir = self.target
        spec = self.specs[choice.policy]
        self.active_policy = choice.policy
        self.log_path = run_dir / "policy-server.log"
        try:
            address = service_address(choice.config)
            if self._listening(address):
                raise RuntimeError(
                    f"Policy endpoint {address[0]}:{address[1]} is already in use; "
                    "stop its external service first"
                )
            command = [
                arg.replace("{config}", str(choice.path)).replace(
                    "{local}", str(choice.config.get("local") or "")
                )
                for arg in spec.command
            ]
            env = os.environ.copy()
            for key, value in spec.env.items():
                if value is None:
                    env.pop(key, None)
                else:
                    env[key] = value
            run_dir.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("ab") as log:
                self.process = subprocess.Popen(
                    command,
                    cwd=spec.cwd,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            self.status = "starting"
            self._deadline = time.monotonic() + spec.startup_timeout_s
        except (OSError, ValueError, RuntimeError) as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.status = "failed"

    def poll(self) -> None:
        if self.process is None:
            return
        code = self.process.poll()
        if self.status == "stopping":
            if code is not None:
                # Reap descendants that outlived the owned launcher before loading another model.
                self._signal(signal.SIGKILL)
                self.process = None
                self.status = "failed" if self.error else "idle"
                if self._after_stop == "start":
                    self._start()
            elif time.monotonic() >= self._deadline and not self._kill_sent:
                self._signal(signal.SIGKILL)
                self._kill_sent = True
            return
        if code is not None:
            self._signal(signal.SIGKILL)
            self.process = None
            self.error = f"Policy server exited with code {code}; see {self.log_path}"
            self.status = "failed"
        elif self.status == "starting":
            if self._listening(service_address(self.target[0].config)):
                self.status = "ready"
            elif time.monotonic() >= self._deadline:
                self.error = f"Policy startup timed out; see {self.log_path}"
                self._stop("failed")

    def close(self) -> None:
        if self.process is not None:
            self._signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=self.specs[self.active_policy].stop_timeout_s)
            except subprocess.TimeoutExpired:
                self._signal(signal.SIGKILL)
                self.process.wait()
            finally:
                self._signal(signal.SIGKILL)
                self.process = None
        self.status = "idle"
