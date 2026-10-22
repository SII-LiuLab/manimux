"""Owned service processes, durable state and graceful shutdown (Linux + tmux)."""

import ctypes
import fcntl
import json
import os
import platform
import shlex
import signal
import subprocess
import sys
import time
import uuid
import webbrowser
from contextlib import contextmanager, suppress
from pathlib import Path

from .plan import doctor, environment, verify_sources

TMUX_SOCKET = f"manimux-deployment-{os.getuid()}"


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None


@contextmanager
def locked(directory):
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (directory / "operation.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another deployment operation is in progress; use status to inspect it"
            ) from exc
        yield


def process_identity(pid):
    """Birth time and boot ID guard against signalling a recycled PID."""
    try:
        tail = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if tail[0] == "Z":
            return None
        return {
            "pid": pid,
            "start_ticks": tail[19],
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        }
    except (FileNotFoundError, ProcessLookupError):
        return None


def alive(service):
    identity = service.get("identity")
    return identity is not None and process_identity(identity["pid"]) == identity


def signal_owned(service):
    """Signal the same kernel process we inspected, even if its PID is recycled."""
    with suppress(ProcessLookupError):
        fd = open_pidfd(service["identity"]["pid"])
        try:
            if alive(service):
                send_pidfd(fd, signal.SIGTERM)
        finally:
            os.close(fd)


def _linux_syscall(number, *args):
    # Some portable CPython builds omit pidfd wrappers. Linux x86-64 and arm64
    # share these syscall numbers (asm/unistd_64.h and asm-generic/unistd.h).
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "aarch64"}:
        raise RuntimeError("Managed process signalling requires Linux x86-64/arm64 pidfds")
    library = ctypes.CDLL(None, use_errno=True)
    library.syscall.restype = ctypes.c_long
    result = library.syscall(ctypes.c_long(number), *(ctypes.c_long(a) for a in args))
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def open_pidfd(pid):
    return os.pidfd_open(pid) if hasattr(os, "pidfd_open") else _linux_syscall(434, pid, 0)


def send_pidfd(fd, signum):
    if hasattr(signal, "pidfd_send_signal"):
        signal.pidfd_send_signal(fd, signum)
    else:
        _linux_syscall(424, fd, signum, 0, 0)


def owns_ports(service, ports):
    if not ports:
        return True
    descriptors = Path(f"/proc/{service['identity']['pid']}/fd")
    inodes = set()
    for entry in descriptors.iterdir():
        with suppress(FileNotFoundError):
            target = os.readlink(entry)
            if target.startswith("socket:["):
                inodes.add(target[8:-1])
    listening = set()
    for filename in ("/proc/net/tcp", "/proc/net/tcp6"):
        for line in Path(filename).read_text().splitlines()[1:]:
            fields = line.split()
            if fields[3] == "0A" and fields[9] in inodes:
                listening.add(int(fields[1].rsplit(":", 1)[1], 16))
    return all(port in listening for _, port in ports)


def tmux(*args, socket_name=TMUX_SOCKET):
    return subprocess.run(
        ["tmux", "-L", socket_name, "-f", "/dev/null", *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    ).stdout.strip()


def attach_command(service):
    return shlex.join(
        [
            "tmux",
            "-L",
            service.get("tmux_socket", TMUX_SOCKET),
            "attach-session",
            "-t",
            service["session"],
        ]
    )


def _spawn(name, plan, state, directory):
    run_dir = Path(state["run_dir"])
    session = f"mx-{state['id']}-{name}"
    # Fresh server inherits this invocation's CUDA/SDK environment, not an older
    # deployment's cached tmux environment.
    socket_name = f"manimux-{os.getuid()}-{state['id']}"
    gate = run_dir / f"{name}.go"
    launch = run_dir / f"{name}.launch.json"
    log = run_dir / f"{name}.log"
    env = environment()
    # An inherited reporter path must never let another service impersonate runtime.
    env.pop("MANIMUX_DEPLOYMENT_STATUS", None)
    if name == "runtime":
        env["MANIMUX_DEPLOYMENT_STATUS"] = str(run_dir / "runtime-status.json")
    atomic_json(
        launch,
        {"command": plan["services"][name]["command"], "gate": str(gate), "cwd": plan["cwd"]},
    )
    # The bootstrap waits until ownership and logging are durable, then execs the
    # existing entry point. stdin remains a PTY for operator-only RELEASE gates.
    command = shlex.join(
        [
            "env",
            "-u",
            "MANIMUX_DEPLOYMENT_STATUS",
            "PYTHONUNBUFFERED=1",
            f"PYTHONPATH={env['PYTHONPATH']}",
            *(
                [f"MANIMUX_DEPLOYMENT_STATUS={env['MANIMUX_DEPLOYMENT_STATUS']}"]
                if name == "runtime"
                else []
            ),
            plan["services"][name]["command"][0],
            "-m",
            "manimux.deployment.worker",
            str(launch),
        ]
    )
    pid = int(
        tmux(
            "new-session",
            "-d",
            "-s",
            session,
            "-c",
            plan["cwd"],
            "-P",
            "-F",
            "#{pane_pid}",
            f"exec {command}",
            socket_name=socket_name,
        )
    )
    service = {
        "session": session,
        "tmux_socket": socket_name,
        "identity": process_identity(pid),
        "log": str(log),
        "state": "starting",
        "detail": "Waiting for service readiness",
    }
    state["services"][name] = service
    atomic_json(directory / "state.json", state)
    if service["identity"] is None:
        raise RuntimeError(f"{name} terminal exited during launch")
    tmux(
        "pipe-pane", "-o", "-t", session, "cat >> " + shlex.quote(str(log)), socket_name=socket_name
    )
    gate.touch()
    return service


def _probe(name, plan, state):
    service = state["services"][name]
    external = service.get("ownership") == "external"
    if not external and not alive(service):
        return {"ok": False, "state": "exited", "detail": f"Process exited; see {service['log']}"}
    try:
        if not external and not owns_ports(service, plan["services"][name]["ports"]):
            return {
                "ok": False,
                "state": "starting",
                "detail": "Waiting for this process to own its configured listeners",
            }
        if name == "runtime":
            value = read_json(Path(state["run_dir"]) / "runtime-status.json")
            if value is None or value.get("pid") != service["identity"]["pid"]:
                return {
                    "ok": False,
                    "state": "starting",
                    "detail": "Awaiting runtime session heartbeat",
                }
            phase = value["phase"]
            if phase == "waiting_for_prepare" and time.time() - value["updated_at"] > 3:
                return {
                    "ok": False,
                    "state": "unresponsive",
                    "detail": "Runtime idle heartbeat expired",
                }
            return {
                "ok": True,
                "state": phase,
                "detail": value.get("detail", ""),
                "run_dir": value.get("run_dir"),
                "last_episode_dir": value.get("last_episode_dir"),
            }
        result = subprocess.run(
            [
                plan["services"][name]["probe_python"],
                "-m",
                "manimux.deployment.probe",
                name,
                str(Path(state["run_dir"]) / "plan.json"),
            ],
            cwd=plan["cwd"],
            env=environment(),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode:
            return {"ok": False, "state": "unresponsive", "detail": result.stderr.strip()[-1800:]}
        return json.loads(result.stdout)
    except Exception as exc:
        return {"ok": False, "state": "unresponsive", "detail": str(exc)}


def _snapshot(directory, state, plan):
    services = {}
    for name, service in state["services"].items():
        services[name] = {
            **service,
            **_probe(name, plan, state),
            **(
                {}
                if service.get("ownership") == "external"
                else {"attach_command": attach_command(service)}
            ),
        }
    ok = (
        state["phase"] == "ready"
        and len(services) == len(plan["services"])
        and all(s["ok"] for s in services.values())
    )
    phase = (
        (
            "demo_ready"
            if plan["mode"] == "demo"
            else services.get("runtime", {}).get("state", "starting")
        )
        if ok
        else "degraded"
    )
    if state["phase"] in {"stopped", "shutdown_blocked", "failed", "starting"}:
        phase = state["phase"]
    if phase == "stopped" and not any(alive(s) for s in services.values()):
        ok = True
        for service in services.values():
            if service.get("ownership") != "external":
                service.update(state="stopped", detail="Owned process has exited")
    result = {
        "ok": ok,
        "state": phase,
        "id": state["id"],
        "mode": plan["mode"],
        "gui_url": plan["gui_url"],
        "services": services,
        "deployment_dir": str(directory),
        "fingerprint": plan["fingerprint"],
        "next_action": (
            "Synthetic preview only; no robot connected"
            if plan["mode"] == "demo"
            else "Operator uses RoboGUI. Prepare may move the robot; verify the workspace first."
        ),
        "evidence": {
            "real_weight_inference": "not_tested_by_launcher",
            "robot_connectivity": "not_tested_by_launcher",
            "task_success": "not_assessed",
        },
    }
    if "runtime" in services:
        result["attach_command"] = services["runtime"]["attach_command"]
    if state.get("error"):
        result["error"] = state["error"]
    return result


def status(directory):
    state = read_json(directory / "state.json")
    if state is None:
        return {"ok": True, "state": "not_started", "deployment_dir": str(directory)}
    return _snapshot(directory, state, read_json(Path(state["run_dir"]) / "plan.json"))


def up(plan, directory, timeout, *, open_browser=False):
    with locked(directory):
        previous = read_json(directory / "state.json")
        if previous and any(alive(s) for s in previous["services"].values()):
            if previous["fingerprint"] != plan["fingerprint"]:
                raise RuntimeError(
                    "This directory owns a different deployment. Inspect status and down first."
                )
            result = _snapshot(directory, previous, plan)
            if not result["ok"]:
                result["error"] = (
                    "Existing deployment is incomplete or unhealthy; inspect logs before down/up"
                )
            elif open_browser:
                result["browser_opened"] = webbrowser.open(plan["gui_url"])
            return result
        preflight = doctor(plan, directory)
        if not preflight["ok"]:
            return preflight
        deployment_id = uuid.uuid4().hex[:12]
        run_dir = directory / deployment_id
        run_dir.mkdir(mode=0o700)
        atomic_json(run_dir / "plan.json", plan)
        state = {
            "id": deployment_id,
            "fingerprint": plan["fingerprint"],
            "run_dir": str(run_dir),
            "phase": "starting",
            "services": {},
        }
        atomic_json(directory / "state.json", state)
        try:
            for name in plan["services"]:
                verify_sources(plan)
                external = plan["services"][name].get("ownership") == "external"
                print(
                    f"{'Checking external' if external else 'Starting'} {name}…",
                    file=sys.stderr,
                    flush=True,
                )
                if external:
                    service = {"ownership": "external"}
                    state["services"][name] = service
                    atomic_json(directory / "state.json", state)
                else:
                    service = _spawn(name, plan, state, directory)
                deadline = time.monotonic() + timeout
                while True:
                    health = _probe(name, plan, state)
                    service.update(health)
                    atomic_json(directory / "state.json", state)
                    if health["ok"]:
                        print(f"{name}: {health['state']}", file=sys.stderr, flush=True)
                        break
                    if (not external and not alive(service)) or time.monotonic() >= deadline:
                        raise RuntimeError(
                            f"{name} failed: {health['detail']}; "
                            f"log: {service.get('log', 'external service owner')}"
                        )
                    time.sleep(0.3)
            state["phase"] = "ready"
            atomic_json(directory / "state.json", state)
        except (Exception, KeyboardInterrupt) as exc:
            startup_error = str(exc) or "Startup interrupted"
            # No process is force-killed; a runtime cleanup gate keeps dependencies alive.
            stopped = _stop(state, directory, timeout=min(timeout, 10.0))
            state["phase"] = "failed" if stopped else "shutdown_blocked"
            state["error"] = startup_error + (
                "" if stopped else "; " + state.get("error", "shutdown incomplete")
            )
            atomic_json(directory / "state.json", state)
            return {**_snapshot(directory, state, plan), "ok": False}
        result = _snapshot(directory, state, plan)
        if open_browser and result["ok"]:
            result["browser_opened"] = webbrowser.open(plan["gui_url"])
        return result


def _stop(state, directory, timeout):
    # Runtime owns hardware cleanup. Keep GUI, cameras and policy available until
    # it exits, including while waiting for physical-support confirmation.
    order = [n for n in ("runtime", "policy", "camera", "robogui") if n in state["services"]]
    for name in order:
        service = state["services"][name]
        if service.get("ownership") == "external":
            continue
        if alive(service):
            if not service.get("stop_requested"):
                signal_owned(service)
                service["stop_requested"] = True
                atomic_json(directory / "state.json", state)
            deadline = time.monotonic() + timeout
            while alive(service) and time.monotonic() < deadline:
                time.sleep(0.1)
            if alive(service):
                state["phase"] = "shutdown_blocked"
                state["error"] = (
                    f"{name} is still alive; dependencies retained. Attach to its terminal."
                )
                atomic_json(directory / "state.json", state)
                return False
        service["state"] = "stopped"
        # tmux removes exited sessions itself. Do not kill by session name after
        # a PID mismatch: the terminal might now contain an unrelated process.
    state["phase"] = "stopped"
    state.pop("error", None)
    atomic_json(directory / "state.json", state)
    return True


def down(directory, timeout):
    with locked(directory):
        state = read_json(directory / "state.json")
        if not state:
            return {"ok": True, "state": "not_started", "deployment_dir": str(directory)}
        ok = _stop(state, directory, timeout)
        result = _snapshot(directory, state, read_json(Path(state["run_dir"]) / "plan.json"))
        result.update(
            ok=ok,
            next_action="Owned services stopped; external dependencies retained"
            if ok
            else "Attach to the blocked service terminal",
        )
        return result


def attach(directory, name):
    state = read_json(directory / "state.json") or {}
    service = state.get("services", {}).get(name)
    if service and service.get("ownership") == "external":
        raise RuntimeError(f"{name} is external; attach through its existing service owner")
    if not service or not alive(service):
        raise RuntimeError(f"No live owned {name} terminal")
    return subprocess.call(
        [
            "tmux",
            "-L",
            service.get("tmux_socket", TMUX_SOCKET),
            "attach-session",
            "-t",
            service["session"],
        ]
    )
