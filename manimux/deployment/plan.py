"""Read-only deployment planning; model and hardware construction stay in services."""

import hashlib
import json
import math
import os
import shutil
import socket
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

REPOSITORY = Path(__file__).resolve().parents[2]
# Hosted model APIs run inside the runtime worker; there is no local model service.
HOSTED_POLICY_WORKERS = {"dream_api", "poke_api"}


def external_camera_sensors(config):
    """Only network readers may be constructed by an external camera probe."""
    sensors = config.get("sensors", [])
    allowed = {
        "camera_server",
        "manimux.embodiments.sensor.camera_server:build_sensor",
        "manimux.embodiments.sensor.camera_server.timestamped:build_sensor",
    }
    if not sensors or any(s["driver"] not in allowed for s in sensors):
        raise ValueError("--external-camera requires camera-server network sensors")
    for sensor in sensors:
        age = sensor.get("options", {}).get("max_frame_age_sec", 0.5)
        if age is None or not math.isfinite(float(age)) or float(age) <= 0:
            raise ValueError("External camera health requires a positive max_frame_age_sec")
    return sensors


def environment():
    return {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(REPOSITORY) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }


def address(endpoint):
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"tcp", "ws"} or not parsed.port:
        raise ValueError(f"Unsupported local endpoint: {endpoint}")
    if parsed.hostname not in {"localhost", "127.0.0.1", "0.0.0.0"}:
        raise ValueError(f"Managed startup currently requires local IPv4 services: {endpoint}")
    return parsed.hostname, parsed.port


def build_plan(args):
    from manimux.cli import _load_config, _trial_config, resolve_local_path
    from manimux.robogui.robot_view import load_robogui_config

    if not 1 <= args.gui_port <= 65535:
        raise ValueError("--gui-port must be between 1 and 65535")
    external_policy = getattr(args, "external_policy", False)
    external_camera = getattr(args, "external_camera", False)
    if args.demo and any(
        (
            args.local,
            args.policy_python,
            args.camera_python,
            args.output_dir,
            external_policy,
            external_camera,
        )
    ):
        raise ValueError("--demo does not accept station, camera, policy or recording options")
    if external_policy and args.policy_python is not None:
        raise ValueError("--external-policy cannot be combined with --policy-python")
    if external_camera and args.camera_python is not None:
        raise ValueError("--external-camera cannot be combined with --camera-python")
    if not args.demo and args.robot:
        raise ValueError("--robot is for --demo; a real deployment reads its runtime recipe")
    # Do not resolve interpreter symlinks: doing so loses the virtualenv identity.
    runtime_python = str(args.runtime_python.expanduser().absolute())
    config, config_path, station = {}, None, None
    hosted_policy = False
    if not args.demo:
        config_path = args.config.expanduser().resolve()
        station = resolve_local_path(config_path, args.local)
        config = _trial_config(
            _load_config(config_path, local=station), config_path, args.output_dir
        )
        if not config["robogui"]["enabled"]:
            raise ValueError("Managed deployment requires robogui.enabled: true")
        hosted_policy = config["policy"]["worker"] in HOSTED_POLICY_WORKERS
        if hosted_policy:
            if args.policy_python is not None or external_policy:
                raise ValueError(
                    "A hosted policy API has no model service; omit --policy-python "
                    "and --external-policy"
                )
        elif config["policy"]["worker"] != "xpolicylab_ws":
            raise ValueError(
                "Managed startup supports XPolicyLab WebSocket recipes and hosted policy APIs"
            )
        elif args.policy_python is None and not external_policy:
            raise ValueError("Select the model environment with --policy-python")
        if external_camera:
            external_camera_sensors(config)
        elif not config.get("camera_server", {}).get("cameras"):
            raise ValueError("Managed startup requires a camera_server recipe")
    robot = (args.robot or "yam") if args.demo else config["robogui"]["robot"]
    load_robogui_config(None, robot=robot)
    gui_command = [
        runtime_python,
        "-m",
        "manimux.robogui.dashboard",
        "--robot",
        robot,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.gui_port),
    ]
    gui_ports = [["127.0.0.1", args.gui_port]]
    if args.demo:
        # Demo has no runtime/control clients and can coexist with a live station.
        gui_command += [
            "--demo",
            "--bridge-endpoint",
            "tcp://127.0.0.1:*",
            "--control-endpoint",
            "tcp://127.0.0.1:*",
        ]
    else:
        gui_ports += [["127.0.0.1", 5568], ["127.0.0.1", 5569]]
    gui = {"command": gui_command, "ports": gui_ports, "probe_python": runtime_python}
    services = {"robogui": gui}
    if not args.demo:
        common = ["--config", str(config_path), "--local", str(station)]
        camera_python = str((args.camera_python or args.runtime_python).expanduser().absolute())
        policy_python = (
            str(args.policy_python.expanduser().absolute()) if args.policy_python else None
        )
        camera = config.get("camera_server", {})
        policy = config.get("policy_server", {})
        if policy.get("protocol", "ws") != "ws":
            raise ValueError("policy_server.protocol must be ws")
        if external_policy:
            endpoint = urlsplit(config["policy"]["options"]["server"])
            if endpoint.scheme not in {"ws", "wss"} or not endpoint.hostname:
                raise ValueError("External policy requires a WebSocket endpoint")
        elif not hosted_policy:
            policy_host, policy_port = address(f"ws://{policy['host']}:{policy['port']}")
            _, client_port = address(config["policy"]["options"]["server"])
            if policy_port != client_port:
                raise ValueError("Policy client/server ports do not match")
        services = {
            "camera": {"ownership": "external", "ports": [], "probe_python": runtime_python}
            if external_camera
            else {
                "command": [camera_python, "-m", "manimux.servers.camera.server", *common],
                "ports": [
                    address(camera.get("rep_endpoint", "tcp://127.0.0.1:5555")),
                    address(camera.get("pub_endpoint", "tcp://127.0.0.1:5556")),
                ],
                "probe_python": runtime_python,
            },
            "robogui": gui,
            "policy": {"ownership": "external", "ports": [], "probe_python": runtime_python}
            if external_policy
            else None
            if hosted_policy
            else {
                "command": [policy_python, "-m", "manimux.servers.policy", *common],
                "ports": [[policy_host, policy_port]],
                "probe_python": runtime_python,
            },
            "runtime": {
                "command": [runtime_python, "-m", "manimux", "run", *common],
                "ports": [],
                "probe_python": runtime_python,
            },
        }
        if hosted_policy:
            del services["policy"]
        if args.output_dir:
            services["runtime"]["command"] += [
                "--output-dir",
                str(args.output_dir.expanduser().resolve()),
            ]
    plan = {
        "schema": 1,
        "mode": "demo" if args.demo else "deployment",
        "robot": robot,
        "cwd": str(Path.cwd().resolve()),
        "repository": str(REPOSITORY),
        "gui_url": f"http://127.0.0.1:{args.gui_port}",
        "services": services,
        "config": config,
        "config_path": str(config_path) if config_path else None,
        "station": str(station) if station else None,
    }
    plan = json.loads(json.dumps(plan, default=str))
    plan["fingerprint"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    return plan


def verify_sources(plan):
    for filename, expected_hash in plan["config"].get("config_sources", {}).items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Configuration changed during startup: {filename}; rerun doctor/up")


def doctor(plan, directory, owned=()):
    checks = []

    def check(name, action):
        try:
            detail = action()
            checks.append({"name": name, "ok": True, "detail": str(detail or "passed")})
        except Exception as exc:
            checks.append({"name": name, "ok": False, "detail": str(exc)})

    def require_tmux():
        if not shutil.which("tmux"):
            raise ValueError("Install tmux; managed services retain attachable operator terminals")
        return "tmux available; terminal access retained during blocked shutdown"

    check("operator terminals", require_tmux)

    def check_signalling():
        from .processes import open_pidfd

        descriptor = open_pidfd(os.getpid())
        os.close(descriptor)
        return "Kernel process handles available; no signal sent"

    check("process ownership", check_signalling)
    check("configuration", lambda: "Resolved without constructing devices or loading weights")
    check("configuration sources", lambda: verify_sources(plan))
    interpreters = set()
    for name, service in plan["services"].items():
        external = service.get("ownership") == "external"
        interpreter = service["probe_python"] if external else service["command"][0]
        if interpreter not in interpreters:

            def inspect_python(interpreter=interpreter):
                result = subprocess.run(
                    [
                        interpreter,
                        "-c",
                        "import sys,manimux; assert (3,11)<=sys.version_info[:2]<(3,13); "
                        "print(sys.executable); print(manimux.__file__)",
                    ],
                    cwd=plan["cwd"],
                    env=environment(),
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode:
                    raise ValueError(result.stderr.strip() or "Interpreter check failed")
                return result.stdout.strip()

            check(f"interpreter: {interpreter}", inspect_python)
            interpreters.add(interpreter)
        if external:
            check(f"{name} ownership", lambda: "External dependency; no process or port ownership")
            continue
        if name in owned:
            continue
        for host, port in service["ports"]:

            def inspect_port(host=host, port=port):
                with socket.socket() as sock:
                    # Recently closed HTTP connections may remain in TIME_WAIT.
                    # Like our servers, allow address reuse, but test listen too
                    # so an active listener still blocks managed startup.
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind((host, port))
                    sock.listen(1)
                return "available"

            check(f"{name} port {host}:{port}", inspect_port)
    ports = [port for s in plan["services"].values() for _, port in s["ports"]]
    check("distinct service ports", lambda: _distinct(ports))
    if plan["mode"] != "demo":
        recipe = plan["config"].get("policy_server", {})
        if "policy" not in plan["services"]:
            key = plan["config"]["policy"].get("options", {}).get("api_key_env")

            def hosted_credentials():
                if key and not os.environ.get(key):
                    raise ValueError(f"Export {key} in the shell that runs up")
                return "Hosted API; credentials stay in the runtime environment"

            check("policy credentials", hosted_credentials)
        elif plan["services"]["policy"].get("ownership") == "external":
            check(
                "policy artifacts",
                lambda: "External host owns artifacts; up/status verify HELLO identity",
            )
        elif recipe.get("policy_name") == "Pi_05":
            from manimux.servers.pi05 import _validate_paths

            check("Pi05 checkpoint and normalization", lambda: _validate_paths(recipe))
        else:
            for key in ("model_path", "checkpoint_path", "ckpt_path", "norm_stats_path"):
                value = recipe.get(key)
                if isinstance(value, str) and Path(value).is_absolute():
                    check(f"artifact: {key}", lambda value=value: _exists(value))
        for name in ("camera", "policy"):
            service = plan["services"].get(name)
            if service is None or service.get("ownership") == "external":
                continue

            def validate_service(name=name):
                result = subprocess.run(
                    [*plan["services"][name]["command"], "--check"],
                    cwd=plan["cwd"],
                    env=environment(),
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                if result.returncode:
                    raise ValueError(result.stderr.strip() or result.stdout.strip())
                return "Configuration check passed; devices and weights unopened"

            check(f"{name} configuration", validate_service)
    return {
        "ok": all(c["ok"] for c in checks),
        "state": "preflight",
        "checks": checks,
        "plan": plan,
        "gui_url": plan["gui_url"],
        "deployment_dir": str(directory),
        "next_action": "up starts services; robot connectivity and inference remain unverified",
    }


def _distinct(ports):
    if len(ports) != len(set(ports)):
        raise ValueError("Two managed services would bind the same port")
    return "No overlapping ports"


def _exists(path):
    if not Path(path).exists():
        raise FileNotFoundError(path)
    return path
