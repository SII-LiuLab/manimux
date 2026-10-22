"""Agent-facing orchestration of existing ManiMux service entry points."""

import argparse
import json
import math
import sys
from pathlib import Path


def positive_seconds(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def add_commands(subparsers):
    for name in ("doctor", "up", "status", "down", "attach"):
        parser = subparsers.add_parser(
            name,
            help={
                "doctor": "Check a local deployment without opening devices",
                "up": "Start managed services and wait for manual Prepare",
                "status": "Inspect managed processes and service health",
                "down": "Gracefully stop owned services, runtime first",
                "attach": "Attach to an owned service's operator terminal",
            }[name],
        )
        parser.add_argument(
            "--deployment-dir",
            type=Path,
            help="Private state/log directory (default: .local/deployment)",
        )
        if name != "attach":
            parser.add_argument("--json", action="store_true", help="Machine-readable result")
        if name in {"doctor", "up"}:
            source = parser.add_mutually_exclusive_group(required=True)
            source.add_argument("--config", type=Path, help="One supported runtime recipe")
            source.add_argument(
                "--demo", action="store_true", help="Synthetic RoboGUI only; no hardware"
            )
            parser.add_argument("--local", type=Path, help="Private station bindings")
            parser.add_argument("--runtime-python", type=Path, default=Path(sys.executable))
            parser.add_argument(
                "--camera-python", type=Path, help="Defaults to runtime interpreter"
            )
            parser.add_argument(
                "--policy-python", type=Path, help="Required for an owned model service"
            )
            parser.add_argument(
                "--external-policy",
                action="store_true",
                help="Connect to the configured policy service; never start or stop its process",
            )
            parser.add_argument(
                "--external-camera",
                action="store_true",
                help="Use configured network camera sensors; never start or stop their servers",
            )
            parser.add_argument("--robot", help="RoboGUI preset for --demo (default: yam)")
            parser.add_argument("--gui-port", type=int, default=8086)
            parser.add_argument("--output-dir", type=Path, help="Separate trial recordings")
        if name == "up":
            parser.add_argument(
                "--timeout",
                type=positive_seconds,
                default=180.0,
                help="Readiness timeout per service, seconds",
            )
            parser.add_argument(
                "--open", action="store_true", help="Open RoboGUI in the default browser"
            )
            parser.add_argument(
                "--detach",
                action="store_true",
                help="Return after startup instead of holding the terminal until Ctrl-C",
            )
        if name == "down":
            parser.add_argument(
                "--timeout",
                type=positive_seconds,
                default=30.0,
                help="Wait per service; never force-kill on timeout",
            )
        if name == "attach":
            parser.add_argument(
                "--service", choices=("runtime", "camera", "policy", "robogui"), default="runtime"
            )


def main(args):
    from . import processes
    from .plan import build_plan, doctor

    directory = (args.deployment_dir or Path.cwd() / ".local/deployment").expanduser().absolute()
    try:
        if args.command == "doctor":
            result = doctor(build_plan(args), directory)
        elif args.command == "up":
            result = processes.up(build_plan(args), directory, args.timeout, open_browser=args.open)
            if result.get("ok") and not args.detach:
                _report(result, args)
                result = processes.hold(directory, timeout=30.0)
        elif args.command == "status":
            result = processes.status(directory)
        elif args.command == "down":
            result = processes.down(directory, args.timeout)
        else:
            return processes.attach(directory, args.service)
    except (Exception, KeyboardInterrupt) as exc:
        result = {
            "ok": False,
            "state": "error",
            "error": str(exc) or "Interrupted",
            "deployment_dir": str(directory),
        }
    _report(result, args)
    return 0 if result.get("ok") else 2


def _report(result, args):
    if getattr(args, "json", False):
        print(json.dumps(result, indent=2, default=str), flush=True)
        return
    print(f"ManiMux · {result.get('state', 'checked')}")
    for check in result.get("checks", []):
        print(f"  {'OK' if check['ok'] else 'BLOCKED'} {check['name']}: {check['detail']}")
    for name, service in result.get("services", {}).items():
        print(f"  {name}: {service.get('state', 'unknown')} — {service.get('detail', '')}")
    for key in ("error", "gui_url", "next_action", "attach_command", "deployment_dir"):
        if result.get(key):
            print(f"{key}: {result[key]}")
    sys.stdout.flush()
