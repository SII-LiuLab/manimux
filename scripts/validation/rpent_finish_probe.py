"""Check that Finish during in-flight RPent calls stops robot commands, without hardware."""

from __future__ import annotations

import argparse
import copy
import json
import threading
import time
from dataclasses import replace
from pathlib import Path

import yaml

from manimux.agents.rpent.cli import load_agent_config
from manimux.cli import _create_run_dir, load_config
from manimux.dispatch import DispatchClient
from manimux.dispatch.server import DispatchServer
from manimux.runtime.dispatch import PolicyDispatch
from manimux.runtime.edge import EdgeRuntime
from manimux.runtime.motion import MotionTools


def scenario(name, recipe, base, motion_options, output, call, finish_via):
    config = copy.deepcopy(base)
    config["run"]["output_dir"] = output / name
    config["policy"]["options"]["response_log_dir"] = str(output / name / "provider-responses")
    dispatch = PolicyDispatch(config)
    runtime = EdgeRuntime(config, _create_run_dir(config, recipe.experiment, mode="rpent"),
                          dispatch=dispatch, launch_mode="rpent")
    dispatch.motion_tools = MotionTools(
        runtime._robot.kinematics, motion_options, config["robot"]["control_hz"]
    )
    # Count every command that reaches the robot; Finish is injected where RoboGUI
    # control is polled, the same branch the dashboard button reaches.
    sends, gui_finish = [], threading.Event()
    send = runtime._robot.send_command
    runtime._robot.send_command = lambda command: (sends.append(time.monotonic()), send(command))
    poll = runtime._robogui.poll_control
    runtime._robogui.poll_control = lambda: (
        replace(poll(), finish_requested=True) if gui_finish.is_set() else poll()
    )
    results, errors, outcome = [], [], {}

    def run():
        try:
            results.append(runtime.run())
        except BaseException as exc:
            errors.append(repr(exc))

    thread = threading.Thread(target=run, daemon=True)
    with DispatchServer(dispatch, port=0) as server:
        client = DispatchClient(f"http://127.0.0.1:{server.port}")
        thread.start()
        started = time.monotonic()
        while not client.status()["ready"]:
            if errors or time.monotonic() - started > 60:
                raise RuntimeError(f"{name}: runtime not ready: {errors}")
            time.sleep(0.05)
        caller = threading.Thread(target=lambda: outcome.update(call(client)), daemon=True)
        caller.start()
        time.sleep(2.0)
        sends_before = len(sends)
        finished_at = time.monotonic()
        if finish_via == "robogui":
            gui_finish.set()
        else:
            client.finish()
        thread.join(30)
        caller.join(30)
        try:
            late = client.submit_motion(
                "move_delta", dict(group="left_arm", delta_xyz=[0, 0, 0.005])
            ).get("state")
        except Exception as exc:
            late = f"rejected: {exc}"
    report = dict(
        scenario=name,
        finish_via=finish_via,
        sends_before_finish=sends_before,
        sends_after_finish=sum(t > finished_at for t in sends),
        call_state=outcome.get("state"),
        call_reason=outcome.get("reason"),
        runtime_exited=not thread.is_alive(),
        runtime_errors=errors,
        terminal_reason=results[0].terminal_reason if results else None,
        late_submission=late,
    )
    report["passed"] = (
        report["sends_before_finish"] > 0
        and report["sends_after_finish"] == 0
        and report["call_state"] not in (None, "completed")
        and report["runtime_exited"]
        and not errors
        and str(late).startswith("rejected")
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("manimux/configs/agents/rpent.yaml"))
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory")
    args = parser.parse_args()
    recipe = load_agent_config(args.config, {})
    base = load_config(recipe.experiment)
    if base["robot"]["type"] != "manimux.dispatch.demo:build_robot" or base["sensors"]:
        raise ValueError("This probe accepts only the dispatch demo robot without physical sensors")
    motion_options = yaml.safe_load(recipe.motion.read_text())
    # Slow primitives keep the call in flight when Finish arrives.
    motion_options["limits"].update(linear_speed_m_s=0.01, max_translation_m=0.1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    motion = lambda c: c.motion(  # noqa: E731
        "move_delta", dict(group="left_arm", delta_xyz=[0.08, 0, 0]), timeout_s=60
    )
    policy = lambda c: c.execute(  # noqa: E731
        base["run"]["task_prompt"], chunks=200, use_length=5, timeout_s=60
    )
    reports = [
        scenario(name, recipe, base, motion_options, output, call, via)
        for name, call, via in [
            ("robogui-finish-motion", motion, "robogui"),
            ("robogui-finish-policy", policy, "robogui"),
            ("planner-finish-motion", motion, "planner"),
        ]
    ]
    status = "passed" if all(r["passed"] for r in reports) else "failed"
    (output / "report.json").write_text(
        json.dumps(dict(status=status, hardware_used=False, scenarios=reports), indent=2) + "\n"
    )
    print(json.dumps({"status": status, "report": str(output / "report.json")}))
    if status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
