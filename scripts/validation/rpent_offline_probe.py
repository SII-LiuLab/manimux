"""Exercise unified RPent motion/VLA tools and ManiMux records without hardware."""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
import traceback
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from manimux.agents.rpent.cli import load_agent_config
from manimux.cli import _create_run_dir, load_config
from manimux.dispatch import DispatchClient
from manimux.dispatch.server import DispatchServer
from manimux.runtime.dispatch import PolicyDispatch
from manimux.runtime.edge import EdgeRuntime
from manimux.runtime.motion import MotionTools


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Shared RPent agent recipe")
    parser.add_argument("--local", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory")
    parser.add_argument("--min-control-steps", type=int, default=500)
    parser.add_argument("--timeout-s", type=float, default=300)
    args = parser.parse_args()
    if args.min_control_steps < 1 or not 0 < args.timeout_s <= 3600:
        parser.error("Use positive control steps and timeout_s in (0, 3600]")

    recipe = load_agent_config(args.config, {})
    if recipe.experiment is None or recipe.motion is None:
        parser.error("agent recipe must select experiment and motion")
    motion_options = yaml.safe_load(recipe.motion.read_text())
    config = load_config(recipe.experiment, local=args.local)
    if config["robot"]["type"] != "manimux.dispatch.demo:build_robot" or config["sensors"]:
        raise ValueError("This probe accepts only the dispatch demo robot without physical sensors")
    if config["robogui"]["enabled"]:
        raise ValueError("Disable RoboGUI for this unattended probe; inspect the saved replay")
    from rpent.memory import MemoryManager

    from manimux.agents.rpent.toolkit import ManiMuxToolkit

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config["run"]["output_dir"] = output / "runs"
    config["policy"]["options"]["response_log_dir"] = str(output / "provider-responses")
    dispatch = PolicyDispatch(config)
    run_dir = _create_run_dir(config, recipe.experiment, mode="rpent")
    runtime = EdgeRuntime(config, run_dir, dispatch=dispatch, launch_mode="rpent")
    dispatch.motion_tools = MotionTools(
        runtime._robot.kinematics, motion_options, config["robot"]["control_hz"]
    )
    report = dict(
        status="running",
        date_utc=datetime.now(UTC).isoformat(),
        experiment=str(recipe.experiment),
        validation_stage="unified_agent_tools_runtime",
        hardware_used=False,
        robot_task_success_tested=False,
        autonomous_planner_used=False,
        policy_worker=config["policy"]["worker"],
        model_evidence=(
            "fake and manimux.dispatch.demo models produce synthetic actions; "
            "other workers require separate provider evidence"
        ),
        reset_scope="rollout; no reset between subtask calls",
        rollout_directory=str(run_dir),
        calls=[],
        checks=[],
    )
    errors, results = [], []
    toolkit = None

    def run() -> None:
        try:
            results.append(runtime.run())
        except BaseException as exc:
            errors.append(repr(exc))

    def check(name: str, condition: bool) -> None:
        report["checks"].append(dict(name=name, passed=bool(condition)))
        if not condition:
            raise AssertionError(name)

    deadline = time.monotonic() + args.timeout_s

    def wait_for(predicate) -> None:
        while not predicate():
            if errors or (results and not results[0].run_completed):
                raise RuntimeError(f"Runtime failed: {errors or results}")
            if time.monotonic() >= deadline:
                raise TimeoutError("Offline agent probe deadline")
            time.sleep(0.02)

    thread = threading.Thread(target=run, daemon=True)
    try:
        with DispatchServer(dispatch, port=0) as server:
            client = DispatchClient(f"http://127.0.0.1:{server.port}")
            thread.start()
            wait_for(lambda: client.status()["ready"])
            toolkit = ManiMuxToolkit(
                client, output_dir=output / "agent", memory=MemoryManager(output / "memory")
            )
            observed = toolkit.execute_tool("observe", {}).data
            check(
                "native_observe_group_layout",
                {k: len(v) for k, v in observed["state"].items()} == config["robot"]["group_dims"],
            )
            result = toolkit.execute_tool("move_to", dict(group="left_arm", xyz=[0.01, 0, 0])).data
            check(
                "preposition_before_vla",
                result.get("state") == "completed" and result["arrival"]["reached"],
            )
            report["calls"].append(dict(tool="move_to", result=result))
            motion_plan_count = 1
            prefix = min(5, dispatch.horizon)
            ticks_per_chunk = (
                prefix * config["policy"]["action_dt_s"] * config["robot"]["control_hz"]
            )
            chunks_for_coverage = max(2, math.ceil(args.min_control_steps / ticks_per_chunk) + 2)
            if chunks_for_coverage > 1000:
                raise ValueError("Requested control-step coverage exceeds the tool's chunk budget")
            for prompt, chunks, length in [
                ("Move toward the object.", chunks_for_coverage, prefix),
                ("Place the object on the table.", 1, min(3, dispatch.horizon)),
            ]:
                result = toolkit.execute_tool(
                    "execute_policy",
                    dict(
                        prompt=prompt,
                        chunks=chunks,
                        use_length=length,
                        timeout_s=max(0.1, deadline - time.monotonic()),
                    ),
                ).data
                report["calls"].append(dict(prompt=prompt, result=result))
                check("bounded_call_completed", result.get("state") == "completed")
                check(
                    "budget_accounted",
                    result["completed_chunks"] == chunks
                    and result["completed_policy_steps"] == chunks * length,
                )
                check(
                    "finite_feedback",
                    all(np.isfinite(v).all() for v in result["observation"]["state"].values()),
                )
                before = client.observe()["state"]
                time.sleep(0.1)
                check("holds_between_calls", client.observe()["state"] == before)
            for tool, arguments in [
                ("set_gripper", dict(group="left_arm", opening=0.5)),
                ("move_delta", dict(group="left_arm", delta_xyz=[0, 0, 0.005])),
            ]:
                result = toolkit.execute_tool(tool, arguments).data
                check(
                    f"{tool}_after_vla",
                    result.get("state") == "completed" and result["arrival"]["reached"],
                )
                report["calls"].append(dict(tool=tool, result=result))
                motion_plan_count += 1
            # Idle holds are not recorded as RUNNING ticks. Check the authoritative
            # result instead of treating observation reads or elapsed time as ticks.
            client.finish()
            thread.join(max(0.0, deadline - time.monotonic()))
            check("runtime_closed", not thread.is_alive() and not errors and len(results) == 1)
            report["runtime"] = asdict(results[0])
            check("minimum_control_steps", results[0].steps >= args.min_control_steps)
            check(
                "clean_plan_acceptance",
                results[0].run_completed
                and results[0].accepted_plans == chunks_for_coverage + 1 + motion_plan_count
                and results[0].rejected_plans == 0,
            )
            check("does_not_claim_task_success", toolkit.solved() is False)
            report["status"] = "passed"
    except Exception as exc:
        report.update(
            status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()
        )
        raise
    finally:
        dispatch.stop_requested = True
        if thread.ident is not None:
            thread.join(30)
        if toolkit is not None:
            toolkit.close()
        report["runtime_errors"] = errors
        (output / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(json.dumps({"status": report["status"], "report": str(output / "report.json")}))


if __name__ == "__main__":
    main()
