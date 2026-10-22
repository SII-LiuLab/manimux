"""Launch an external RPent planner with ManiMux tools and a recorded agent recipe."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from contextlib import suppress
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictBool


class AgentConfig(BaseModel):
    """One recipe selects the runtime, motion tools and external planner."""

    model_config = ConfigDict(extra="forbid")

    agent: Literal["rpent"] = "rpent"
    endpoint: str = "http://127.0.0.1:8088"
    task: str = Field(min_length=1)
    memory_root: Path
    output_dir: Path
    planner: Literal["api", "codex", "claude_code"] = "api"
    model: str | None = None
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh"] = "none"
    operator_control: bool = False
    max_turns: int = Field(default=30, gt=0)
    experiment: Path | None = None
    motion: Path | None = None
    policy_enabled: StrictBool = True


def load_agent_config(path: Path | None, overrides: dict[str, object]) -> AgentConfig:
    """Resolve recipe paths relative to YAML; explicit CLI paths use the working directory."""
    values = {}
    if path is not None:
        values = yaml.safe_load(path.read_text())
        if not isinstance(values, dict):
            raise ValueError("Agent configuration must be a YAML mapping")
        for key in ("memory_root", "output_dir", "experiment", "motion"):
            if values.get(key) is not None:
                value = Path(values[key]).expanduser()
                values[key] = (path.resolve().parent / value).resolve()
    values.update(overrides)
    config = AgentConfig.model_validate(values)
    config.memory_root = config.memory_root.expanduser().resolve()
    config.output_dir = config.output_dir.expanduser().resolve()
    return config


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        return serve(sys.argv[2:])
    argv = sys.argv[2:] if len(sys.argv) > 1 and sys.argv[1] == "run" else sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="RPent motion and VLA tools: serve or run one agent recipe"
    )
    parser.add_argument("--config", type=Path, help="Agent YAML; CLI fields override its values")
    parser.add_argument("--endpoint", default=argparse.SUPPRESS)
    parser.add_argument("--task", default=argparse.SUPPRESS)
    parser.add_argument("--memory-root", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, default=argparse.SUPPRESS)
    parser.add_argument(
        "--planner", choices=["api", "codex", "claude_code"], default=argparse.SUPPRESS
    )
    parser.add_argument("--model", default=argparse.SUPPRESS)
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh"],
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--max-turns", type=int, default=argparse.SUPPRESS)
    args = vars(parser.parse_args(argv))
    source = args.pop("config")
    try:
        config = load_agent_config(source, args)
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))

    # Basic package imports and CLI help never require the separately installed SDK.
    try:
        from rpent.dashboard.events import NullDashboardEventSink
        from rpent.memory import MemoryManager
        from rpent.planner.base import build_planner
        from rpent.utils.logging import init_output_dir

        import rpent
        from manimux.agents.rpent.toolkit import ManiMuxToolkit
    except ModuleNotFoundError as exc:
        parser.error(f"Install RPent in the agent environment; missing dependency: {exc.name}")

    from manimux.dispatch.client import DispatchClient

    source_root = Path(rpent.__file__).resolve().parent.parent
    revision = None
    if (source_root / ".git").exists():
        with suppress(OSError, subprocess.SubprocessError):
            revision = subprocess.check_output(
                ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True, timeout=10
            ).strip()
    config.output_dir.mkdir(parents=True, exist_ok=False)
    # RPent binds tool-schema templates to this native output context.
    init_output_dir(config.output_dir)
    config.memory_root.mkdir(parents=True, exist_ok=True)
    events = NullDashboardEventSink()
    client = DispatchClient(config.endpoint, token=os.environ.get("MANIMUX_DISPATCH_TOKEN"))
    toolkit = ManiMuxToolkit(
        client,
        output_dir=config.output_dir,
        memory=MemoryManager(config.memory_root),
        dashboard_events=events,
        planner_label=f"RPent / {config.model or config.planner}",
        operator_control=config.operator_control,
    )
    try:
        # Resolve native tool schemas before paying for a planner request.
        available_tools = [declaration.name for declaration in toolkit.list_tools()]
        if toolkit.policy_enabled != config.policy_enabled:
            raise ValueError("Agent policy_enabled must match the active ManiMux runtime")
        (config.output_dir / "agent_config.json").write_text(
            json.dumps(
                {
                    "config": config.model_dump(mode="json"),
                    "config_source": str(source.resolve()) if source else None,
                    "rpent_version": version("rpent"),
                    "rpent_module": str(Path(rpent.__file__).resolve()),
                    "rpent_git_sha": revision,
                    "manimux_session_id": client.session_id,
                    "policy_contract": toolkit.policy_contract,
                    "available_tools": available_tools,
                    "execute_policy_schema": (
                        None if toolkit.execute_policy is None
                        else toolkit.execute_policy.input_schema
                    ),
                },
                indent=2,
            )
            + "\n"
        )
        planner = build_planner(
            config.planner,
            model=config.model,
            reasoning_effort=config.reasoning_effort,
            output_dir=config.output_dir,
            recipe_tag=config.output_dir.name,
            robot_name="manimux",
            memory_dir=config.memory_root,
            dashboard_events=events,
        )
        result = planner.solve(
            system_prompt=(
                (
                    "Use observe, configured motion tools and execute_policy to complete the task. "
                    "Read policy_contract from observe and respect "
                    "execute_policy's use_length bound. "
                    "The model and robot are selected by the ManiMux experiment. "
                    "Observe before acting and after each bounded policy call. "
                    "Use available task memory to choose short subtask prompts "
                    "and retry when needed. "
                    "A completed policy budget is not proof of task success. "
                    if toolkit.policy_enabled else
                    "Complete the task using observe and the configured motion primitives. "
                    "Choose motion targets and gripper actions yourself "
                    "using observations and task memory. "
                    "Observe before acting and after each motion to assess progress. "
                    "A completed motion is not proof of task success. "
                )
                + "Read group names and coordinate frames from observe before planning motion. "
                "Operator pause, failure or cancellation must not be bypassed. "
                "Call finish with an honest status and summary when done."
                + (" ManiMux manages the GUI, robot lifecycle, cameras and recording. "
                   "Do not launch or stop services, create recording processes "
                   "or modify configuration. "
                   "Use the finish tool to deliver your verdict; ManiMux ends the rollout. "
                   "Tools wait for operator Resume after Pause; an operator_paused cancellation "
                   "is not task failure. After Resume, observe again before choosing actions."
                   if config.operator_control else "")
            ),
            user_message=config.task,
            toolkit=toolkit,
            max_turns=config.max_turns,
        )
        error = result.error
        if result.finish_result is None and error is None:
            error = "RPent planner exited without a finish verdict; inspect its tool transcript"
        (config.output_dir / "planner_result.json").write_text(
            json.dumps(
                {
                    "finish_result": result.finish_result,
                    "stats": result.stats,
                    "error": error,
                    "manimux_session_id": client.session_id,
                },
                indent=2,
            )
            + "\n"
        )
        return 1 if error else 0
    finally:
        toolkit.close()


def serve(argv: list[str]) -> int:
    """Own one ordinary runtime; both motion and policy calls use its mailbox."""
    from urllib.parse import urlsplit

    from manimux.cli import (
        _create_run_dir,
        _runtime_lock,
        load_config,
        read_yaml,
        resolve_local_path,
    )
    from manimux.dispatch.server import DispatchServer
    from manimux.runtime.dispatch import PolicyDispatch
    from manimux.runtime.edge import EdgeRuntime
    from manimux.runtime.motion import MotionTools

    parser = argparse.ArgumentParser(
        description="Serve RPent motion and VLA tools in one ManiMux rollout"
    )
    parser.add_argument("--config", type=Path, required=True, help="Shared RPent agent recipe")
    parser.add_argument("--local", type=Path)
    parser.add_argument("--motion", type=Path, help="Private workspace bounds and path checks")
    args = parser.parse_args(argv)
    recipe = load_agent_config(args.config, {})
    motion_path = args.motion or recipe.motion
    if recipe.experiment is None or motion_path is None:
        parser.error("agent recipe requires experiment and motion (or --motion)")
    raw = read_yaml(recipe.experiment)
    offline = (
        raw.get("robot", {}).get("type") == "manimux.dispatch.demo:build_robot"
        and args.local is None
        and raw.get("local") is None
    )
    config = load_config(
        recipe.experiment,
        local=None if offline else resolve_local_path(recipe.experiment, args.local),
    )
    motion_options = yaml.safe_load(motion_path.read_text())
    if not isinstance(motion_options, dict):
        parser.error("motion configuration must be a mapping")
    guard = motion_options.get("path_validator", {}).get("options", {})
    if "extrinsics_path" in guard:
        guard["extrinsics_path"] = str(
            (motion_path.resolve().parent / Path(guard["extrinsics_path"]).expanduser()).resolve()
        )
    import hashlib

    for path in (args.config, motion_path, guard.get("extrinsics_path")):
        if path is not None:
            source = Path(path).resolve()
            config["config_sources"][str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    endpoint = urlsplit(recipe.endpoint)
    if (
        endpoint.scheme != "http"
        or not endpoint.hostname
        or not endpoint.port
        or endpoint.path not in ("", "/")
    ):
        parser.error("endpoint must be http://host:port")
    if not recipe.policy_enabled:
        config["run"]["warmup_before_start"] = False
    dispatch = PolicyDispatch(config, policy_enabled=recipe.policy_enabled)
    with _runtime_lock(config, recipe.experiment, mode="rpent"):
        run_dir = _create_run_dir(config, recipe.experiment, mode="rpent")
        config["policy"]["options"]["response_log_dir"] = str(run_dir / "provider-responses")
        runtime = EdgeRuntime(config, run_dir, dispatch=dispatch, launch_mode="rpent")
        dispatch.motion_tools = MotionTools(
            runtime._robot.kinematics, motion_options, config["robot"]["control_hz"]
        )
        with DispatchServer(
            dispatch,
            host=endpoint.hostname,
            port=endpoint.port,
            token=os.environ.get("MANIMUX_DISPATCH_TOKEN"),
        ):
            try:
                runtime.run()
            except KeyboardInterrupt:
                return 130
            finally:
                dispatch.close("runtime_stopped")
    return 0
