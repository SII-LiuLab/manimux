from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import signal
import subprocess
import uuid
from copy import deepcopy
from datetime import UTC, datetime
from json import dumps, loads
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import yaml

if TYPE_CHECKING:
    from manimux.runtime.lock import RuntimeInstanceLock

DEFAULT_LOCAL_PATH = Path(__file__).resolve().parent / "configs/local/station.yaml"


def resolve_local_path(
    experiment: str | Path | None = None, local: str | Path | None = None
) -> Path:
    """Select the station for a deployment entry point; never fall back on missing files.

    Explicit CLI paths are relative to the working directory. A station reference
    saved in an experiment is relative to that experiment. Offline config readers
    do not call this resolver, so inspecting a model does not require a station.
    """
    if local is not None:
        return Path(local).expanduser().resolve()
    if experiment is not None:
        source = Path(experiment).expanduser().resolve()
        reference = read_yaml(source).get("local")
        if reference is not None:
            return (source.parent / Path(reference).expanduser()).resolve()
    return DEFAULT_LOCAL_PATH


def read_yaml(path: str | Path) -> dict:
    """Read a YAML mapping without interpreting robot, policy or execution fields."""
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _merge(base: dict, overrides: dict) -> dict:
    """Merge mappings; replace lists instead of guessing component matches by index."""
    result = deepcopy(base)
    for key, value in overrides.items():
        result[key] = (
            _merge(result[key], value)
            if isinstance(result.get(key), dict) and isinstance(value, dict)
            else deepcopy(value)
        )
    return result


def load_local(path: str | Path) -> dict:
    """Read device, service and path bindings; execution flags belong to experiments."""
    source = Path(path).expanduser().resolve()
    bindings = read_yaml(source)
    # Retain the existing boundary check so misplaced execution flags are not ignored.
    extra = set(bindings) - {"robot", "services", "paths"}
    extra |= set(bindings.get("robot", {})) - {"hardware", "components"}
    if extra:
        raise ValueError(f"unsupported local bindings: {sorted(extra)}")
    # Resolve paths beside the station file; components interpret device parameters.
    bindings["paths"] = {
        name: (source.parent / Path(value).expanduser()).resolve()
        for name, value in bindings.get("paths", {}).items()
    }
    return bindings


def read_experiment(
    path: str | Path, *, local: str | Path | None = None, bind_local: bool = True
) -> dict:
    """Expand component references and explicit station bindings.

    Each section references at most one base YAML, without recursive inheritance.
    robot.config remains the assembly path; reading does not construct RobotModel.
    """
    source = Path(path).expanduser().resolve()
    raw = read_yaml(source)
    from manimux.embodiments.robot import apply_action_contract
    from manimux.policies.base import backend_identity_from_recipe

    backend_identity = None
    for name in ("policy", "inference", "executor", "policy_server"):
        section = raw.get(name, {})
        if "config" in section:
            reference = (source.parent / section.pop("config")).resolve()
            raw[name] = _merge(read_yaml(reference), section)
        if name == "policy_server" and isinstance(raw.get(name), dict):
            backend_identity = raw[name].pop("backend_identity", None)
    adapter = raw.get("policy", {}).get("adapter", {})
    diff_ik = adapter.get("diff_ik", {})
    if isinstance(diff_ik, dict) and "config" in diff_ik:
        reference = (source.parent / diff_ik.pop("config")).resolve()
        adapter["diff_ik"] = _merge(read_yaml(reference), diff_ik)
    robot = raw.setdefault("robot", {})
    if robot.get("config") is not None:
        robot["config"] = str((source.parent / robot["config"]).resolve())
        contract = read_yaml(robot["config"]).get("action_contract")
        if contract is not None:
            apply_action_contract(raw, contract)
    if raw.get("control_profile") is not None:
        raw["control_profile"] = str((source.parent / raw["control_profile"]).resolve())

    # CLI station paths are relative to cwd; saved references are experiment-relative.
    selected = Path(local).expanduser().resolve() if local is not None else None
    if selected is None and raw.get("local") is not None:
        selected = (source.parent / raw["local"]).resolve()
    if selected is not None:
        raw["local"] = str(selected)
    # Exported experiments retain their station reference instead of copying devices.
    if bind_local and selected is not None:
        raw = bind_station(raw, selected)
    if backend_identity is not None:
        generated = backend_identity_from_recipe(raw["policy_server"], backend_identity)
        policy = raw.setdefault("policy", {})
        configured = policy.get("expected_backend")
        if configured is not None and configured != generated:
            raise ValueError("policy.expected_backend conflicts with policy_server recipe")
        policy["expected_backend"] = generated
    return raw


def bind_station(config: dict, local: str | Path) -> dict:
    """Bind device addresses and file locations without changing experiment semantics.

    Both experiment and standalone model-server entry points use this operation.
    Keep the caller's configuration intact so a deployment can be inspected before
    binding another station. Missing station files raise the normal file-read error.
    """
    raw = deepcopy(config)
    selected = Path(local).expanduser().resolve()
    raw["local"] = str(selected)
    bindings = load_local(selected)
    robot = raw.setdefault("robot", {})
    hardware = bindings.get("robot", {}).get("hardware", {})
    components = bindings.get("robot", {}).get("components", {})
    services = bindings.get("services", {})
    paths = bindings.get("paths", {})
    options = robot.setdefault("options", {})
    if hardware:
        options["hardware"] = _merge(options.get("hardware", {}), hardware)
    if components:
        options["component_hardware"] = _merge(
            options.get("component_hardware", {}), components
        )
    for sensor in raw.get("sensors", []):
        service = sensor.get("service")
        if service in services:
            sensor.setdefault("options", {})["endpoint"] = services[service]["endpoint"]
    policy = raw.setdefault("policy", {})
    policy_service = policy.get("service", "policy")
    if policy_service in services:
        service = services[policy_service]
        policy.setdefault("options", {})["server"] = service["endpoint"]
        if "policy_server" in raw:
            address = urlsplit(service["endpoint"])
            # Clients use reachable addresses; servers may bind to a different host.
            raw["policy_server"].update(
                host=service.get("bind_host", address.hostname), port=address.port
            )
    if "camera" in services and "camera_server" in raw:
        service = services["camera"]
        raw["camera_server"].update(
            pub_endpoint=service.get("bind_endpoint", service["endpoint"]),
            rep_endpoint=service.get("bind_request_endpoint", service["request_endpoint"]),
        )
    server = raw.get("policy_server")
    if server is not None:
        # Recipe-only metadata configures ManiMux's handshake check and is not
        # part of the model deployment arguments.
        server.pop("backend_identity", None)
        # Provider field names differ; station paths do not change the selected
        # checkpoint variant, normalization identity, horizon or action contract.
        pi05 = server.get("policy_name") == "Pi_05"
        if pi05:
            # The experiment selects a checkpoint; the station only supplies
            # its storage root, so switching tasks cannot reuse one fixed model.
            for key in ("model_path", "norm_stats_path"):
                server[key] = str((paths["checkpoints"] / server[key]).resolve())
        elif "checkpoint" in paths:
            server["checkpoint_path"] = str(paths["checkpoint"])
        if "norm_stats" in paths:
            server["norm_stats_path"] = str(paths["norm_stats"])
        if "vlm_processor" in paths:
            server["vlm_processor_path"] = str(paths["vlm_processor"])
        expected = raw.get("policy", {}).get("expected_backend") or {}
        model = expected.get("model", {})
        if pi05 and model.get("policy_family") == "pi05":
            model["model_root"] = server["model_path"]
            model["norm_stats_path"] = server["norm_stats_path"]
    if "output_dir" in paths and "run" in raw:
        raw["run"]["output_dir"] = str(paths["output_dir"])
    return raw


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


def _load_config(
    config_path: Path, executor: str | None = None, *, local: Path | None = None
) -> dict:
    config = load_config(config_path, local=resolve_local_path(config_path, local))
    if executor is not None:
        config["executor"]["type"] = executor
    return config


def _create_run_dir(config: dict, config_path: Path, *, mode: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"session-{timestamp}-{uuid.uuid4().hex[:8]}"
    run_dir = config["run"]["output_dir"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    resolved_config = config_path.expanduser().resolve()
    config_sha256 = hashlib.sha256(resolved_config.read_bytes()).hexdigest()
    repository_root = Path(__file__).resolve().parents[1]
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
                "config": loads(dumps(deepcopy(config), default=str)),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    return run_dir


def _run(config_path: Path, executor: str | None = None, *, local: Path | None = None) -> int:
    from manimux.runtime import build_runtime
    from manimux.runtime.lock import RuntimeLockError

    config = _load_config(config_path, executor, local=local)
    try:
        with _runtime_lock(config, config_path, mode="run"):
            run_dir = _create_run_dir(config, config_path, mode="run")
            result = build_runtime(config, run_dir, launch_mode="run").run()
    except KeyboardInterrupt:
        print("interrupted; robot shutdown and partial episode save completed")
        return 130
    except RuntimeLockError as exc:
        print(f"runtime not started: {exc}")
        return 2
    status = "completed" if result.success else "FAULT"
    print(
        f"{status} {result.steps} steps; reason={result.terminal_reason}; "
        f"accepted={result.accepted_plans} rejected={result.rejected_plans}; "
        f"episode={result.episode_dir}"
    )
    return 0 if result.success else 2


def _serve(config_path: Path, executor: str | None = None, *, local: Path | None = None) -> int:
    # Session/recovery dependencies are only needed for the interactive service.
    from manimux.runtime.lock import RuntimeLockError
    from manimux.session import RuntimeSessionService

    config = _load_config(config_path, executor, local=local)
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
    config: dict,
    config_path: Path,
    *,
    mode: str,
) -> RuntimeInstanceLock:
    from manimux.runtime.lock import RuntimeInstanceLock

    identity = config["viewer"]["robot"].strip() or config["robot"]["type"]
    return RuntimeInstanceLock(identity, mode=mode, config_path=config_path)


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--local", type=Path,
        help="Station bindings (default: manimux/configs/local/station.yaml)",
    )
    parser.add_argument("--executor", choices=("direct", "smooth", "mpc"))
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Runtime diagnostic verbosity (default: INFO)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="manimux")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run one local robot-policy session")
    _add_runtime_arguments(run_parser)
    serve_parser = subparsers.add_parser(
        "serve", help="keep one runtime service available for Viewer-controlled rollouts"
    )
    _add_runtime_arguments(serve_parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    mp.freeze_support()
    signal.signal(signal.SIGTERM, _handle_termination)
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command == "run":
        return _run(args.config, args.executor, local=args.local)
    if args.command == "serve":
        return _serve(args.config, args.executor, local=args.local)
    raise AssertionError(f"unhandled command {args.command}")


def run_parameters(**options) -> dict:
    """补齐实验记录目录和控制步数。"""

    if "max_steps" in options:
        raise ValueError("unsupported run field: max_steps")
    values = {
        "output_dir": Path("data"),
        "max_control_steps": 500,
        "experiment_mode": False,
        "layout_id": "",
        **options,
    }
    if values.get("output_dir") is not None:
        values["output_dir"] = Path(values["output_dir"])
    return values


def control_profile_parameters(**options) -> dict:
    """展开采集与部署共用的本体布局和运动限制；动作时间由 policy 声明。"""
    from manimux.embodiments.robot import shared_robot_parameters
    from manimux.runtime.executors.limits import motion_limits_parameters
    from manimux.runtime.safety import command_safety_parameters

    values = {
        "command_safety": None,
        "motion_limits": None,
        **options,
    }
    if "control_profile" in options:
        raise ValueError("recursive control_profile is not supported")
    values["robot"] = shared_robot_parameters(**values["robot"])
    if values.get("command_safety") is not None:
        values["command_safety"] = command_safety_parameters(**values["command_safety"])
    if values.get("motion_limits") is not None:
        values["motion_limits"] = motion_limits_parameters(**values["motion_limits"])
    return values


def prepare_experiment(**options) -> dict:
    """把各模块处理过的参数组合成实验字典；不创建机器人或连接硬件。"""
    from manimux.embodiments.robot import robot_parameters
    from manimux.embodiments.sensor import sensor_parameters
    from manimux.policies.base import policy_parameters
    from manimux.recording import recording_parameters
    from manimux.runtime import (
        executor_parameters,
        inference_parameters,
        validate_runtime_parameters,
    )
    from manimux.viewer import viewer_parameters

    options = deepcopy(options)
    values = {
        "control_profile": None,
        "local": None,
        "policy_server": {},
        "camera_server": {},
        "sensors": [],
        "inference": {},
        "executor": {},
        "viewer": {},
        "recording": {},
        **options,
    }
    if values.get("control_profile") is not None:
        values["control_profile"] = Path(values["control_profile"])
    if values.get("local") is not None:
        values["local"] = Path(values["local"])
    values["run"] = run_parameters(**values["run"])
    values["robot"] = robot_parameters(**values["robot"])
    values["policy"] = policy_parameters(**values["policy"])
    values["executor"] = executor_parameters(**values["executor"])
    values["inference"] = inference_parameters(executor=values["executor"], **values["inference"])
    if values.get("viewer") is not None:
        values["viewer"] = viewer_parameters(**values["viewer"])
    if values.get("recording") is not None:
        values["recording"] = recording_parameters(**values["recording"])
    values["sensors"] = [sensor_parameters(**sensor) for sensor in values["sensors"]]
    validate_runtime_parameters(values)
    return values


def load_config(path: str | Path, *, local: str | Path | None = None) -> dict:
    """主入口使用的完整加载流程；保留共享控制参数原有的合并规则。"""
    from manimux.embodiments.robot import (
        action_contract_group_indices,
        shared_robot_parameters,
    )
    from manimux.runtime.executors.limits import motion_limits_parameters
    from manimux.runtime.safety import command_safety_parameters

    config_path = Path(path)
    raw = read_experiment(config_path, local=local)
    gripper_indices = action_contract_group_indices(raw)
    # read_experiment 已解析文件引用；这里只组合共享控制参数和实验参数。
    robot = raw["robot"] = shared_robot_parameters(**raw.get("robot", {}))
    profile = None
    if raw.get("control_profile") is not None:
        profile_path = Path(raw["control_profile"])
        profile_raw = read_yaml(profile_path)
        if profile_raw.get("rate_contract") is not None:
            from manimux.embodiments.arm.tianji.arm import resolve_rate_contract

            assembly = read_yaml(raw["robot"]["config"])
            controller = _merge(
                assembly.get("hardware", {}),
                raw.get("robot", {}).get("options", {}).get("hardware", {}),
            )
            profile_raw = resolve_rate_contract(profile_raw, controller, gripper_indices)
        if gripper_indices is not None and profile_raw.get("motion_limits") is not None:
            profile_gripper = profile_raw["motion_limits"].setdefault("gripper", {})
            _set_shared_value(
                profile_gripper,
                "group_indices",
                deepcopy(gripper_indices),
                "control_profile.motion_limits.gripper.group_indices",
            )
        profile = control_profile_parameters(**profile_raw)
        raw["control_profile"] = profile_path
        robot = raw.setdefault("robot", {})
        _set_shared_value(robot, "type", profile["robot"]["type"], "robot.type")
        _set_shared_value(robot, "group_dims", profile["robot"]["group_dims"], "robot.group_dims")
        options = robot.setdefault("options", {})
        for name, value in profile["robot"]["options"].items():
            _set_shared_value(options, name, value, f"robot.options.{name}")
        # 模型动作间隔属于 checkpoint 的动作约定，实验显式填写 policy.action_dt_s。
        # 共享本体 profile 只提供布局和限制，不再隐式决定轨迹的时间轴。
        execution = raw.setdefault("executor", {})
        envelope = profile["command_safety"] or command_safety_parameters()
        if "command_safety" in execution:
            local = command_safety_parameters(**execution["command_safety"] or {})
            if local != envelope:
                raise ValueError("executor.command_safety conflicts with control_profile")
        execution["command_safety"] = deepcopy(envelope)
        if profile["motion_limits"] is not None:
            if execution.get("motion_limits") is not None:
                execution["motion_limits"] = deepcopy(
                    motion_limits_parameters(**execution["motion_limits"])
                )
            _set_shared_value(
                execution,
                "motion_limits",
                deepcopy(profile["motion_limits"]),
                "executor.motion_limits",
            )
    execution = raw.get("executor", {})
    if execution.get("motion_limits") is not None:
        motion = motion_limits_parameters(**execution["motion_limits"])
        smooth = execution.setdefault("smooth", {})
        for name, value in deepcopy(motion["arm"]).items():
            _set_shared_value(smooth, name, value, f"executor.smooth.{name}")
        gripper = smooth.setdefault("gripper", {"mode": "continuous"})
        for name, value in deepcopy(motion["gripper"]).items():
            _set_shared_value(gripper, name, value, f"executor.smooth.gripper.{name}")
    return prepare_experiment(**raw)


def _set_shared_value(target: dict, name: str, value: object, label: str) -> None:
    if name in target and target[name] != value:
        raise ValueError(f"{label} conflicts with control_profile; edit the shared profile")
    target[name] = value
