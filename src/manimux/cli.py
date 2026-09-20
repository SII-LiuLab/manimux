from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
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


def read_yaml(path: str | Path) -> dict:
    """读取 YAML 映射；不解释其中的机器人、策略或执行参数。"""
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _merge(base: dict, overrides: dict) -> dict:
    """合并配置字典；列表整体替换，避免按位置猜测组件对应关系。"""
    result = deepcopy(base)
    for key, value in overrides.items():
        result[key] = (
            _merge(result[key], value)
            if isinstance(result.get(key), dict) and isinstance(value, dict)
            else deepcopy(value)
        )
    return result


def load_local(path: str | Path) -> dict:
    """只读取设备、服务和路径绑定；工位文件不能打开执行开关。"""
    source = Path(path).expanduser().resolve()
    bindings = read_yaml(source)
    # 保留这一处边界检查，防止把执行参数误写入 local 后被静默忽略。
    extra = set(bindings) - {"robot", "services", "paths"}
    extra |= set(bindings.get("robot", {})) - {"hardware", "components"}
    if extra:
        raise ValueError(f"unsupported local bindings: {sorted(extra)}")
    # 路径相对工位文件；设备参数保持原样，交给具体组件解释。
    bindings["paths"] = {
        name: (source.parent / Path(value).expanduser()).resolve()
        for name, value in bindings.get("paths", {}).items()
    }
    return bindings


def read_experiment(
    path: str | Path, *, local: str | Path | None = None, bind_local: bool = True
) -> dict:
    """展开明确的 policy/execution/policy_server 引用，再应用工位绑定。

    每个 section 最多引用一份基础 YAML，不递归继承。robot.config 始终是
    整机装配文件；这里不展开组件或创建 RobotModel。
    """
    source = Path(path).expanduser().resolve()
    raw = read_yaml(source)
    for name in ("policy", "execution", "policy_server"):
        section = raw.get(name, {})
        if "config" in section:
            reference = (source.parent / section.pop("config")).resolve()
            raw[name] = _merge(read_yaml(reference), section)
    robot = raw.setdefault("robot", {})
    if robot.get("config") is not None:
        robot["config"] = str((source.parent / robot["config"]).resolve())
    if raw.get("control_profile") is not None:
        raw["control_profile"] = str((source.parent / raw["control_profile"]).resolve())

    # CLI --local 相对当前目录；YAML 中的 local 相对实验文件。
    selected = Path(local).expanduser().resolve() if local is not None else None
    if selected is None and raw.get("local") is not None:
        selected = (source.parent / raw["local"]).resolve()
    if selected is not None:
        raw["local"] = str(selected)
    # 导出已绑定 checkpoint 的实验时保留 local 引用，不复制一份设备参数。
    if not bind_local:
        return raw
    bindings = load_local(selected) if selected is not None else {}
    hardware = bindings.get("robot", {}).get("hardware", {})
    components = bindings.get("robot", {}).get("components", {})
    services = bindings.get("services", {})
    paths = bindings.get("paths", {})
    if selected is not None:
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
    if "policy" in services:
        service = services["policy"]
        raw.setdefault("policy", {}).setdefault("options", {})["server"] = service["endpoint"]
        if "policy_server" in raw:
            address = urlsplit(service["endpoint"])
            # 客户端使用可达地址，服务端可以另选 bind_host（例如 0.0.0.0）。
            raw["policy_server"].update(
                host=service.get("bind_host", address.hostname), port=address.port
            )
    if "camera" in services and "camera_server" in raw:
        service = services["camera"]
        raw["camera_server"].update(
            pub_endpoint=service.get("bind_endpoint", service["endpoint"]),
            rep_endpoint=service.get("bind_request_endpoint", service["request_endpoint"]),
        )
    if "checkpoint" in paths:
        raw["policy_server"]["checkpoint_path"] = str(paths["checkpoint"])
    if "norm_stats" in paths:
        raw["policy_server"]["norm_stats_path"] = str(paths["norm_stats"])
    if "vlm_processor" in paths:
        raw["policy_server"]["vlm_processor_path"] = str(paths["vlm_processor"])
    if "output_dir" in paths:
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
    config = load_config(config_path, local=local)
    if executor is not None:
        config["execution"]["executor"] = executor
    return config


def _create_run_dir(config: dict, config_path: Path, *, mode: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"session-{timestamp}-{uuid.uuid4().hex[:8]}"
    run_dir = config["run"]["output_dir"] / run_id
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
    parser.add_argument("--local", type=Path, help="Local robot, device and service bindings")
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
    """补齐实验记录目录和步数；保持旧入口的路径含义。"""

    values = {
        "output_dir": Path("data"),
        "max_steps": 500,
        "experiment_mode": False,
        "layout_id": "",
        **options,
    }
    if values.get("output_dir") is not None:
        values["output_dir"] = Path(values["output_dir"])
    return values


def control_profile_parameters(**options) -> dict:
    """展开旧采集/部署共用控制参数，保留原合并约束。"""
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
    from manimux.runtime import execution_parameters, validate_runtime_parameters
    from manimux.viewer import viewer_parameters

    options = deepcopy(options)
    values = {
        "control_profile": None,
        "local": None,
        "policy_server": {},
        "camera_server": {},
        "sensors": [],
        "execution": {},
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
    if values.get("execution") is not None:
        values["execution"] = execution_parameters(**values["execution"])
    if values.get("viewer") is not None:
        values["viewer"] = viewer_parameters(**values["viewer"])
    if values.get("recording") is not None:
        values["recording"] = recording_parameters(**values["recording"])
    values["sensors"] = [sensor_parameters(**sensor) for sensor in values["sensors"]]
    validate_runtime_parameters(values)
    return values


def load_config(path: str | Path, *, local: str | Path | None = None) -> dict:
    """主入口使用的完整加载流程；保留共享控制参数原有的合并规则。"""
    from manimux.embodiments.robot import shared_robot_parameters
    from manimux.policies.base import action_interval
    from manimux.runtime.executors.limits import motion_limits_parameters
    from manimux.runtime.safety import command_safety_parameters

    config_path = Path(path)
    raw = read_experiment(config_path, local=local)
    # read_experiment 已解析文件引用；这里只组合共享控制参数和实验参数。
    robot = raw["robot"] = shared_robot_parameters(**raw.get("robot", {}))
    profile = None
    if raw.get("control_profile") is not None:
        profile_path = Path(raw["control_profile"])
        profile = control_profile_parameters(**read_yaml(profile_path))
        raw["control_profile"] = profile_path
        robot = raw.setdefault("robot", {})
        _set_shared_value(robot, "type", profile["robot"]["type"], "robot.type")
        _set_shared_value(robot, "group_dims", profile["robot"]["group_dims"], "robot.group_dims")
        options = robot.setdefault("options", {})
        for name, value in profile["robot"]["options"].items():
            _set_shared_value(options, name, value, f"robot.options.{name}")
        policy = raw.setdefault("policy", {})
        _set_shared_value(policy, "action_dt_s", profile["action_dt_s"], "policy.action_dt_s")
        execution = raw.setdefault("execution", {})
        envelope = profile["command_safety"] or command_safety_parameters()
        if "command_safety" in execution:
            local = command_safety_parameters(**execution["command_safety"] or {})
            if local != envelope:
                raise ValueError("execution.command_safety conflicts with control_profile")
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
                "execution.motion_limits",
            )
    execution = raw.get("execution", {})
    if execution.get("motion_limits") is not None:
        motion = motion_limits_parameters(**execution["motion_limits"])
        smooth = execution.setdefault("smooth", {})
        for name, value in deepcopy(motion["arm"]).items():
            _set_shared_value(smooth, name, value, f"execution.smooth.{name}")
        gripper = smooth.setdefault("gripper", {"mode": "continuous"})
        for name, value in deepcopy(motion["gripper"]).items():
            _set_shared_value(gripper, name, value, f"execution.smooth.gripper.{name}")
    config = prepare_experiment(**raw)
    if profile is not None and not math.isclose(
        action_interval(config["policy"]), profile["action_dt_s"], rel_tol=1e-9
    ):
        raise ValueError("policy effective action interval conflicts with control_profile")
    return config


def _set_shared_value(target: dict, name: str, value: object, label: str) -> None:
    if name in target and target[name] != value:
        raise ValueError(f"{label} conflicts with control_profile; edit the shared profile")
    target[name] = value
