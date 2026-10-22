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
        from manimux.configuration import compose
        reference = compose(source)[0].get("local")
        if reference is not None:
            return (source.parent / Path(reference).expanduser()).resolve()
    return DEFAULT_LOCAL_PATH


def read_yaml(path: str | Path) -> dict:
    """Read a YAML mapping without interpreting robot, policy or execution fields."""
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _resolve_camera_references(recipe: dict, directory: Path) -> dict:
    """Resolve each camera's component file relative to its declaring YAML."""
    result = deepcopy(recipe)
    for camera in result.get("cameras", {}).values():
        if "config" in camera:
            camera["config"] = str((directory / camera["config"]).resolve())
    return result


def read_camera_recipe(path: str | Path) -> dict:
    source = Path(path).expanduser().resolve()
    return _resolve_camera_references(read_yaml(source), source.parent)


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


def _merge_executor(base: dict, overrides: dict) -> dict:
    """Merge an executor recipe while replacing a changed tagged gripper mode."""

    prepared = deepcopy(base)
    base_gripper = prepared.get("smooth", {}).get("gripper")
    override_gripper = overrides.get("smooth", {}).get("gripper")
    if (
        isinstance(base_gripper, dict)
        and isinstance(override_gripper, dict)
        and "mode" in override_gripper
        and override_gripper["mode"] != base_gripper.get("mode")
    ):
        prepared.setdefault("smooth", {})["gripper"] = {}
    return _merge(prepared, overrides)


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

    Experiment/task bases compose first; component sections reference one recipe.
    robot.config remains the assembly path; reading does not construct RobotModel.
    """
    source = Path(path).expanduser().resolve()
    from manimux.configuration import apply_initial_pose, compose
    raw, sources = compose(source)
    if "task" in raw.get("run", {}):
        raise ValueError("Rename run.task to run.task_prompt; experiment.task remains the task ID")
    from manimux.config_validation import EXPERIMENT_FIELDS, check_fields
    check_fields(raw, EXPERIMENT_FIELDS, "experiment")
    raw["config_sources"] = sources
    from manimux.embodiments.layout import apply_action_contract, assembly_action_contract
    from manimux.policies.base import backend_identity_from_recipe

    backend_identity = None
    for name in ("policy", "inference", "executor", "policy_server", "camera_server", "evaluation"):
        section = raw.get(name, {})
        if name == "evaluation" and section is None:
            continue
        if name == "camera_server":
            section = _resolve_camera_references(section, source.parent)
            if section:
                raw[name] = section
        if "config" in section:
            reference = (source.parent / section.pop("config")).resolve()
            sources[str(reference)] = hashlib.sha256(reference.read_bytes()).hexdigest()
            base = (
                read_camera_recipe(reference) if name == "camera_server" else read_yaml(reference)
            )
            raw[name] = (
                _merge_executor(base, section) if name == "executor" else _merge(base, section)
            )
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
        contract = assembly_action_contract(robot["config"])
        if contract is not None:
            apply_action_contract(raw, contract)
    if raw.get("control_profile") is not None:
        raw["control_profile"] = str((source.parent / raw["control_profile"]).resolve())

    apply_initial_pose(raw)

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
    # Snapshot every referenced YAML, including assembly/driver and station files.
    def record_references(value, directory: Path) -> None:
        if isinstance(value, list):
            for item in value:
                record_references(item, directory)
        elif isinstance(value, dict):
            for key, item in value.items():
                if key in {"config", "control_profile", "local"} and isinstance(item, str):
                    dependency = (directory / Path(item).expanduser()).resolve()
                    if dependency.is_file() and str(dependency) not in visited:
                        visited.add(str(dependency))
                        raw["config_sources"][str(dependency)] = hashlib.sha256(dependency.read_bytes()).hexdigest()
                        record_references(read_yaml(dependency), dependency.parent)
                else:
                    record_references(item, directory)
    visited: set[str] = set()
    for dependency in list(raw["config_sources"]):
        record_references(read_yaml(dependency), Path(dependency).parent)
    record_references(raw, source.parent)
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
    from manimux.policies.recipe import resolve_recipe_paths
    raw = resolve_recipe_paths(raw, bindings.get("paths", {}))
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
    camera_service = raw.get("camera_server", {}).get("service", "camera")
    if camera_service in services and "camera_server" in raw:
        service = services[camera_service]
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
        if pi05 and server.get("observation_profile") == "tianji_taccap_pi05_zero_pose":
            # New EE profile: explicit checkpoint directories or a station storage root.
            if "checkpoints" in paths:
                for key in ("model_path", "norm_stats_path"):
                    server[key] = str((paths["checkpoints"] / server[key]).resolve())
            elif "checkpoint" in paths:
                server["model_path"] = str(paths["checkpoint"])
                server["norm_stats_path"] = str(paths["checkpoint"].parent.parent / "normalization")
        elif pi05:
            # The experiment selects a checkpoint; the station only supplies
            # its storage root, so switching tasks cannot reuse one fixed model.
            for key in ("model_path", "norm_stats_path"):
                server[key] = str((paths["checkpoints"] / server[key]).resolve())
        elif "checkpoint" in paths:
            candidates = [key for key in ("checkpoint_path", "ckpt_dir", "model_path", "model_dir", "model_root")
                          if key in server]
            # Recipes may carry both a model directory and its selected weight
            # file. The explicit file field is the checkpoint override target.
            if "checkpoint_path" in candidates:
                candidates = ["checkpoint_path"]
            if len(candidates) != 1:
                raise ValueError("paths.checkpoint is ambiguous for this recipe; use explicit ${paths.<root>} references")
            server[candidates[0]] = str(paths["checkpoint"])
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
    config_path: Path, executor: str | None = None, *, local: Path | None = None,
    control_timing: bool = False,
) -> dict:
    config = load_config(config_path, local=resolve_local_path(config_path, local))
    if executor is not None:
        config["executor"]["type"] = executor
    if control_timing:
        config["run"]["control_timing"] = True
    return config


def _create_run_dir(config: dict, config_path: Path, *, mode: str,
                    experiment_selection: dict | None = None) -> Path:
    from manimux.recording.provenance import (
        capture_source_provenance,
        deployment_artifact_provenance,
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"session-{timestamp}-{uuid.uuid4().hex[:8]}"
    run_dir = config["run"]["output_dir"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    resolved_config = config_path.expanduser().resolve()
    config_sha256 = (experiment_selection["config_sha256"] if experiment_selection else
                     hashlib.sha256(resolved_config.read_bytes()).hexdigest())
    repository_root = Path(__file__).resolve().parents[1]
    recipe = config.get("policy_server", {})
    sources = capture_source_provenance(repository_root, policy_name=recipe.get("policy_name"))
    repositories = sources["repositories"]
    with (run_dir / "session-manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "session_id": run_id,
                "mode": mode,
                "workflow": "experiment" if experiment_selection else "trial",
                "created_at": datetime.now(UTC).isoformat(),
                "git_sha": repositories["manimux"].get("head"),
                "xpolicylab_git_sha": (
                    repositories["xpolicylab"].get("head") if "xpolicylab" in repositories
                    else _git_sha(repository_root / "third_party" / "XPolicyLab")
                ),
                "source_provenance": sources,
                "deployment_artifacts": deployment_artifact_provenance(recipe),
                "config_path": str(resolved_config),
                "config_sha256": config_sha256,
                "config": loads(dumps(deepcopy(config), default=str)),
                "config_sources": config.get("config_sources", {}),
                **({"collection": config["collection"]} if "collection" in config else {}),
                **({"experiment_selection": experiment_selection} if experiment_selection else {}),
            },
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    return run_dir


def _trial_config(config: dict, config_path: Path, output_dir: Path | None = None) -> dict:
    """Reuse a runtime recipe without borrowing study identity, progress or storage."""
    from manimux.collections import safe_id

    trial = deepcopy(config)
    trial.pop("collection", None)
    run = trial.setdefault("run", {})
    root = (output_dir if output_dir is not None
            else Path(run.get("output_dir") or "data") / "trials")
    identity = trial.get("experiment") or {}
    if identity:
        destination = (root / safe_id(identity.get("task"), "task id")
                       / safe_id(identity.get("variant"), "variant id"))
    else:
        # Equal filenames in separate directories must not share a trial namespace.
        source_id = hashlib.sha256(str(config_path.resolve()).encode()).hexdigest()[:8]
        destination = root / f"{config_path.stem}-{source_id}"
    run.update(
        output_dir=destination.expanduser().resolve(), experiment_mode=False,
        experiment_name="", condition="", notes="", layout_id="", repeat_id=None,
        reference_layout=None, experiment_template=None, history_sessions=[],
    )
    return trial


def _run_once(
    config_path: Path, executor: str | None = None, *, local: Path | None = None,
    control_timing: bool = False, output_dir: Path | None = None,
) -> int:
    from manimux.runtime import build_runtime
    from manimux.runtime.lock import RuntimeLockError

    config = _trial_config(
        _load_config(config_path, executor, local=local, control_timing=control_timing),
        config_path, output_dir,
    )
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
    status = "completed" if result.run_completed else "FAULT"
    print(
        f"{status} {result.steps} steps; reason={result.terminal_reason}; "
        f"accepted={result.accepted_plans} rejected={result.rejected_plans}; "
        f"episode={result.episode_dir}"
    )
    return 0 if result.run_completed else 2


def _run_interactive(
    config_path: Path | None, executor: str | None = None, *, local: Path | None = None,
    control_timing: bool = False,
    collection: Path | None = None, experiment_id: str | None = None,
    output_dir: Path | None = None,
) -> int:
    # Session/recovery dependencies are only needed for the interactive service.
    from manimux.runtime.lock import RuntimeLockError
    from manimux.session import RuntimeSessionService

    catalog = None
    if collection is not None:
        from manimux.collections import load_collection
        if executor is not None or control_timing or output_dir is not None:
            raise ValueError(
                "Config-list runs own their output root and variant parameters; edit the YAML"
            )
        catalog = load_collection(collection, local=local, experiment=experiment_id)
        first = catalog.choices[catalog.default_id]
        config_path, config = first.path, deepcopy(first.config)
    else:
        config = _trial_config(
            _load_config(config_path, executor, local=local, control_timing=control_timing),
            config_path, output_dir,
        )
    if experiment_id is not None and collection is None:
        raise ValueError("--experiment-id requires --config-list")

    # Preserve the historical record/protocol mode; the public command is now run.
    def create_selection_dir(choice_id: str) -> Path:
        choice = catalog.choices[choice_id]
        return _create_run_dir(choice.config, choice.path, mode="serve",
                               experiment_selection=catalog.selection(choice_id))

    def create_pose_dir(selected: dict) -> Path:
        from manimux.configuration import initial_pose_metadata
        if catalog:
            choice_id = selected["collection"]["experiment_id"]
            choice = catalog.choices[choice_id]
            selection = {**catalog.selection(choice_id), "initial_pose": initial_pose_metadata(selected)}
            return _create_run_dir(selected, choice.path, mode="serve", experiment_selection=selection)
        return _create_run_dir(selected, config_path, mode="serve")

    try:
        with _runtime_lock(config, config_path, mode="serve"):
            run_dir = (create_selection_dir(catalog.default_id) if catalog else
                       _create_run_dir(config, config_path, mode="serve"))
            RuntimeSessionService(
                config, run_dir, experiment_list=catalog,
                selection_dir_factory=create_selection_dir if catalog else None,
                pose_dir_factory=create_pose_dir,
                allow_experiments=catalog is not None,
            ).serve()
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

    identity = config["robogui"]["robot"].strip() or config["robot"]["type"]
    return RuntimeInstanceLock(identity, mode=mode, config_path=config_path)


def _add_runtime_arguments(parser: argparse.ArgumentParser, *, allow_list: bool = False) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    if allow_list:
        source.add_argument(
            "--config-list", type=Path, help="Config list with trial plan and recording root"
        )
        parser.add_argument("--experiment-id", help="Initially select this config-list entry")
    parser.add_argument(
        "--output-dir", type=Path,
        help="Trial recording root for --config only; default: <run.output_dir>/trials",
    )
    parser.add_argument(
        "--local", type=Path,
        help="Station bindings (default: manimux/configs/local/station.yaml)",
    )
    parser.add_argument("--executor", choices=("direct", "smooth", "mpc"))
    parser.add_argument(
        "--control-timing", action="store_true",
        help="Collect bounded per-cycle timings; save and summarize when the rollout ends",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Runtime diagnostic verbosity (default: INFO)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="manimux")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser(
        "run", help="try one config or manage a config list; --once exits after one trial"
    )
    _add_runtime_arguments(run_parser, allow_list=True)
    run_parser.add_argument(
        "--once", action="store_true",
        help=("Run one standalone --config rollout and exit; connects immediately "
              "(GUI still gates Start)"),
    )
    inspect_parser = subparsers.add_parser(
        "inspect", help="Validate configuration without starting services"
    )
    source = inspect_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--config-list", type=Path)
    inspect_parser.add_argument("--output-dir", type=Path, help="Trial recording root for --config")
    inspect_parser.add_argument("--local", type=Path)
    inspect_parser.add_argument("--experiment-id")
    inspect_parser.add_argument("--index", action="store_true", help="Refresh derived record index")
    from manimux.deployment import add_commands
    add_commands(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    mp.freeze_support()
    signal.signal(signal.SIGTERM, _handle_termination)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in {"doctor", "up", "status", "down", "attach"}:
        from manimux.deployment import main as deployment_main
        return deployment_main(args)
    if args.command == "run":
        if args.once and args.config_list is not None:
            parser.error(
                "--once requires a standalone --config; "
                "select a concrete variant with --config for a standalone trial"
            )
        if args.experiment_id is not None and args.config_list is None:
            parser.error("--experiment-id requires --config-list")
    if args.command == "inspect":
        if args.config is not None:
            if args.index or args.experiment_id:
                parser.error("--index and --experiment-id require --config-list")
            config = _trial_config(_load_config(args.config, local=args.local),
                                   args.config, args.output_dir)
            print(json.dumps({"workflow": "trial", "config": config},
                             indent=2, default=str, ensure_ascii=False))
            return 0
        if args.output_dir is not None:
            parser.error("--output-dir is for --config trials; config lists own recording.root")
        from manimux.collections import load_collection, write_inventory
        catalog = load_collection(args.config_list, local=args.local, experiment=args.experiment_id)
        rows = []
        for choice in catalog.choices.values():
            c = choice.config
            rows.append({**choice.metadata(), "output_dir": str(c['run']['output_dir']),
                         "inference": c['inference'],
                         "history_sessions": len(c['run']['history_sessions']),
                         "history_rejected": c['collection']['history_rejected']})
        print(json.dumps(rows, indent=2, default=str, ensure_ascii=False))
        if args.index:
            first = catalog.choices[catalog.default_id].config['collection']
            print(write_inventory(Path(first['root']) / first['id']))
        return 0
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command == "run":
        if args.once:
            return _run_once(
                args.config, args.executor, local=args.local, control_timing=args.control_timing,
                output_dir=args.output_dir,
            )
        return _run_interactive(
            args.config, args.executor, local=args.local, control_timing=args.control_timing,
            collection=args.config_list, experiment_id=args.experiment_id,
            output_dir=args.output_dir,
        )
    raise AssertionError(f"unhandled command {args.command}")


def run_parameters(**options) -> dict:
    """Fill in the experiment recording directory and control step count."""
    from manimux.config_validation import merge_parameters
    from manimux.embodiments.sensor.reader import sensor_reading_parameters
    from manimux.evaluation.identity import research_template

    # Saved manifests used run.task before task_prompt became the public field.
    # Normalize only at this boundary; live YAML rejects the old spelling above.
    if "task" in options:
        if "task_prompt" in options:
            raise ValueError("Specify task_prompt only; both task and task_prompt were supplied")
        options["task_prompt"] = options.pop("task")
    if "max_steps" in options:
        raise ValueError("unsupported run field: max_steps")
    values = merge_parameters({
        "output_dir": Path("data"),
        # Per-rollout RoboGUI selection, retained in recorded configs; not a YAML setting.
        "execution_mode": None,
        "max_control_steps": 500,
        "warmup_before_start": False,
        "control_timing": False,
        "timing_max_cycles": 20000,
        "sensor_reading": {},
        "experiment_template": None,
        "experiment_name": "",
        "condition": "",
        "notes": "",
        "experiment_mode": False,
        "layout_id": "",
        "repeat_id": None,
        "reference_layout": None,
    }, options, 'run', extra=('task_prompt', 'initial_pose', 'initial_pose_options', 'history_sessions', 'agent_run_dir'))
    if values.get("output_dir") is not None:
        values["output_dir"] = Path(values["output_dir"])
    if values["execution_mode"] not in {None, "vla", "rpent"}:
        raise ValueError("run.execution_mode must be vla or rpent")
    if type(values["max_control_steps"]) is not int or values["max_control_steps"] < 0:
        raise ValueError("run.max_control_steps must be a non-negative integer; 0 disables the limit")
    if not isinstance(values["warmup_before_start"], bool):
        raise ValueError("run.warmup_before_start must be boolean")
    if not isinstance(values["control_timing"], bool):
        raise ValueError("run.control_timing must be boolean")
    if type(values["timing_max_cycles"]) is not int or values["timing_max_cycles"] <= 0:
        raise ValueError("run.timing_max_cycles must be a positive integer")
    values["experiment_template"] = research_template(values["experiment_template"])
    values["sensor_reading"] = sensor_reading_parameters(**values["sensor_reading"])
    return values


def control_profile_parameters(**options) -> dict:
    """Resolve the embodiment layout and motion limits shared by collection and deployment; policy declares action timing."""
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
    """Combine resolved component parameters into an experiment dictionary without creating robots or connecting hardware."""
    from manimux.embodiments.robot import robot_parameters
    from manimux.embodiments.sensor import sensor_parameters
    from manimux.evaluation.rubric import evaluation_parameters
    from manimux.policies.base import policy_parameters
    from manimux.recording import recording_parameters
    from manimux.robogui import robogui_parameters
    from manimux.runtime import (
        executor_parameters,
        inference_parameters,
        validate_runtime_parameters,
    )

    if "viewer" in options:
        raise ValueError("Rename the experiment section 'viewer' to 'robogui'.")
    from manimux.config_validation import EXPERIMENT_FIELDS, check_fields
    check_fields(options, EXPERIMENT_FIELDS, "experiment")
    options = deepcopy(options)
    values = {
        "control_profile": None,
        "local": None,
        "policy_server": {},
        "camera_server": {},
        "sensors": [],
        "inference": {},
        "executor": {},
        "robogui": {},
        "recording": {},
        "evaluation": {},
        **options,
    }
    if values.get("control_profile") is not None:
        values["control_profile"] = Path(values["control_profile"])
    if values.get("local") is not None:
        values["local"] = Path(values["local"])
    from manimux.runtime.ik import ik_parameters
    execution = values.setdefault("execution", {})
    if set(execution) - {"ik"}:
        raise ValueError("execution supports only the ik section")
    execution["ik"] = ik_parameters(**(execution.get("ik") or {}))
    values["run"] = run_parameters(**values["run"])
    if values.get("agent") is not None:
        from manimux.agents.execution import agent_parameters
        values["agent"] = agent_parameters(values["agent"])
    if values["run"]["execution_mode"] is None:
        # An agent profile opens RoboGUI on RPent; Direct VLA stays selectable.
        values["run"]["execution_mode"] = "rpent" if values.get("agent") else "vla"
    if values["run"]["execution_mode"] == "rpent" and not values.get("agent"):
        raise ValueError("RPent execution requires agent.config")
    values["robot"] = robot_parameters(**values["robot"])
    values["policy"] = policy_parameters(**values["policy"])
    values["executor"] = executor_parameters(**values["executor"])
    values["inference"] = inference_parameters(executor=values["executor"], **values["inference"])
    if values.get("robogui") is not None:
        values["robogui"] = robogui_parameters(**values["robogui"])
    if values.get("recording") is not None:
        values["recording"] = recording_parameters(**values["recording"])
    values["evaluation"] = evaluation_parameters(values["evaluation"])
    values["sensors"] = [sensor_parameters(**sensor) for sensor in values["sensors"]]
    validate_runtime_parameters(values)
    return values


def load_config(path: str | Path, *, local: str | Path | None = None) -> dict:
    """Run the complete entry-point loading flow, preserving the shared control parameter merge rules."""
    from manimux.embodiments.robot import (
        action_contract_group_indices,
        shared_robot_parameters,
    )
    from manimux.runtime.executors.limits import motion_limits_parameters
    from manimux.runtime.safety import command_safety_parameters

    config_path = Path(path)
    raw = read_experiment(config_path, local=local)
    gripper_indices = action_contract_group_indices(raw)
    # read_experiment has resolved file references; combine shared control and experiment parameters here.
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
        # The model action interval belongs to the checkpoint contract; experiments specify policy.action_dt_s explicitly.
        # Shared embodiment profiles provide layout and limits without implicitly setting the trajectory timeline.
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
    resolved = prepare_experiment(**raw)
    gripper = (
        resolved["executor"]["smooth"].get("gripper")
        if resolved["executor"]["type"] == "smooth"
        else None
    )
    if gripper is not None:
        adapter = resolved["policy"]["adapter"]
        if "gripper_mapping" in adapter:
            raise ValueError(
                "policy.adapter.gripper_mapping is derived from executor.smooth.gripper"
            )
        if gripper["mode"] == "curve":
            adapter["gripper_mapping"] = {
                "mode": "curve",
                "deadzone": gripper["deadzone"],
                "exponent": gripper["exponent"],
            }
    return resolved


def _set_shared_value(target: dict, name: str, value: object, label: str) -> None:
    if name in target and target[name] != value:
        raise ValueError(f"{label} conflicts with control_profile; edit the shared profile")
    target[name] = value
