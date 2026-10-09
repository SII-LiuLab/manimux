"""Shared execution runtime with replaceable inference strategies.

``inference.algorithm`` selects the default latest-chunk strategy, ACT temporal
ensembling, Physical Intelligence real-time chunking, or a strategy plugin.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from manimux.runtime.decode_forecast import FORECAST_MODES
from manimux.runtime.inference import build_inference_strategy
from manimux.types import ACTION_START_MODES

if TYPE_CHECKING:
    from manimux.runtime.edge import EdgeRuntime, RunResult


def __getattr__(name: str):
    # Workers unpickle runtime request types. Do not make the first inference
    # import all executors/robogui code and count that cost as model latency.
    if name in {"EdgeRuntime", "RunResult"}:
        from manimux.runtime import edge

        return getattr(edge, name)
    raise AttributeError(name)


def build_runtime(
    config: dict,
    run_dir: Path,
    *,
    launch_mode: str = "run",
) -> EdgeRuntime:
    from manimux.runtime.edge import EdgeRuntime

    strategy = build_inference_strategy(config)
    return EdgeRuntime(config, run_dir, strategy=strategy, launch_mode=launch_mode)


__all__ = ["EdgeRuntime", "RunResult", "build_runtime"]


def executor_parameters(**options) -> dict:
    """补齐命令平滑和限幅参数；保留各执行器原有数值与物理含义。"""
    from manimux.runtime.executors.limits import motion_limits_parameters
    from manimux.runtime.executors.mpc import mpc_parameters
    from manimux.runtime.executors.smooth import smooth_parameters
    from manimux.runtime.safety import command_safety_parameters

    values = {
        "type": "smooth",
        "motion_limits": None,
        "smooth": {},
        "mpc": {},
        "command_safety": {},
        **options,
    }
    if values["type"] not in {"direct", "smooth", "mpc"}:
        raise ValueError(f"unknown executor type: {values['type']!r}")
    if values.get("motion_limits") is not None:
        values["motion_limits"] = motion_limits_parameters(**values["motion_limits"])
    if values.get("smooth") is not None:
        values["smooth"] = smooth_parameters(**values["smooth"])
    if values.get("mpc") is not None:
        values["mpc"] = mpc_parameters(**values["mpc"])
    if values.get("command_safety") is not None:
        values["command_safety"] = command_safety_parameters(**values["command_safety"])
    return values


def inference_parameters(*, executor: dict, **options) -> dict:
    """补齐推理调度参数；算法与执行器在实验中分别选择。"""
    from manimux.runtime.aac import aac_parameters
    from manimux.runtime.paint import paint_parameters
    from manimux.runtime.rtc.strategy import rtc_parameters
    from manimux.runtime.temporal_ensemble import temporal_ensemble_parameters

    unsupported = {
        "chunk_steps", "blend_steps", "max_chunk_steps", "commit_lead_s", "dvac",
    }.intersection(options)
    if unsupported:
        raise ValueError(f"unsupported inference fields: {sorted(unsupported)}")
    if options.get("algorithm") == "dvac":
        raise ValueError("inference.algorithm=dvac is no longer supported")
    # chunk_policy_steps 只是各调度方式已有步数参数的统一入口。
    options = dict(options)
    if options.get("algorithm") == "serial":
        options.setdefault("inference_schedule", "serial")
    if options.get("chunk_policy_steps") is not None:
        paths = {
            "manimux": (None, "max_chunk_policy_steps"),
            "async": (None, "max_chunk_policy_steps"),
            "serial": (None, "max_chunk_policy_steps"),
            "rtc": ("rtc", "min_execute_policy_steps"),
            "paint": ("paint", "execution_policy_steps"),
            "act_temporal_ensemble": ("temporal_ensemble", "query_interval_policy_steps"),
        }
        section, field = paths[options.get("algorithm", "manimux")]
        destination = options if section is None else dict(options.get(section, {}))
        if (
            destination.get(field) is not None
            and destination[field] != options["chunk_policy_steps"]
        ):
            raise ValueError(
                "inference.chunk_policy_steps conflicts with "
                f"inference.{section or field}"
            )
        destination[field] = options["chunk_policy_steps"]
        if section is not None:
            options[section] = destination

    values = {
        "algorithm": "manimux",
        "strategy": None,
        "chunk_policy_steps": None,
        "inference_schedule": "deadline",
        "refill_threshold_s": 0.4,
        "handoff_skip_steps": 0,
        "max_plan_age_s": 1.0,
        "blend_policy_steps": 2,
        "action_start_mode": "drop_infer_latency",
        # Chunk handoff: blend joints at commit, or an adapter EE waypoint before dense IK.
        "handoff": "blend",
        "handoff_margin_s": 0.0,
        "max_chunk_policy_steps": None,
        "independent_group_decoding": False,
        "decode_budget_ms": 40.0,
        "expected_decode_s": 0.0,
        "decode_forecast_size": 0,
        "decode_forecast_mode": "max",
        "rtc": {},
        "temporal_ensemble": {},
        "aac": {},
        "paint": {},
        **options,
    }
    if values.get("rtc") is not None:
        values["rtc"] = rtc_parameters(**values["rtc"])
    if values.get("temporal_ensemble") is not None:
        values["temporal_ensemble"] = temporal_ensemble_parameters(**values["temporal_ensemble"])
    if values.get("aac") is not None:
        values["aac"] = aac_parameters(**values["aac"])
    if values.get("paint") is not None:
        values["paint"] = paint_parameters(**values["paint"])
    # 字典可能已补齐默认值；默认调度字段不应被误认成用户为其他模式新增的参数。
    inactive_defaults = {"inference_schedule": "deadline", "refill_threshold_s": 0.4}
    provided = {
        key
        for key in options
        if key not in inactive_defaults or options[key] != inactive_defaults[key]
    }
    validate_inference_parameters(values, executor, provided=provided)
    return values


def validate_runtime_parameters(config: dict) -> None:
    """保留调度、动作解码和执行限位之间的必要约束。"""
    if config["run"].get("warmup_before_start", False):
        if not config["robogui"]["enabled"]:
            raise ValueError("run.warmup_before_start requires RoboGUI Start control")
        if config["policy"]["action_decoding"] != "inline":
            raise ValueError("pre-Start warmup currently requires inline action decoding")
    motion = config["executor"]["motion_limits"]
    if motion is not None:
        for group, index in motion["gripper"]["group_indices"].items():
            if (
                group not in config["robot"]["group_dims"]
                or not 0 <= index < config["robot"]["group_dims"][group]
            ):
                raise ValueError("motion_limits gripper indices must match robot.group_dims")
        if config["executor"]["type"] == "mpc":
            raise ValueError("shared motion_limits currently support direct and smooth, not mpc")
    if (
        config["inference"]["inference_schedule"] == "serial"
        and config["policy"]["action_decoding"] != "inline"
    ):
        raise ValueError(
            "serial scheduling requires inline action decoding without latency trimming"
        )
    if (
        config["inference"]["chunk_policy_steps"] is not None
        and config["inference"]["chunk_policy_steps"] > config["policy"]["horizon_policy_steps"]
    ):
        raise ValueError("inference.chunk_policy_steps must not exceed policy.horizon_policy_steps")
    if (
        config["inference"]["handoff"] == "waypoint"
        and config["policy"]["action_decoding"] != "process"
    ):
        raise ValueError("inference.handoff=waypoint requires process action decoding")
    if (
        config["inference"]["independent_group_decoding"]
        and config["policy"]["action_decoding"] != "process"
    ):
        raise ValueError("independent group decoding requires process action decoding")
    if (
        config["inference"]["expected_decode_s"] > 0
        and config["policy"]["action_decoding"] != "process"
    ):
        raise ValueError("inference.expected_decode_s requires process action decoding")
    # A positive expected_decode_s already implies process decoding, checked above.
    if (
        config["inference"]["decode_forecast_size"]
        and not config["inference"]["expected_decode_s"]
    ):
        raise ValueError(
            "inference.decode_forecast_size requires a positive expected_decode_s "
            "to use as its initial estimate and lower bound"
        )
    if (
        config["inference"]["max_chunk_policy_steps"] is not None
        and config["inference"]["max_chunk_policy_steps"] > config["policy"]["horizon_policy_steps"]
    ):
        raise ValueError("max_chunk_policy_steps must not exceed policy.horizon_policy_steps")
    command_safety = config["executor"]["command_safety"]
    if bool(command_safety["position_lower"]):
        expected_groups = set(config["robot"]["group_dims"])
        if set(command_safety["position_lower"]) != expected_groups:
            raise ValueError("executor.command_safety groups must exactly match robot.group_dims")
        for group, dimension in config["robot"]["group_dims"].items():
            if len(command_safety["position_lower"][group]) != dimension:
                raise ValueError(
                    f"executor.command_safety group {group!r} must have {dimension} values"
                )
    if config["inference"]["algorithm"] == "rtc":
        delay = config["inference"]["rtc"]["initial_delay_policy_steps"]
        skip = (
            config["inference"]["handoff_skip_steps"]
            if config["inference"]["handoff"] == "waypoint"
            else 0
        )
        horizon = config["policy"]["horizon_policy_steps"]
        if skip >= horizon:
            raise ValueError("RTC waypoint handoff_skip_steps must be smaller than the horizon")
        if delay is not None and 2 * delay > config["policy"]["horizon_policy_steps"]:
            raise ValueError(
                "RTC requires 2 * initial_delay_policy_steps "
                "<= policy.horizon_policy_steps"
            )
        if delay is not None and skip and 2 * delay > horizon - skip:
            raise ValueError(
                "RTC requires 2 * initial_delay_policy_steps <= "
                "policy.horizon_policy_steps - handoff_skip_steps for waypoint handoff"
            )
    if config["inference"]["algorithm"] == "act_temporal_ensemble":
        query_interval = config["inference"]["temporal_ensemble"]["query_interval_policy_steps"]
        if query_interval >= config["policy"]["horizon_policy_steps"]:
            raise ValueError(
                "ACT temporal ensembling requires query_interval_policy_steps < "
                "policy.horizon_policy_steps so consecutive chunks overlap"
            )
    if config["inference"]["algorithm"] == "paint":
        execution = config["inference"]["paint"]["execution_policy_steps"]
        delay = config["inference"]["paint"]["initial_delay_policy_steps"]
        horizon = config["policy"]["horizon_policy_steps"]
        if not delay <= execution <= horizon - delay:
            raise ValueError(
                "PAINT requires initial_delay_policy_steps <= execution_policy_steps <= "
                "horizon_policy_steps - initial_delay_policy_steps"
            )


def validate_inference_parameters(values: dict, executor: dict, *, provided=frozenset()) -> None:
    """检查调度和执行方式的组合；provided 仅用于识别 YAML 中明确给出的字段。"""
    skip_steps = values["handoff_skip_steps"]
    if type(skip_steps) is not int or skip_steps < 0:
        raise ValueError("inference.handoff_skip_steps must be a non-negative integer")
    if values["action_start_mode"] not in ACTION_START_MODES:
        raise ValueError(
            "inference.action_start_mode must be one of "
            f"{sorted(ACTION_START_MODES)}"
        )
    if values["decode_forecast_mode"] not in FORECAST_MODES:
        raise ValueError(f"inference.decode_forecast_mode must be one of {FORECAST_MODES}")
    if values["handoff"] not in {"blend", "waypoint"}:
        raise ValueError("inference.handoff must be blend or waypoint")
    if not values["handoff_margin_s"] >= 0:
        raise ValueError("inference.handoff_margin_s must be non-negative")
    if values["handoff"] == "waypoint" and (
        values["algorithm"] not in {"manimux", "rtc"} or values["blend_policy_steps"] != 0
    ):
        raise ValueError(
            "inference.handoff=waypoint requires the manimux or rtc algorithm and "
            "blend_policy_steps=0"
        )
    # The handoff time lies on the source row grid, which first_step_when_ready discards.
    if values["handoff"] == "waypoint" and values["action_start_mode"] != "drop_infer_latency":
        raise ValueError("inference.handoff=waypoint requires action_start_mode=drop_infer_latency")
    if values["inference_schedule"] == "serial":
        if values["algorithm"] not in {"manimux", "serial"}:
            raise ValueError("serial scheduling requires inference.algorithm=serial")
        if "refill_threshold_s" in provided:
            raise ValueError("serial scheduling does not use refill_threshold_s")
    elif values["algorithm"] == "serial":
        raise ValueError("inference.algorithm=serial requires inference_schedule=serial")
    if values["independent_group_decoding"] and (
        values["algorithm"] not in {"manimux", "async"}
        or executor["type"] != "smooth"
        or executor["smooth"]["tracking_mode"] != "braking"
        or (executor["smooth"]["gripper"] is None)
        or (executor["smooth"]["gripper"]["mode"] != "continuous")
    ):
        raise ValueError(
            "independent group decoding requires braking smooth with continuous grippers"
        )
    if values["max_chunk_policy_steps"] is not None and (
        values["algorithm"] not in {"manimux", "async", "serial"}
        or executor["type"] not in {"smooth", "direct", "mpc"}
    ):
        raise ValueError("max_chunk_policy_steps requires the ordinary ManiMux joint timeline")
    # 这些策略自行决定请求时机，不使用普通 timeline 的补充调度参数。
    names = {
        "rtc": "RTC",
        "act_temporal_ensemble": "ACT temporal ensembling",
        "aac": "AAC",
        "paint": "PAINT",
        "autohorizon": "AutoHorizon",
    }
    runtime = values["algorithm"]
    if runtime == "aac" and not values["aac"]["ee_stats_path"]:
        raise ValueError("inference.aac.ee_stats_path is required by AAC")
    if runtime in names:
        ignored = {"inference_schedule", "refill_threshold_s"}.intersection(provided)
        if ignored:
            fields = ", ".join(sorted(ignored))
            raise ValueError(f"inference fields are not used by {names[runtime]}: {fields}")
