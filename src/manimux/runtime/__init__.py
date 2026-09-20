"""Shared execution runtime with replaceable inference strategies.

``execution.runtime`` selects the default latest-chunk strategy, ACT temporal
ensembling, Physical Intelligence real-time chunking, or a strategy plugin.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from manimux.runtime.decode_forecast import FORECAST_MODES
from manimux.runtime.inference import build_inference_strategy

if TYPE_CHECKING:
    from manimux.runtime.edge import EdgeRuntime, RunResult


def __getattr__(name: str):
    # Workers unpickle runtime request types. Do not make the first inference
    # import all executors/viewer code and count that cost as model latency.
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
    # A history plugin can delegate to RTC while retaining its observation and
    # condition alignment hooks. Only the literal built-in uses RtcRuntime.
    if config["execution"]["runtime"] != "rtc":
        return EdgeRuntime(config, run_dir, strategy=strategy, launch_mode=launch_mode)
    from manimux.runtime.rtc import RtcRuntime

    return RtcRuntime(config, run_dir, strategy=strategy, launch_mode=launch_mode)


__all__ = ["EdgeRuntime", "RunResult", "build_runtime"]


def execution_parameters(**options) -> dict:
    """补齐各调度和执行方式的原默认值，输出普通字典。"""
    from manimux.runtime.aac import aac_parameters
    from manimux.runtime.dvac import dvac_parameters
    from manimux.runtime.executors.limits import motion_limits_parameters
    from manimux.runtime.executors.mpc import mpc_parameters
    from manimux.runtime.executors.smooth import smooth_parameters
    from manimux.runtime.paint import paint_parameters
    from manimux.runtime.rtc.strategy import rtc_parameters
    from manimux.runtime.safety import command_safety_parameters
    from manimux.runtime.temporal_ensemble import temporal_ensemble_parameters

    # chunk_steps 只是各调度方式已有步数参数的统一入口。
    options = dict(options)
    if options.get("chunk_steps") is not None:
        paths = {
            "manimux": (None, "max_chunk_steps"),
            "rtc": ("rtc", "min_execute_steps"),
            "paint": ("paint", "execution_steps"),
            "act_temporal_ensemble": ("temporal_ensemble", "query_interval_steps"),
            "dvac": ("dvac", "max_execution_steps"),
        }
        section, field = paths[options.get("runtime", "manimux")]
        destination = options if section is None else dict(options.get(section, {}))
        if destination.get(field) is not None and destination[field] != options["chunk_steps"]:
            raise ValueError(f"execution.chunk_steps conflicts with execution.{section or field}")
        destination[field] = options["chunk_steps"]
        if section is not None:
            options[section] = destination

    values = {
        "motion_limits": None,
        "runtime": "manimux",
        "chunk_steps": None,
        "executor": "smooth",
        "inference_schedule": "deadline",
        "refill_threshold_s": 0.4,
        "commit_lead_s": 0.02,
        "max_plan_age_s": 1.0,
        "blend_steps": 2,
        "max_chunk_steps": None,
        "independent_group_decoding": False,
        "decode_budget_ms": 40.0,
        "expected_decode_s": 0.0,
        "decode_forecast_size": 0,
        "decode_forecast_mode": "max",
        "smooth": {},
        "mpc": {},
        "command_safety": {},
        "rtc": {},
        "temporal_ensemble": {},
        "aac": {},
        "paint": {},
        "dvac": {},
        **options,
    }
    if values.get("motion_limits") is not None:
        values["motion_limits"] = motion_limits_parameters(**values["motion_limits"])
    if values.get("smooth") is not None:
        values["smooth"] = smooth_parameters(**values["smooth"])
    if values.get("mpc") is not None:
        values["mpc"] = mpc_parameters(**values["mpc"])
    if values.get("command_safety") is not None:
        values["command_safety"] = command_safety_parameters(**values["command_safety"])
    if values.get("rtc") is not None:
        values["rtc"] = rtc_parameters(**values["rtc"])
    if values.get("temporal_ensemble") is not None:
        values["temporal_ensemble"] = temporal_ensemble_parameters(**values["temporal_ensemble"])
    if values.get("aac") is not None:
        values["aac"] = aac_parameters(**values["aac"])
    if values.get("paint") is not None:
        values["paint"] = paint_parameters(**values["paint"])
    if values.get("dvac") is not None:
        values["dvac"] = dvac_parameters(**values["dvac"])
    # 字典可能已补齐默认值；默认调度字段不应被误认成用户为其他模式新增的参数。
    inactive_defaults = {"inference_schedule": "deadline", "refill_threshold_s": 0.4}
    provided = {
        key
        for key in options
        if key not in inactive_defaults or options[key] != inactive_defaults[key]
    }
    validate_execution_parameters(values, provided=provided)
    return values


def validate_runtime_parameters(config: dict) -> None:
    """保留调度、动作解码和执行限位之间的必要约束。"""
    motion = config["execution"]["motion_limits"]
    if motion is not None:
        for group, index in motion["gripper"]["group_indices"].items():
            if (
                group not in config["robot"]["group_dims"]
                or not 0 <= index < config["robot"]["group_dims"][group]
            ):
                raise ValueError("motion_limits gripper indices must match robot.group_dims")
        if config["execution"]["executor"] == "mpc":
            raise ValueError("shared motion_limits currently support direct and smooth, not mpc")
    if (
        config["execution"]["inference_schedule"] == "serial"
        and config["policy"]["action_decoding"] != "inline"
    ):
        raise ValueError(
            "serial scheduling requires inline action decoding without latency trimming"
        )
    if (
        config["execution"]["chunk_steps"] is not None
        and config["execution"]["chunk_steps"] > config["policy"]["horizon_steps"]
    ):
        raise ValueError("execution.chunk_steps must not exceed policy.horizon_steps")
    if (
        config["execution"]["independent_group_decoding"]
        and config["policy"]["action_decoding"] != "process"
    ):
        raise ValueError("independent group decoding requires process action decoding")
    if (
        config["execution"]["expected_decode_s"] > 0
        and config["policy"]["action_decoding"] != "process"
    ):
        raise ValueError("execution.expected_decode_s requires process action decoding")
    # A positive expected_decode_s already implies process decoding, checked above.
    if config["execution"]["decode_forecast_size"] and not config["execution"]["expected_decode_s"]:
        raise ValueError(
            "execution.decode_forecast_size requires a positive expected_decode_s "
            "to use as its initial estimate and lower bound"
        )
    if (
        config["execution"]["max_chunk_steps"] is not None
        and config["execution"]["max_chunk_steps"] > config["policy"]["horizon_steps"]
    ):
        raise ValueError("max_chunk_steps must not exceed policy.horizon_steps")
    command_safety = config["execution"]["command_safety"]
    if bool(command_safety["position_lower"]):
        expected_groups = set(config["robot"]["group_dims"])
        if set(command_safety["position_lower"]) != expected_groups:
            raise ValueError("execution.command_safety groups must exactly match robot.group_dims")
        for group, dimension in config["robot"]["group_dims"].items():
            if len(command_safety["position_lower"][group]) != dimension:
                raise ValueError(
                    f"execution.command_safety group {group!r} must have {dimension} values"
                )
    if config["execution"]["runtime"] == "rtc":
        delay = config["execution"]["rtc"]["initial_delay_steps"]
        if 2 * delay > config["policy"]["horizon_steps"]:
            raise ValueError("RTC requires 2 * initial_delay_steps <= policy.horizon_steps")
    if config["execution"]["runtime"] == "act_temporal_ensemble":
        query_interval = config["execution"]["temporal_ensemble"]["query_interval_steps"]
        if query_interval >= config["policy"]["horizon_steps"]:
            raise ValueError(
                "ACT temporal ensembling requires query_interval_steps < "
                "policy.horizon_steps so consecutive chunks overlap"
            )
    if config["execution"]["runtime"] == "paint":
        execution = config["execution"]["paint"]["execution_steps"]
        delay = config["execution"]["paint"]["initial_delay_steps"]
        horizon = config["policy"]["horizon_steps"]
        if not delay <= execution <= horizon - delay:
            raise ValueError(
                "PAINT requires initial_delay_steps <= execution_steps <= "
                "horizon_steps - initial_delay_steps"
            )
    if config["execution"]["runtime"] == "dvac":
        dvac = config["execution"]["dvac"]
        maximum = dvac["max_execution_steps"] or config["policy"]["horizon_steps"]
        if not dvac["min_execution_steps"] <= maximum <= config["policy"]["horizon_steps"]:
            raise ValueError(
                "DVAC requires min_execution_steps <= max_execution_steps <= policy.horizon_steps"
            )


def validate_execution_parameters(values: dict, *, provided=frozenset()) -> None:
    """检查调度和执行方式的组合；provided 仅用于识别 YAML 中明确给出的字段。"""
    if values["decode_forecast_mode"] not in FORECAST_MODES:
        raise ValueError(f"execution.decode_forecast_mode must be one of {FORECAST_MODES}")
    if values["inference_schedule"] == "serial":
        if values["runtime"] != "manimux":
            raise ValueError("serial scheduling requires execution.runtime=manimux")
        if "refill_threshold_s" in provided:
            raise ValueError("serial scheduling does not use refill_threshold_s")
    if values["independent_group_decoding"] and (
        values["runtime"] != "manimux"
        or values["executor"] != "smooth"
        or values["smooth"]["tracking_mode"] != "braking"
        or (values["smooth"]["gripper"] is None)
        or (values["smooth"]["gripper"]["mode"] != "continuous")
    ):
        raise ValueError(
            "independent group decoding requires braking smooth with continuous grippers"
        )
    if values["max_chunk_steps"] is not None and (
        values["runtime"] != "manimux" or values["executor"] not in {"smooth", "direct", "mpc"}
    ):
        raise ValueError("max_chunk_steps requires the ordinary ManiMux joint timeline")
    # 这些策略自行决定请求时机，不使用普通 timeline 的补充调度参数。
    names = {
        "rtc": "RTC",
        "act_temporal_ensemble": "ACT temporal ensembling",
        "aac": "AAC",
        "paint": "PAINT",
        "autohorizon": "AutoHorizon",
        "dvac": "DVAC",
    }
    runtime = values["runtime"]
    if runtime == "aac" and not values["aac"]["ee_stats_path"]:
        raise ValueError("execution.aac.ee_stats_path is required by AAC")
    if runtime in names:
        ignored = {"inference_schedule", "refill_threshold_s"}.intersection(provided)
        if ignored:
            fields = ", ".join(sorted(ignored))
            raise ValueError(f"execution fields are not used by {names[runtime]}: {fields}")
        # 除 RTC 外，保留策略选定的轨迹；提交时不能再次插值改写。
        if runtime != "rtc" and values["blend_steps"] != 0:
            raise ValueError(
                f"{names[runtime]} requires execution.blend_steps=0 "
                "to preserve the strategy's trajectory"
            )
