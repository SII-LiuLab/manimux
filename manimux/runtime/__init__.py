"""Shared execution runtime with replaceable inference strategies.

``inference.algorithm`` selects the default latest-chunk strategy, ACT temporal
ensembling, Physical Intelligence real-time chunking, or a strategy plugin.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from manimux.runtime.decode_forecast import FORECAST_MODES
from manimux.runtime.inference import INFERENCE_ALGORITHMS, build_inference_strategy
from manimux.types import ACTION_START_MODES

if TYPE_CHECKING:
    from manimux.agents.rpent.session import ManagedRPentRuntime
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
) -> EdgeRuntime | ManagedRPentRuntime:
    if config["run"].get("execution_mode", "vla") == "rpent":
        from manimux.agents.rpent.session import ManagedRPentRuntime
        return ManagedRPentRuntime(config, run_dir, launch_mode=launch_mode)
    from manimux.runtime.edge import EdgeRuntime

    strategy = build_inference_strategy(config)
    return EdgeRuntime(config, run_dir, strategy=strategy, launch_mode=launch_mode)


__all__ = ["EdgeRuntime", "RunResult", "build_runtime"]


def executor_parameters(**options) -> dict:
    """Fill in smoothing and limits, preserving each executor's values and meaning."""
    from manimux.config_validation import merge_parameters
    from manimux.runtime.executors.limits import motion_limits_parameters
    from manimux.runtime.executors.mpc import mpc_parameters
    from manimux.runtime.executors.smooth import smooth_parameters
    from manimux.runtime.safety import command_safety_parameters

    values = merge_parameters({
        "type": "smooth",
        "motion_limits": None,
        "smooth": {},
        "mpc": {},
        "command_safety": {},
    }, options, 'executor', extra=())
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
    """Fill in scheduling parameters; algorithms and executors are selected independently."""
    from manimux.config_validation import merge_parameters
    from manimux.runtime.aac import aac_parameters
    from manimux.runtime.bid import bid_parameters
    from manimux.runtime.paint import paint_parameters
    from manimux.runtime.rtc.strategy import rtc_parameters
    from manimux.runtime.scheduling import resolve_request_schedule
    from manimux.runtime.temporal_ensemble import temporal_ensemble_parameters
    from manimux.runtime.train_time_rtc import train_time_rtc_parameters

    if "min_execute_policy_steps" in options:
        raise ValueError(
            "inference.min_execute_policy_steps is ambiguous; use "
            "inference.min_plan_hold_policy_steps for replacement timing, or "
            "inference.rtc.min_execute_policy_steps for RTC request timing"
        )
    unsupported = {
        "chunk_steps", "blend_steps", "max_chunk_steps", "commit_lead_s", "dvac",
    }.intersection(options)
    if unsupported:
        raise ValueError(f"unsupported inference fields: {sorted(unsupported)}")
    if options.get("algorithm") == "dvac":
        raise ValueError("inference.algorithm=dvac is no longer supported")
    # chunk_policy_steps is the shared entry point for existing scheduling step counts.
    options = dict(options)
    options.setdefault(
        "algorithm", "serial" if options.get("inference_schedule") == "serial" else "async"
    )
    if options["algorithm"] not in INFERENCE_ALGORITHMS:
        raise ValueError(f"inference.algorithm must be one of {sorted(INFERENCE_ALGORITHMS)}")
    chunktrust = options["algorithm"] in {"chunktrust_ahs", "chunktrust_ahs_rtc"}
    if chunktrust:
        if options.get("chunk_policy_steps") is not None:
            raise ValueError("ChunkTrust selects K; omit inference.chunk_policy_steps")
        options.setdefault("action_start_mode", "first_step_when_ready")
        options.setdefault("blend_policy_steps", 0)
        if options["algorithm"] == "chunktrust_ahs_rtc":
            rtc = options.get("rtc", {})
            if not isinstance(rtc, Mapping):
                raise ValueError("inference.rtc must be a mapping for ChunkTrust RTC")
            options["rtc"] = {
                "beta": 10.0,
                "initial_delay_policy_steps": None,
                "delay_buffer_size": 10,
                **rtc,
            }
    elif options.get("chunktrust") is not None:
        raise ValueError("inference.chunktrust requires a ChunkTrust algorithm")
    if options.get("algorithm") in {"serial", "bid_backward"}:
        options.setdefault("inference_schedule", "serial")
    serial_act = (
        options.get("algorithm") == "act_temporal_ensemble"
        and options.get("inference_schedule") == "serial"
    )
    progress_async = options["algorithm"] == "async" and options.get("request_trigger") == "progress"
    # Serial ACT and progress-triggered Async consume the shared step count directly.
    if options.get("chunk_policy_steps") is not None and not (serial_act or progress_async):
        paths = {
            "async": (None, "max_chunk_policy_steps"),
            "serial": (None, "max_chunk_policy_steps"),
            "rtc": ("rtc", "min_execute_policy_steps"),
            "train_time_rtc": ("train_time_rtc", "min_execute_policy_steps"),
            "paint": ("paint", "execution_policy_steps"),
            "bid_backward": ("bid", "replan_policy_steps"),
            "act_temporal_ensemble": ("temporal_ensemble", "query_interval_policy_steps"),
        }
        section, field = paths[options.get("algorithm", "async")]
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

    values = merge_parameters({
        "algorithm": "async",
        "strategy": None,
        "chunk_policy_steps": None,
        "inference_schedule": "deadline",
        "refill_threshold_s": 0.4,
        "observation_hz": None,
        "request_trigger": None,
        "min_plan_hold_policy_steps": 0,
        "handoff_skip_steps": 0,
        "max_plan_age_s": 1.0,
        "blend_policy_steps": 2,
        "blend_mode": "linear",
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
        "train_time_rtc": {},
        "temporal_ensemble": {},
        "aac": {},
        "bid": {},
        "paint": {},
        "chunktrust": None,
    }, options, 'inference', extra=('history',))
    if chunktrust:
        from manimux.runtime.chunktrust.config import chunktrust_parameters

        settings = values["chunktrust"]
        if settings is None:
            settings = {}
        if not isinstance(settings, Mapping):
            raise ValueError("inference.chunktrust must be a mapping")
        if values["algorithm"] == "chunktrust_ahs_rtc":
            settings = {"candidates": [10, 20, 30, 40], **settings}
        values["chunktrust"] = chunktrust_parameters(**settings)
    if values.get("rtc") is not None:
        values["rtc"] = rtc_parameters(**values["rtc"])
    values["train_time_rtc"] = train_time_rtc_parameters(**values["train_time_rtc"])
    if values.get("temporal_ensemble") is not None:
        values["temporal_ensemble"] = temporal_ensemble_parameters(**values["temporal_ensemble"])
    if values.get("aac") is not None:
        values["aac"] = aac_parameters(**values["aac"])
    if values.get("bid") is not None:
        values["bid"] = bid_parameters(**values["bid"])
    if values.get("paint") is not None:
        values["paint"] = paint_parameters(**values["paint"])
    # The dictionary may already contain defaults. Do not treat scheduling defaults
    # as user-supplied options for other modes.
    inactive_defaults = {
        "inference_schedule": "deadline", "refill_threshold_s": 0.4,
        "observation_hz": None, "request_trigger": None,
    }
    provided = {
        key
        for key in options
        if key not in inactive_defaults or options[key] != inactive_defaults[key]
    }
    resolve_request_schedule(values)
    validate_inference_parameters(values, executor, provided=provided)
    return values


def validate_runtime_parameters(config: dict) -> None:
    """Preserve constraints between scheduling, action decoding and execution limits."""
    from manimux.runtime.ik import validate_ik_config
    validate_ik_config(config)
    settings = config["inference"]
    if settings["algorithm"] in {"chunktrust_ahs", "chunktrust_ahs_rtc"}:
        _validate_chunktrust_runtime(config)
    if settings["min_plan_hold_policy_steps"] and config["policy"]["action_decoding"] != "inline":
        raise ValueError("min_plan_hold_policy_steps requires inline action decoding")
    ensemble = (settings.get("temporal_ensemble") or {}).get("enabled", False)
    candidate_aac = settings["algorithm"] == "aac" and (
        settings["inference_schedule"] != "serial" or ensemble
    )
    bid_aac = (
        settings["algorithm"] == "bid_backward" and settings["bid"]["execution_horizon"] == "aac"
    )
    adaptive_async = settings["algorithm"] in {"aac", "autohorizon"} and (
        settings["inference_schedule"] != "serial"
    )
    if (ensemble or adaptive_async) and config["policy"]["action_decoding"] != "inline":
        raise ValueError("ACT composition and async adaptive horizons require inline decoding")
    if (candidate_aac or bid_aac) and config["policy"]["worker"] != "xpolicylab_ws":
        raise ValueError("AAC composition currently requires the xpolicylab_ws client")
    if config["inference"]["algorithm"] == "bid_backward":
        if config["policy"]["action_decoding"] != "inline":
            raise ValueError("BID backward-only currently requires inline action decoding")
        if not bid_aac and config["inference"]["bid"]["replan_policy_steps"] >= config["policy"][
            "horizon_policy_steps"
        ]:
            raise ValueError("BID replan_policy_steps must be smaller than the model horizon")
    if (
        config["inference"]["inference_schedule"] == "multi_inflight"
        and config["policy"]["action_decoding"] != "inline"
    ):
        raise ValueError("multi_inflight currently requires inline action decoding")
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
        and config["inference"]["action_start_mode"] != "first_step_when_ready"
    ):
        raise ValueError(
            "serial scheduling requires action_start_mode=first_step_when_ready"
        )
    if (
        config["inference"]["chunk_policy_steps"] is not None
        and config["inference"]["chunk_policy_steps"] > config["policy"]["horizon_policy_steps"]
    ):
        raise ValueError("inference.chunk_policy_steps must not exceed policy.horizon_policy_steps")
    if (
        config["inference"]["request_trigger"] == "progress"
        and config["inference"]["chunk_policy_steps"] >= config["policy"]["horizon_policy_steps"]
    ):
        raise ValueError("progress chunk_policy_steps must be smaller than policy.horizon_policy_steps")
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
    if (config["inference"].get("rtc") or {}).get("condition_source") == "executor":
        smooth = config["executor"].get("smooth") or {}
        gripper = smooth.get("gripper")
        if (config["inference"]["algorithm"] != "rtc"
                or (config["inference"].get("temporal_ensemble") or {}).get("enabled")
                or config["executor"]["type"] != "smooth"
                or smooth.get("release_guard") or smooth.get("grasp_guard")
                or (gripper and gripper["mode"] not in {"continuous", "curve"})):
            raise ValueError(
                "RTC executor conditions require standalone RTC and smooth without gripper guards"
            )
    if (
        (config["inference"].get("rtc") or {}).get("condition_source", "canonical") != "canonical"
        and config["inference"]["handoff_skip_steps"]
    ):
        raise ValueError("timeline/executor RTC conditions do not support handoff row skipping")
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
    if config["inference"]["algorithm"] in {"rtc", "train_time_rtc"}:
        section = config["inference"]["algorithm"]
        delay_field = (
            "fixed_prefix_steps"
            if section == "train_time_rtc"
            and config["inference"][section]["delay_estimator"] == "fixed"
            else "initial_delay_policy_steps"
        )
        delay = config["inference"][section][delay_field]
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
                f"RTC requires 2 * {delay_field} "
                "<= policy.horizon_policy_steps"
            )
    if (
        config["inference"]["algorithm"] == "act_temporal_ensemble"
        and config["inference"]["inference_schedule"] != "serial"
    ):
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
    """Check scheduling/execution combinations; provided identifies explicit YAML fields."""
    from manimux.runtime.scheduling import validate_request_schedule

    validate_request_schedule(values, provided=provided)
    if values["algorithm"] in {"chunktrust_ahs", "chunktrust_ahs_rtc"}:
        _validate_chunktrust_inference(values, executor)
    minimum = values["min_plan_hold_policy_steps"]
    if type(minimum) is not int or minimum < 0:
        raise ValueError("inference.min_plan_hold_policy_steps must be a non-negative integer")
    if minimum and values["inference_schedule"] not in {"single_inflight", "multi_inflight"}:
        raise ValueError("min_plan_hold_policy_steps requires single_inflight or multi_inflight")
    ensemble = (values.get("temporal_ensemble") or {}).get("enabled", False)
    if type(ensemble) is not bool:
        raise ValueError("temporal_ensemble.enabled must be boolean")
    serial_act = (
        values["algorithm"] == "act_temporal_ensemble" and values["inference_schedule"] == "serial"
    )
    if serial_act:
        steps = values["chunk_policy_steps"]
        if steps is not None and (type(steps) is not int or steps < 1):
            raise ValueError("serial ACT chunk_policy_steps must be a positive integer or null")
        if values["action_start_mode"] != "first_step_when_ready":
            raise ValueError("serial ACT requires first_step_when_ready")
    if ensemble or serial_act:
        if values["strategy"] is not None or values["algorithm"] not in {
            "act_temporal_ensemble", "rtc", "paint", "bid_backward", "aac", "autohorizon",
        }:
            raise ValueError("ACT composition requires a supported built-in algorithm")
        if (
            values["handoff_skip_steps"] or values["blend_policy_steps"]
            or values["handoff"] != "blend"
        ):
            raise ValueError("ACT composition requires zero blending and no waypoint/skip handoff")
        if values["algorithm"] in {"rtc", "paint"} and (
            values["action_start_mode"] != "drop_infer_latency"
        ):
            raise ValueError("ACT with conditioned sampling requires drop_infer_latency")
    if values["algorithm"] == "paint" and values["inference_schedule"] == "multi_inflight" and (
        values["action_start_mode"] != "drop_infer_latency"
        or values["blend_policy_steps"] or values["handoff_skip_steps"]
    ):
        raise ValueError("streaming PAINT requires drop_infer_latency, zero blending and skip")
    bid_aac = values["algorithm"] == "bid_backward" and values["bid"]["execution_horizon"] == "aac"
    adaptive_async = values["inference_schedule"] != "serial" and (
        values["algorithm"] in {"aac", "autohorizon"} or bid_aac
    )
    if adaptive_async and (
        values["action_start_mode"] != "drop_infer_latency"
        or values["handoff_skip_steps"] or values["blend_policy_steps"]
    ):
        raise ValueError(
            "asynchronous adaptive horizons require drop_infer_latency, zero blend and skip"
        )
    if bid_aac and values["chunk_policy_steps"] is not None:
        raise ValueError("BID with AAC chooses its execution horizon; omit chunk_policy_steps")
    if values["algorithm"] == "bid_backward":
        if values["handoff_skip_steps"]:
            raise ValueError("BID requires handoff_skip_steps=0")
        if (
            values["inference_schedule"] == "serial"
            and values["action_start_mode"] != "first_step_when_ready"
        ):
            raise ValueError("serial BID requires first_step_when_ready")
    skip_steps = values["handoff_skip_steps"]
    if type(skip_steps) is not int:
        raise ValueError("inference.handoff_skip_steps must be an integer")
    if skip_steps < 0 and not (
        values["algorithm"] == "async"
        and values["inference_schedule"] == "single_inflight"
        and values["action_start_mode"] == "drop_infer_latency"
        and values["handoff"] == "blend"
    ):
        raise ValueError("negative handoff_skip_steps requires single-inflight Async with latency drop and blend handoff")
    if values["action_start_mode"] not in ACTION_START_MODES:
        raise ValueError(
            "inference.action_start_mode must be one of "
            f"{sorted(ACTION_START_MODES)}"
        )
    if values["decode_forecast_mode"] not in FORECAST_MODES:
        raise ValueError(f"inference.decode_forecast_mode must be one of {FORECAST_MODES}")
    if values["handoff"] not in {"blend", "waypoint"}:
        raise ValueError("inference.handoff must be blend or waypoint")
    if values["blend_mode"] not in {"linear", "command_cubic"}:
        raise ValueError("inference.blend_mode must be linear or command_cubic")
    if values["blend_mode"] == "command_cubic" and (
        values["handoff"] != "blend" or values["blend_policy_steps"] < 2
        or values["algorithm"] not in {"async", "rtc"}
        or values["inference_schedule"] == "serial"
        or executor["type"] != "smooth"
        or (values.get("temporal_ensemble") or {}).get("enabled")
        or values["independent_group_decoding"]
    ):
        raise ValueError(
            "command_cubic requires standalone async/rtc with smooth, complete groups, "
            "and at least two blend rows"
        )
    if not values["handoff_margin_s"] >= 0:
        raise ValueError("inference.handoff_margin_s must be non-negative")
    if values["handoff"] == "waypoint" and (
        values["algorithm"] not in {"async", "rtc", "train_time_rtc"}
        or values["blend_policy_steps"] != 0
    ):
        raise ValueError(
            "inference.handoff=waypoint requires async, rtc or train_time_rtc and "
            "blend_policy_steps=0"
        )
    # The handoff time lies on the source row grid, which first_step_when_ready discards.
    if values["handoff"] == "waypoint" and values["action_start_mode"] != "drop_infer_latency":
        raise ValueError("inference.handoff=waypoint requires action_start_mode=drop_infer_latency")
    if values["independent_group_decoding"] and (
        values["algorithm"] != "async"
        or executor["type"] != "smooth"
        or executor["smooth"]["tracking_mode"] != "braking"
        or (executor["smooth"]["gripper"] is None)
        or (executor["smooth"]["gripper"]["mode"] != "continuous")
    ):
        raise ValueError(
            "independent group decoding requires braking smooth with continuous grippers"
        )
    if values["max_chunk_policy_steps"] is not None and (
        values["algorithm"] not in {"async", "serial"}
        or executor["type"] not in {"smooth", "direct", "mpc"}
    ):
        raise ValueError("max_chunk_policy_steps requires the Serial/Async joint timeline")
    if values["algorithm"] == "train_time_rtc" and (
        values["blend_policy_steps"] != 0 or values["handoff_skip_steps"] != 0
        or values["action_start_mode"] != "drop_infer_latency"
    ):
        raise ValueError(
            "train_time_rtc requires drop_infer_latency, blend_policy_steps=0 "
            "and handoff_skip_steps=0 to preserve the fixed prefix clock"
        )
    if (values["algorithm"] == "aac" or bid_aac) and not values["aac"]["ee_stats_path"]:
        raise ValueError("inference.aac.ee_stats_path is required by AAC")
    if values["algorithm"] == "aac":
        if values["aac"]["chunk_id_selector"] not in {"0", "mean", "backward"}:
            raise ValueError("AAC chunk_id_selector must be 0, mean or backward")
        if not 0 < values["aac"]["backward_beta"] <= 1:
            raise ValueError("AAC backward_beta must be in (0, 1]")


def _validate_chunktrust_inference(values: dict[str, Any], executor: dict[str, Any]) -> None:
    """Keep adaptive consumption separate from fixed-window RTC and latency trimming."""
    import math

    from manimux.runtime.rtc.strategy import rtc_parameters

    for field in ("strategy", "chunk_policy_steps", "max_chunk_policy_steps", "history"):
        if values.get(field) is not None:
            raise ValueError(f"ChunkTrust does not support inference.{field}")
    for field in ("min_plan_hold_policy_steps", "handoff_skip_steps", "blend_policy_steps"):
        if type(values[field]) is not int or values[field] != 0:
            raise ValueError(f"ChunkTrust requires inference.{field}=0")
    if (
        values["action_start_mode"] != "first_step_when_ready"
        or values["handoff"] != "blend"
        or values["blend_mode"] != "linear"
        or isinstance(values["handoff_margin_s"], bool)
        or values["handoff_margin_s"] != 0
    ):
        raise ValueError(
            "ChunkTrust requires first_step_when_ready, linear blend handoff and zero margin; "
            "the strategy aligns consumed rows before committing, without wall-clock trimming"
        )
    if values["independent_group_decoding"] is not False:
        raise ValueError("ChunkTrust does not support independent group decoding")
    if (values.get("temporal_ensemble") or {}).get("enabled", False) is not False:
        raise ValueError("ChunkTrust does not support ACT temporal ensembling")
    if executor["type"] != "direct" or executor["motion_limits"] is not None:
        raise ValueError("ChunkTrust requires executor.type=direct and motion_limits=null")
    rtc = values["rtc"]
    if not isinstance(rtc, Mapping):
        raise ValueError("ChunkTrust requires a mapping for inference.rtc")
    beta = rtc["beta"]
    if isinstance(beta, bool) or not isinstance(beta, int | float) or not (
        math.isfinite(beta) and beta > 0
    ):
        raise ValueError("ChunkTrust rtc.beta must be finite and positive")
    size = rtc["delay_buffer_size"]
    if type(size) is not int or size < 1:
        raise ValueError("ChunkTrust rtc.delay_buffer_size must be a positive integer")
    initial = rtc["initial_delay_policy_steps"]
    if initial is not None and (type(initial) is not int or initial < 0):
        raise ValueError(
            "ChunkTrust rtc.initial_delay_policy_steps must be null or nonnegative integer"
        )
    if values["algorithm"] == "chunktrust_ahs":
        if rtc != rtc_parameters():
            raise ValueError("inference.rtc overrides are inactive for chunktrust_ahs")
        return
    if rtc["min_execute_policy_steps"] is not None:
        raise ValueError("ChunkTrust RTC chooses K; rtc.min_execute_policy_steps must be null")
    if rtc["condition_source"] != "canonical":
        raise ValueError("ChunkTrust RTC requires rtc.condition_source=canonical")


def _validate_chunktrust_runtime(config: dict[str, Any]) -> None:
    """Validate the first supported target-preserving client/adapter/executor path."""
    policy = config["policy"]
    adapter = policy["adapter"]
    if policy["worker"] != "xpolicylab_ws" or policy["action_decoding"] != "inline":
        raise ValueError("ChunkTrust requires xpolicylab_ws with inline action decoding")
    if policy["options"].get("action_format", "native") != "joint":
        raise ValueError("ChunkTrust requires policy.options.action_format=joint")
    if adapter["type"] != "manimux.policy_adapter.joint:JointAdapter":
        raise ValueError("ChunkTrust requires the standard absolute-joint adapter")
    if tuple(config["robot"]["group_dims"].items()) != (("left_arm", 7), ("right_arm", 7)):
        raise ValueError(
            "ChunkTrust requires ordered left_arm/right_arm groups with seven joints each"
        )
    if (
        adapter.get("gripper_mapping") is not None
        or adapter.get("allow_short_horizon", False) is not False
    ):
        raise ValueError("ChunkTrust requires complete joint chunks without gripper remapping")
    if config.get("execution", {}).get("ik", {}).get("mode", "chunk") != "chunk":
        raise ValueError("ChunkTrust does not support step IK")
    settings = config["inference"]
    if settings["algorithm"] == "chunktrust_ahs_rtc" and max(
        settings["chunktrust"]["candidates"]
    ) >= policy["horizon_policy_steps"]:
        raise ValueError("ChunkTrust RTC candidates must leave at least one future policy row")
    if (
        isinstance(settings["expected_decode_s"], bool) or settings["expected_decode_s"] != 0
        or type(settings["decode_forecast_size"]) is not int
        or settings["decode_forecast_size"] != 0
    ):
        raise ValueError("ChunkTrust does not support process-decode latency forecasting")
    if config["run"].get("execution_mode", "vla") != "vla":
        raise ValueError("ChunkTrust requires the ordinary VLA runtime")
