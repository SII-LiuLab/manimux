#!/usr/bin/env python3
"""Generate selecting YAMLs from one existing task/checkpoint; never start services."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import yaml

from manimux.cli import load_config, read_experiment, read_yaml

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("serial", "async", "act", "rtc", "paint", "aac", "autohorizon")
CHUNKTRUST_METHODS = ("chunktrust_ahs", "chunktrust_ahs_rtc")
ALL_METHODS = (*METHODS, *CHUNKTRUST_METHODS)
FAMILIES = {"Pi_05", "Xiaomi_Robotics_1", "LingBot_VLA2", "DP", "OpenWAM"}


def build_matrix(
    base,
    *,
    output_dir,
    execution_steps=None,
    methods=METHODS,
    policy_server=None,
    paint_delay_steps=None,
    aac_stats=None,
    async_trigger="refill",
):
    """Keep the base embodiment/checkpoint/timing and select compatible algorithms.

    Generated settings are engineering smoke-test presets, not a frozen study.
    The live server's sampling_modes remains authoritative for a loaded model.
    """
    if async_trigger not in {"refill", "progress"}:
        raise ValueError("async_trigger must be refill or progress")
    source = read_experiment(base, bind_local=False)
    if policy_server is not None:
        source["policy_server"] = read_yaml(policy_server)
    server = source.get("policy_server", {})
    family = server.get("policy_name")
    if family not in FAMILIES:
        raise ValueError("Provide a policy_server recipe for one of the five supported families")
    if (
        family == "Pi_05"
        and source["policy"].get("options", {}).get("action_format", "joint") != "joint"
    ):
        raise ValueError(
            "This matrix targets Pi05 JAX joint actions; Pi05 EEF needs separate validation"
        )
    horizon = source["policy"]["horizon_policy_steps"]
    fixed_methods = set(methods) - {"aac", "autohorizon", *CHUNKTRUST_METHODS}
    if fixed_methods and (type(execution_steps) is not int or not 1 <= execution_steps < horizon):
        raise ValueError(f"fixed-window methods require execution_steps in [1, {horizon - 1}]")
    unknown = set(methods) - set(ALL_METHODS)
    if unknown or len(set(methods)) != len(methods):
        raise ValueError(f"Unknown or duplicate methods: {methods}")
    skipped = {
        m: "U-Net DDPM has no flow inversion/action attention"
        for m in methods
        if family == "DP" and m in {"paint", "autohorizon"}
    }
    if family == "LingBot_VLA2" and "rtc" in methods:
        skipped["rtc"] = "LingBot RTC withdrawn: conditioned latency exceeds the validated real-time budget"
    for method in set(methods).intersection(CHUNKTRUST_METHODS):
        reason = _chunktrust_skip_reason(source, server, family, method)
        if reason is not None:
            skipped[method] = reason
    selected = [m for m in methods if m not in skipped]
    if "paint" in selected and (
        paint_delay_steps is None
        or not 0 < paint_delay_steps <= execution_steps
        or paint_delay_steps + execution_steps > horizon
    ):
        raise ValueError(
            "PAINT requires 0 < delay <= execution_steps and delay + execution_steps <= H"
        )
    if "aac" in selected and (aac_stats is None or not Path(aac_stats).is_file()):
        raise ValueError("AAC requires an explicit existing EE-increment statistics file")
    output_dir = Path(output_dir).resolve()
    paths = [output_dir / f"{method}.yaml" for method in selected]
    if any(path.exists() for path in paths):
        raise FileExistsError("Matrix output already exists; choose a new directory")
    prepared = []
    for method, path in zip(selected, paths, strict=True):
        config = deepcopy(source)
        old = source.get("inference", {})
        # Observation-history wrappers are independent of the selected scheduler.
        inference = {k: deepcopy(old[k]) for k in ("strategy", "history") if k in old}
        inference.update(
            algorithm="act_temporal_ensemble" if method == "act" else method,
            max_plan_age_s=old.get("max_plan_age_s", 2.0),
            blend_policy_steps=0,
            action_start_mode="drop_infer_latency",
        )
        if method in {"serial", "async"}:
            inference.update(
                chunk_policy_steps=execution_steps,
                inference_schedule="serial" if method == "serial" else "single_inflight",
            )
            if method == "serial":
                inference["action_start_mode"] = "first_step_when_ready"
            else:
                inference["request_trigger"] = async_trigger
                if async_trigger == "refill":
                    inference["refill_threshold_s"] = execution_steps * source["policy"]["action_dt_s"]
        elif method == "act":
            inference["temporal_ensemble"] = {"coefficient": 0.01, "query_interval_policy_steps": 1}
        elif method == "rtc":
            inference["rtc"] = dict(
                min_execute_policy_steps=execution_steps,
                initial_delay_policy_steps=None,
                delay_buffer_size=10,
                beta=5.0,
            )
        elif method == "paint":
            inference["paint"] = dict(
                execution_policy_steps=execution_steps,
                initial_delay_policy_steps=paint_delay_steps,
                delay_buffer_size=10,
            )
        elif method in CHUNKTRUST_METHODS:
            from manimux.runtime.chunktrust.config import chunktrust_parameters
            from manimux.runtime.rtc.strategy import rtc_parameters

            if old.get("strategy") is not None or old.get("history") is not None:
                raise ValueError(
                    "ChunkTrust matrix variants cannot inherit a strategy/history plugin"
                )
            settings = old.get("chunktrust")
            if settings is None:
                settings = {}
            if not isinstance(settings, Mapping):
                raise ValueError("inference.chunktrust must be a mapping")
            if method == "chunktrust_ahs_rtc":
                settings = {"candidates": [10, 20, 30, 40], **settings}
            inference.update(
                strategy=None,
                chunktrust=chunktrust_parameters(**settings),
                inference_schedule="serial" if method == "chunktrust_ahs" else "single_inflight",
                request_trigger="algorithm",
                action_start_mode="first_step_when_ready",
                chunk_policy_steps=None,
                max_chunk_policy_steps=None,
                handoff_skip_steps=0,
            )
            if server.get("backend_identity") is not None:
                from manimux.policies.base import backend_identity_from_recipe

                config["policy"]["expected_backend"] = backend_identity_from_recipe(
                    server, server["backend_identity"],
                )
            if method == "chunktrust_ahs_rtc":
                rtc = old.get("rtc", {})
                if not isinstance(rtc, Mapping):
                    raise ValueError("inference.rtc must be a mapping")
                # Only guidance/delay carry over: the source algorithm's fixed K
                # is not ChunkTrust's adaptive old-plan consumption budget.
                rtc_parameters(**rtc)
                inference["rtc"] = {
                    "beta": 10.0,
                    "initial_delay_policy_steps": None,
                    "delay_buffer_size": 10,
                    **{key: deepcopy(rtc[key]) for key in (
                        "beta", "initial_delay_policy_steps", "delay_buffer_size",
                    ) if key in rtc and old.get("algorithm") != "chunktrust_ahs"},
                    "min_execute_policy_steps": None,
                    "condition_source": "canonical",
                }
        else:
            inference["action_start_mode"] = "first_step_when_ready"
            if method == "aac":
                inference["aac"] = dict(
                    num_samples=20,
                    motion_threshold=0.2,
                    ee_stats_path=str(Path(aac_stats).resolve()),
                    chunk_id_selector="0",
                    backward_beta=0.99,
                )
        config["inference"] = inference
        # Storage belongs to the selecting collection, never to a variant.
        config["run"].pop("output_dir", None)
        config["run"].pop("history_sessions", None)
        config.pop("collection", None)
        config.pop("config_sources", None)
        if "experiment" in config:
            config["experiment"]["variant"] = method
        config.setdefault("robogui", {})["policy_label"] = f"{family} - {method}"
        prepared.append((path, config))
    # Validate all variants through the real loader before publishing any output.
    import tempfile

    with tempfile.TemporaryDirectory(prefix="manimux-matrix-") as directory:
        for path, config in prepared:
            check = Path(directory) / path.name
            check.write_text(yaml.safe_dump(config, sort_keys=False))
            load_config(check)
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, config in prepared:
        with path.open("x") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
    return paths, skipped


def _chunktrust_skip_reason(source, server, family, method):
    """Require an explicit trace recipe; the live backend is checked at startup."""
    if family != "Pi_05":
        return "ChunkTrust currently supports only Pi05 JAX joint14 profiles"
    identity = server.get("backend_identity")
    if identity is not None:
        from manimux.policies.base import backend_identity_from_recipe

        model = backend_identity_from_recipe(server, identity)["model"]
    else:
        model = (source["policy"].get("expected_backend") or {}).get("model", {})
    trace = model.get("denoise_trace")
    if (
        model.get("action_type") != "joint"
        or (model.get("execution") or {}).get("backend") != "jax"
    ):
        return "ChunkTrust requires explicit JAX joint-action backend identity"
    modes = {"default", "rtc"} if method == "chunktrust_ahs_rtc" else {"default"}
    if not isinstance(trace, Mapping) or trace.get("enabled") is not True:
        return "requires an explicit denoise_trace backend identity and trace-enabled deployment"
    if server.get("return_denoise_trace") is not True:
        return "requires policy_server.return_denoise_trace=true"
    supported = trace.get("supported_modes")
    if not isinstance(supported, list) or not all(isinstance(mode, str) for mode in supported):
        return "denoise_trace.supported_modes must explicitly list the trace-capable sampling modes"
    if not modes.issubset(supported):
        return (
            f"requires denoising trace for sampling modes {sorted(modes)}; "
            "default trace is insufficient for RTC"
        )
    if method == "chunktrust_ahs_rtc":
        guidance = model.get("rtc_guidance")
        if (
            not isinstance(guidance, Mapping)
            or guidance.get("padding_mask") != "valid_action_indices"
        ):
            return "requires an explicit RTC guidance mask excluding padded model coordinates"
        indices = guidance.get("valid_action_indices")
        if (
            not isinstance(indices, list) or any(type(index) is not int for index in indices)
            or indices != list(range(14))
        ):
            return "RTC guidance valid_action_indices must be the ordered integer list 0..13"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-server", type=Path)
    parser.add_argument(
        "--execution-steps", type=int, help="Required only for fixed-window methods",
    )
    parser.add_argument("--paint-delay-steps", type=int)
    parser.add_argument(
        "--async-trigger", choices=("refill", "progress"), default="refill",
        help="Async step meaning: refill caps the source prefix; progress retains the full horizon and replans after K source steps",
    )
    parser.add_argument("--aac-stats", type=Path)
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=METHODS)
    args = parser.parse_args()
    paths, skipped = build_matrix(**vars(args))
    for path in paths:
        print(path)
    for method, reason in skipped.items():
        print(f"Not generated: {method}: {reason}")


if __name__ == "__main__":
    main()
