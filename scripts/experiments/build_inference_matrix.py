#!/usr/bin/env python3
"""Generate selecting YAMLs from one existing task/checkpoint; never start services."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml

from manimux.cli import load_config, read_experiment, read_yaml

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("serial", "async", "act", "rtc", "paint", "aac", "autohorizon")
FAMILIES = {"Pi_05", "Xiaomi_Robotics_1", "LingBot_VLA2", "DP", "OpenWAM"}


def build_matrix(
    base,
    *,
    output_dir,
    execution_steps,
    methods=METHODS,
    policy_server=None,
    paint_delay_steps=None,
    aac_stats=None,
):
    """Keep the base embodiment/checkpoint/timing and select compatible algorithms.

    Generated settings are engineering smoke-test presets, not a frozen study.
    The live server's sampling_modes remains authoritative for a loaded model.
    """
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
    if isinstance(execution_steps, bool) or not 1 <= execution_steps < horizon:
        raise ValueError(f"execution_steps must be in [1, {horizon - 1}]")
    unknown = set(methods) - set(METHODS)
    if unknown or len(set(methods)) != len(methods):
        raise ValueError(f"Unknown or duplicate methods: {methods}")
    skipped = {
        m: "U-Net DDPM has no flow inversion/action attention"
        for m in methods
        if family == "DP" and m in {"paint", "autohorizon"}
    }
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
    if family == "OpenWAM":
        server.update(compile_enabled=False, dit_cache_enabled=False, inference_seed=0)
    elif family == "LingBot_VLA2":
        server["use_compile"] = False
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
        config["run"]["output_dir"] = str(Path(source["run"]["output_dir"]) / "matrix" / method)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-server", type=Path)
    parser.add_argument("--execution-steps", type=int, required=True)
    parser.add_argument("--paint-delay-steps", type=int)
    parser.add_argument("--aac-stats", type=Path)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    args = parser.parse_args()
    paths, skipped = build_matrix(**vars(args))
    for path in paths:
        print(path)
    for method, reason in skipped.items():
        print(f"Not generated: {method}: {reason}")


if __name__ == "__main__":
    main()
