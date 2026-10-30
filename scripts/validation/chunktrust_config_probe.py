#!/usr/bin/env python3
"""Exercise ChunkTrust's public YAML/config path without models, services or hardware."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from manimux.cli import load_config, read_yaml  # noqa: E402
from manimux.runtime import inference_parameters  # noqa: E402
from manimux.runtime.chunktrust.config import ChunkTrustOptions  # noqa: E402

EXAMPLES = ROOT / "manimux/configs/experiments/example-yam-pi05/put_bottles"
METHODS = ("chunktrust_ahs", "chunktrust_ahs_rtc")


def variant(directory: Path, label: str, base: Path, overrides: dict) -> Path:
    target = directory / f"{label}.yaml"
    target.write_text(yaml.safe_dump({"_base_": str(base), **overrides}, sort_keys=False))
    return target


def rejected(directory: Path, label: str, base: Path, overrides: dict) -> None:
    path = variant(directory, label, base, overrides)
    try:
        load_config(path)
    except (TypeError, ValueError):
        return
    raise AssertionError(f"invalid configuration was accepted: {label}")


def load_matrix_module():
    path = ROOT / "scripts/experiments/build_inference_matrix.py"
    spec = importlib.util.spec_from_file_location("chunktrust_matrix_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    matrix = load_matrix_module()
    checked = 0
    with tempfile.TemporaryDirectory(prefix="manimux-chunktrust-config-") as task:
        directory = Path(task)
        for method in METHODS:
            config = load_config(EXAMPLES / f"{method}.yaml")
            settings = config["inference"]
            assert len(settings["chunktrust"]) == 19
            assert settings["chunktrust"] == ChunkTrustOptions.from_mapping(
                settings["chunktrust"],
            ).to_dict()
            assert settings["inference_schedule"] == (
                "serial" if method == "chunktrust_ahs" else "single_inflight"
            )
            assert settings["chunk_policy_steps"] is None
            assert settings["action_start_mode"] == "first_step_when_ready"
            assert inference_parameters(executor=config["executor"], **settings) == settings
            checked += 1

        base = EXAMPLES / "chunktrust_ahs.yaml"
        # Each published field crosses real _base_ + inference.config merges.
        alternatives = {
            "candidates": [10, 30, 50],
            "exec_mode": "thompson",
            "temperature": 0.3,
            "epsilon": 0.0,
            "seed": 17,
            "update_mode": "independent",
            "kernel_bandwidth": 7.0,
            "forget_rho": 0.8,
            "update_eta": 0.0,
            "intra_cut_t": 0.5,
            "intra_z_mode": "trend",
            "intra_tau_window": 4,
            "intra_alpha": 3.0,
            "inter_use_speed_uniformity": False,
            "mix_mode": "inter",
            "history_window": 80,
            "inter_weight": 0.2,
            "inter_beta": 2.5,
            "inter_fallback": "neutral",
        }
        assert set(alternatives) == set(ChunkTrustOptions().to_dict())
        defaults = ChunkTrustOptions().to_dict()
        for field, value in alternatives.items():
            path = variant(directory, f"field-{field}", base, {
                "inference": {"chunktrust": {field: value}},
            })
            actual = load_config(path)["inference"]["chunktrust"]
            expected = ChunkTrustOptions.from_mapping({**defaults, field: value}).to_dict()
            assert actual == expected, field
            checked += 1

        bad_fields = (
            {"unknown": 1}, {"candidates": [True]}, {"seed": False},
            {"temperature": False}, {"temperature": 0}, {"epsilon": 1.1},
            {"history_window": 3}, {"inter_use_speed_uniformity": 0},
            {"exec_mode": "qmix_expected_round", "seed": 1},
            {"update_mode": "independent", "forget_rho": 0.8},
            {"mix_mode": "intra", "inter_weight": 0.2},
        )
        for index, fields in enumerate(bad_fields):
            rejected(directory, f"bad-field-{index}", base, {
                "inference": {"chunktrust": fields},
            })
            checked += 1
        for field, value in (
            ("chunktrust", False), ("strategy", "some.module:factory"),
            ("chunk_policy_steps", 50), ("max_chunk_policy_steps", 10),
            ("blend_policy_steps", 1), ("handoff_skip_steps", False),
            ("action_start_mode", "drop_infer_latency"),
            ("inference_schedule", "single_inflight"), ("request_trigger", "refill"),
            ("temporal_ensemble", {"enabled": True}),
            ("rtc", {"beta": 9}), ("independent_group_decoding", True),
        ):
            rejected(directory, f"bad-inference-{field}", base, {"inference": {field: value}})
            checked += 1
        for label, overrides in (
            ("inactive", {"inference": {"algorithm": "serial"}}),
            ("smooth", {"executor": {"type": "smooth"}}),
            ("process", {"policy": {"action_decoding": "process"}}),
            ("wrong-worker", {"policy": {"worker": "fake"}}),
            ("gripper-map", {"policy": {"adapter": {"gripper_mapping": {"mode": "curve"}}}}),
        ):
            rejected(directory, label, base, overrides)
            checked += 1

        rtc_base = EXAMPLES / "chunktrust_ahs_rtc.yaml"
        for field, value in (
            ("beta", False), ("beta", 0), ("delay_buffer_size", False),
            ("initial_delay_policy_steps", False), ("condition_source", "timeline"),
            ("min_execute_policy_steps", 20),
        ):
            rejected(directory, f"bad-rtc-{field}-{value}", rtc_base, {
                "inference": {"rtc": {field: value}},
            })
            checked += 1
        rejected(directory, "rtc-no-suffix", rtc_base, {
            "inference": {"chunktrust": {"candidates": [10, 50]}},
        })
        checked += 1

        # A default-only trace recipe must never generate an advertised RTC deployment.
        paths, skipped = matrix.build_matrix(
            base, output_dir=directory / "default-trace", methods=METHODS,
        )
        assert [path.stem for path in paths] == ["chunktrust_ahs"]
        assert "chunktrust_ahs_rtc" in skipped
        checked += 1

        unsafe_guidance = variant(directory, "unmasked-combined-profile", rtc_base, {
            "policy_server": {"backend_identity": {"model": {
                "rtc_guidance": {"padding_mask": "unmasked"},
            }}},
        })
        paths, skipped = matrix.build_matrix(
            unsafe_guidance, output_dir=directory / "unmasked-matrix", methods=METHODS,
        )
        assert [path.stem for path in paths] == ["chunktrust_ahs"]
        assert "chunktrust_ahs_rtc" in skipped
        checked += 1

        # This is a synthetic FUTURE identity, not a live RTC+trace backend claim.
        combined = variant(directory, "combined-profile", rtc_base, {
            "inference": {
                "chunktrust": {"temperature": 0.31, "history_window": 88, "seed": 13},
                "rtc": {"beta": 7.0, "initial_delay_policy_steps": 0, "delay_buffer_size": 3},
            },
        })
        source = load_config(combined)
        paths, skipped = matrix.build_matrix(
            combined, output_dir=directory / "combined-matrix", methods=METHODS,
        )
        assert not skipped and len(paths) == 2
        for path in paths:
            actual = load_config(path)["inference"]
            assert actual["chunktrust"] == source["inference"]["chunktrust"]
            if path.stem == "chunktrust_ahs_rtc":
                assert actual["rtc"] == source["inference"]["rtc"]
        checked += 1

        # Existing default methods remain unchanged; adaptive choices need no fixed K.
        assert matrix.METHODS == ("serial", "async", "act", "rtc", "paint", "aac", "autohorizon")
        stats = directory / "stats.json"
        stats.write_text("{}")
        paths, skipped = matrix.build_matrix(
            EXAMPLES / "serial.yaml", output_dir=directory / "legacy-matrix",
            execution_steps=16, paint_delay_steps=4, aac_stats=stats,
        )
        assert not skipped and [path.stem for path in paths] == list(matrix.METHODS)
        for path in paths:
            assert "chunktrust" not in read_yaml(path)["inference"]
        checked += 1
    print(f"PASS: {checked} ChunkTrust config/merge/matrix cases; no model or hardware calls.")
    print("This verifies resolved config routing; model/selector behavior needs other probes.")


if __name__ == "__main__":
    main()
