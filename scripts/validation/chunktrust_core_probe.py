"""CPU-only PI AHS oracle, parameter-consumer and transaction checks.

Run from an installed ManiMux environment. The committed golden outputs were
generated from the pinned PI05 integration, independently of the ManiMux adapter.
--reference optionally re-executes those upstream numerical methods using AST
extraction, without importing OpenPI, JAX, Torch or a hardware backend.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
FIXTURES = Path(__file__).with_name("chunktrust_fixtures")
REFERENCE_PATH = "third_party/overlays/robotwin/policy/pi05_horizon/pi_model.py"
INFO_FIELDS = (
    "horizons",
    "z_intra",
    "z_intra_norm",
    "u_inter_raw",
    "u_inter_norm",
    "q_intra",
    "p_inter",
    "q_mix",
    "scores",
    "expected_horizon",
    "choose_mode",
    "chosen_horizon",
    "update_center",
    "update_feedback",
)
PARAMETER_MAP = {
    "candidates": "horizon_candidates",
    "exec_mode": "horizon_exec_mode",
    "temperature": "horizon_expected_temp",
    "epsilon": "horizon_ts_epsilon",
    "update_mode": "horizon_ts_update_mode",
    "kernel_bandwidth": "horizon_ts_kernel_bandwidth",
    "forget_rho": "horizon_ts_forget_rho",
    "update_eta": "horizon_ts_update_eta",
    "intra_cut_t": "horizon_intra_cut_t",
    "intra_z_mode": "horizon_intra_z_mode",
    "intra_tau_window": "horizon_intra_tau_window",
    "intra_alpha": "horizon_intra_alpha",
    "inter_use_speed_uniformity": "horizon_inter_use_speed_uniformity",
    "mix_mode": "horizon_mix_mode",
    "history_window": "horizon_inter_window",
    "inter_weight": "horizon_inter_weight",
    "inter_beta": "horizon_inter_beta",
    "inter_fallback": "horizon_inter_fallback",
}


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_reference(checkout):
    from manimux.runtime.chunktrust.provenance import UPSTREAM_COMMIT

    source = subprocess.check_output(
        ["git", "show", f"{UPSTREAM_COMMIT}:{REFERENCE_PATH}"],
        cwd=checkout,
        text=True,
    )
    original = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "PI0"
    )
    # Select by the vendored numerical interface, not by copying its implementation.
    frozen = ast.parse((REPO / "manimux/runtime/chunktrust/_pi_ahs.py").read_text())
    names = {
        node.name
        for node in frozen.body[-1].body
        if isinstance(node, ast.FunctionDef) and node.name not in {"__init__", "select"}
    }
    methods = [
        node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in methods} == names
    body = "\n".join("    " + line for node in methods for line in ast.unparse(node).splitlines())
    namespace = {"np": np, "math": math, "Any": Any}
    exec("from __future__ import annotations\nclass Oracle:\n" + body, namespace)
    return namespace["Oracle"], hashlib.sha256(source.encode()).hexdigest()


def inputs(data, case, step):
    velocity = np.roll(data["velocity"], step, axis=0)
    if case.get("single_step"):
        velocity = velocity[:1]
    rows = case.get("history_rows", 100)
    history = data["history"][-rows:] if rows else data["history"][:0]
    return velocity, data["actions"] + step * 0.025, history + step * 0.025


def oracle_outputs(oracle_type, options, data, case):
    oracle = oracle_type()
    oracle.__dict__.update({target: options[name] for name, target in PARAMETER_MAP.items()})
    oracle._horizon_ts_state = {int(k): [1.0, 1.0] for k in options["candidates"]}
    oracle._horizon_ts_rng = np.random.default_rng(options["seed"])
    records = []
    for step in range(3):
        velocity, actions, history = inputs(data, case, step)
        oracle._executed_action_history = history[-options["history_window"] :].copy()
        info = oracle._compute_horizon_info(velocity, 50, xt_step=actions)
        k, info = oracle._select_exec_k_horizon_from_info(info, len(actions))
        records.append(
            json_value(
                {
                    "k": k,
                    "info": {key: info[key] for key in INFO_FIELDS},
                    "posterior": oracle._horizon_ts_state,
                    "rng": oracle._horizon_ts_rng.bit_generator.state,
                }
            )
        )
    return records


def assert_close(actual, expected):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), (set(actual), set(expected))
        for key in expected:
            assert_close(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_close(left, right)
    elif isinstance(expected, float):
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
    else:
        assert actual == expected, (actual, expected)


def must_reject(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except (TypeError, ValueError):
        return
    raise AssertionError("An invalid or stale operation was accepted")


def parameter_probe(golden, data, reference):
    from manimux.runtime.chunktrust.config import ChunkTrustOptions, chunktrust_parameters
    from manimux.runtime.chunktrust.core import ChunkTrustAdapter

    records = {}
    for case in golden["cases"]:
        options = ChunkTrustOptions.from_mapping(case["options"])
        adapter = ChunkTrustAdapter(options)
        got = []
        for step, expected in enumerate(case["expected"]):
            velocity, actions, history = inputs(data, case, step)
            before = adapter.snapshot()
            adapter.evaluate(velocity, actions, history)
            assert adapter.snapshot() == before
            proposal = adapter.prepare(velocity, actions, history, request_key=step)
            assert adapter.snapshot() == before
            adapter.commit(proposal)
            snapshot = adapter.snapshot()
            actual = json_value(
                {
                    "k": proposal.k,
                    "info": {key: proposal.info[key] for key in INFO_FIELDS},
                    "posterior": snapshot["posterior"],
                    "rng": snapshot["rng"],
                }
            )
            assert_close(actual, expected)
            got.append(actual)
        if reference is not None:
            assert_close(got, oracle_outputs(reference, options.to_dict(), data, case))
        records[case["name"]] = got

    # Every parameter must change numerical outputs/state in an applicable branch.
    # Comparing only the integer K would miss temperature and update parameters.
    consumers = {}
    for case in golden["cases"]:
        name = case.get("parameter")
        if name is None:
            continue
        baseline = records["baseline_empty" if name == "inter_fallback" else "baseline"]
        result = records[case["name"]]
        assert result != baseline, f"No numerical/state effect for {name}"
        consumers[name] = "upstream numerical output or RNG/posterior differs"
    assert set(consumers) == set(chunktrust_parameters())
    assert len(consumers) == 19
    return consumers


def transaction_probe(data):
    from manimux.runtime.chunktrust.config import ChunkTrustOptions, chunktrust_parameters
    from manimux.runtime.chunktrust.core import ChunkTrustAdapter

    options = ChunkTrustOptions(
        candidates=(5, 15, 25, 35, 45),
        temperature=0.45,
        epsilon=0.0,
        seed=71,
        kernel_bandwidth=3.0,
        forget_rho=0.7,
        update_eta=2.0,
        intra_cut_t=0.45,
        intra_tau_window=1,
        intra_alpha=7.0,
        history_window=85,
        inter_weight=0.8,
        inter_beta=7.0,
        inter_fallback="neutral",
    )
    adapter = ChunkTrustAdapter(options)
    args = inputs(data, {}, 0)
    initial = adapter.snapshot()
    first = adapter.prepare(*args, request_key="first", context={"source_offset": 0})
    assert adapter.snapshot() == initial
    must_reject(adapter.prepare, *args, request_key="first")
    adapter.discard("first")
    assert adapter.snapshot() == initial
    must_reject(adapter.commit, first)
    second = adapter.prepare(*args, request_key="second")
    sibling = adapter.prepare(*args, request_key="sibling")
    # Returned dictionaries cannot edit the private state to be committed.
    second.after["posterior"]["5"] = [999, 999]
    adapter.commit(second)
    assert adapter.snapshot()["posterior"]["5"] != [999, 999]
    must_reject(adapter.commit, second)
    must_reject(adapter.commit, sibling)
    checkpoint = json.loads(json.dumps(adapter.snapshot(), allow_nan=False))
    clone = adapter.clone()
    restored = ChunkTrustAdapter(options)
    restored.restore(checkpoint)
    for step in range(1, 4):
        args = inputs(data, {}, step)
        proposals = [
            instance.prepare(*args, request_key=step) for instance in (adapter, clone, restored)
        ]
        assert len({json.dumps(item.to_dict(), sort_keys=True) for item in proposals}) == 1
        for instance, proposal in zip((adapter, clone, restored), proposals, strict=True):
            instance.commit(proposal)
    pending = adapter.prepare(*args, request_key="before-reset")
    adapter.reset()
    must_reject(adapter.commit, pending)
    fresh = ChunkTrustAdapter(options)
    left = adapter.prepare(*args, request_key="after-reset")
    right = fresh.prepare(*args, request_key="after-reset")
    assert_close(json_value(left.info), json_value(right.info))
    assert left.after["rng"] == right.after["rng"]
    assert adapter.options == options
    # Complete normalized export/reload does not make filled defaults explicit overrides.
    assert chunktrust_parameters(**options.to_dict()) == options.to_dict()
    corrupted = copy.deepcopy(checkpoint)
    corrupted["config"]["intra_alpha"] = 4.0
    must_reject(restored.restore, corrupted)
    unchanged = adapter.snapshot()
    bad = list(args)
    bad[0] = bad[0].copy()
    bad[0][0, 0, 0] = np.nan
    must_reject(adapter.prepare, *bad, request_key="nan")
    assert adapter.snapshot() == unchanged
    empty = np.zeros((0, 14), np.float32)
    must_reject(
        adapter.prepare, data["velocity"][:, :39], data["actions"][:39], empty, request_key="short"
    )
    return {
        "speculative_state": True,
        "duplicate_stale_rejected": True,
        "hidden_nondefaults_survive_clone_reset_restore": True,
        "json_replay": True,
        "bad_inputs_do_not_advance_state": True,
    }


def config_probe():
    from manimux.runtime.chunktrust.config import chunktrust_parameters

    cases = [
        {"unknown": 1},
        {"seed": True},
        {"seed": 0.0},
        {"seed": -1},
        {"history_window": 3},
        {"candidates": [10, 10]},
        {"candidates": [True]},
        {"temperature": np.inf},
        {"epsilon": np.nan},
        {"intra_alpha": "7"},
        {"inter_use_speed_uniformity": 1},
        {"forget_rho": 0},
        {"exec_mode": "thompson", "temperature": 0.5},
        {"update_mode": "independent", "kernel_bandwidth": 3},
        {"exec_mode": "qmix_expected_round", "seed": 3},
        {"intra_z_mode": "trend", "intra_tau_window": 1},
        {"mix_mode": "intra", "inter_beta": 1},
    ]
    for case in cases:
        must_reject(chunktrust_parameters, **case)
    # This is conditional, not inactive: inter can need intra fallback.
    chunktrust_parameters(mix_mode="inter", intra_alpha=7, inter_fallback="ehh_only")
    normalized = chunktrust_parameters(seed=np.int64(3), intra_alpha=np.float32(7))
    json.dumps(normalized, allow_nan=False)
    return {"invalid_config_cases": len(cases), "conditional_fallback_allowed": True}


def duck_probe():
    from manimux.runtime.chunktrust._pi_ahs import Selector

    recording = json.loads((FIXTURES / "duck_replans.json").read_text())
    for scores, candidates, selected in zip(
        recording["ahs_replan_scores"],
        recording["ahs_replan_candidates"],
        recording["selected_K"],
        strict=True,
    ):
        probability = Selector._normalize_score_distribution(
            np.asarray(scores),
            recording["temperature"],
        )
        assert int(round(float(np.dot(probability, candidates)))) == selected
    return {
        "recorded_replans": len(recording["selected_K"]),
        "scope": "Recorded score-to-K arithmetic only; no raw trace or physical success claim",
    }


def main():
    from manimux.runtime.chunktrust.provenance import core_provenance, verify_frozen_core

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--reference", type=Path, help="Optional ChunkTrust Git checkout for live oracle"
    )
    args = parser.parse_args()
    verify_frozen_core()
    golden = json.loads((FIXTURES / "core_golden.json").read_text())
    input_path = FIXTURES / "core_inputs.npz"
    assert hashlib.sha256(input_path.read_bytes()).hexdigest() == golden["input_sha256"]
    reference = None
    if args.reference is not None:
        reference, source_hash = load_reference(args.reference)
        assert source_hash == golden["reference_sha256"]
    with np.load(input_path, allow_pickle=False) as data:
        report = {
            "scope": "CPU numerical/config/transaction checks; no model, services or hardware",
            "python": sys.version,
            "numpy": np.__version__,
            "source": core_provenance(),
            "live_upstream_oracle": reference is not None,
            "golden_cases": len(golden["cases"]),
            "parameter_consumers": parameter_probe(golden, data, reference),
            "transactions": transaction_probe(data),
            "configuration": config_probe(),
            "recorded_golden": duck_probe(),
            "success": True,
        }
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
