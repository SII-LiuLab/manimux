#!/usr/bin/env python3
"""Disconnected real-weight MolmoAct2 parity checks and adapter latency benchmark.

No robot, camera driver, service shutdown, or motor connection is constructed.
Wall latency includes processor, VLM, sampler, output conversion and BID scoring;
it excludes WS transport and runtime scheduling. Inputs are labelled in the report.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import yaml

from XPolicyLab.policy.MolmoAct2.model import Model
from XPolicyLab.policy.MolmoAct2.sampling import MolmoAct2Sampler
from manimux.runtime.bid import select_backward


def packed(result):
    if isinstance(result, dict):
        result = result["actions"]
    keys = ("left_arm_joint_state", "left_ee_joint_state", "right_arm_joint_state", "right_ee_joint_state")
    return np.stack([np.concatenate([np.asarray(step[k]).reshape(-1) for k in keys]) for step in result])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observation", type=Path)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--runtime-local", type=Path)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--candidates", type=int, default=16)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load((root / "manimux/configs/policy/molmoact2/yam/latest300-step30000.yaml").read_text())
    cfg.update(model_path=str(args.checkpoint.resolve()), implementation="local", enable_cuda_graph=False, num_steps=args.steps)
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    state = np.zeros(14, dtype=np.float32)
    images = {name: rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
              for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")}
    provenance = "synthetic 480x640 RGB arrays and zero joint state; performance only"
    if args.observation:
        with np.load(args.observation, allow_pickle=False) as saved:
            state = np.concatenate([saved["state__left_arm"], saved["state__right_arm"]]).astype(np.float32)
            images = {role: saved["image__" + name] for role, name in (
                ("cam_high", "d405_front"), ("cam_left_wrist", "d405_left"), ("cam_right_wrist", "d405_right"))}
        provenance = str(args.observation.resolve())
    obs = {"state": {"left_arm_joint_state": state[:6], "left_ee_joint_state": state[6:7],
                     "right_arm_joint_state": state[7:13], "right_ee_joint_state": state[13:]},
           "vision": images, "instruction": "put the coin into the bin"}
    report = {"checkpoint": str(args.checkpoint.resolve()), "observation": provenance,
              "precision": "bfloat16", "steps": args.steps, "horizon": 30, "torch": torch.__version__,
              "device": torch.cuda.get_device_name(), "cuda_graph": False, "compile": False,
              "scope": "adapter end-to-end; excludes transport and runtime", "samples_ms": {},
              "gpu_before": subprocess.check_output(["nvidia-smi"], text=True)}
    report["source_sha256"] = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            root / "third_party/XPolicyLab/policy/MolmoAct2/model.py",
            root / "third_party/XPolicyLab/policy/MolmoAct2/sampling.py",
            root / "third_party/XPolicyLab/policy/MolmoAct2/molmoact2/experiments/olmo/hf_model/modeling_molmoact2.py",
        )
    }
    if args.observation:
        provenance_file = args.observation.parent / "input-provenance.json"
        if provenance_file.exists():
            report["input_provenance"] = json.loads(provenance_file.read_text())
    def persist():
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
    persist()
    print("Loading real weights", flush=True)
    policy = Model(cfg)
    print("Loaded", policy.sampling_modes(), flush=True)
    def reset():
        policy.reset()
        policy.update_obs(obs)
    reset()
    baseline = packed(policy.get_action())
    np.savez(args.output / "baseline.npz", actions=baseline)
    reset()
    bridge = packed(policy._predict(obs, 0, sampler=MolmoAct2Sampler(policy, {"mode": "default"})))
    np.testing.assert_allclose(baseline, bridge, rtol=0, atol=0)
    reset()
    auto = policy.get_action_autohorizon({"mode": "autohorizon"})
    np.testing.assert_allclose(baseline, packed(auto), rtol=0, atol=0)
    report["autohorizon"] = auto["autohorizon"]
    rtc = {"mode": "rtc", "action_condition": baseline, "condition_weights": np.zeros(30), "beta": 10.0}
    reset()
    zero = packed(policy.get_action_rtc(rtc))
    np.testing.assert_allclose(baseline, zero, rtol=0, atol=0)
    report["parity"] = "bitwise baseline == bridge == AutoHorizon == zero-weight RTC"
    # A nontrivial prefix from an earlier chunk, shifted as at a replan boundary.
    target = np.concatenate([baseline[10:], np.repeat(baseline[-1:], 10, axis=0)])
    rtc.update(action_condition=target, condition_weights=np.r_[np.ones(5), np.linspace(1, 0, 10), np.zeros(15)])
    paint = {"mode": "paint", "action_prefix": target[:5], "delay_steps": 5}
    reset()
    changed = packed(policy.get_action_rtc(rtc))
    assert not np.array_equal(baseline, changed), "RTC did not affect the action chunk"
    reset()
    conditioned = packed(policy.get_action_paint(paint))
    assert not np.array_equal(baseline, conditioned), "PAINT did not affect the action chunk"
    np.savez(args.output / "conditioned.npz", target=target, rtc=changed, paint=conditioned)
    report["conditioned_finite"] = bool(np.isfinite(changed).all() and np.isfinite(conditioned).all())
    report["condition"] = {"delay_steps": 5, "rtc_beta": 10, "replan_steps": 10}
    def bid():
        result = policy.get_action_aac({"mode": "aac", "num_samples": args.candidates})
        rows = np.stack([packed(item) for item in result["actions"]])
        assert not np.array_equal(rows[0], rows[1])
        selected, scores = select_backward(rows, baseline, replan_steps=10, rho=0.9)
        return {"candidates": rows, "selected": selected, "scores": scores}
    calls = {"async_default": policy.get_action,
             "rtc": lambda: policy.get_action_rtc(rtc),
             "paint": lambda: policy.get_action_paint(paint),
             "autohorizon": lambda: policy.get_action_autohorizon({"mode": "autohorizon"}),
             f"bid_{args.candidates}": bid}
    report["samples_ms"] = {name: [] for name in calls}
    report["peak_allocated_GB"] = {name: [] for name in calls}
    persist()
    for i in range(args.warmup + args.repeats):
        # Rotate order to reduce algorithm/order bias on a shared GPU.
        names = list(calls)
        names = names[i % len(names):] + names[:i % len(names)]
        for name in names:
            reset()
            policy._generator_for_env(0).manual_seed(1000 + i)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            result = calls[name]()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            if i >= args.warmup:
                report["samples_ms"][name].append(elapsed)
                report["peak_allocated_GB"][name].append(torch.cuda.max_memory_allocated() / 1e9)
            if i == args.warmup:
                if name.startswith("bid"):
                    np.savez(args.output / f"{name}.npz", **result)
                else:
                    np.savez(args.output / f"{name}.npz", actions=packed(result))
            print(i, name, round(elapsed, 2), "ms", flush=True)
            persist()
    report["summary_ms"] = {name: {"median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
                                   "min": min(values), "max": max(values), "n": len(values)}
                            for name, values in report["samples_ms"].items()}
    report["gpu_after"] = subprocess.check_output(["nvidia-smi"], text=True)
    persist()
    print(json.dumps(report["summary_ms"], indent=2), flush=True)
    if args.runtime_local:
        from types import SimpleNamespace
        from scripts.validation.molmoact2_runtime_probe import run_with_model
        variants = ["rtc_5steps", "paint_5steps"] if args.steps == 5 else ["async", "rtc", "paint", "autohorizon", "bid"]
        run_with_model(policy, cfg, SimpleNamespace(
            local=args.runtime_local, checkpoint=args.checkpoint, observation=args.observation,
            output=args.output / "runtime", seconds=10, variants=variants))



if __name__ == "__main__":
    main()
