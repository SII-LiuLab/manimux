#!/usr/bin/env python3
"""Check the LingBot-VLA2 XPolicy adapter and ManiMux timing contract."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[2]
XPOLICY_ROOT = REPO_ROOT / "XPolicyLab"
DEFAULT_CONFIG = REPO_ROOT / "manimux/configs/policy/lingbot-vla2/yam/base.yaml"
DEFAULT_INFRA_CONFIG = (
    REPO_ROOT
    / "manimux/configs/experiments/pick_red_object/yam_lingbot_vla2_manimux.yaml"
)


def _load_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"server config must be a mapping: {path}")
    return loaded


def _prepare_imports() -> None:
    for path in (REPO_ROOT, XPOLICY_ROOT):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _validate(
    config: dict[str, Any], infra_config: dict[str, Any] | None = None
) -> dict[str, Any]:
    if config.get("policy_name") != "LingBot_VLA2":
        raise ValueError("policy_name must be LingBot_VLA2")
    if config.get("protocol") != "ws":
        raise ValueError("protocol must be ws")
    if config.get("action_type") != "joint":
        raise ValueError("action_type must be joint")
    _prepare_imports()
    from XPolicyLab.policy.LingBot_VLA2.model import validate_deployment

    report = validate_deployment(config)
    infra_errors: list[str] = []
    if infra_config is not None:
        policy = infra_config.get("policy", {})
        execution = infra_config.get("inference", {})
        configured_horizon = int(policy.get("horizon_policy_steps", 0))
        if report["status"] == "ready":
            native_hz = float(report["native_hz"])
            action_horizon = int(report["action_horizon"])
            if configured_horizon != action_horizon:
                infra_errors.append(
                    "infra policy.horizon_policy_steps must equal server action_horizon "
                    f"{action_horizon}"
                )
            action_dt_s = float(policy.get("action_dt_s", 0.0))
            if abs(action_dt_s - 1.0 / native_hz) > 1e-9:
                infra_errors.append(
                    f"infra policy.action_dt_s must equal 1/native_hz ({1.0 / native_hz})"
                )
            expected_adapter_type = (
                "manimux.policy_adapter.lingbot_vla2.yam:LingBotVLA2YamAdapter"
                if report.get("action_semantics")
                == "anchor_relative_arm_absolute_gripper"
                else "manimux.policy_adapter.joint:JointAdapter"
            )
            adapter = policy.get("adapter", {})
            if not isinstance(adapter, dict) or adapter.get("type") != expected_adapter_type:
                infra_errors.append(
                    "infra policy.adapter.type must be "
                    f"{expected_adapter_type} for this checkpoint"
                )
        runtime = execution.get("algorithm")
        if runtime not in {"manimux", "rtc"}:
            infra_errors.append("infra inference.algorithm must be manimux or rtc")
        if runtime == "rtc":
            if report.get("rtc_capability") != "pi_guided_v1_sampler":
                infra_errors.append("RTC config requires sampler-level pi_guided_v1 support")
            rtc = execution.get("rtc", {})
            delay = int(rtc.get("initial_delay_policy_steps", 0))
            execute = int(rtc.get("min_execute_policy_steps", 0))
            beta = float(rtc.get("beta", 0.0))
            if delay <= 0 or not delay <= execute <= configured_horizon - delay:
                infra_errors.append(
                    "RTC requires delay <= min_execute_policy_steps <= action_horizon - delay"
                )
            if int(rtc.get("delay_buffer_size", 0)) <= 0:
                infra_errors.append("RTC delay_buffer_size must be positive")
            if not math.isfinite(beta) or beta <= 0:
                infra_errors.append("RTC beta must be finite and positive")
    if infra_errors:
        report["errors"].extend(infra_errors)
        report["status"] = "blocked"
    report.update(
        {
            "policy_name": "LingBot_VLA2",
            "checkpoint_variant": config.get("checkpoint_variant", "yam_finetuned"),
            "official_repository": "https://github.com/Robbyant/lingbot-vla-v2",
            "model_action_shape": [int(report["action_horizon"]), 14],
            "manimux_action_space": "absolute_joint_position",
            "rtc_capability": report.get("rtc_capability", "blocked"),
            "inference_status": "not_verified",
            "policy_status": (
                "base_checkpoint_capability_unvalidated"
                if "base_with_yam_stats" in str(config.get("checkpoint_variant"))
                else "requires_yam_posttraining_evidence"
            ),
            "norm_stats_role": config.get("norm_stats_role"),
        }
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--infra-config", type=Path, default=DEFAULT_INFRA_CONFIG)
    args = parser.parse_args()

    config = _load_config(args.config.resolve())
    infra_config = _load_config(args.infra_config.resolve())
    report = _validate(config, infra_config)
    report["infra_config_path"] = str(args.infra_config.resolve())
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
