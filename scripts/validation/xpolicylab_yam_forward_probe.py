#!/usr/bin/env python3
"""Run one hardware-free XPolicyLab inference through the ManiMux YAM adapter."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import yaml

from manimux.config import load_config
from manimux.policies import build_policy_adapter, build_policy_model
from manimux.policies.base import prepare_policy_request
from manimux.runtime.aac import AacInferenceRequest
from manimux.runtime.autohorizon import AutoHorizonInferenceRequest
from manimux.runtime.dvac import DvacInferenceRequest
from manimux.runtime.paint import PaintInferenceRequest
from manimux.types import (
    ActionContext,
    InferenceRequest,
    ObservationSnapshot,
    RobotState,
    SensorFrame,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _repo_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _start_joints(path: Path, expected_dim: int) -> np.ndarray:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    values = np.asarray(payload["agent"]["start_joints"], dtype=np.float64)
    if values.shape != (expected_dim,) or not np.isfinite(values).all():
        raise ValueError(f"{path} agent.start_joints must contain {expected_dim} finite values")
    return values


def _synthetic_rgb(height: int, width: int, offset: int) -> np.ndarray:
    horizontal = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    vertical = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    red = np.broadcast_to((horizontal + offset).astype(np.uint8), (height, width))
    green = np.broadcast_to((vertical + offset).astype(np.uint8), (height, width))
    blue = ((red.astype(np.uint16) + green.astype(np.uint16)) // 2).astype(np.uint8)
    return np.ascontiguousarray(np.stack([red, green, blue], axis=-1))


def _snapshot(config_path: Path, height: int, width: int) -> ObservationSnapshot:
    config = load_config(config_path)
    group_order = list(config.policy.options["group_order"])
    if group_order != ["left_arm", "right_arm"]:
        raise ValueError(f"YAM probe requires left_arm/right_arm, got {group_order}")
    robot_configs = {
        "left_arm": _repo_path(config.robot.config),
        "right_arm": _repo_path(config.robot.options["right_config"]),
    }
    configured_start = config.robot.options.get("start_joints")
    if configured_start is None:
        groups = {
            name: _start_joints(robot_configs[name], int(config.robot.group_dims[name]))
            for name in group_order
        }
    else:
        packed_start = np.asarray(configured_start, dtype=np.float64)
        expected_dim = sum(int(config.robot.group_dims[name]) for name in group_order)
        if packed_start.shape != (expected_dim,) or not np.isfinite(packed_start).all():
            raise ValueError(
                f"robot.options.start_joints must contain {expected_dim} finite values"
            )
        groups = {}
        offset = 0
        for name in group_order:
            width = int(config.robot.group_dims[name])
            groups[name] = packed_start[offset : offset + width]
            offset += width

    camera_map = config.policy.options["camera_map"]
    camera_names = list(dict.fromkeys(str(name) for name in camera_map.values()))
    if len(camera_names) != 3:
        raise ValueError(f"YAM probe requires three camera roles, got {camera_names}")
    monotonic_ns = time.monotonic_ns()
    frames = {
        name: SensorFrame(
            name=name,
            data=_synthetic_rgb(height, width, 17 + 56 * index),
            capture_monotonic_ns=monotonic_ns,
            sequence=1,
        )
        for index, name in enumerate(camera_names)
    }
    return ObservationSnapshot(
        state=RobotState(groups=groups, monotonic_ns=monotonic_ns, sequence=1),
        frames=frames,
    )


def _native_action_summary(raw: object) -> dict[str, object]:
    if isinstance(raw, Mapping) and "actions" in raw:
        raw = raw["actions"]
    if isinstance(raw, Sequence) and raw and all(isinstance(step, Mapping) for step in raw):
        rows = [
            np.concatenate([np.asarray(value).reshape(-1) for value in step.values()])
            for step in raw
        ]
        actions = np.stack(rows)
        native_format = "action_step_dicts"
    else:
        try:
            actions = np.asarray(raw)
        except (TypeError, ValueError):
            return {"native_shape": None, "native_format": "unknown"}
        if actions.dtype == object:
            return {"native_shape": None, "native_format": "unknown"}
        native_format = "packed_actions"
    return {
        "native_shape": list(actions.shape),
        "native_finite": bool(np.isfinite(actions).all()),
        "native_format": native_format,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--instruction", default="Pick the red block up.")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()
    if args.height <= 0 or args.width <= 0:
        raise ValueError("height and width must be positive")

    config_path = args.config.resolve()
    config = load_config(config_path)
    if config.policy.worker != "xpolicylab_ws":
        raise ValueError("forward probe requires policy.worker: xpolicylab_ws")
    snapshot = _snapshot(config_path, args.height, args.width)
    session_id = f"xpolicy-probe-{uuid.uuid4().hex[:8]}"
    observation_time_ns = time.monotonic_ns()
    request_fields = {
        "session_id": session_id,
        "request_seq": 1,
        "observation_time_ns": observation_time_ns,
        "deadline_ns": observation_time_ns
        + int(config.policy.timeout_s * 1_000_000_000),
        "observation": snapshot,
        "instruction": args.instruction,
    }
    if config.execution.runtime == "aac":
        aac = config.execution.aac
        request = AacInferenceRequest(
            **request_fields,
            aac_num_samples=aac.num_samples,
            aac_motion_threshold=aac.motion_threshold,
            aac_ee_stats_path=aac.ee_stats_path,
            aac_chunk_id_selector=aac.chunk_id_selector,
            aac_backward_beta=aac.backward_beta,
        )
    elif config.execution.runtime == "paint":
        paint = config.execution.paint
        group_order = list(config.policy.options["group_order"])
        packed_state = np.concatenate(
            [snapshot.state.groups[name] for name in group_order]
        )
        prefix = np.repeat(
            packed_state[None, :],
            paint.initial_delay_steps,
            axis=0,
        )
        request = PaintInferenceRequest(
            **request_fields,
            paint_action_prefix=prefix,
            paint_delay_steps=paint.initial_delay_steps,
            paint_execution_steps=paint.execution_steps,
        )
    elif config.execution.runtime == "autohorizon":
        request = AutoHorizonInferenceRequest(**request_fields)
    elif config.execution.runtime == "dvac":
        dvac = config.execution.dvac
        request = DvacInferenceRequest(
            **request_fields,
            dvac_tail_steps=dvac.tail_steps,
            dvac_alpha=dvac.alpha,
            dvac_rolling_window_size=dvac.rolling_window_size,
            dvac_min_execution_steps=dvac.min_execution_steps,
            dvac_max_execution_steps=(
                dvac.max_execution_steps or config.policy.horizon_steps
            ),
        )
    else:
        request = InferenceRequest(**request_fields)

    model = build_policy_model(config.policy)
    adapter = build_policy_adapter(config.robot, config.policy)
    request = prepare_policy_request(adapter, request)
    started = time.perf_counter()
    try:
        model.reset(session_id)
        raw = model.infer(request)
    finally:
        model.close()
    round_trip_ms = (time.perf_counter() - started) * 1000.0
    native_summary = _native_action_summary(raw)
    if native_summary.get("native_finite") is False:
        raise ValueError("model returned non-finite native actions")

    chunk = adapter.decode_action(
        raw,
        ActionContext(
            request_seq=1,
            observation_time_ns=observation_time_ns,
            created_time_ns=time.monotonic_ns(),
        ),
    )
    group_order = list(config.policy.options["group_order"])
    packed = np.concatenate([chunk.groups[name] for name in group_order], axis=1)
    expected_width = sum(int(config.robot.group_dims[name]) for name in group_order)
    if config.execution.runtime == "aac":
        if packed.ndim != 2 or packed.shape[1] != expected_width:
            raise ValueError(
                f"expected an AAC chunk with width {expected_width}, got {packed.shape}"
            )
        if not 2 <= packed.shape[0] <= config.policy.horizon_steps:
            raise ValueError(
                "expected AAC selected horizon in "
                f"[2, {config.policy.horizon_steps}], got {packed.shape[0]}"
            )
    else:
        expected_shape = (config.policy.horizon_steps, expected_width)
        if packed.shape != expected_shape:
            raise ValueError(f"expected a {expected_shape} chunk, got {packed.shape}")
    if not np.isfinite(packed).all():
        raise ValueError("ManiMux adapter returned non-finite joint targets")

    print(
        json.dumps(
            {
                "status": "ok",
                "config": str(config_path),
                "server": config.policy.options["server"],
                "action_codec": config.policy.options.get("action_codec", "joint_position"),
                **native_summary,
                "action_space": chunk.action_space,
                "canonical_shape": list(packed.shape),
                "dt_s": chunk.dt_ns / 1_000_000_000,
                "round_trip_ms": round(round_trip_ms, 1),
                "minimum": float(packed.min()),
                "maximum": float(packed.max()),
                "first_action": packed[0].tolist(),
                "aac": raw.get("aac") if isinstance(raw, Mapping) else None,
                "paint": raw.get("paint") if isinstance(raw, Mapping) else None,
                "autohorizon": raw.get("autohorizon") if isinstance(raw, Mapping) else None,
                "dvac": raw.get("dvac") if isinstance(raw, Mapping) else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
