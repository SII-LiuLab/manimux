"""Offline comparison of submitted targets at configurable lower sample rates.

No hardware execution: interpolation uses future saved knots. Grippers retain
the reference's event timing and never participate in subsampling/interpolation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .rate_resampling import resample_joint_streams, sampling_strides
from .replay import JointReplay, _series, load_replay_cameras


def load_command_rate_replay(
    episode: Path | str, *, target_hz: float = 60.0,
    low_rates_hz: tuple[float, ...] = (10.0, 30.0), max_gap_s: float = 0.1,
) -> JointReplay:
    """Sample saved commands onto one grid, then keep every k-th arm target.

    The reference is causal hold on a regular target_hz grid, not an invented
    stream of new commands. Lower rates must divide that grid exactly. Endpoints
    are shared knots across *all* rates; no extrapolation or angle adjustment.
    Source timestamps define sampling, independently of the configured collection
    frequency. Gaps exceeding max_gap_s remain hidden in every method.
    """
    sampling_strides(target_hz, low_rates_hz)
    episode = Path(episode).expanduser().resolve()
    if not (episode / "write_complete.flag").is_file():
        raise ValueError("Episode is not finalized (missing write_complete.flag)")
    meta = json.loads((episode / "metadata.json").read_text())
    arms = meta.get("arm_names", [])
    if (meta.get("schema_version") != 2 or meta.get("num_arm_joints") != 6
            or not arms or not set(arms) <= {"left", "right"} or len(set(arms)) != len(arms)):
        raise ValueError("Command rate comparison requires a schema 2 YAM joint episode")
    if not math.isfinite(float(meta["control_hz"])) or float(meta["control_hz"]) <= 0:
        raise ValueError("Configured source collection frequency must be finite and positive")
    trace_lines = (episode / "manimux-control.jsonl").read_text().splitlines()
    trace = [json.loads(line) for line in trace_lines if line.strip()]
    if len(trace) != meta["num_frames"]:
        raise ValueError("Command trace count does not match recorded control samples")
    raw = {}
    source_rates = {}
    for arm in arms:
        ts, arm_q = _series(
            episode, f"controller-{arm}-joint.npy", f"controller-{arm}-timestamp-ns.npy", 6,
        )
        if np.any(np.diff(ts) <= 0) or len(ts) != len(trace):
            raise ValueError(f"Command timestamps must increase strictly and match trace: {arm}")
        q = np.asarray([r["command"][f"{arm}_arm"] for r in trace], dtype=float)
        if (q.shape != (len(ts), 7) or not np.isfinite(q).all()
                or np.any((q[:, 6] < 0) | (q[:, 6] > 1))):
            raise ValueError(f"Invalid submitted command or gripper in trace: {arm}")
        if (not np.array_equal(ts, np.asarray([r["unix_ns"] for r in trace], dtype=np.int64))
                or not np.allclose(q[:, :6], arm_q, atol=1e-6, rtol=0)):
            raise ValueError(f"Controller arrays and command trace disagree: {arm}")
        raw[arm] = (ts, q)
        source_rates[arm] = (len(ts) - 1) * 1e9 / int(ts[-1] - ts[0])
    origin, time_s, reference, variants = resample_joint_streams(
        raw, target_hz=target_hz, low_rates_hz=low_rates_hz, max_gap_s=max_gap_s,
    )
    camera_data = load_replay_cameras(episode, meta)
    low_label = "、".join(f"{r:g}" for r in low_rates_hz)
    actual_label = " / ".join(f"{arm} {hz:.2f}" for arm, hz in source_rates.items())
    return JointReplay(
        episode=episode, source="command", origin_ns=origin, time_s=time_s,
        held={}, linear={}, cameras=camera_data, source_rates_hz=source_rates,
        target_hz=target_hz, low_hz=min(low_rates_hz),
        native=reference,
        source_samples=raw,
        reference_label="已录 command",
        resampled=variants,
        resample_rates_hz=dict(zip(variants, low_rates_hz, strict=True)),
        source_note=(
            f"采集设置 {meta['control_hz']:g} Hz；已保存 command 实际平均 {actual_label} Hz。\n\n"
            f"基准按真实提交时间取最近已下发 command 到 {target_hz:g} Hz 网格；"
            f"两类数据的频率不能混为一谈。虚影取同一基准的 {low_label} Hz 子集，"
            f"再线性插值到 {target_hz:g} Hz。夹爪共用原 command 开合时刻，不降采样、不插值。\n\n"
            "首尾裁到所有网格的共有采样点，保证端点相同；"
            f"超过 {max_gap_s * 1000:g} ms 的缺口不补造。"
            "插值使用已录未来点，没有另加低通。这里只显示目标姿态，不预测从臂反馈或接触效果；"
            "在线执行还需要定义未来目标的可用时间及延迟。"
        ),
    )
