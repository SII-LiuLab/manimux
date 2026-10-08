"""Reconstruct saved achieved joints; no commands or controller simulation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .rate_resampling import resample_joint_streams, sampling_strides
from .replay import JointReplay, _series, load_replay_cameras


def load_feedback_rate_replay(
    episode: Path | str, *, target_hz: float = 60.0,
    low_rates_hz: tuple[float, ...] = (10.0, 30.0), max_gap_s: float = 0.1,
) -> JointReplay:
    """Use each follower's own saved snapshot timestamps, independent of config Hz.

    Preserve raw observations for plots. The reference on the comparison grid is
    latest-at-or-before sampling, not interpolation or a fresh measured stream.
    """
    sampling_strides(target_hz, low_rates_hz)
    episode = Path(episode).expanduser().resolve()
    if not (episode / "write_complete.flag").is_file():
        raise ValueError("Episode is not finalized (missing write_complete.flag)")
    meta = json.loads((episode / "metadata.json").read_text())
    arms = meta.get("arm_names", [])
    if (meta.get("schema_version") != 2 or meta.get("num_arm_joints") != 6
            or not arms or not set(arms) <= {"left", "right"} or len(set(arms)) != len(arms)):
        raise ValueError("Achieved rate comparison requires a schema 2 YAM joint episode")
    raw, original, rates = {}, {}, {}
    for arm in arms:
        time_file = f"{arm}-feedback-timestamp-ns.npy"
        ts, q = _series(episode, f"{arm}-joint_pos.npy", time_file, 6)
        _, grip = _series(episode, f"{arm}-gripper_pos.npy", time_file, 1)
        if len(ts) != meta["num_frames"] or np.any((grip < 0) | (grip > 1)):
            raise ValueError(f"Invalid achieved count or gripper range: {arm}")
        original[arm] = (ts, np.column_stack([q, grip]))
        # Duplicate snapshot times do not represent an extra temporal sample.
        keep = np.r_[np.diff(ts) > 0, True]
        ts, values = ts[keep], original[arm][1][keep]
        if len(ts) < 2:
            raise ValueError(f"Need at least two distinct achieved timestamps: {arm}")
        raw[arm] = (ts, values)
        rates[arm] = (len(ts) - 1) * 1e9 / int(ts[-1] - ts[0])
    origin, time_s, reference, variants = resample_joint_streams(
        raw, target_hz=target_hz, low_rates_hz=low_rates_hz, max_gap_s=max_gap_s,
    )
    actual = " / ".join(f"{arm} {rate:.2f}" for arm, rate in rates.items())
    low_label = "、".join(f"{r:g}" for r in low_rates_hz)
    return JointReplay(
        episode=episode, source="feedback", origin_ns=origin, time_s=time_s,
        held={}, linear={}, native=reference, resampled=variants,
        target_hz=target_hz, low_hz=min(low_rates_hz),
        cameras=load_replay_cameras(episode, meta), source_rates_hz=rates,
        reference_label="原始 achieved", source_samples=original,
        resample_rates_hz=dict(zip(variants, low_rates_hz, strict=True)),
        source_note=(
            f"采集设置 {meta['control_hz']:g} Hz；从臂 achieved 实际保存平均 {actual} Hz。\n\n"
            f"蓝色：各臂按自己的反馈时间戳，取此前最近实测值到 {target_hz:g} Hz 比较网格；"
            f"橙／绿色：同一份 achieved 按时间取 {low_label} Hz 采样点，"
            f"再对六个关节线性插值到 {target_hz:g} Hz。不是每隔几个原始行抽一个点。"
            "原始采样点另保留在关节报告中；比较网格重复取值不算新测量。\n\n"
            "夹爪三组共用已记录的实测开合值，不降采样、不插值、不改写旧记录。"
            f"首尾裁到共有采样点；超过 {max_gap_s * 1000:g} ms 的源时间缺口不补造。"
            "使用离线未来点，没有额外低通或抗混叠滤波；误差衡量低频保存的重建损失。"
            "这些是实测轨迹及其重建，不是三次真机下发实验，也不是 MIT 控制精度。"
        ),
    )
