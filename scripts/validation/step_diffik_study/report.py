#!/usr/bin/env python3
# ruff: noqa: E501
"""Build compact task conclusions and detailed offline evidence reports."""

import html
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import run as s
import yaml

sys.path.append(str(s.OUT / "plot-deps"))
os.environ.setdefault("MPLCONFIGDIR", str(s.OUT / "matplotlib-cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text())


def esc(x):
    return html.escape(str(x))


def pct(x):
    return f"{(x - 1) * 100:+.1f}%"


def table(head, rows):
    return (
        "<table><thead><tr>"
        + "".join("<th>" + esc(x) + "</th>" for x in head)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join("<td>" + esc(x) + "</td>" for x in row) + "</tr>" for row in rows
        )
        + "</tbody></table>"
    )


def get_metrics(row, name="baseline", historical=False):
    p = s.OUT / "episodes" / row["id"]
    return (
        read(p / ("historical_metrics.json" if historical else "baseline_sim_metrics.json"))
        if name == "baseline"
        else read(p / "candidates" / name / "metrics.json")
    )


def aggregate(rows, name):
    """Equal episode averages, never pooled percentiles or a success estimate."""
    values = defaultdict(list)
    counts = defaultdict(int)
    for row in rows:
        m = get_metrics(row, name)
        for region in ["all", "handoff_100ms", "stable", "stable_success", "ik_hold"]:
            if region not in m:
                continue
            counts[region] += m[region]["ticks"]
            for layer, arms in m[region]["errors"].items():
                for arm, metrics in arms.items():
                    for metric, stat in metrics.items():
                        if stat:
                            for kind in ["rms", "p95", "max"]:
                                values[
                                    "/".join([region, "error", layer, arm, metric, kind])
                                ].append(stat[kind])
            for layer, arms in m[region]["motion"].items():
                for arm, metrics in arms.items():
                    for metric, stat in metrics.items():
                        if stat:
                            for kind in ["rms", "p95", "max"]:
                                values[
                                    "/".join([region, "motion", layer, arm, metric, kind])
                                ].append(stat[kind])
        for reason, n in m["counts"]["reasons"].items():
            counts["reason/" + reason] += n
    return dict(
        episodes=len(rows),
        counts=dict(counts),
        metrics={
            k: dict(equal_episode_mean=float(np.mean(v)), n_episodes=len(v))
            for k, v in values.items()
        },
    )


def plot_task(task, rows, selected):
    # The bottle has one development session; plate plot explicitly uses holdout.
    row = next((r for r in rows if r["split"] == "holdout" and r["status"] == "complete"), rows[0])
    d = s.load(row)
    model = s.make_model(d["c"])
    base = s.OUT / "episodes" / row["id"]
    names = ["baseline"] + list(selected)
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True, layout="constrained")
    t = (d["times"] - d["times"][0]) / 1e9
    for name in names:
        r = dict(
            np.load(
                base / "baseline.npz"
                if name == "baseline"
                else base / "candidates" / name / "trace.npz"
            )
        )
        poses = s.poses(model, r["command"])
        pos, rot = s.error(d["target"], poses)
        der = s.derivatives(poses, d["times"])
        for j, _g in enumerate(s.GROUPS):
            axes[0, j].plot(t, pos[:, j], label=name, lw=1)
            axes[1, j].plot(t, rot[:, j], label=name, lw=1)
            axes[2, j].plot(t, der["linear_speed"][:, j], label=name, lw=1)
    for j, g in enumerate(s.GROUPS):
        axes[0, j].set_title(g)
        for i, label in enumerate(
            [
                "Target to command TCP error (mm)",
                "Target to command rotation (deg)",
                "Command TCP speed (m/s)",
            ]
        ):
            axes[i, j].set_ylabel(label)
            axes[i, j].grid(alpha=0.2)
        axes[2, j].set_xlabel("Recorded tick time (s)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(task + " | " + row["id"] + " | " + row["split"], fontsize=10)
    path = s.OUT / f"{task}_trace.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return row["id"]


def matched_windows(rows, selection):
    output = []
    for row in rows:
        d = s.load(row)
        model = s.make_model(d["c"])
        dest = s.OUT / "episodes" / row["id"]
        b = dict(np.load(dest / "baseline.npz"))
        active = np.isfinite(d["target"][:, 0, 0])
        failed = np.flatnonzero(active & (b["reason"] != "ok"))
        mask = active.copy()
        if len(failed):
            mask[failed[0] :] = False
        entry = dict(
            id=row["id"],
            task=row["task"],
            split=row["split"],
            status=row["status"],
            ticks=int(mask.sum()),
            baseline_first_hold=int(failed[0]) if len(failed) else None,
            metrics={},
        )
        for name in ["baseline"] + list(selection[row["task"]]):
            r = b if name == "baseline" else dict(np.load(dest / "candidates" / name / "trace.npz"))
            poses = s.poses(model, r["command"])
            pos, rot = s.error(d["target"], poses)
            der = s.derivatives(poses, d["times"])
            entry["metrics"][name] = {
                g: dict(
                    position_mm=s.stats(pos[mask, j]),
                    rotation_deg=s.stats(rot[mask, j]),
                    **{k: s.stats(v[mask, j]) for k, v in der.items()},
                )
                for j, g in enumerate(s.GROUPS)
            }
        output.append(entry)
    s.save(s.OUT / "matched_pre_hold_metrics.json", output)
    return output


def main():
    rows = [r for r in s.discover() if r["included"]]
    selection = read(s.OUT / "frozen_selection.json")
    roles = read(s.OUT / "frozen_selection_roles.json")
    matched = matched_windows(rows, selection)
    summary = {}
    review = {
        "kind": "offline-review-only",
        "deployable": False,
        "constraints": [
            "No production config changed",
            "No hardware or model service used",
            "Fixed historical TCP targets/times and handoff_skip_steps=1",
            "All physical joint bounds, IK margins, J6/J7 checks, lag policy and historical command guard preserved",
        ],
        "tasks": {},
    }
    style = "body{font:15px system-ui;max-width:1250px;margin:36px auto;padding:0 24px;color:#222}h1{font-size:26px}h2{font-size:20px;margin-top:32px}table{border-collapse:collapse;width:100%;font-size:12px;margin:16px 0}th,td{padding:7px;border-bottom:1px solid #ddd;text-align:left}th{background:#f3f3f3}img{max-width:100%}details{margin:16px 0}pre{white-space:pre-wrap;background:#f5f5f5;padding:12px}a{color:#145e9e}.note{border-left:3px solid #777;padding:12px}"
    for task in ["pack_plate", "pack_bottle"]:
        task_rows = [r for r in rows if r["task"] == task]
        hold = [r for r in task_rows if r["split"] == "holdout"]
        rank = read(s.OUT / f"{task}_final_development_ranking.json")
        chosen = roles[task]["selected_by_protocol"]
        dev_rank = {x["candidate"]: x for x in rank}
        hold_rank = s.ranking(hold, ["baseline"] + list(selection[task])) if hold else []
        s.save(s.OUT / f"{task}_holdout_evaluation.json", hold_rank)
        groups = defaultdict(list)
        for row in task_rows:
            info = read(s.OUT / "episodes" / row["id"] / "identity.json")
            groups[(row["split"], info["stratum"], row["status"])].append(row)
        strata = []
        for (split, stratum, status), subset in groups.items():
            strata.append(
                dict(
                    split=split,
                    stratum=stratum,
                    status=status,
                    baseline=aggregate(subset, "baseline"),
                    candidates={name: aggregate(subset, name) for name in selection[task]},
                )
            )
        summary[task] = dict(
            eligible=len(task_rows),
            complete=sum(r["status"] == "complete" for r in task_rows),
            aborted=sum(r["status"] == "aborted" for r in task_rows),
            sessions=len({r["session"] for r in task_rows}),
            primary_candidate=chosen,
            development=dev_rank[chosen],
            holdout=hold_rank,
            strata=strata,
        )
        review["tasks"][task] = dict(
            primary_candidate=chosen,
            alternatives=selection[task],
            roles=roles[task],
            historical_strata=[
                dict(
                    split=x["split"],
                    stratum=x["stratum"],
                    status=x["status"],
                    episodes=x["baseline"]["episodes"],
                )
                for x in strata
            ],
            requires_step_code_change={
                name: values.get("study.objective") == "tcp"
                for name, values in selection[task].items()
            },
            holdout_cannot_be_used_for_selection=True,
        )
        if task == "pack_plate":
            conclusion = (
                "step DiffIK × pack_plate：8 条有效记录（4 完整、4 中止），3 个 session；不足 20 条。完整不表示成功，无人工成功标签。\n"
                "Baseline：8 条 reference/command 逐元素完全复现，QP hold 2,335 次、预算 hold 0 次。\n"
                "已证实瓶颈：右臂 TCP 目标到限速输入的积压；左臂 J4 在执行器加速度整形后越过 IK 内侧 -139°，下一 tick 速度盒无解，双臂持续 hold。其 command 仍未越过 -140° command guard；放宽安全阈值不能修复此问题。\n"
                "小改候选 mu_nullspace=5000：开发集平移/旋转 RMS 均约改善 1%，hold 1434→1338；留出集改善约 0.8%/1.1%，hold 901→901。通过预定筛选，但收益小、未解决卡死。\n"
                "TCP 目标 + acceleration=40 rad/s² + angular limiter=0.9 rad/s：开发集平移/旋转 RMS -22.4%/-23.4%，hold 1434→0；留出集 -30.5%/-31.6%，hold 901→0，但线性 jerk RMS 平均 +35.6%（最差臂/记录 +55.8%），未通过平顺性筛选。\n"
                "TCP 单因素在多数轨迹改善，但一条开发记录更早卡死，不能把目标函数改善等同于端到端改善。当前没有兼顾显著精度提升、平顺性和所有记录无卡死的已验证组合。\n"
                "全程 jerk 增幅包含 baseline 长 hold 的影响；固定双方首次 hold 前时窗后，留出集线性 jerk RMS 仍平均增加 12.2%（最差 +18.3%）。\n"
                "仍需真机验证：控制器 fault 13 的原因、控制器跟随和接触、真实周期 deadline、TCP 标定与任务成功。留出 session 是 v=0.7/a=25 的独立配置层，不能代表同配置重复验证。\n"
            )
        else:
            conclusion = (
                "step DiffIK × pack_bottle：3 条有效记录，全部中止且来自同一 session；另 1 条正常结束记录为 0 tick，排除。不足 20 条，没有独立留出集或成功证据。\n"
                "Baseline：3 条 reference/command 逐元素完全复现；QP/预算 hold 均 0。左臂在前两条记录有目标→限速输入与 DiffIK 速度盒造成的积压，executor arm shaping 为 0；三条均以 left controller fault 13 中止，离线未确定其物理原因。\n"
                "TCP 目标单因素：等记录/等臂平均平移 RMS -39.3%、旋转 RMS -19.8%、线性 jerk RMS -20.0%，无新增 hold。支持 TCP 优化目标的收益。\n"
                "探索候选 TCP 目标 + shared max_velocity=0.6 rad/s + angular limiter=0.9 rad/s；保留 linear=0.4 m/s、arm acceleration=null、dt cap=0.016 s、所有边界和 handoff_skip=1。\n"
                "候选平均平移 RMS -41.9%、旋转 RMS -49.1%、线性 jerk RMS -23.6%，无新增 QP hold；所有臂/记录通过预定精度和平顺性筛选，最大单项运动指标增幅约 9.0%。\n"
                "此结论只适用于固定目标/时间轴的命令生成。不能预测改参后的实测 state、fault 13 是否消失或任务成功；需要新 session、真实控制器/接触和 deadline 验证。\n"
            )
        (s.OUT / f"{task}_conclusion.txt").write_text(conclusion)
        doc = f'<!doctype html><html lang="zh"><meta charset="utf-8"><title>{task} step DiffIK review</title><style>{style}</style><body><h1>{task} · step DiffIK offline review</h1><pre>{esc(conclusion)}</pre>'
        doc += '<p class="note">所有改善百分比均为等记录、等臂的指标比值均值；不是合并采样点的误差，也不是成功率。只有 baseline 使用历史实测关节 state；TCP state 是关节 FK，非外部测得的末端位姿。图和导数使用历史 tick 时间，IK/executor 使用固定 10 ms。TCP CAD 近似不代表真实工具亚毫米精度。</p>'
        doc += "<h2>分层 baseline 与样本身份</h2>"
        baseline_rows = []
        for row in task_rows:
            m = get_metrics(row, historical=True)
            info = read(s.OUT / "episodes" / row["id"] / "identity.json")
            for g in s.GROUPS:
                e = m["all"]["errors"]

                def val(pair, e=e, g=g):
                    return f"{e[pair][g]['translation_mm']['rms']:.2f} / {e[pair][g]['rotation_deg']['rms']:.2f}"

                baseline_rows.append(
                    [
                        row["id"],
                        row["status"],
                        row["split"],
                        info["stratum"],
                        g,
                        val("target_to_limited"),
                        val("limited_to_reference"),
                        val("reference_to_command"),
                        val("target_to_command"),
                        val("command_to_state"),
                    ]
                )
        doc += table(
            [
                "Sample",
                "Status",
                "Split",
                "Frozen stratum",
                "Arm",
                "Target→limited mm/deg RMS",
                "Limited→ref",
                "Ref→command",
                "Target→command",
                "Command→state (pre-send)",
            ],
            baseline_rows,
        )
        doc += "<p>层间误差是不同向量的范数，不能相加。command→state 比较同 tick 发出前读取的关节 state，有采样时序差；previous-command→state 及 sent snapshot 有效性见 baseline_forensics.json。</p>"
        doc += "<h2>候选各层变化（按配置、完整/中止和 split 分开）</h2><p>每格为平均每记录 RMS：baseline→candidate 的 mm / deg。</p>"
        layer_rows = []
        for (split, stratum, status), subset in groups.items():
            for name in selection[task]:
                for g in s.GROUPS:
                    cells = []
                    for layer in [
                        "target_to_limited",
                        "limited_to_reference",
                        "reference_to_command",
                        "target_to_command",
                    ]:
                        b = []
                        a = []
                        for key in ["translation_mm", "rotation_deg"]:
                            b.append(
                                np.mean(
                                    [
                                        get_metrics(r)["all"]["errors"][layer][g][key]["rms"]
                                        for r in subset
                                    ]
                                )
                            )
                            a.append(
                                np.mean(
                                    [
                                        get_metrics(r, name)["all"]["errors"][layer][g][key]["rms"]
                                        for r in subset
                                    ]
                                )
                            )
                        cells.append(f"{b[0]:.2f}→{a[0]:.2f} / {b[1]:.2f}→{a[1]:.2f}")
                    layer_rows.append([split, stratum, status, name, g] + cells)
        doc += table(
            [
                "Split",
                "Config stratum",
                "Status",
                "Candidate",
                "Arm",
                "Target→limited",
                "Limited→reference",
                "Reference→command",
                "Target→command",
            ],
            layer_rows,
        )
        plot_row = plot_task(task, task_rows, selection[task])
        doc += f'<h2>固定历史输入上的 TCP 轨迹</h2><img src="{task}_trace.png" alt="TCP target error and command speed"><p>Source: {esc(plot_row)}. No candidate state is synthesized.</p>'
        doc += "<h2>开发集筛选与收益代价</h2>"
        rr = []
        for x in rank:
            q = x["mean_ratios"]
            rr.append(
                [
                    x["candidate"],
                    x["passes"],
                    x["holds"],
                    pct(q["position"]),
                    pct(q["rotation"]),
                    pct(q["jerk"]),
                    pct(q["angular_jerk"]),
                    pct(max(x["worst_ratios"].values())),
                ]
            )
        doc += table(
            [
                "Candidate",
                "Pass all constraints",
                "Hold ticks",
                "TCP pos RMS",
                "TCP rot RMS",
                "Linear jerk RMS",
                "Angular jerk RMS",
                "Worst metric / arm / episode",
            ],
            rr,
        )
        if hold_rank:
            doc += "<h2>冻结候选后的留出评估（无再选参）</h2>" + table(
                [
                    "Candidate",
                    "Pass",
                    "Hold ticks",
                    "Position RMS",
                    "Rotation RMS",
                    "Linear jerk RMS",
                ],
                [
                    [
                        x["candidate"],
                        x["passes"],
                        x["holds"],
                        pct(x["mean_ratios"]["position"]),
                        pct(x["mean_ratios"]["rotation"]),
                        pct(x["mean_ratios"]["jerk"]),
                    ]
                    for x in hold_rank
                ],
            )
        doc += "<h2>匹配 baseline 首次 hold 前的时窗</h2><p>全程 jerk 对比会受到 baseline 长时间静止影响。以下固定共同前缀的补充诊断不参与选参，也不取代全程结果。</p>"
        matched_rows = []
        for split in sorted({r["split"] for r in task_rows}):
            group = [r for r in matched if r["task"] == task and r["split"] == split]
            for name in selection[task]:
                ratios = []
                for key in ["position_mm", "rotation_deg", "linear_jerk", "angular_jerk"]:
                    ratios.append(
                        pct(
                            np.mean(
                                [
                                    r["metrics"][name][g][key]["rms"]
                                    / r["metrics"]["baseline"][g][key]["rms"]
                                    for r in group
                                    for g in s.GROUPS
                                ]
                            )
                        )
                    )
                matched_rows.append([split, name] + ratios)
        doc += table(
            [
                "Split",
                "Candidate",
                "Position RMS",
                "Rotation RMS",
                "Linear jerk RMS",
                "Angular jerk RMS",
            ],
            matched_rows,
        )
        doc += "<h2>参数有效范围与边界</h2><ul><li>shared max_velocity 单因素 0.3/0.6/0.9 rad/s；DiffIK 七关节分别使用同一标量上界，executor 同步使用该共享值。历史 guard 每关节 1.00531 rad/s 始终保留。</li><li>max_step_dt_s 8/10/16/32 ms；10、16、32 ms 在 100 Hz 下等价，8 ms 只收紧 executor，不改变 IK dt。</li><li>arm acceleration null/10/20/25/40 rad/s²；direct 的该项真实整形。smooth.cutoff_hz、smooth tracking mode 未启用。controller acceleration_ratio=100 属硬件行为，不等于离线滤波。</li><li>pose linear 0.2/0.4/0.8 m/s，angular 0.9/1.8/3.6 rad/s，另测试禁用 limiter。平移/旋转共用一个缩放比例。</li><li>lambda 0.001/0.01/0.1；mu_nullspace 0/1000/5000/50000。限位、J6/J7 和 lag_policy=report 不变；max_lag 是报告/拒绝语义，不是轨迹整形。</li><li>velocity_ratio 的 40→32 在固定 resolved runtime caps 下离线命令完全相同；真实控制器收益未知。command_margin 是 loader 的乘数，不是这批冻结配置里的独立运行时字段；见 rate_contract_audit.json，不能用当前 profile 反推历史。</li></ul>"
        doc += "<h2>逐样本、逐层、交接/稳定区间证据</h2><p>handoff_100ms 为首次有效目标或 plan 切换后 100 ms；stable 为其补集（含 hold），stable_success 排除失败 hold。ik_hold 单列。速度 m/s、rad/s；加速度 m/s²、rad/s²；jerk m/s³、rad/s³。导数为非均匀时间上的离散差分，无后处理平滑。</p>"
        for row in task_rows:
            prefix = "episodes/" + row["id"]
            m = get_metrics(row, historical=True)
            doc += f'<details><summary>{esc(row["id"])} · {row["status"]}</summary><p><a href="{prefix}/identity.json">Frozen config, source hashes, timeline audit, sample path</a> · <a href="{prefix}/baseline_parity.json">Baseline parity</a> · <a href="{prefix}/historical_metrics.json">All historical layers and motion statistics</a> · <a href="{prefix}/baseline.npz">Baseline tick arrays</a></p>'
            doc += "<pre>" + esc(json.dumps(m["counts"], indent=2)) + "</pre>"
            for name in selection[task]:
                doc += f'<p>{esc(name)}: <a href="{prefix}/candidates/{name}/metrics.json">all layers/regions/derivatives</a> · <a href="{prefix}/candidates/{name}/trace.npz">endogenous candidate seed/reference/command trace</a></p>'
            doc += "</details>"
        doc += "<h2>证据边界</h2><p>已证实：冻结配置、代码核心哈希匹配，11 条 baseline 的每 tick reference/command 与成败完全复现；延迟 trim、skip=1 且不延迟交接时刻均逐 plan 核验；QP 和预算失败的双臂 hold/reset 通过独立测试。仅有迹象：硬件跟随滞后与 fault 13 的关联。必须真机验证：控制周期 deadline、真实 state/接触、控制器 velocity_ratio/acceleration_ratio、工具标定与成功率。离线耗时受本机并行和调度影响，仅供参考。</p>"
        doc += "<p>配置层与 session 在此样本中混杂；plate 无同配置跨 session 的开发/留出重复，bottle 完全没有独立留出。没有用候选留出结果调参。历史中止尾部可能存在已求解但未成功 dispatch/record 的事件，它们列于 identity.json/events_without_recorded_tick，不伪造成状态样本。</p>"
        doc += "<p>历史 source manifest 未记录 TCP 资产和离线 FK 二进制的哈希；当前资产已独立归档。逐 tick 完全复现提供功能一致性证据，不能替代历史资产字节身份。只加载离线 FK 库，不创建硬件控制器。</p>"
        doc += '<p><a href="candidate_parameters_review.yaml">Candidate review</a> · <a href="cohort.json">Cohort and exclusions</a> · <a href="verification.json">Baseline and hold tests</a> · <a href="tcp_verification.json">TCP Jacobian/bound tests</a> · <a href="baseline_forensics.json">Failure causal windows</a> · <a href="matched_pre_hold_metrics.json">Matched windows</a> · <a href="selection_protocol.json">Selection protocol</a> · <a href="summary.json">Stratified summary</a></p></body></html>'
        (s.OUT / f"{task}_report.html").write_text(doc)
    s.save(s.OUT / "summary.json", summary)
    (s.OUT / "candidate_parameters_review.yaml").write_text(
        yaml.safe_dump(review, sort_keys=False, allow_unicode=True)
    )
    # Preserve the actual pure-code dependency surface and geometry for later replay.
    archive = s.OUT / "source_snapshot"
    hashes = {}
    for p in (s.CODE_ROOT / "manimux").rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        target = archive / p.relative_to(s.CODE_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        hashes[str(p.relative_to(s.CODE_ROOT))] = s.sha(p)
    for pattern in [
        "manimux/configs/embodiment/**/*.yaml",
        "manimux/embodiments/end_effector/taccap/assets/**/*.yaml",
        "manimux/embodiments/end_effector/taccap/assets/**/*.urdf",
    ]:
        for p in s.CODE_ROOT.glob(pattern):
            target = archive / p.relative_to(s.CODE_ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
            hashes[str(p.relative_to(s.CODE_ROOT))] = s.sha(p)
    # RobotModel uses the vendor's offline FK library; no controller is created.
    for name in ["libKine.so", "ccs_m6_40.MvKDCfg", "LICENSE"]:
        p = s.CODE_ROOT / "manimux/embodiments/arm/tianji/sdk/marvin" / name
        target = archive / p.relative_to(s.CODE_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        hashes[str(p.relative_to(s.CODE_ROOT))] = s.sha(p)
    s.save(
        s.OUT / "asset_provenance_limit.json",
        dict(
            note="The historical selected-source manifest does not hash the TCP YAML/URDF or native FK library/config. Current assets are archived and hashed; exact full baseline replay supports functional parity, not independent historical byte identity.",
            offline_fk_library_loaded=True,
            controller_created=False,
            assets={
                k: v
                for k, v in hashes.items()
                if "/assets/" in k or k.endswith((".so", ".MvKDCfg"))
            },
        ),
    )
    for p in Path(__file__).parent.iterdir():
        if p.suffix not in {".py", ".md"}:
            continue
        target = s.OUT / "script_snapshot" / p.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    s.save(s.OUT / "source_snapshot_sha256.json", hashes)
    s.save(
        s.OUT / "script_identity.json",
        {str(p.relative_to(s.ROOT)): s.sha(p) for p in Path(__file__).parent.glob("*.py")},
    )
    print("Reports, conclusions, candidate review and source snapshots written.")


if __name__ == "__main__":
    main()
