"""Chinese presentation of reports; preserve evidence JSON, code and file paths."""

# Translation keys retain the exact source sentences.
# ruff: noqa: E501
from __future__ import annotations

import html
import re
from html.parser import HTMLParser

TEXT = {
    "Execution-qualified seam details": "实际执行交接与跳变明细",
    "end_hold": "保持后接手",
    "continuous": "连续交接",
    "Continuous": "连续交接",
    "End hold": "保持后接手",
    "Left reference seam / mm": "左臂参考轨迹跳变 / mm",
    "Right reference seam / mm": "右臂参考轨迹跳变 / mm",
    "Seams are qualified from executed tick ownership and lifecycle events. PRM result import remains pending.": "跳变已按实际执行的 plan 与暂停、重启事件筛选。PRM 暂缓。",
    "Execution-qualified windows use their own valid handoffs at each N, with equal rollout weighting. Counts can differ across windows. Without execution evidence, attached candidate diagnostics remain labeled as such.": "按实际执行交接筛选。10、20、30 步各自统计有完整覆盖的边界，再按 rollout 等权平均；各窗口样本数可以不同。无执行证据时只保留候选计划诊断。",
    "Pause, reset, missing identity and long gaps split segments. Verified expired-plan holds use the recorded constant command because scheduled can contain measured-state placeholders. Segment metrics are duration-weighted within each channel; episodes remain equally weighted. All exclusions are exported.": "暂停、重启、plan ID 缺失及长时间断点处切段。计划耗尽后，核实为保持的区间使用记录下来的恒定 command，避免 scheduled 中实测 state 占位值引入假变化。段内计算，按有效段时长加权，再按通道、rollout 等权汇总。明细保留全部排除原因。",
    "Continuous: old committed reference versus incoming first point at the incoming start time. End hold: verified held command versus incoming first point, reported separately. Mean/P95 use equal rollout weighting; Max is the global maximum. n is rollout count; b is boundary count.": "连续交接：同一切换时刻的旧参考值与新计划首点之差。保持后接手：记录且核实的保持 command 与新计划首点之差，两类分开列出。Mean/P95 按 rollout 等权，Max 为全局最大值；n 为 rollout 数，b 为交接数。",
    "Missing completion annotation": "缺少完成时刻标注",
    "Human / M25–75 / P-SR: %; seams: mm. — means unmeasured. Historical data is listed separately.": "H-Score、H-SR、M25–75、P-SR 单位为 %；接缝单位为 mm。— 表示未测，历史数据单列。",
    "Time: manually labeled seconds from Start to task completion for successful attempts. Smoothness: TCP translation/rotation from FK of the scheduled reference, not measured state; average equally across valid active channels, then rollouts. Do not combine different active-channel sets.": "完成时间需人工标注 Start 到任务成功的秒数。平滑性取 scheduled 参考轨迹的 TCP 平移与转动，先对有效通道等权平均，再对试验等权平均；不同通道集合不混算。",
    "Candidate-plan diagnostics; windows use shared valid handoffs with equal rollout weighting. N denotes model action steps; the table retains valid sample counts.": "候选轨迹诊断：10/20/30 步使用共同有效的切换点，试验间等权平均。N 为模型动作步数；表中保留有效样本数。",
    "ln(T³ ∫‖jerk‖²dt / peak_speed²), excluding differentiation stencils across plans; ": "ln(T³ ∫‖jerk‖²dt / peak_speed²)，排除跨计划的差分窗口；",
    "Positive spectral arc length of speed magnitude. Both are dimensionless and use standard speed normalization T³, unlike the T⁵ printed in the Legato appendix; do not directly compare paper values. Windows include all RUNNING ticks and normal pauses, without interpolating long gaps. Details retain sample rates, spectral cutoffs and missing-channel reasons.": "速度幅值的正谱弧长。两项均无量纲，使用标准的 T³ 归一化，与 Legato 附录的 T⁵ 不同，不直接比较论文数值。保留运行窗口内的正常停顿，不跨长缺口插值；采样率、截止频率和缺失原因见明细。",
    "TCP position differences between old and new plans at the same time. Mean/P95 use equal rollout weighting; Max is the global maximum.": "新旧计划在同一时刻的 TCP 位置差。Mean/P95 对各试验等权平均；Max 取全局最大值。",
    "Position RMSE, origin-aligned shape RMSE, endpoint error and velocity-direction cosine over the same future window.": "在相同未来窗口内计算位置 RMSE、起点对齐后的形状 RMSE、终点误差、速度方向余弦。",
    "Fraction of episodes with maximum predicted progress reaching 25% / 50% / 75%.": "预测最大进度达到 25% / 50% / 75% 的试验比例。",
    "PRM success rate from a progress threshold or specified labels, reported separately from Human SR.": "按进度阈值或指定标签得到的 PRM 成功率，与人工成功率分开报告。",
    "Mean of per-episode maximum predicted progress.": "各试验最大预测进度的平均值。",
    "Progress path efficiency: MP² / Σ|Δprogress|.": "进度路径效率：MP² / Σ|Δprogress|。",
    "Mean regression below the running maximum progress, normalized by MP.": "相对历史最大进度的平均回退量，以 MP 归一化。",
    "Fraction of adjacent progress changes below the threshold.": "相邻进度变化低于阈值的比例。",
    "Recovery after the largest regression, using only episodes with regression.": "最大回退后的恢复程度，仅统计发生回退的试验。",
    "Proximity to success for failed episodes.": "失败试验接近成功的程度。",
    "Overall quality of successful episodes, combining efficiency, regression and stagnation.": "成功试验的综合质量，结合效率、回退和停滞。",
    "Formal seam filtering and PRM result import are not yet integrated.": "正式接缝筛选与 PRM 结果导入尚未接通。",
    "Formal seams: actual execution-boundary filtering is not integrated; PRM: matching profile/case import is not integrated.": "正式接缝尚未完成实际执行边界筛选；PRM 尚未导入匹配的评估结果。",
    "Candidate reference seam: blue is the old plan, red is the new plan, and black marks the switch time.": "候选参考轨迹接缝：蓝色为旧计划，红色为新计划，黑线为切换时刻。",
    "Legacy records with unknown layout/repeat IDs are not assigned to slots.": "缺少布局或重复编号的历史记录不分配试验槽位。",
    "Slots indicate recorded attempts, not success or acceptance.": "槽位表示已录制尝试，不代表成功或验收通过。",
    "Points are rollouts; vertical lines are means.": "圆点为各次试验，竖线为均值。",
    "No completion-time or smoothness values available.": "暂无可用的完成时间或平滑性数值。",
    "No matching seam plots available.": "暂无匹配的接缝图。",
    "No trajectory metrics available.": "暂无轨迹指标。",
    "No similarity plots available.": "暂无相似度图。",
    "No evaluation records available.": "暂无评估记录。",
    "Task registered; model/method settings still need to be bound.": "任务已登记，模型与方法配置待绑定。",
    "First front frame with the demonstration manipulation scope.": "首帧叠加示范数据的操作范围。",
    "Mean completion score (bottles: mean(k/6)).": "人工平均完成分数（瓶子任务为 mean(k/6)）。",
    "Success rate; ": "人工成功率；",
    "Similarity over 10 / 20 / 30 steps after handoff": "切换后 10 / 20 / 30 步的轨迹相似度",
    "Completion time and smoothness": "完成时间与平滑性",
    "Completion-time / smoothness details": "完成时间与平滑性明细",
    "Successful completion time / s": "成功完成时间 / 秒",
    "Position deviation between old and new plans": "新旧计划的位置差异",
    "Motion-direction agreement between old and new plans": "新旧计划的运动方向一致性",
    "Velocity cosine / dimensionless": "速度方向余弦 / 无量纲",
    "Window N / model action steps": "窗口 N / 模型动作步数",
    "Sources, sample details and downloads": "来源、样本明细与下载",
    "Sources and downloads": "来源与下载",
    "Filter by setting, path or status": "按配置、路径或状态筛选",
    "Evaluation report": "评估报告",
    "Experiment results": "实验结果",
    "Export data": "导出数据",
    "Print / PDF": "打印 / PDF",
    "Report contents": "报告目录",
    "Seam plots": "接缝诊断图",
    "Manipulation coverage": "夹爪开合位置与轨迹覆盖",
    "Trajectory similarity": "轨迹相似度",
    "Metric definitions": "指标定义",
    "Model / method": "模型 / 方法",
    "N steps": "N 步",
    "Position RMSE": "位置 RMSE",
    "Shape RMSE": "形状 RMSE",
    "Endpoint error": "终点误差",
    "Velocity cosine": "速度方向余弦",
    "N rollout": "有效试验数",
    "Position / cosine": "位置 / 方向",
    "Handoffs": "切换点数",
    "Left arm": "左臂",
    "Right arm": "右臂",
    "Method": "方法",
    "Arm": "机械臂",
    "Metric": "指标",
    "Definition": "定义",
    "Output directory": "输出目录",
    "Summary JSON": "汇总 JSON",
    "Per-episode JSON": "逐次试验 JSON",
    "Cohort": "样本清单",
    "Provenance / SHA-256": "来源 / SHA-256",
    "Task configuration": "任务配置",
    "Samples and layouts": "样本与布局",
    "Rollout details": "逐次试验明细",
    "Filter rollouts": "筛选试验",
    "Matching fields:": "身份核对字段：",
    "Trajectory source:": "轨迹来源：",
    "Not provided": "未提供",
    "Download SVG": "下载 SVG",
    "Live": "现场评分",
    "Video": "录像评分",
    "Human:": "人工结果：",
    "overlap:": "重叠诊断：",
    "Layout": "布局",
    "Repeat": "重复",
    "repeat": "重复",
    "attempt ID:": "试验 ID：",
    "attempts": "次尝试",
    "Recorded": "已录制",
    "target": "目标",
    "Known layout slots": "已覆盖槽位",
    "Duplicate slots": "重复槽位",
    "Identity mismatches": "身份不匹配",
    "Metadata processing errors": "元数据处理错误",
    "Available trajectory artifacts": "可用轨迹产物",
    "episodes.": "条试验。",
    "Not registered": "未登记",
    "Unspecified": "未指定",
    "Partial results": "部分结果",
    "Pending evaluation": "待评估",
    "missing_tick_plan_identity": "运行中缺少计划标识",
    "completion_time_not_recorded": "未记录完成时间",
}
WORDS = {
    "success": "成功",
    "failure": "失败",
    "invalid": "无效",
    "available": "可用",
    "missing": "缺失",
    "excluded": "已排除",
    "error": "错误",
    "unreviewed": "未评分",
}


def translate(text: str) -> str:
    for source in sorted(TEXT, key=len, reverse=True):
        if source.isascii() and source.replace("_", "").isalpha():
            text = re.sub(r"(?<![\w/])" + re.escape(source) + r"(?![\w/])", TEXT[source], text)
        else:
            text = text.replace(source, TEXT[source])
    for source, target in WORDS.items():
        text = re.sub(r"(?<![\w/])" + source + r"(?![\w/])", target, text)
    return text


def localize_html(document: str) -> str:
    """Translate presentation only; scripts, JSON, code, URLs and identifiers stay intact."""

    class ChineseHTML(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=False)
            self.parts = []
            self.protected = []

        def handle_starttag(self, tag, attrs):
            raw = self.get_starttag_text()
            if tag == "html":
                raw = re.sub(r'lang="[^"]*"', 'lang="zh-CN"', raw)
            if not self.protected:
                for name, value in attrs:
                    if value is not None and name in ("title", "placeholder", "aria-label", "alt"):
                        raw = raw.replace(
                            html.escape(value, quote=True),
                            html.escape(translate(value), quote=True),
                        )
            self.parts.append(raw)
            if tag in ("script", "style", "pre", "code"):
                self.protected.append(tag)

        def handle_startendtag(self, tag, attrs):
            self.handle_starttag(tag, attrs)

        def handle_endtag(self, tag):
            self.parts.append(f"</{tag}>")
            if self.protected and self.protected[-1] == tag:
                self.protected.pop()

        def handle_data(self, data):
            self.parts.append(data if self.protected else translate(data))

        def handle_entityref(self, name):
            self.parts.append("&" + name + ";")

        def handle_charref(self, name):
            self.parts.append("&#" + name + ";")

        def handle_decl(self, decl):
            self.parts.append("<!" + decl + ">")

        def handle_comment(self, data):
            self.parts.append("<!--" + data + "-->")

    parser = ChineseHTML()
    parser.feed(document)
    return "".join(parser.parts)
