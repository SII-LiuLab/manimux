"use strict";
const DATA = JSON.parse(document.getElementById("report-data").textContent);
const $ = (id) => document.getElementById(id),
  NS = "http://www.w3.org/2000/svg";
const NAMES = {
  scheduled: "计划参考 · scheduled",
  command: "执行器命令 · command",
  state: "实测反馈 FK · state",
  saved: "保存的解码轨迹",
};
const COLORS = {
  scheduled: "--roll",
  command: "--ds",
  state: "--ref",
  saved: "--roll",
};
const ARM = { left_arm: "左臂", right_arm: "右臂" },
  SOURCES = ["scheduled", "command", "state"];
const state = {
  arm: "left_arm",
  scope: "all",
  threshold: 5,
  units: "relative",
  ep: 0,
  window: 10,
  start: 0,
  focus: null,
};
const css = (n) =>
  getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const num = (v, n = 2) =>
  v == null || !Number.isFinite(v) ? "—" : v.toFixed(n);
const quant = (a, q) => {
  const b = a.filter(Number.isFinite).sort((x, y) => x - y);
  if (!b.length) return null;
  const x = (b.length - 1) * q,
    i = Math.floor(x);
  return b[i] + (b[Math.min(i + 1, b.length - 1)] - b[i]) * (x - i);
};
const median = (a) => quant(a, 0.5),
  clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const esc = (s) =>
  String(s).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
function S(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
function text(svg, x, y, label, attrs = {}) {
  const e = S(
    "text",
    {
      x,
      y,
      fill: css("--muted"),
      "font-size": 11,
      "font-family": "system-ui,sans-serif",
      ...attrs,
    },
    svg,
  );
  e.textContent = label;
  return e;
}
function eligiblePlans(scope = state.scope) {
  return DATA.episodes
    .filter((e) => scope === "all" || e.id === scope)
    .flatMap((e) => e.plans)
    .filter((p) => p.execution?.eligible);
}
function rows(source, trim = "0", plans = eligiblePlans()) {
  return plans
    .map((p) => ({
      p,
      s:
        trim === "0"
          ? p.arms[state.arm][source].stats.tcp
          : p.arms[state.arm][source].sensitivity[trim],
    }))
    .filter((r) => r.s && r.s.mean >= state.threshold && r.s.ratio != null);
}
function agg(rs) {
  if (!rs.length) return { n: 0 };
  return {
    n: rs.length,
    ratio: median(rs.map((r) => r.s.ratio)),
    slow: rs.filter((r) => r.s.ratio <= 0.8).length,
    fast: rs.filter((r) => r.s.ratio >= 1.2).length,
    early: median(rs.map((r) => r.s.early)),
    late: median(rs.map((r) => r.s.late)),
  };
}
function profile(rs) {
  return Array.from({ length: 12 }, (_, i) => {
    const v = rs.map(
      (r) => r.s.profile[i] / (state.units === "relative" ? r.s.mean : 1),
    );
    return {
      x: ((i + 0.5) / 12) * 100,
      y: median(v),
      lo: quant(v, 0.25),
      hi: quant(v, 0.75),
    };
  });
}
function legend(target, items) {
  $(target).innerHTML = items
    .map(
      ([label, color, dash]) =>
        `<span class="it"><i class="key-line ${dash ? "dash" : ""}" style="color:${color}"></i>${esc(label)}</span>`,
    )
    .join("");
}
function axes(
  id,
  {
    height = 320,
    xmin = 0,
    xmax = 100,
    ymin = 0,
    ymax = 2,
    xlabel = "执行 chunk 内进度 (%)",
    ylabel = "相对本 chunk 平均速度 (×)",
    xticks = 5,
  },
) {
  const box = $(id);
  box.replaceChildren();
  const W = 1050,
    H = height,
    L = 70,
    R = 25,
    T = 28,
    B = 48,
    w = W - L - R,
    h = H - T - B;
  const svg = S(
    "svg",
    { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": ylabel },
    box,
  );
  S("rect", { x: 0, y: 0, width: W, height: H, fill: css("--surface") }, svg);
  const X = (x) => L + ((x - xmin) / (xmax - xmin)) * w,
    Y = (y) => T + h - ((y - ymin) / (ymax - ymin)) * h;
  for (let i = 0; i <= 5; i++) {
    const v = ymin + ((ymax - ymin) * i) / 5;
    S(
      "line",
      { x1: L, x2: L + w, y1: Y(v), y2: Y(v), stroke: css("--grid") },
      svg,
    );
    text(svg, L - 10, Y(v) + 4, num(v, ymax > 20 ? 0 : 1), {
      "text-anchor": "end",
    });
  }
  for (let i = 0; i <= xticks; i++) {
    const v = xmin + ((xmax - xmin) * i) / xticks;
    text(svg, X(v), H - 26, num(v, xmax - xmin <= 2 ? 2 : xmax > 100 ? 0 : 1), {
      "text-anchor": "middle",
    });
  }
  text(svg, L, 16, ylabel);
  text(svg, L + w / 2, H - 5, xlabel, { "text-anchor": "middle" });
  const clip = S("clipPath", { id: "clip-" + id }, S("defs", {}, svg));
  S("rect", { x: L, y: T, width: w, height: h }, clip);
  const layer = S("g", { "clip-path": `url(#clip-${id})` }, svg);
  return { svg, layer, X, Y, L, T, w, h, W, H, xmin, xmax, ymax };
}
function line(chart, points, color, width = 2, dash = false, opacity = 1) {
  const valid = points.filter(
    (p) => p[0] != null && p[1] != null && Number.isFinite(p[1]),
  );
  if (!valid.length) return;
  const d = valid
    .map(
      (p, i) =>
        `${i ? "L" : "M"}${chart.X(p[0]).toFixed(2)},${chart.Y(p[1]).toFixed(2)}`,
    )
    .join(" ");
  return S(
    "path",
    {
      d,
      fill: "none",
      stroke: color,
      "stroke-width": width,
      "stroke-dasharray": dash ? "5 4" : "",
      "stroke-opacity": opacity,
    },
    chart.layer,
  );
}
function band(chart, p, color) {
  p = p.filter((v) => Number.isFinite(v.lo) && Number.isFinite(v.hi));
  if (!p.length) return;
  const points = p
    .map((v) => [v.x, v.hi])
    .concat([...p].reverse().map((v) => [v.x, v.lo]));
  S(
    "path",
    {
      d:
        points
          .map((v, i) => `${i ? "L" : "M"}${chart.X(v[0])},${chart.Y(v[1])}`)
          .join(" ") + " Z",
      fill: color,
      "fill-opacity": 0.1,
      stroke: "none",
    },
    chart.layer,
  );
}
function showTip(event, html) {
  const tip = $("tip");
  tip.innerHTML = html;
  tip.style.display = "block";
  tip.style.left = Math.min(event.clientX + 15, innerWidth - 350) + "px";
  tip.style.top = Math.min(event.clientY + 10, innerHeight - 160) + "px";
}
function hideTip() {
  $("tip").style.display = "none";
}
function drawProfiles(id, sets) {
  const ymax =
    Math.max(
      state.units === "relative" ? 1.5 : 10,
      ...sets.flatMap((s) => s.data.map((p) => p.hi ?? 0)),
    ) * 1.06;
  const ch = axes(id, {
    ymax,
    ylabel:
      state.units === "relative"
        ? "速度 / 本 chunk 平均速度 (×)"
        : "TCP 速度 (mm/s)",
    xlabel:
      id === "saved-profile"
        ? "已保存解码轨迹内进度 (%)"
        : "实际执行 chunk 内进度 (%)",
  });
  if (state.units === "relative")
    line(
      ch,
      [
        [0, 1],
        [100, 1],
      ],
      css("--axis"),
      1,
      true,
    );
  for (const s of sets) {
    band(ch, s.data, s.color);
    line(
      ch,
      s.data.map((p) => [p.x, p.y]),
      s.color,
      2.5,
    );
  }
  const overlay = S(
    "rect",
    { x: ch.L, y: ch.T, width: ch.w, height: ch.h, fill: "transparent" },
    ch.svg,
  );
  overlay.addEventListener("mousemove", (e) => {
    const r = ch.svg.getBoundingClientRect(),
      x = ((e.clientX - r.left) / r.width) * ch.W;
    const i = clamp(Math.floor(((x - ch.L) / ch.w) * 12), 0, 11);
    showTip(
      e,
      `<b>进度 ${num((i / 12) * 100, 0)}–${num(((i + 1) / 12) * 100, 0)}%</b><br>` +
        sets
          .map(
            (s) =>
              `${esc(s.label)}：${num(s.data[i]?.y)}${state.units === "relative" ? "×" : " mm/s"}<br>25–75%：${num(s.data[i]?.lo)}–${num(s.data[i]?.hi)}`,
          )
          .join("<br>"),
    );
  });
  overlay.addEventListener("mouseleave", hideTip);
  return ch;
}
function renderHeader() {
  const c = DATA.recorded_config;
  const verdict = Object.entries(ARM).map(([arm, label]) => {
    const a = DATA.summary.arms[arm].state;
    if (!a.n) return `${label}：没有满足默认门槛的完整执行窗口。`;
    const trend =
      a.median_ratio <= 0.8
        ? "后段减速至少 20%"
        : a.median_ratio >= 1.2
          ? "后段加速至少 20%"
          : "前后差异未达 20%";
    return `${label}：实测后段 / 前段中位数 ${num(a.median_ratio)}×（${trend}），${a.slow20_count}/${a.n} 个 chunk 减速至少 20%。`;
  });
  $("identity-label").textContent =
    DATA.identity.model?.checkpoint_source || DATA.session_id;
  $("verdict").textContent = verdict.join(" ");
  $("snapshot").textContent =
    `固定快照 · ${DATA.session_id} · ${DATA.episodes.length} 次已完成 rollout · ${DATA.summary.all_plans} 个接受计划 / ${DATA.summary.eligible_execution} 个完整执行窗口 · 算法 ${c.algorithm} · 动作 ${num(1 / c.action_dt_s, 1)} Hz / 控制 ${num(c.control_hz, 1)} Hz。结论只覆盖这些记录，不自动跟随之后的运行配置。`;
  for (const [i, e] of DATA.episodes.entries()) {
    $("scope").insertAdjacentHTML(
      "beforeend",
      `<option value="${esc(e.id)}">${esc(e.id)} · ${num(e.duration, 1)} s</option>`,
    );
    $("timeline-ep").insertAdjacentHTML(
      "beforeend",
      `<option value="${i}">${esc(e.id)} · ${num(e.duration, 1)} s</option>`,
    );
  }
  const lag = DATA.summary.ik_lag;
  $("methods").innerHTML =
    `<p><b>记录配置。</b> ${c.horizon} 步模型 horizon，${num(c.action_dt_s * 1000, 2)} ms/步；算法 ${esc(c.algorithm)}、skip=${c.skip}。交接、动作解码和执行器设置见下方记录配置。</p>
 <p><b>前后速度。</b> v = ‖FK(qᵢ₊₁).xyz − FK(qᵢ).xyz‖ / Δt。按每个 chunk 自身执行时间的前后 ⅓ 作时间加权均值。默认平均速度 ≥5 mm/s 且前段速度 >1 mm/s；比值取每个 chunk 的 R 再取中位数，不是两个全局中位数相除。旋转、夹爪动作不混入 TCP 平移速度。</p>
 <p><b>执行证据。</b> 同一 plan_id 的连续 tick 为一个执行窗口；两端各遗漏至多一个采样间隔，不跨切换点差分。剔除每次 rollout 的首个启动计划、末个截断计划、缺少前后计划归属、样本不足和大于 50 ms 的时间缺口。正常完成记录不等于任务成功。</p>
 <p><b>预测范围。</b> canonical_raw 是已解码关节轨迹：保存行数 ${esc(JSON.stringify(DATA.summary.saved_rows_counts))}。它不能还原现场观测未经裁剪、交接和 IK 的原始模型输出。可选 offline 部分使用数据集图片且没有 RTC 条件。预测曲线与执行窗口采用不同时间域，不能直接互换。</p>
 <p><b>IK 记录。</b> 左臂 ${lag.left_arm.plans_over_5mm}/${DATA.summary.all_plans}、右臂 ${lag.right_arm.plans_over_5mm}/${DATA.summary.all_plans} 个计划记录了超过 5 mm 的最大位置滞后（已记录最大值 ${num(lag.left_arm.max_mm, 1)} / ${num(lag.right_arm.max_mm, 1)} mm）。缺少 IK 元数据的计划不提供此项证据。</p>
 <p><b>判断范围。</b> 对比 scheduled、command 与 state 可定位减速出现在哪一层。本分析没有做因果对照，不能单凭相关曲线判定模型、RTC、交接或 IK 是原因。</p>`;
  $("evidence").innerHTML =
    `<p>session：<code>${esc(DATA.session)}</code></p><p>checkpoint：<code>${esc(DATA.identity.model?.checkpoint_source || "unknown")}</code><br>backend revision：<code>${esc(DATA.identity.server_revision || "unknown")}</code></p>
 <p>FK 源码 / 配置与记录时指纹核对：${DATA.fk_source_check.checked_files} 个文件，${DATA.fk_source_check.mismatches.length} 个差异。外部 SDK 和模型资产未包含在此检查中。原始证据摘要、逐 chunk 数据与 SHA-256 在同目录 analysis.json。</p>
 <p>复现入口：<code>scripts/validation/plot_chunk_speed.py --session SESSION --out OUTPUT [--offline OFFLINE]</code>，用法见 <code>docs/usage/chunk-speed.md</code>。</p>
 <pre>${esc(JSON.stringify(c, null, 2))}</pre><p>排除 ${DATA.exclusions.length} 项：${esc(DATA.exclusions.map((x) => (x.plan || x.episode) + " → " + x.reason).join("；"))}</p>`;
}
function renderStats() {
  const sets = SOURCES.map((source) => {
    const r = rows(source);
    return { source, r, ...agg(r) };
  });
  $("facts").innerHTML = sets
    .map(
      (s) =>
        `<div class="fact" style="--k:var(${COLORS[s.source]})"><span class="lab">${ARM[state.arm]} · ${NAMES[s.source]}</span><span class="val">${num(s.ratio)}×</span><span class="lab">后 ⅓ ÷ 前 ⅓ · n=${s.n} · 减速 ≥20%：${s.slow || 0}/${s.n}</span></div>`,
    )
    .join("");
  $("scope-note").innerHTML =
    `当前图表：<b>${ARM[state.arm]} · ${state.scope === "all" ? "全部 rollout 合并" : esc(state.scope)}</b>；各层按自身平均速度 ≥${state.threshold} mm/s 筛选，故分母可能不同。顶部结论固定为全部 rollout、5 mm/s 门槛。`;
  legend(
    "profile-legend",
    sets.map((s) => [`${NAMES[s.source]} (n=${s.n})`, css(COLORS[s.source])]),
  );
  const ps = sets
    .filter((s) => s.n)
    .map((s) => ({
      label: NAMES[s.source],
      color: css(COLORS[s.source]),
      data: profile(s.r),
    }));
  drawProfiles("profile", ps);
  $("profile-table").innerHTML =
    "<table><tr><th>进度</th>" +
    ps.map((s) => `<th>${s.label} 中位数</th>`).join("") +
    "</tr>" +
    Array.from(
      { length: 12 },
      (_, i) =>
        `<tr><td>${num((i / 12) * 100, 0)}–${num(((i + 1) / 12) * 100, 0)}%</td>${ps.map((s) => `<td>${num(s.data[i].y)}</td>`).join("")}</tr>`,
    ).join("") +
    "</table>";
  const savedPlans = DATA.episodes
    .filter((e) => state.scope === "all" || e.id === state.scope)
    .flatMap((e) => e.plans)
    .filter((p) => !p.startup);
  const sr = rows("saved", "0", savedPlans),
    sa = agg(sr);
  drawProfiles("saved-profile", [
    { label: NAMES.saved, color: css("--roll"), data: profile(sr) },
  ]);
  $("saved-note").textContent =
    `${ARM[state.arm]} · 保存整段轨迹后 / 前中位数 ${num(sa.ratio)}×（n=${sa.n}），减速至少 20% ${sa.slow || 0}/${sa.n}；实际执行窗口中位时长 ${num(DATA.summary.duration_median, 3)} s。保存整段的形状不能用来替代实际执行段的形状。`;
  const groups = [
    ["合并", eligiblePlans()],
    ...DATA.episodes
      .filter((e) => state.scope === "all" || state.scope === e.id)
      .map((e) => [e.id, eligiblePlans(e.id)]),
  ];
  $("sensitivity").innerHTML =
    "<table><tr><th>范围 · 实测 FK</th><th>原窗口</th><th>去掉前 50 ms</th><th>去掉前 100 ms</th><th>重采样 + 平滑</th></tr>" +
    groups
      .map(
        ([label, plans]) =>
          "<tr><td>" +
          esc(label) +
          "</td>" +
          ["0", "0.05", "0.1", "resampled"]
            .map((trim) => {
              const a = agg(rows("state", trim, plans));
              return `<td>${num(a.ratio)}× <span class="muted">n=${a.n}</span></td>`;
            })
            .join("") +
          "</tr>",
      )
      .join("") +
    "</table>";
}
function heatColor(r) {
  const rgb = (s) =>
    s
      .replace("#", "")
      .match(/../g)
      .map((x) => parseInt(x, 16));
  const mid = rgb(css("--div-mid")),
    edge = rgb(css(r < 1 ? "--div-b2" : "--div-r2"));
  const f = clamp(Math.abs(Math.log2(Math.max(r, 0.001))) / Math.log2(3), 0, 1);
  return `rgb(${mid.map((x, i) => Math.round(x + (edge[i] - x) * f)).join(",")})`;
}
function renderHeatmaps() {
  const plans = eligiblePlans();
  $("heatmaps").replaceChildren();
  $("heat-key").innerHTML =
    "相对本 chunk 均速： " +
    [1 / 3, 0.5, 1, 2, 3]
      .map(
        (r) => `<span style="background:${heatColor(r)}"></span>${num(r, 1)}×`,
      )
      .join(" ") +
    " · 灰色：低于门槛";
  for (const src of SOURCES) {
    const col = document.createElement("div");
    col.innerHTML = `<div class="heat-head"><b>${NAMES[src]}</b><span>${plans.length} 行</span></div><div class="heat-scroll"></div>`;
    $("heatmaps").appendChild(col);
    const W = 350,
      L = 54,
      T = 24,
      rowH = 7,
      w = 282;
    const svg = S(
      "svg",
      {
        viewBox: `0 0 ${W} ${T + plans.length * rowH + 20}`,
        class: "heat-chart",
        role: "img",
        "aria-label": NAMES[src] + "速度热图",
      },
      col.lastChild,
    );
    for (let k = 0; k <= 4; k++)
      text(svg, L + (k / 4) * w, 14, k * 25 + "%", {
        "text-anchor": "middle",
        "font-size": 10,
      });
    plans.forEach((p, i) => {
      const s = p.arms[state.arm][src].stats.tcp,
        active = s && s.mean >= state.threshold;
      const g = S(
        "g",
        {
          "data-key": p.key,
          tabindex: 0,
          role: "button",
          "aria-label": p.key + " chunk " + p.seq,
        },
        svg,
      );
      for (let j = 0; j < 12; j++)
        S(
          "rect",
          {
            x: L + (j * w) / 12,
            y: T + i * rowH,
            width: w / 12 - 0.5,
            height: rowH - 0.4,
            fill: active ? heatColor(s.profile[j] / s.mean) : css("--still"),
          },
          g,
        );
      if (
        i % 10 === 0 ||
        i === 0 ||
        plans[i - 1].key.split("/")[0] !== p.key.split("/")[0]
      )
        text(
          svg,
          L - 7,
          T + i * rowH + 6,
          p.key.split("/")[0].slice(-3) + ":" + p.seq,
          { "text-anchor": "end", "font-size": 9 },
        );
      const choose = () => {
        state.ep = DATA.episodes.findIndex((e) => p.key.startsWith(e.id + "/"));
        state.focus = p.key;
        state.start = Math.max(0, p.execution.start - 2);
        $("timeline-ep").value = state.ep;
        renderTimeline();
        $("timeline-section").scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      };
      g.addEventListener("click", choose);
      g.addEventListener("keydown", (e) => {
        if (e.key === "Enter") choose();
      });
      g.addEventListener("mousemove", (e) =>
        showTip(
          e,
          `${esc(p.key)}<br>${NAMES[src]}<br>前 ${num(s?.early, 1)} / 后 ${num(s?.late, 1)} mm/s<br>R = ${num(s?.ratio)}×`,
        ),
      );
      g.addEventListener("mouseleave", hideTip);
    });
  }
}
function plotTickSegments(ch, ep, src, start, end, color) {
  const tick = ep.ticks.arms[state.arm][src],
    t = ep.ticks.t;
  for (const p of ep.plans) {
    const ex = p.execution;
    if (!ex || ex.end < start || ex.start > end) continue;
    const a = ex.tick_start,
      b = ex.tick_end;
    const pts = [];
    for (let i = a; i < b - 1; i++) {
      const tm = (t[i] + t[i + 1]) / 2;
      if (tm >= start && tm <= end) pts.push([tm, tick.speed.tcp[i]]);
    }
    line(
      ch,
      pts,
      color,
      src === "state" ? 1.5 : 1.8,
      false,
      src === "state" ? 0.8 : 1,
    );
  }
}
function renderTimeline() {
  const ep = DATA.episodes[state.ep];
  let win = state.window === "all" ? ep.duration : Number(state.window);
  win = Math.min(win, ep.duration);
  state.start = clamp(state.start, 0, Math.max(0, ep.duration - win));
  const start = state.start,
    end = start + win;
  $("pan").max = Math.max(0, ep.duration - win);
  $("pan").value = start;
  $("window-label").textContent =
    `${num(start, 1)}–${num(end, 1)} s / ${num(ep.duration, 1)} s`;
  let max = 10;
  for (const src of SOURCES) {
    const v = ep.ticks.arms[state.arm][src].speed.tcp;
    ep.ticks.t.forEach((t, i) => {
      if (t >= start && t <= end && v[i] != null) max = Math.max(max, v[i]);
    });
  }
  for (const p of ep.plans) {
    const v = p.arms[state.arm].committed;
    v.speed.tcp.forEach((s, i) => {
      const t = p.start + (v.t[i] + v.t[i + 1]) / 2;
      if (t >= start && t <= end) max = Math.max(max, s);
    });
  }
  const ch = axes("timeline", {
    height: 350,
    xmin: start,
    xmax: end,
    ymax: max * 1.07,
    xlabel: "rollout 时间 (s)",
    ylabel: `${ARM[state.arm]} TCP 速度 (mm/s)`,
  });
  for (const p of ep.plans) {
    const ex = p.execution;
    if (!ex) continue;
    const v = p.arms[state.arm].committed;
    line(
      ch,
      v.speed.tcp
        .map((s, i) => [p.start + (v.t[i] + v.t[i + 1]) / 2, s])
        .filter((x) => x[0] >= start && x[0] <= end),
      css("--roll"),
      1,
      true,
      0.22,
    );
    if (ex.start >= start && ex.start <= end) {
      S(
        "line",
        {
          x1: ch.X(ex.start),
          x2: ch.X(ex.start),
          y1: ch.T,
          y2: ch.T + ch.h,
          stroke: css("--axis"),
          "stroke-opacity": 0.55,
        },
        ch.layer,
      );
      text(ch.svg, ch.X(ex.start) + 2, ch.T + 12, "#" + p.seq, {
        "font-size": 9,
      });
    }
  }
  for (const src of ["state", "scheduled", "command"])
    plotTickSegments(ch, ep, src, start, end, css(COLORS[src]));
  legend("timeline-legend", [
    ...SOURCES.map((s) => [NAMES[s], css(COLORS[s])]),
    ["保存点的后续预测", css("--roll"), true],
  ]);
  const overlay = S(
    "rect",
    { x: ch.L, y: ch.T, width: ch.w, height: ch.h, fill: "transparent" },
    ch.svg,
  );
  overlay.addEventListener("mousemove", (e) => {
    const r = ch.svg.getBoundingClientRect(),
      tt =
        start + ((((e.clientX - r.left) / r.width) * ch.W - ch.L) / ch.w) * win;
    const p = ep.plans.find(
      (p) => p.execution && p.execution.start <= tt && p.execution.end >= tt,
    );
    if (p)
      showTip(
        e,
        `${esc(ep.id)} · chunk ${p.seq}<br>t=${num(tt, 3)} s<br>实测前 / 后：${num(p.arms[state.arm].state.stats.tcp?.early, 1)} / ${num(p.arms[state.arm].state.stats.tcp?.late, 1)} mm/s<br>点击查看该 chunk`,
      );
  });
  overlay.addEventListener("mouseleave", hideTip);
  overlay.addEventListener("click", (e) => {
    const r = ch.svg.getBoundingClientRect(),
      tt =
        start + ((((e.clientX - r.left) / r.width) * ch.W - ch.L) / ch.w) * win;
    const p = ep.plans.find(
      (p) => p.execution && p.execution.start <= tt && p.execution.end >= tt,
    );
    if (p) {
      state.focus = p.key;
      renderDetail();
    }
  });
  $("chunk-select").innerHTML = ep.plans
    .map(
      (p) =>
        `<option value="${esc(p.key)}">#${p.seq} · ${num(p.execution?.start, 2)} s${p.execution?.eligible ? "" : " · 排除统计"}</option>`,
    )
    .join("");
  if (!ep.plans.some((p) => p.key === state.focus))
    state.focus =
      ep.plans.find((p) => p.execution?.eligible)?.key || ep.plans[0].key;
  renderDetail();
}
function renderDetail() {
  const ep = DATA.episodes[state.ep],
    p = ep.plans.find((p) => p.key === state.focus);
  if (!p) return;
  $("chunk-select").value = p.key;
  const ex = p.execution,
    cm = p.arms[state.arm].committed;
  if (!ex) {
    $("detail").replaceChildren();
    $("chunk-label").textContent = p.key;
    $("detail-note").textContent =
      "该保存计划没有 tick 执行证据，不纳入执行窗口统计。";
    return;
  }
  const shift = p.start - ex.start,
    xmax = Math.max(cm.t.at(-1) + shift, ex.duration, 0.01);
  let ymax = Math.max(...cm.speed.tcp, 10);
  for (const src of SOURCES)
    ymax = Math.max(
      ymax,
      ...ep.ticks.arms[state.arm][src].speed.tcp.slice(
        ex.tick_start,
        ex.tick_end - 1,
      ),
    );
  const ch = axes("detail", {
    height: 300,
    xmin: 0,
    xmax,
    ymax: ymax * 1.1,
    xlabel: "从 tick 计划切换起的时间 (s)",
    ylabel: `chunk ${p.seq} · ${ARM[state.arm]} TCP 速度 (mm/s)`,
  });
  S(
    "rect",
    {
      x: ch.X(ex.duration),
      y: ch.T,
      width: Math.max(0, ch.X(xmax) - ch.X(ex.duration)),
      height: ch.h,
      fill: css("--band"),
    },
    ch.layer,
  );
  line(
    ch,
    cm.speed.tcp.map((v, i) => [shift + (cm.t[i] + cm.t[i + 1]) / 2, v]),
    css("--roll"),
    1.8,
    true,
    0.5,
  );
  for (const src of SOURCES) {
    const t = ep.ticks.t,
      v = ep.ticks.arms[state.arm][src].speed.tcp;
    line(
      ch,
      Array.from({ length: ex.tick_end - ex.tick_start - 1 }, (_, k) => {
        const i = ex.tick_start + k;
        return [(t[i] + t[i + 1]) / 2 - ex.start, v[i]];
      }),
      css(COLORS[src]),
      2,
    );
  }
  S(
    "line",
    {
      x1: ch.X(ex.duration),
      x2: ch.X(ex.duration),
      y1: ch.T,
      y2: ch.T + ch.h,
      stroke: css("--ink"),
      "stroke-dasharray": "4 3",
    },
    ch.layer,
  );
  text(ch.svg, ch.X(ex.duration) + 7, ch.T + 15, "此后未执行 / 已换 chunk");
  $("chunk-label").textContent =
    `${ep.id} · request ${p.seq} · 保存 ${p.saved_rows} 行 · source_offset=${p.source_offset} · lead-in=${p.lead_in_steps} 子步`;
  const a = p.arms[state.arm].state.stats.tcp;
  $("detail-note").textContent =
    `执行区间 ${num(ex.start, 3)}–${num(ex.end, 3)} s（${num(ex.duration, 3)} s）；实测前段 ${num(a?.early, 1)}、后段 ${num(a?.late, 1)} mm/s，R=${num(a?.ratio)}×。${ex.eligible ? "纳入执行窗口统计。" : "不纳入统计：" + ex.reasons.join("；")} 阴影只表示预测上下文，不能当作实测运动。`;
}
function render() {
  renderStats();
  renderHeatmaps();
  renderTimeline();
  if (globalThis.renderOffline) globalThis.renderOffline();
}
renderHeader();
render();
$("arm").addEventListener("change", (e) => {
  state.arm = e.target.value;
  render();
});
$("scope").addEventListener("change", (e) => {
  state.scope = e.target.value;
  if (state.scope !== "all")
    state.ep = DATA.episodes.findIndex((x) => x.id === state.scope);
  $("timeline-ep").value = state.ep;
  render();
});
$("threshold").addEventListener("change", (e) => {
  state.threshold = +e.target.value;
  render();
});
$("units").addEventListener("change", (e) => {
  state.units = e.target.value;
  renderStats();
  if (globalThis.renderOffline) globalThis.renderOffline();
});
$("timeline-ep").addEventListener("change", (e) => {
  state.ep = +e.target.value;
  state.start = 0;
  state.focus = null;
  renderTimeline();
});
$("window").addEventListener("change", (e) => {
  state.window = e.target.value;
  renderTimeline();
});
$("pan").addEventListener("input", (e) => {
  state.start = +e.target.value;
  renderTimeline();
});
$("chunk-select").addEventListener("change", (e) => {
  state.focus = e.target.value;
  renderDetail();
});
$("download").addEventListener("click", () => {
  const blob = new Blob(
      [
        JSON.stringify(
          {
            session: DATA.session,
            method: DATA.method,
            summary: DATA.summary,
            offline: DATA.offline
              ? {
                  provenance: DATA.offline.provenance,
                  summary: DATA.offline.summary,
                }
              : null,
          },
          null,
          2,
        ),
      ],
      { type: "application/json" },
    ),
    a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "chunk-speed-summary.json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});
globalThis.__chunkReport = {
  DATA,
  state,
  render,
  renderStats,
  renderTimeline,
  renderDetail,
  rows,
  agg,
};
