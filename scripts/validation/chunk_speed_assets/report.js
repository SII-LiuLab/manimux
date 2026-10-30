"use strict";
const DATA = JSON.parse(document.getElementById("report-data").textContent);
const $ = (id) => document.getElementById(id),
  NS = "http://www.w3.org/2000/svg";
const NAMES = {
  scheduled: "Scheduled reference",
  command: "Executor command",
  state: "Measured feedback FK",
  saved: "Saved decoded trajectory",
};
const COLORS = {
  scheduled: "--roll",
  command: "--ds",
  state: "--ref",
  saved: "--roll",
};
const ARM = { left_arm: "Left arm", right_arm: "Right arm" },
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
    xlabel = "Progress within executed chunk (%)",
    ylabel = "Relative to this chunk's mean speed (×)",
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
        ? "Speed / this chunk's mean speed (×)"
        : "TCP speed (mm/s)",
    xlabel:
      id === "saved-profile"
        ? "Progress within saved decoded trajectory (%)"
        : "Progress within actual executed chunk (%)",
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
      `<b>Progress ${num((i / 12) * 100, 0)}–${num(((i + 1) / 12) * 100, 0)}%</b><br>` +
        sets
          .map(
            (s) =>
              `${esc(s.label)}: ${num(s.data[i]?.y)}${state.units === "relative" ? "×" : " mm/s"}<br>25–75%: ${num(s.data[i]?.lo)}–${num(s.data[i]?.hi)}`,
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
    if (!a.n) return `${label}: no complete execution windows meet the default threshold.`;
    const trend =
      a.median_ratio <= 0.8
        ? "late segment slows by at least 20%"
        : a.median_ratio >= 1.2
          ? "late segment speeds up by at least 20%"
          : "early/late difference is below 20%";
    return `${label}: median measured late/early ratio ${num(a.median_ratio)}× (${trend}); ${a.slow20_count}/${a.n} chunks slow by at least 20%.`;
  });
  $("identity-label").textContent =
    DATA.identity.model?.checkpoint_source || DATA.session_id;
  $("verdict").textContent = verdict.join(" ");
  $("snapshot").textContent =
    `Fixed snapshot · ${DATA.session_id} · ${DATA.episodes.length} finalized rollouts · ${DATA.summary.all_plans} accepted plans / ${DATA.summary.eligible_execution} complete execution windows · Algorithm ${c.algorithm} · Actions ${num(1 / c.action_dt_s, 1)} Hz / control ${num(c.control_hz, 1)} Hz. Conclusions cover these records only and do not track later configuration changes.`;
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
    `<p><b>Recorded configuration.</b> Model horizon ${c.horizon} steps, ${num(c.action_dt_s * 1000, 2)} ms/step; algorithm ${esc(c.algorithm)}, skip=${c.skip}. See the recorded configuration below for handoff, action decoding and executor settings.</p>
  <p><b>Early/late speed.</b> v = ‖FK(qᵢ₊₁).xyz − FK(qᵢ).xyz‖ / Δt. Use time-weighted means over the first and last ⅓ of each chunk's execution time. Defaults require mean speed ≥5 mm/s and early speed >1 mm/s. Report the median of per-chunk ratios R, not a ratio of two global medians. Rotation and gripper motion are excluded from TCP translation speed.</p>
  <p><b>Execution evidence.</b> Consecutive ticks with one plan_id define an execution window; at most one sample interval may be missing at each end. Do not differentiate across plan switches. Exclude the initial startup plan, final truncated plan, missing neighboring plan assignments, insufficient samples and gaps longer than 50 ms. A finalized recording does not imply task success.</p>
  <p><b>Prediction scope.</b> canonical_raw contains decoded joint trajectories with saved row counts ${esc(JSON.stringify(DATA.summary.saved_rows_counts))}. It cannot reconstruct raw model outputs before cropping, handoff and IK at the live observation. Optional offline analysis uses dataset images without RTC conditions. Prediction curves and execution windows use different time domains and are not interchangeable.</p>
  <p><b>IK records.</b> ${lag.left_arm.plans_over_5mm}/${DATA.summary.all_plans} left-arm and ${lag.right_arm.plans_over_5mm}/${DATA.summary.all_plans} right-arm plans recorded maximum position lag above 5 mm (recorded maxima ${num(lag.left_arm.max_mm, 1)} / ${num(lag.right_arm.max_mm, 1)} mm). Plans without IK metadata provide no evidence for this metric.</p>
  <p><b>Interpretation.</b> Comparing scheduled, command and state helps locate the layer where slowdown appears. This analysis has no causal control; correlated curves alone cannot identify the model, RTC, handoff or IK as the cause.</p>`;
  $("evidence").innerHTML =
    `<p>session: <code>${esc(DATA.session)}</code></p><p>checkpoint: <code>${esc(DATA.identity.model?.checkpoint_source || "unknown")}</code><br>backend revision: <code>${esc(DATA.identity.server_revision || "unknown")}</code></p>
  <p>FK source/configuration fingerprints compared with recording time: ${DATA.fk_source_check.checked_files} files, ${DATA.fk_source_check.mismatches.length} mismatches. External SDKs and model assets are outside this check. Evidence summaries, per-chunk data and SHA-256 hashes are in analysis.json in the same directory.</p>
  <p>Reproduce with <code>scripts/validation/plot_chunk_speed.py --session SESSION --out OUTPUT [--offline OFFLINE]</code>; see <code>skills/manimux-experiments/references/chunk-speed.md</code>.</p>
  <pre>${esc(JSON.stringify(c, null, 2))}</pre><p>${DATA.exclusions.length} exclusions: ${esc(DATA.exclusions.map((x) => (x.plan || x.episode) + " → " + x.reason).join("; "))}</p>`;
}
function renderStats() {
  const sets = SOURCES.map((source) => {
    const r = rows(source);
    return { source, r, ...agg(r) };
  });
  $("facts").innerHTML = sets
    .map(
      (s) =>
        `<div class="fact" style="--k:var(${COLORS[s.source]})"><span class="lab">${ARM[state.arm]} · ${NAMES[s.source]}</span><span class="val">${num(s.ratio)}×</span><span class="lab">Last ⅓ ÷ first ⅓ · n=${s.n} · slowdown ≥20%: ${s.slow || 0}/${s.n}</span></div>`,
    )
    .join("");
  $("scope-note").innerHTML =
    `Current chart: <b>${ARM[state.arm]} · ${state.scope === "all" ? "All rollouts combined" : esc(state.scope)}</b>; each layer is filtered by its own mean speed ≥${state.threshold} mm/s, so denominators may differ. Top-level conclusions always use all rollouts and a 5 mm/s threshold.`;
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
    "<table><tr><th>Progress</th>" +
    ps.map((s) => `<th>${s.label} median</th>`).join("") +
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
    `${ARM[state.arm]} · Median saved full-trajectory late/early ratio ${num(sa.ratio)}× (n=${sa.n}), slowdown ≥20% in ${sa.slow || 0}/${sa.n}; median actual execution-window duration ${num(DATA.summary.duration_median, 3)} s. Full saved trajectory shape cannot substitute for executed segment shape.`;
  const groups = [
    ["Combined", eligiblePlans()],
    ...DATA.episodes
      .filter((e) => state.scope === "all" || state.scope === e.id)
      .map((e) => [e.id, eligiblePlans(e.id)]),
  ];
  $("sensitivity").innerHTML =
    "<table><tr><th>Scope · measured FK</th><th>Original window</th><th>Exclude first 50 ms</th><th>Exclude first 100 ms</th><th>Resampled + smoothed</th></tr>" +
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
    "Relative to this chunk's mean speed: " +
    [1 / 3, 0.5, 1, 2, 3]
      .map(
        (r) => `<span style="background:${heatColor(r)}"></span>${num(r, 1)}×`,
      )
      .join(" ") +
    " · Gray: below threshold";
  for (const src of SOURCES) {
    const col = document.createElement("div");
    col.innerHTML = `<div class="heat-head"><b>${NAMES[src]}</b><span>${plans.length} rows</span></div><div class="heat-scroll"></div>`;
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
        "aria-label": NAMES[src] + " speed heatmap",
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
          `${esc(p.key)}<br>${NAMES[src]}<br>Early ${num(s?.early, 1)} / late ${num(s?.late, 1)} mm/s<br>R = ${num(s?.ratio)}×`,
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
    xlabel: "Rollout time (s)",
    ylabel: `${ARM[state.arm]} TCP speed (mm/s)`,
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
    ["Future prediction from saved points", css("--roll"), true],
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
        `${esc(ep.id)} · chunk ${p.seq}<br>t=${num(tt, 3)} s<br>Measured early / late: ${num(p.arms[state.arm].state.stats.tcp?.early, 1)} / ${num(p.arms[state.arm].state.stats.tcp?.late, 1)} mm/s<br>Click to inspect this chunk`,
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
        `<option value="${esc(p.key)}">#${p.seq} · ${num(p.execution?.start, 2)} s${p.execution?.eligible ? "" : " · Excluded from statistics"}</option>`,
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
      "This saved plan has no tick-level execution evidence and is excluded from execution-window statistics.";
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
    xlabel: "Time since tick-level plan switch (s)",
    ylabel: `chunk ${p.seq} · ${ARM[state.arm]} TCP speed (mm/s)`,
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
  text(ch.svg, ch.X(ex.duration) + 7, ch.T + 15, "Not executed / chunk replaced after this point");
  $("chunk-label").textContent =
    `${ep.id} · request ${p.seq} · ${p.saved_rows} saved rows · source_offset=${p.source_offset} · lead-in=${p.lead_in_steps} substeps`;
  const a = p.arms[state.arm].state.stats.tcp;
  $("detail-note").textContent =
    `Execution interval ${num(ex.start, 3)}–${num(ex.end, 3)} s (${num(ex.duration, 3)} s); measured early ${num(a?.early, 1)}, late ${num(a?.late, 1)} mm/s, R=${num(a?.ratio)}×. ${ex.eligible ? "Included in execution-window statistics." : "Excluded: " + ex.reasons.join("; ")} Shading shows prediction context only, not measured motion.`;
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
