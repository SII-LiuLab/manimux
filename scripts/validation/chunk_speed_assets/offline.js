(function () {
  if (!DATA.offline) return;
  const D = DATA.offline,
    P = D.provenance,
    OS = {
      ep: 0,
      scope: "all",
      domain: "full",
      chunk: 0,
      window: 10,
      start: 0,
    };
  $("offline-panel").hidden = false;
  const ep = () => D.episodes[OS.ep];
  const chunks = () =>
    OS.scope === "all" ? D.episodes.flatMap((e) => e.chunks) : ep().chunks;
  const named = { pred: "Offline model prediction", gt: "Human demonstration GT" };
  function offlineRows(source, selected = chunks()) {
    return selected
      .map((c) => ({ p: c, s: c.arms[state.arm].stats[source][OS.domain] }))
      .filter((r) => r.s && r.s.mean >= state.threshold && r.s.ratio != null);
  }
  function curveSet(source) {
    const rs = offlineRows(source);
    return {
      label: named[source] + " (n=" + rs.length + ")",
      color: css(source === "pred" ? "--ds" : "--ref"),
      data: profile(rs),
      r: rs,
    };
  }
  function renderComparison() {
    const real = rows("state"),
      pred = curveSet("pred"),
      gt = curveSet("gt"),
      sets = [
        {
          label: "Measured robot FK (n=" + real.length + ")",
          color: css("--roll"),
          data: profile(real),
          r: real,
        },
        pred,
        gt,
      ];
    $("offline-facts").innerHTML = sets
      .map((s) => {
        const a = agg(s.r);
        return `<div class="fact" style="--k:${s.color}"><span class="lab">${ARM[state.arm]} · ${esc(s.label)}</span><span class="val">${num(a.ratio)}×</span><span class="lab">Last ⅓ / first ⅓ · slowdown ≥20%: ${a.slow || 0}/${a.n}</span></div>`;
      })
      .join("");
    legend(
      "offline-legend",
      sets.map((s) => [s.label, s.color]),
    );
    const ch = drawProfiles(
      "offline-compare",
      sets.filter((s) => s.r.length),
    );
    const texts = ch.svg.querySelectorAll("text");
    texts[texts.length - 1].textContent = "Progress within each comparison window (%)";
    $("offline-domain-note").textContent =
      `${ARM[state.arm]} · Offline ${OS.scope === "all" ? D.episodes.length + " episodes combined" : "episode " + ep().id} · ${OS.domain === "full" ? `${P.horizon - 1} adjacent speed intervals in the native ${P.horizon}-step action (${num((P.horizon - 1) / P.fps, 3)} s)` : `First 12 speed intervals of the native prediction (${num(12 / P.fps, 3)} s)`}. Robot scope: ${state.scope === "all" ? DATA.episodes.length + " rollouts" : state.scope}. Checkpoint and normalization files match; input distributions and inference conditions differ. This is a shape comparison.`;
  }
  function choose(index, scroll = false) {
    OS.chunk = clamp(index, 0, ep().chunks.length - 1);
    OS.start = Math.max(0, ep().chunks[OS.chunk].t - 2);
    renderImages();
    renderOfflineTimeline();
    if (scroll)
      $("offline-frame-label").scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
  }
  function renderOfflineHeat() {
    const E = ep(),
      N = OS.domain === "full" ? 31 : 12,
      W = 490,
      L = 54,
      T = 26,
      w = 418,
      h = 6;
    $("offline-heatmaps").replaceChildren();
    for (const source of ["pred", "gt"]) {
      const box = document.createElement("div");
      box.innerHTML = `<div class="heat-head"><b>${named[source]}</b><span>episode ${E.id} · ${E.chunks.length} chunks</span></div><div class="heat-scroll"></div>`;
      $("offline-heatmaps").appendChild(box);
      const svg = S(
        "svg",
        {
          viewBox: `0 0 ${W} ${T + E.chunks.length * h + 20}`,
          class: "heat-chart",
          role: "img",
          "aria-label": named[source] + " per-chunk speed",
        },
        box.lastChild,
      );
      for (let k = 0; k <= 4; k++)
        text(svg, L + (w * k) / 4, 15, num((k * (N - 1)) / 4 + 1, 0) + " steps", {
          "font-size": 10,
          "text-anchor": "middle",
        });
      E.chunks.forEach((c, i) => {
        const s = c.arms[state.arm].stats[source][OS.domain],
          v = c.arms[state.arm][source],
          active = s.mean >= state.threshold;
        const g = S(
          "g",
          {
            "data-key": "offline/" + E.id + "/" + c.frame,
            tabindex: 0,
            role: "button",
            "aria-label": `episode ${E.id} frame ${c.frame}`,
          },
          svg,
        );
        for (let k = 0; k < N; k++)
          S(
            "rect",
            {
              x: L + (k * w) / N,
              y: T + i * h,
              width: w / N - 0.3,
              height: h - 0.3,
              fill: active ? heatColor(v[k] / s.mean) : css("--still"),
            },
            g,
          );
        if (i % 15 === 0)
          text(svg, L - 6, T + i * h + 5, num(c.t, 1) + "s", {
            "font-size": 9,
            "text-anchor": "end",
          });
        g.addEventListener("click", () => choose(i, true));
        g.addEventListener("keydown", (e) => {
          if (e.key === "Enter") choose(i, true);
        });
        g.addEventListener("mousemove", (e) =>
          showTip(
            e,
            `episode ${E.id} · frame ${c.frame} · ${num(c.t, 2)} s<br>${named[source]}<br>Early ${num(s.early, 1)} / late ${num(s.late, 1)} mm/s<br>R=${num(s.ratio)}× · Click to inspect input images`,
          ),
        );
        g.addEventListener("mouseleave", hideTip);
      });
    }
  }
  function renderImages() {
    const E = ep(),
      c = E.chunks[OS.chunk],
      arm = c.arms[state.arm];
    $("offline-chunk").value = OS.chunk;
    $("offline-frame-slider").value = OS.chunk;
    $("offline-left-image").src = c.images.left;
    $("offline-right-image").src = c.images.right;
    $("offline-frame-label").textContent =
      `episode ${E.id} · frame ${c.frame} · t=${num(c.t, 3)} s`;
    $("offline-image-caption").textContent =
      `Input: original RGB (left: ${c.image_source.left.rgb_shape.join("×")}; right: ${c.image_source.right.rgb_shape.join("×")}; JPEG previews shown here) · Prompt: ${P.prompt} · ${P.num_steps} sampling steps · ${num(c.latency_ms, 1)} ms for this prediction${c.warmup ? " (includes initial JIT compilation; excluded from steady-state timing)" : ""} · seed ${c.seed}`;
    const max = Math.max(10, ...arm.pred, ...arm.gt),
      ch = axes("offline-detail", {
        height: 310,
        xmin: 1,
        xmax: 31,
        ymax: max * 1.08,
        xlabel: `Speed interval k within chunk (action[k] → action[k+1], ${num(1000 / P.fps, 1)} ms per step)`,
        ylabel: `${ARM[state.arm]} native TCP speed (mm/s)`,
        xticks: 6,
      });
    if (OS.domain === "prefix") {
      S(
        "rect",
        {
          x: ch.X(12.5),
          y: ch.T,
          width: ch.X(31) - ch.X(12.5),
          height: ch.h,
          fill: css("--band"),
        },
        ch.layer,
      );
      text(ch.svg, ch.X(13), ch.T + 14, "Gray segment excluded from current prefix statistics");
    }
    line(
      ch,
      arm.pred.map((v, i) => [i + 1, v]),
      css("--ds"),
      2.5,
    );
    line(
      ch,
      arm.gt.map((v, i) => [i + 1, v]),
      css("--ref"),
      2,
    );
    legend("offline-detail-legend", [
      [named.pred, css("--ds")],
      [named.gt, css("--ref")],
    ]);
    const a = arm.stats.pred[OS.domain],
      b = arm.stats.gt[OS.domain];
    $("offline-detail-note").textContent =
      `Current window: prediction R=${num(a.ratio)}×, GT R=${num(b.ratio)}×. Reference-page full-chunk metric (last 8 intervals / first 8 intervals): prediction ${num(arm.stats.pred.ratio8)}×, GT ${num(arm.stats.gt.ratio8)}×. Trajectory speed uses adjacent predicted points; it excludes displacement from the observation anchor to the first predicted point.`;
  }
  function renderOfflineTimeline() {
    const E = ep(),
      win = Math.min(
        OS.window === "all" ? E.duration : Number(OS.window),
        E.duration,
      );
    OS.start = clamp(OS.start, 0, Math.max(0, E.duration - win));
    const start = OS.start,
      end = start + win;
    $("offline-pan").max = Math.max(0, E.duration - win);
    $("offline-pan").value = start;
    $("offline-window-label").textContent =
      `${num(start, 1)}–${num(end, 1)} s / ${num(E.duration, 1)} s`;
    const gt = E.gt[state.arm],
      gpts = gt.t
        .map((t, i) => [t, gt.speed[i]])
        .filter((x) => x[0] >= start && x[0] <= end),
      visible = E.chunks.filter((c) => c.t + 32 / P.fps >= start && c.t <= end);
    const max = Math.max(
      10,
      ...gpts.map((x) => x[1]),
      ...visible.flatMap((c) => c.arms[state.arm].pred),
    );
    const ch = axes("offline-timeline", {
      height: 320,
      xmin: start,
      xmax: end,
      ymax: max * 1.07,
      xlabel: "Dataset episode time (s)",
      ylabel: `episode ${E.id} · ${ARM[state.arm]} TCP speed (mm/s)`,
    });
    line(ch, gpts, css("--ref"), 1.8);
    for (const c of visible) {
      const points = c.arms[state.arm].pred
        .map((v, k) => [c.t + (k + 1.5) / P.fps, v])
        .filter((x) => x[0] >= start && x[0] <= end);
      line(ch, points, css("--ds"), 1, false, 0.18);
      line(
        ch,
        points.filter((x) => x[0] <= c.t + P.stride_frames / P.fps),
        css("--ds"),
        2.2,
      );
    }
    const selected = E.chunks[OS.chunk];
    if (selected.t >= start && selected.t <= end) {
      S(
        "line",
        {
          x1: ch.X(selected.t),
          x2: ch.X(selected.t),
          y1: ch.T,
          y2: ch.T + ch.h,
          stroke: css("--ink"),
          "stroke-dasharray": "4 3",
        },
        ch.layer,
      );
      text(ch.svg, ch.X(selected.t) + 5, ch.T + 13, "Selected image");
    }
    const overlay = S(
      "rect",
      { x: ch.L, y: ch.T, width: ch.w, height: ch.h, fill: "transparent" },
      ch.svg,
    );
    const nearest = (e) => {
      const r = ch.svg.getBoundingClientRect(),
        t =
          start +
          ((((e.clientX - r.left) / r.width) * ch.W - ch.L) / ch.w) * win;
      return E.chunks.reduce(
        (best, c, i) =>
          Math.abs(c.t - t) < Math.abs(E.chunks[best].t - t) ? i : best,
        0,
      );
    };
    overlay.addEventListener("mousemove", (e) => {
      const c = E.chunks[nearest(e)];
      showTip(
        e,
        `episode ${E.id} · frame ${c.frame}<br>t=${num(c.t, 3)} s<br>Click to select these input images`,
      );
    });
    overlay.addEventListener("mouseleave", hideTip);
    overlay.addEventListener("click", (e) => {
      OS.chunk = nearest(e);
      renderImages();
      renderOfflineTimeline();
    });
  }
  function renderTable() {
    $("offline-episode-table").innerHTML =
      "<table><tr><th>episode</th><th>Predictions</th><th>Prediction R</th><th>GT R</th><th>Inference p50 / p95 (ms)</th></tr>" +
      D.episodes
        .map((e) => {
          const pred = agg(offlineRows("pred", e.chunks)),
            gt = agg(offlineRows("gt", e.chunks)),
            ms = e.chunks.filter((c) => !c.warmup).map((c) => c.latency_ms);
          return `<tr><td>${e.id}</td><td>${e.chunks.length}</td><td>${num(pred.ratio)}× · n=${pred.n}</td><td>${num(gt.ratio)}× · n=${gt.n}</td><td>${num(median(ms), 1)} / ${num(quant(ms, 0.95), 1)}</td></tr>`;
        })
        .join("") +
      "</table>";
    $("offline-method").textContent =
      `Dataset ${P.dataset}, metadata split ${JSON.stringify(P.dataset_split)}; independence from training data requires a separate check. Checkpoint ${P.checkpoint_source}, normalization SHA-256 ${P.norm_stats_sha256}. ${P.chunks} independent predictions, sampled every ${P.stride_frames} frames; both wrist images use that frame's video PTS. GT is state[t+1:t+33]; action[t]=state[t+1] has been verified. No RTC prefix or IK. Steady-state inference p50=${num(P.latency_ms.median, 1)} ms, p95=${num(P.latency_ms.p95, 1)} ms (excluding initial JIT compilation; including model preprocessing, sampling and output synchronization; excluding video decoding and preview encoding).`;
  }
  function renderOffline() {
    renderComparison();
    renderOfflineHeat();
    renderImages();
    renderOfflineTimeline();
    renderTable();
  }
  function resetEpisode() {
    const E = ep();
    OS.chunk = 0;
    OS.start = 0;
    $("offline-chunk").innerHTML = E.chunks
      .map(
        (c, i) =>
          `<option value="${i}">frame ${c.frame} · ${num(c.t, 2)} s</option>`,
      )
      .join("");
    $("offline-frame-slider").max = E.chunks.length - 1;
    renderOffline();
  }
  $("offline-episode").innerHTML = D.episodes
    .map(
      (e, i) =>
        `<option value="${i}">${e.id} · ${num(e.duration, 1)} s</option>`,
    )
    .join("");
  $("offline-episode").addEventListener("change", (e) => {
    OS.ep = +e.target.value;
    resetEpisode();
  });
  $("offline-scope").addEventListener("change", (e) => {
    OS.scope = e.target.value;
    renderComparison();
  });
  $("offline-domain").addEventListener("change", (e) => {
    OS.domain = e.target.value;
    renderOffline();
  });
  $("offline-chunk").addEventListener("change", (e) => choose(+e.target.value));
  $("offline-frame-slider").addEventListener("input", (e) =>
    choose(+e.target.value),
  );
  $("offline-window").addEventListener("change", (e) => {
    OS.window = e.target.value;
    renderOfflineTimeline();
  });
  $("offline-pan").addEventListener("input", (e) => {
    OS.start = +e.target.value;
    renderOfflineTimeline();
  });
  globalThis.renderOffline = renderOffline;
  globalThis.__offlineReport = {
    state: OS,
    render: renderOffline,
    choose,
    resetEpisode,
    rows: offlineRows,
    DATA: D,
  };
  resetEpisode();
})();
