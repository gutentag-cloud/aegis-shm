/* app.js — Aegis site logic: tab routing, live forward-physics simulator,
 * and the scenario explorer driven by real saved model outputs. */
(function () {
  "use strict";
  const R = window.AEGIS_RESULTS;            // may be null before first run
  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));

  // ---- palette (matches CSS) ------------------------------------------- //
  const C = { gold: "#f4c45a", blue: "#5aa6f4", green: "#56d39a", red: "#f47878",
              grey: "rgba(148,163,184,0.35)", greyLine: "#94a3b8" };

  // ---------------------------------------------------------------- tabs //
  const renderers = {};
  function showTab(name) {
    $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
    $$(".panel").forEach((p) => p.classList.toggle("active", p.id === name));
    if (renderers[name]) renderers[name]();
    try { history.replaceState(null, "", "#" + name); } catch (e) {}
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
  $$(".tab[data-tab]").forEach((t) => t.addEventListener("click", () => showTab(t.dataset.tab)));
  $$("[data-goto]").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.goto)));

  // ------------------------------------------------------- headline stats //
  function initHeadline() {
    if (!R || !R.headline) return;
    const h = R.headline;
    const ls = h.loc_error_std != null ? "±" + h.loc_error_std.toFixed(1) : "";
    if ($("#hs-loc")) $("#hs-loc").textContent = h.loc_error_pct.toFixed(1) + ls + "%";
    if ($("#hs-auc")) $("#hs-auc").textContent = h.detection_auc != null ? h.detection_auc.toFixed(2) : "—";
    if ($("#hs-sensors")) $("#hs-sensors").textContent = h.n_sensors;
    if ($("#hs-improve")) $("#hs-improve").textContent = h.improvement_factor.toFixed(1) + "× better";
    if ($("#foot-generated") && R.meta) $("#foot-generated").textContent = "generated " + R.meta.generated;
  }

  // ------------------------------------------------------- hero canvas ---- //
  function initHero() {
    const cv = $("#hero-canvas");
    if (!cv) return;
    const plot = window.AegisPlot.makePlot(cv);
    renderers.overview = () => {
      let x, dTrue, dPred, sx;
      if (R && R.scenarios && R.scenarios.length) {
        const s = R.scenarios.find((z) => z.category === "single") || R.scenarios[0];
        x = s.x; dTrue = s.d_true; dPred = s.d_pred; sx = s.sensor_x;
      } else {
        x = []; dTrue = []; dPred = [];
        for (let i = 0; i <= 200; i++) { const xi = i / 200; x.push(xi);
          dTrue.push(Math.abs(xi - 0.6) < 0.04 ? 0.4 : 0);
          dPred.push(0.4 * Math.exp(-0.5 * Math.pow((xi - 0.6) / 0.05, 2))); }
        sx = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875];
      }
      const ymax = Math.max(0.1, Math.max(...dTrue, ...dPred) * 1.25);
      plot.render({
        xlim: [0, 1], ylim: [0, ymax], xlabel: "position  x / L", ylabel: "damage  d",
        areas: [{ x, y: dTrue, color: C.grey }],
        series: [{ x, y: dPred, color: C.gold, width: 2.5 }],
        markers: [{ x: sx, y: sx.map(() => 0.0), color: C.blue, r: 4 }],
        legend: [{ label: "true", color: C.greyLine }, { label: "Aegis", color: C.gold },
                 { label: "sensors", color: C.blue }],
      });
    };
    renderers.overview();
  }

  // -------------------------------------------------- how-it-works sim ---- //
  function initSim() {
    if (!window.AegisBeam || !$("#defl-canvas")) return;
    const beam = new window.AegisBeam.Beam(1.0, 60, 1.0);
    const loads = window.AegisBeam.loadLibrary(1.0, 1.0);
    const sel = $("#s-load");
    loads.forEach((l, i) => { const o = document.createElement("option");
      o.value = i; o.textContent = l.name; sel.appendChild(o); });
    sel.value = 0;

    const deflPlot = window.AegisPlot.makePlot($("#defl-canvas"));
    const stiffPlot = window.AegisPlot.makePlot($("#stiff-canvas"));

    function read() {
      return { loc: +$("#r-loc").value, sev: +$("#r-sev").value,
               wid: +$("#r-wid").value, sen: +$("#r-sen").value, load: +sel.value };
    }
    function draw() {
      const p = read();
      $("#o-loc").textContent = p.loc.toFixed(2);
      $("#o-sev").textContent = Math.round(p.sev * 100) + "%";
      $("#o-wid").textContent = p.wid.toFixed(2);
      $("#o-sen").textContent = p.sen;
      const segs = [{ center: p.loc, halfWidth: p.wid / 2, severity: p.sev }];
      const EIe = beam.EIelem(segs);
      const EIeHealthy = beam.EIelem([]);
      const q = loads[p.load].f;
      const u = beam.solveStatic(EIe, q);
      const uH = beam.solveStatic(EIeHealthy, q);

      const xs = []; const w = []; const wH = []; const EIx = []; const dmg = [];
      const N = 200;
      for (let i = 0; i <= N; i++) {
        const x = i / N; xs.push(x);
        w.push(beam.deflectionAt(x, u));
        wH.push(beam.deflectionAt(x, uH));
        let d = 0;
        for (const s of segs) if (Math.abs(x - s.center) <= s.halfWidth) d = Math.max(d, s.severity);
        dmg.push(d); EIx.push(1 - d);
      }
      // sensors
      const sx = []; const sw = [];
      for (let i = 1; i <= p.sen; i++) { const x = i / (p.sen + 1); sx.push(x);
        sw.push(beam.deflectionAt(x, u)); }

      const allw = w.concat(wH);
      const wmin = Math.min(...allw), wmax = Math.max(...allw);
      const pad = 0.12 * (wmax - wmin || 1);
      deflPlot.render({
        xlim: [0, 1], ylim: [wmin - pad, wmax + pad],
        xlabel: "position  x / L", ylabel: "deflection  w",
        series: [{ x: xs, y: wH, color: C.greyLine, width: 1.5, dash: [5, 4] },
                 { x: xs, y: w, color: C.blue, width: 2.5 }],
        markers: [{ x: sx, y: sw, color: C.gold, r: 4.5 }],
        legend: [{ label: "healthy", color: C.greyLine }, { label: "damaged", color: C.blue },
                 { label: "sensors", color: C.gold }],
      });
      stiffPlot.render({
        xlim: [0, 1], ylim: [0, 1.15],
        xlabel: "position  x / L", ylabel: "stiffness  EI / EI₀",
        areas: [{ x: xs, y: dmg.map((d) => 1), color: "rgba(86,211,154,0.08)" }],
        series: [{ x: xs, y: EIx, color: C.green, width: 2.5 }],
        legend: [{ label: "true EI(x)", color: C.green }],
      });
    }
    ["#r-loc", "#r-sev", "#r-wid", "#r-sen"].forEach((id) =>
      $(id).addEventListener("input", draw));
    sel.addEventListener("change", draw);
    renderers.how = draw;
    draw();
  }

  // ------------------------------------------------------ live demo ------- //
  function initDemo() {
    const sel = $("#scenario-select");
    if (!sel) return;
    if (!R || !R.scenarios) {
      $("#scn-meta").innerHTML = "Run <code>python src/run_experiments.py</code> to generate live results.";
      return;
    }
    R.scenarios.forEach((s, i) => { const o = document.createElement("option");
      o.value = i; o.textContent = s.label; sel.appendChild(o); });

    const dmgPlot = window.AegisPlot.makePlot($("#demo-damage"));
    const deflPlot = window.AegisPlot.makePlot($("#demo-defl"));

    function mc(k, v, alt) {
      return `<div class="mc"><div class="v${alt ? " alt" : ""}">${v}</div><div class="k">${k}</div></div>`;
    }
    function draw() {
      const s = R.scenarios[+sel.value];
      const m = s.metrics, mb = s.metrics_baseline, ms = s.metrics_std || {};
      const std = s.d_pred_std || s.d_pred.map(() => 0);
      const lo = s.d_pred.map((v, i) => Math.max(0, v - std[i]));
      const hi = s.d_pred.map((v, i) => v + std[i]);
      const locstd = (ms.localization_error_pct || 0).toFixed(1);
      $("#scn-meta").innerHTML =
        `<b>${s.label}</b><br/>${s.n_sensors} sensors · ${Math.round(s.noise_frac * 100)}% noise · `
        + `${R.meta.n_load_cases} load cases · ${s.n_seeds || R.meta.n_seeds || 1} noise seeds`;
      $("#metric-cards").innerHTML =
        mc("Localization err", m.localization_error_pct.toFixed(1) + "±" + locstd + "%") +
        mc("Severity err", m.severity_error_pct.toFixed(1) + " pts") +
        mc("Detection IoU", m.detection_iou.toFixed(2)) +
        mc("Baseline loc err", mb.localization_error_pct.toFixed(1) + "%", true);

      const ymax = Math.max(0.12, Math.max(...hi, ...s.d_true, ...s.d_baseline) * 1.25);
      dmgPlot.render({
        xlim: [0, 1], ylim: [0, ymax], xlabel: "position  x / L", ylabel: "damage  d = 1 − EI/EI₀",
        areas: [{ x: s.x, y: s.d_true, color: C.grey }],
        bands: [{ x: s.x, ylo: lo, yhi: hi, color: "rgba(244,196,90,0.22)" }],
        series: [{ x: s.x, y: s.d_baseline, color: C.red, width: 1.8, dash: [6, 4] },
                 { x: s.x, y: s.d_pred, color: C.gold, width: 2.8 }],
        markers: [{ x: s.sensor_x, y: s.sensor_x.map(() => 0), color: C.blue, r: 4 }],
        legend: [{ label: "true", color: C.greyLine }, { label: "Aegis ±1σ", color: C.gold },
                 { label: "classical", color: C.red }, { label: "sensors", color: C.blue }],
      });
      const d = s.defl;
      const allw = d.w_true.concat(d.w_pred);
      const wmin = Math.min(...allw), wmax = Math.max(...allw);
      const pad = 0.12 * (wmax - wmin || 1);
      deflPlot.render({
        xlim: [0, 1], ylim: [wmin - pad, wmax + pad],
        xlabel: "position  x / L", ylabel: "deflection  w",
        series: [{ x: d.x, y: d.w_true, color: C.greyLine, width: 1.5, dash: [5, 4] },
                 { x: d.x, y: d.w_pred, color: C.gold, width: 2.2 }],
        markers: [{ x: d.sensor_x, y: d.sensor_w, color: C.blue, r: 4.5 }],
        legend: [{ label: "true w (load: " + d.load_name + ")", color: C.greyLine },
                 { label: "Aegis fit", color: C.gold }, { label: "measurements", color: C.blue }],
      });
    }
    sel.addEventListener("change", draw);
    renderers.demo = draw;
    draw();
  }

  // -------------------------------------------------------- results ------ //
  function initResults() {
    if (!R) return;
    if (R.table && $("#results-table")) {
      const cols = ["scenario", "sensors", "noise", "loc err %", "sev err pts",
                    "IoU", "FPR", "baseline loc %"];
      $("#results-table thead").innerHTML =
        "<tr>" + cols.map((c) => `<th>${c}</th>`).join("") + "</tr>";
      $("#results-table tbody").innerHTML = R.table.map((row) =>
        "<tr>" + [row.scenario, row.n_sensors, Math.round(row.noise * 100) + "%",
          row.loc_err.toFixed(1) + "±" + (row.loc_err_std || 0).toFixed(1),
          row.sev_err.toFixed(1), row.iou.toFixed(2),
          (row.fpr * 100).toFixed(0) + "%", row.loc_err_base.toFixed(1)]
          .map((c, i) => `<td>${c}</td>`).join("") + "</tr>").join("");
    }
    if (R.figures && $("#figgrid")) {
      $("#figgrid").innerHTML = R.figures.map((f) =>
        `<div class="figitem"><img src="assets/${f.file}" alt="${f.caption}"/>` +
        `<div class="cap">${f.caption}</div></div>`).join("");
    }
  }

  // ---------------------------------------------------------- paper ------ //
  function initPaper() {
    if (R && R.abstract && $("#abstract-box")) {
      $("#abstract-box").innerHTML = "<h3>Abstract</h3><p>" + R.abstract + "</p>";
    }
  }

  // ------------------------------------------------------------ boot ----- //
  initHeadline();
  initHero();
  initSim();
  initDemo();
  initResults();
  initPaper();
  if (!R && $("#foot-generated"))
    $("#foot-generated").textContent = "results not yet generated";

  // honor a deep link such as index.html#demo on first load
  const boot = (location.hash || "#overview").slice(1);
  if (document.getElementById(boot) && boot !== "overview") showTab(boot);
})();
