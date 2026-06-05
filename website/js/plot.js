/* plot.js — a tiny dependency-free canvas plotting helper.
 * Supports line series, filled areas (damage bands), scatter markers, axes,
 * grid, and legends. Kept deliberately small so the site runs offline with no
 * external libraries. */
(function (global) {
  "use strict";

  const DPR = Math.max(1, window.devicePixelRatio || 1);

  function makePlot(canvas, opts) {
    opts = opts || {};
    const pad = Object.assign({ l: 52, r: 16, t: 18, b: 40 }, opts.pad || {});
    const ctx = canvas.getContext("2d");

    function resize() {
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, rect.width * DPR);
      canvas.height = Math.max(1, rect.height * DPR);
    }
    resize();

    return {
      ctx,
      resize,
      render(state) {
        resize();
        const W = canvas.width, H = canvas.height;
        const L = pad.l * DPR, R = W - pad.r * DPR;
        const T = pad.t * DPR, B = H - pad.b * DPR;
        ctx.clearRect(0, 0, W, H);

        const xmin = state.xlim[0], xmax = state.xlim[1];
        const ymin = state.ylim[0], ymax = state.ylim[1];
        const sx = (x) => L + ((x - xmin) / (xmax - xmin || 1)) * (R - L);
        const sy = (y) => B - ((y - ymin) / (ymax - ymin || 1)) * (B - T);

        // grid + axes
        ctx.lineWidth = 1 * DPR;
        ctx.strokeStyle = "rgba(148,163,184,0.18)";
        ctx.fillStyle = "#94a3b8";
        ctx.font = `${12 * DPR}px ui-monospace, monospace`;
        ctx.textAlign = "center"; ctx.textBaseline = "top";
        const nx = 5;
        for (let i = 0; i <= nx; i++) {
          const xv = xmin + (i / nx) * (xmax - xmin);
          const X = sx(xv);
          ctx.beginPath(); ctx.moveTo(X, T); ctx.lineTo(X, B); ctx.stroke();
          ctx.fillText(xv.toFixed(2), X, B + 6 * DPR);
        }
        ctx.textAlign = "right"; ctx.textBaseline = "middle";
        const ny = 4;
        for (let i = 0; i <= ny; i++) {
          const yv = ymin + (i / ny) * (ymax - ymin);
          const Y = sy(yv);
          ctx.beginPath(); ctx.moveTo(L, Y); ctx.lineTo(R, Y); ctx.stroke();
          ctx.fillText(yv.toFixed(2), L - 8 * DPR, Y);
        }
        // axis labels
        ctx.fillStyle = "#cbd5e1";
        ctx.textAlign = "center"; ctx.textBaseline = "bottom";
        if (state.xlabel) ctx.fillText(state.xlabel, (L + R) / 2, H - 6 * DPR);
        if (state.ylabel) {
          ctx.save();
          ctx.translate(14 * DPR, (T + B) / 2);
          ctx.rotate(-Math.PI / 2);
          ctx.textBaseline = "top";
          ctx.fillText(state.ylabel, 0, 0);
          ctx.restore();
        }

        // uncertainty bands (fill between ylo and yhi)
        for (const bd of state.bands || []) {
          ctx.fillStyle = bd.color;
          ctx.beginPath();
          for (let i = 0; i < bd.x.length; i++) {
            const X = sx(bd.x[i]), Y = sy(bd.yhi[i]);
            if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
          }
          for (let i = bd.x.length - 1; i >= 0; i--) ctx.lineTo(sx(bd.x[i]), sy(bd.ylo[i]));
          ctx.closePath();
          ctx.fill();
        }

        // filled areas (e.g., damage bands)
        for (const a of state.areas || []) {
          ctx.fillStyle = a.color;
          ctx.beginPath();
          ctx.moveTo(sx(a.x[0]), sy(0));
          for (let i = 0; i < a.x.length; i++) ctx.lineTo(sx(a.x[i]), sy(a.y[i]));
          ctx.lineTo(sx(a.x[a.x.length - 1]), sy(0));
          ctx.closePath();
          ctx.fill();
        }

        // line series
        for (const s of state.series || []) {
          ctx.strokeStyle = s.color;
          ctx.lineWidth = (s.width || 2) * DPR;
          if (s.dash) ctx.setLineDash(s.dash.map((d) => d * DPR));
          ctx.beginPath();
          for (let i = 0; i < s.x.length; i++) {
            const X = sx(s.x[i]), Y = sy(s.y[i]);
            if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
          }
          ctx.stroke();
          ctx.setLineDash([]);
        }

        // scatter markers (e.g., sensors)
        for (const m of state.markers || []) {
          ctx.fillStyle = m.color;
          for (let i = 0; i < m.x.length; i++) {
            ctx.beginPath();
            ctx.arc(sx(m.x[i]), sy(m.y[i]), (m.r || 4) * DPR, 0, 2 * Math.PI);
            ctx.fill();
          }
        }

        // legend
        if (state.legend) {
          ctx.font = `${12 * DPR}px ui-monospace, monospace`;
          ctx.textAlign = "left"; ctx.textBaseline = "middle";
          let ly = T + 6 * DPR;
          for (const item of state.legend) {
            ctx.fillStyle = item.color;
            ctx.fillRect(R - 150 * DPR, ly - 5 * DPR, 16 * DPR, 10 * DPR);
            ctx.fillStyle = "#e2e8f0";
            ctx.fillText(item.label, R - 128 * DPR, ly);
            ly += 18 * DPR;
          }
        }
      },
    };
  }

  global.AegisPlot = { makePlot };
})(window);
