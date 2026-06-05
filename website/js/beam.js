/* beam.js — live Euler–Bernoulli beam FEM in the browser.
 * A faithful port of src/beam_simulation.py (cubic-Hermite elements), used by
 * the "How it works" interactive so visitors can apply loads, inject damage,
 * and watch the deflection and sensor readings update in real time.
 * No external dependencies — runs offline. */
(function (global) {
  "use strict";

  function elementStiffness(EI, le) {
    const c = EI / (le * le * le);
    const le2 = le * le;
    return [
      [12 * c, 6 * le * c, -12 * c, 6 * le * c],
      [6 * le * c, 4 * le2 * c, -6 * le * c, 2 * le2 * c],
      [-12 * c, -6 * le * c, 12 * c, -6 * le * c],
      [6 * le * c, 2 * le2 * c, -6 * le * c, 4 * le2 * c],
    ];
  }

  function hermite(xi, le) {
    const x2 = xi * xi, x3 = x2 * xi;
    return [
      1 - 3 * x2 + 2 * x3,
      le * (xi - 2 * x2 + x3),
      3 * x2 - 2 * x3,
      le * (-x2 + x3),
    ];
  }

  // 4-point Gauss–Legendre on [0,1]
  const GLX = [0.0694318442, 0.3300094782, 0.6699905218, 0.9305681558];
  const GLW = [0.1739274226, 0.3260725774, 0.3260725774, 0.1739274226];

  // Dense linear solve A x = b (Gaussian elimination, partial pivoting)
  function solve(A, b) {
    const n = b.length;
    const M = A.map((row, i) => row.slice().concat(b[i]));
    for (let col = 0; col < n; col++) {
      let piv = col;
      for (let r = col + 1; r < n; r++)
        if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
      const tmp = M[col]; M[col] = M[piv]; M[piv] = tmp;
      const d = M[col][col] || 1e-30;
      for (let r = 0; r < n; r++) {
        if (r === col) continue;
        const f = M[r][col] / d;
        for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
      }
    }
    return M.map((row, i) => row[n] / (row[i] || 1e-30));
  }

  class Beam {
    constructor(L, nElem, EI0) {
      this.L = L; this.nElem = nElem; this.EI0 = EI0;
      this.nNode = nElem + 1;
      this.nDof = 2 * this.nNode;
      this.le = L / nElem;
      this.nodes = Array.from({ length: this.nNode }, (_, i) => i * this.le);
    }

    // damageSegments: [{center, halfWidth, severity}] in fraction of span
    EIelem(damageSegments) {
      const EI = new Array(this.nElem);
      for (let e = 0; e < this.nElem; e++) {
        const xc = ((e + 0.5) * this.le) / this.L;
        let d = 0;
        for (const s of damageSegments)
          if (Math.abs(xc - s.center) <= s.halfWidth) d = Math.max(d, s.severity);
        EI[e] = this.EI0 * (1 - d);
      }
      return EI;
    }

    solveStatic(EIelem, qFunc) {
      const n = this.nDof;
      const K = Array.from({ length: n }, () => new Array(n).fill(0));
      const f = new Array(n).fill(0);
      for (let e = 0; e < this.nElem; e++) {
        const ke = elementStiffness(EIelem[e], this.le);
        const dofs = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3];
        for (let a = 0; a < 4; a++) {
          for (let b = 0; b < 4; b++) K[dofs[a]][dofs[b]] += ke[a][b];
          // consistent load vector
          let fe = 0;
          for (let g = 0; g < 4; g++) {
            const xq = this.nodes[e] + GLX[g] * this.le;
            fe += GLW[g] * hermite(GLX[g], this.le)[a] * qFunc(xq);
          }
          f[dofs[a]] += this.le * fe;
        }
      }
      // simply supported: w=0 at both ends
      const fixed = new Set([0, this.nDof - 2]);
      const free = [];
      for (let i = 0; i < n; i++) if (!fixed.has(i)) free.push(i);
      const Kff = free.map((i) => free.map((j) => K[i][j]));
      const ff = free.map((i) => f[i]);
      const uf = solve(Kff, ff);
      const u = new Array(n).fill(0);
      free.forEach((idx, i) => (u[idx] = uf[i]));
      return u;
    }

    deflectionAt(x, u) {
      let e = Math.min(Math.floor(x / this.le), this.nElem - 1);
      if (e < 0) e = 0;
      const xi = (x - this.nodes[e]) / this.le;
      const N = hermite(xi, this.le);
      const dofs = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3];
      return N[0] * u[dofs[0]] + N[1] * u[dofs[1]] +
             N[2] * u[dofs[2]] + N[3] * u[dofs[3]];
    }
  }

  // Smooth distributed-load library matching the Python load_library()
  function loadLibrary(L, amplitude) {
    const loads = [];
    for (const n of [1, 2, 3, 4]) {
      const a = amplitude * Math.pow((n * Math.PI) / L, 4);
      loads.push({ name: `sin ${n}`, f: (x) => a * Math.sin((n * Math.PI * x) / L) });
    }
    for (const x0 of [0.33 * L, 0.66 * L]) {
      const s = 0.08 * L, a = amplitude * 120;
      loads.push({ name: `patch @${(x0 / L).toFixed(2)}`,
                   f: (x) => a * Math.exp(-0.5 * Math.pow((x - x0) / s, 2)) });
    }
    return loads;
  }

  global.AegisBeam = { Beam, loadLibrary };
})(window);
