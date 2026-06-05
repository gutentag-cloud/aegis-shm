"""
pinn_model.py
=============
Aegis: physics-informed neural inverse model for sparse-sensor structural
damage localisation.

Formulation
-----------
A simply-supported Euler-Bernoulli beam is load-tested with K known smooth
distributed loads q_k(x).  We measure the deflection only at a handful of sparse,
noisy sensors and want the continuous flexural-rigidity field EI(x)
(equivalently the damage field d(x)=1-EI/EI0).

Aegis represents the unknown stiffness by a single neural network

        EI_phi(x) = EI0 * (1 - d_max * sigmoid(g_phi(x)))            (>0, <= EI0)

and computes the structural response with an *exact, differentiable* beam solver
rather than a soft PDE residual.  Because the bending moment of a simply-supported
beam under a known load is statically determinate,

        m_k(x) = (d^2/dx^2)^{-1} q_k    with   m_k(0)=m_k(L)=0,       (known)

the curvature is  w_k'' = m_k / EI(x)  and the deflection follows by double
integration with the support conditions:

        w_k(x) = P_k(x) - (x/L) P_k(L),   P_k = (d^2/dx^2)^{-1}[ m_k/EI ].

The whole map  EI(x) -> w_k(x_sensor)  is differentiable, so we fit EI(x) to the
sparse measurements by gradient descent under sparsity + total-variation priors
(real damage is localised and piecewise).

This *hard* physics constraint is the key robustness property: driving EI->0 makes
the predicted deflection blow up and is immediately punished by the data term, so
the "zero-stiffness collapse" that destroys naive soft-residual inverse PINNs
cannot occur here (see paper, Sec. Method).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Callable, Sequence


def set_seed(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- #
#  Networks                                                                    #
# --------------------------------------------------------------------------- #
class FourierFeatures(nn.Module):
    """x in [0,L] -> [2x/L-1, sin(m*pi*x/L), cos(m*pi*x/L) for m=1..M]."""
    def __init__(self, n_freq: int = 5, L: float = 1.0):
        super().__init__()
        self.n_freq, self.L = n_freq, L
        self.out_dim = 1 + 2 * n_freq

    def forward(self, x):
        feats = [2.0 * x / self.L - 1.0]
        for m in range(1, self.n_freq + 1):
            feats.append(torch.sin(m * np.pi * x / self.L))
            feats.append(torch.cos(m * np.pi * x / self.L))
        return torch.cat(feats, dim=-1)


class StiffnessNet(nn.Module):
    """EI(x) = EI0 * (1 - d_max * sigmoid(g(x))); starts near healthy (d~0)."""
    def __init__(self, EI0=1.0, d_max=0.9, L=1.0, n_freq=5, width=64, depth=4,
                 init_bias=-3.5):
        super().__init__()
        self.EI0, self.d_max = EI0, d_max
        self.ff = FourierFeatures(n_freq, L)
        layers, d = [], self.ff.out_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        layers += [nn.Linear(d, 1)]
        self.mlp = nn.Sequential(*layers)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
        last = [m for m in self.mlp if isinstance(m, nn.Linear)][-1]
        nn.init.constant_(last.bias, init_bias)

    def damage(self, x):
        return self.d_max * torch.sigmoid(self.mlp(self.ff(x)))

    def forward(self, x):
        return self.EI0 * (1.0 - self.damage(x))


# --------------------------------------------------------------------------- #
#  Differentiable beam operator                                               #
# --------------------------------------------------------------------------- #
def _cumtrapz(f, h):
    """Cumulative trapezoidal integral of f (..., N) along last axis; returns
    g with g[...,0]=0 and same length N."""
    seg = 0.5 * (f[..., 1:] + f[..., :-1]) * h
    z = torch.zeros(f.shape[:-1] + (1,), dtype=f.dtype, device=f.device)
    return torch.cat([z, torch.cumsum(seg, dim=-1)], dim=-1)


def _rev_cumtrapz(f, h):
    """Cumulative trapezoidal integral from the RIGHT support: g(x)=int_x^L f, g(L)=0."""
    return torch.flip(_cumtrapz(torch.flip(f, [-1]), h), [-1])


def double_integral_bc(f, x, L, h):
    """Return g with g'' = f and g(0)=g(L)=0 (simply-supported double integral)."""
    P = _cumtrapz(_cumtrapz(f, h), h)
    return P - (x / L) * P[..., -1:]


def moment_field(q, x, L, h, bc):
    """Statically determinate moment m (m''=q) for the given boundary condition.

    ss          : pinned-pinned, m(0)=m(L)=0
    cantilever  : fixed at x=0, free at x=L, so m(L)=0 and m'(L)=0
    """
    if bc == "ss":
        return double_integral_bc(q, x, L, h)
    if bc == "cantilever":
        return _rev_cumtrapz(_rev_cumtrapz(q, h), h)
    raise ValueError(f"unknown bc {bc!r}")


def deflection_field(kappa, x, L, h, bc):
    """Deflection w (w''=kappa) satisfying the support conditions exactly.

    ss          : w(0)=w(L)=0
    cantilever  : w(0)=w'(0)=0
    """
    if bc == "ss":
        return double_integral_bc(kappa, x, L, h)
    if bc == "cantilever":
        return _cumtrapz(_cumtrapz(kappa, h), h)
    raise ValueError(f"unknown bc {bc!r}")


# --------------------------------------------------------------------------- #
#  Config + model                                                             #
# --------------------------------------------------------------------------- #
@dataclass
class AegisConfig:
    L: float = 1.0
    EI0: float = 1.0
    d_max: float = 0.8
    bc: str = "ss"           # "ss" (pinned-pinned) or "cantilever"
    n_grid: int = 400        # integration / stiffness grid
    width: int = 64
    depth: int = 4
    n_freq: int = 5
    w_l1: float = 1.5e-3     # damage sparsity ("prefer healthy") prior
    w_tv: float = 7e-3       # total variation (piecewise damage)
    use_physics: bool = True  # if False -> data-only smoothing baseline (ablation)
    adam_iters: int = 3000
    lbfgs_iters: int = 120
    lr: float = 5e-3
    seed: int = 0


class AegisPINN:
    def __init__(self, cfg: AegisConfig, load_funcs: Sequence[Callable]):
        self.cfg = cfg
        set_seed(cfg.seed)
        self.K = len(load_funcs)
        self.load_funcs = load_funcs
        self.d_net = StiffnessNet(cfg.EI0, cfg.d_max, cfg.L, cfg.n_freq,
                                  cfg.width, cfg.depth)
        N = cfg.n_grid
        self.x = torch.linspace(0, cfg.L, N).reshape(-1, 1)      # (N,1)
        self.xrow = self.x.reshape(1, -1)                        # (1,N)
        self.h = cfg.L / (N - 1)
        # statically-determinate moment m_k(x) for each load (constant in EI)
        m_rows = []
        for k in range(self.K):
            q = torch.tensor(load_funcs[k](self.x.numpy().ravel()),
                             dtype=torch.float32).reshape(1, -1)
            m_rows.append(moment_field(q, self.xrow, cfg.L, self.h, cfg.bc))
        self.m = torch.cat(m_rows, dim=0)                        # (K,N)
        self.sensors = None

    # ---- differentiable forward map  EI(x) -> w_k(x) ---------------------- #
    def deflection_grid(self):
        """Return w (K,N): deflection of every load case on the grid."""
        EI = self.d_net(self.x).reshape(1, -1)                  # (1,N)
        if self.cfg.use_physics:
            kappa = self.m / EI.clamp_min(1e-4)                 # (K,N) curvature
        else:
            # ablation: ignore physics, treat EI as a free smoother of data only
            kappa = self.m / self.cfg.EI0
        return deflection_field(kappa, self.xrow, self.cfg.L, self.h, self.cfg.bc)

    # ---- data -------------------------------------------------------------- #
    def set_data(self, sensor_x, measurements, w_scale):
        xs = np.asarray(sensor_x, float)
        idx = np.clip((xs / self.h).astype(int), 0, self.cfg.n_grid - 2)
        frac = (xs - idx * self.h) / self.h
        self.s_idx = torch.tensor(idx, dtype=torch.long)
        self.s_frac = torch.tensor(frac, dtype=torch.float32).reshape(1, -1)
        self.meas = torch.tensor(np.vstack([np.asarray(measurements[k]).reshape(1, -1)
                                            for k in range(self.K)]), dtype=torch.float32)
        self.w_scale = torch.tensor(np.asarray(w_scale).reshape(-1, 1),
                                    dtype=torch.float32).clamp_min(1e-6)
        self.sensors = True

    def _sample_sensors(self, w_grid):
        a = w_grid[:, self.s_idx]
        b = w_grid[:, self.s_idx + 1]
        return a * (1 - self.s_frac) + b * self.s_frac          # (K,S)

    # ---- losses ------------------------------------------------------------ #
    def loss(self):
        cfg = self.cfg
        w_grid = self.deflection_grid()
        w_s = self._sample_sensors(w_grid)
        L_data = torch.mean(((w_s - self.meas) / self.w_scale) ** 2)
        d = self.d_net.damage(self.x).reshape(-1)
        L_l1 = torch.mean(torch.abs(d))
        L_tv = torch.mean(torch.abs(d[1:] - d[:-1]))
        total = L_data + cfg.w_l1 * L_l1 + cfg.w_tv * L_tv
        return total, dict(data=L_data.item(), l1=L_l1.item(),
                           tv=L_tv.item(), total=total.item())

    # ---- training ---------------------------------------------------------- #
    def parameters(self):
        return list(self.d_net.parameters())

    def train(self, verbose=True, log_every=1000):
        cfg = self.cfg
        opt = torch.optim.Adam(self.parameters(), lr=cfg.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.adam_iters)
        history = []
        for it in range(cfg.adam_iters):
            opt.zero_grad()
            total, parts = self.loss()
            total.backward()
            opt.step(); sched.step()
            if it % 25 == 0:
                history.append(parts["total"])
            if verbose and (it % log_every == 0 or it == cfg.adam_iters - 1):
                print(f"  [adam {it:5d}] total={parts['total']:.3e} "
                      f"data={parts['data']:.2e} l1={parts['l1']:.2e} "
                      f"tv={parts['tv']:.2e}", flush=True)
        if cfg.lbfgs_iters > 0:
            opt2 = torch.optim.LBFGS(self.parameters(), max_iter=cfg.lbfgs_iters,
                                     history_size=50, line_search_fn="strong_wolfe",
                                     tolerance_grad=1e-10, tolerance_change=1e-12)

            def closure():
                opt2.zero_grad()
                total, _ = self.loss()
                total.backward()
                return total
            opt2.step(closure)
            if verbose:
                _, parts = self.loss()
                print(f"  [lbfgs end] total={parts['total']:.3e} "
                      f"data={parts['data']:.2e}", flush=True)
        return history

    # ---- inference --------------------------------------------------------- #
    @torch.no_grad()
    def predict_damage(self, x_grid):
        xt = torch.tensor(np.asarray(x_grid).reshape(-1, 1), dtype=torch.float32)
        return self.d_net.damage(xt).numpy().ravel()

    @torch.no_grad()
    def predict_EI(self, x_grid):
        xt = torch.tensor(np.asarray(x_grid).reshape(-1, 1), dtype=torch.float32)
        return self.d_net(xt).numpy().ravel()

    @torch.no_grad()
    def predict_deflection(self, k, x_grid):
        w_grid = self.deflection_grid()[k].numpy()
        xg = self.x.numpy().ravel()
        return np.interp(np.asarray(x_grid), xg, w_grid)
