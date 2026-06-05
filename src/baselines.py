"""
baselines.py
============
Classical (non-learning) damage-identification baseline: the modal/static
*curvature method*.

Because a simply-supported beam under a known load is statically determinate,
the bending-moment-like field m_k(x)=EI(x) w_k''(x) is known exactly from
equilibrium (m_k'' = q_k, m_k(0)=m_k(L)=0).  The classical method estimates the
curvature w_k''(x) by numerically double-differentiating an interpolant of the
sparse deflection measurements and then forms

        EI(x) = argmin_EI  sum_k ( m_k(x) - EI * w_k''(x) )^2 ,

which is the standard curvature-based stiffness identification.  Its weakness --
amplification of measurement noise by second differentiation of sparse data --
is exactly what the physics-informed network is designed to overcome.
"""
from __future__ import annotations

import warnings
import numpy as np
from scipy.interpolate import UnivariateSpline


def known_moment(q_func, x, L, n_quad=2001):
    """Statically-determinate field m(x)=EI w'' for a simply-supported beam,
    solved from m'' = q with m(0)=m(L)=0."""
    s = np.linspace(0, L, n_quad)
    qs = np.asarray(q_func(s), float)
    # particular solution of m'' = q with m(0)=0, m'(0)=0 via double integration
    I1 = np.concatenate([[0], np.cumsum(0.5 * (qs[1:] + qs[:-1]) * np.diff(s))])
    m_part = np.concatenate([[0], np.cumsum(0.5 * (I1[1:] + I1[:-1]) * np.diff(s))])
    # enforce m(L)=0 by adding linear term  a*x  (a chosen so m(L)=0)
    a = -m_part[-1] / L
    m = m_part + a * s
    return np.interp(x, s, m)


def curvature_baseline(data, smooth=None):
    """Return (EI_est, damage_est) on data['x_dense'] using the curvature method."""
    beam = data["beam"]
    x = data["x_dense"]
    L, EI0 = beam.L, beam.EI0
    xs = data["sensor_x"]
    # include the supports (w=0) as known data points
    xfull = np.concatenate([[0.0], xs, [L]])
    order = np.argsort(xfull)
    xfull = xfull[order]

    num = np.zeros_like(x)   # sum_k m_k * kappa_k
    den = np.zeros_like(x)   # sum_k kappa_k^2
    for k, q in enumerate(data["loads"]):
        wfull = np.concatenate([[0.0], data["measurements"][k], [0.0]])[order]
        # smoothing spline; default smoothing scaled to #points
        s = smooth if smooth is not None else len(xfull) * (1e-3) ** 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                spl = UnivariateSpline(xfull, wfull, k=3, s=s)
            except Exception:
                spl = UnivariateSpline(xfull, wfull, k=3, s=0)
        kappa = spl.derivative(2)(x)
        m = known_moment(q, x, L)
        num += m * kappa
        den += kappa ** 2
    EI_est = np.where(den > 1e-9, num / np.maximum(den, 1e-9), EI0)
    EI_est = np.clip(EI_est, 0.05 * EI0, 1.5 * EI0)
    damage_est = np.clip(1.0 - EI_est / EI0, 0.0, 1.0)
    return EI_est, damage_est
