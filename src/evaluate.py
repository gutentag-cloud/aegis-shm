"""
evaluate.py
===========
Quantitative metrics for damage-field recovery.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def _true_region_centers(x_grid, d_true, thresh):
    """Peak location of each connected true-damage region."""
    mask = d_true >= max(thresh, 1e-6)
    centers, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            centers.append(x_grid[i + int(np.argmax(d_true[i:j]))])
            i = j
        else:
            i += 1
    return centers


def localization_error(x_grid, d_pred, d_true, L=1.0, thresh=0.05):
    """Distance (in % of span) from the predicted *global* damage peak to the
    nearest true-damage region.  This is fair across methods: a method that
    sprays spurious damage everywhere (e.g. the noisy curvature baseline) places
    its global peak away from the real damage and is penalized, while a clean
    multi-damage reconstruction whose strongest peak sits on any real damage
    scores well.  False detection of the *other* damages is captured separately
    by the detection-IoU and false-positive-rate metrics."""
    centers = _true_region_centers(x_grid, d_true, thresh)
    if not centers:
        return 0.0
    xp = x_grid[int(np.argmax(d_pred))]
    return float(min(abs(xp - c) for c in centers) / L * 100.0)


def severity_error(d_pred, d_true):
    """Absolute error of peak severity (in %-points of stiffness loss)."""
    return abs(np.max(d_pred) - np.max(d_true)) * 100.0


def rmse(d_pred, d_true):
    return float(np.sqrt(np.mean((d_pred - d_true) ** 2)))


def detection_iou(d_pred, d_true, thresh=0.05):
    """Intersection-over-union of the detected vs. true damaged regions."""
    p = d_pred >= thresh
    t = d_true >= max(thresh, 1e-6)
    inter = np.sum(p & t)
    union = np.sum(p | t)
    return float(inter / union) if union > 0 else 1.0


def false_positive_rate(d_pred, d_true, thresh=0.05):
    """Fraction of *healthy* locations wrongly flagged as damaged."""
    healthy = d_true < 1e-6
    if np.sum(healthy) == 0:
        return 0.0
    return float(np.mean(d_pred[healthy] >= thresh))


def integrated_damage(x_grid, d):
    """Total stiffness loss  integral d(x) dx -- the aggregate severity, which is
    far better constrained by sparse data than the peak (width/depth trade-off)."""
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(_trapz(d, x_grid))


def all_metrics(x_grid, d_pred, d_true, L=1.0, thresh=0.05):
    it = integrated_damage(x_grid, d_true)
    ip = integrated_damage(x_grid, d_pred)
    return dict(
        localization_error_pct=localization_error(x_grid, d_pred, d_true, L),
        severity_error_pct=severity_error(d_pred, d_true),
        damage_rmse=rmse(d_pred, d_true),
        detection_iou=detection_iou(d_pred, d_true, thresh),
        false_positive_rate=false_positive_rate(d_pred, d_true, thresh),
        peak_severity_pred=float(np.max(d_pred)),
        peak_severity_true=float(np.max(d_true)),
        integrated_damage_pred=ip,
        integrated_damage_true=it,
        integrated_damage_error_pct=float(abs(ip - it) / max(it, 1e-6) * 100.0),
    )
