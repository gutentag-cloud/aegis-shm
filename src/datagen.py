"""
datagen.py
==========
Builds physics-exact datasets for the Aegis study from the FEM beam model:
ground-truth damage fields, multi-load-case forward solves, and sparse noisy
sensor measurements.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Sequence

from beam_simulation import BeamModel, load_library, sample_sensors


@dataclass
class DamageSegment:
    center: float       # fraction of span [0,1]
    half_width: float   # fraction of span
    severity: float     # stiffness loss fraction in [0,1)


def damage_field(beam: BeamModel, segments: Sequence[DamageSegment]) -> np.ndarray:
    """Per-element damage severity from a list of rectangular damage segments."""
    xc = beam.elem_centers() / beam.L
    d = np.zeros(beam.n_elem)
    for s in segments:
        mask = np.abs(xc - s.center) <= s.half_width
        d[mask] = np.maximum(d[mask], s.severity)
    return d


def damage_on_grid(x_grid: np.ndarray, segments: Sequence[DamageSegment],
                   L: float = 1.0) -> np.ndarray:
    xn = x_grid / L
    d = np.zeros_like(xn, dtype=float)
    for s in segments:
        mask = np.abs(xn - s.center) <= s.half_width
        d[mask] = np.maximum(d[mask], s.severity)
    return d


def build_dataset(beam: BeamModel, segments: Sequence[DamageSegment],
                  n_sensors: int = 7, noise_frac: float = 0.02,
                  amplitude: float = 1.0, seed: int = 0,
                  sensor_x: np.ndarray | None = None):
    """Forward-simulate all load cases and return the full data bundle."""
    rng = np.random.default_rng(seed)
    d_elem = damage_field(beam, segments)
    EI_elem = beam.EI_from_damage(d_elem)
    loads = load_library(beam.L, amplitude)

    if sensor_x is None:
        if beam.bc == "cantilever":
            # exclude only the clamped end (w=0); include the free end (max signal)
            sensor_x = np.linspace(0, beam.L, n_sensors + 1)[1:]
        else:
            # interior sensors, evenly spaced (avoid the supports where w=0)
            sensor_x = np.linspace(0, beam.L, n_sensors + 2)[1:-1]

    x_dense = np.linspace(0, beam.L, 400)
    measurements, w_true_dense, w_scale = [], [], []
    for q in loads:
        u = beam.solve_static(EI_elem, q)
        w_dense = beam.deflection_at(x_dense, u)
        w_true_dense.append(w_dense)
        scale = np.max(np.abs(w_dense)) + 1e-12
        w_scale.append(scale)
        meas = sample_sensors(beam, u, sensor_x, noise_frac=noise_frac, rng=rng)
        measurements.append(meas)

    return dict(
        beam=beam, segments=list(segments), loads=loads,
        d_elem=d_elem, EI_elem=EI_elem,
        sensor_x=sensor_x, measurements=measurements,
        w_scale=np.array(w_scale), x_dense=x_dense,
        w_true_dense=w_true_dense,
        d_true_dense=damage_on_grid(x_dense, segments, beam.L),
        EI_true_dense=beam.EI0 * (1.0 - damage_on_grid(x_dense, segments, beam.L)),
        noise_frac=noise_frac, n_sensors=len(sensor_x),
    )
