"""Test suite for Aegis. Run with:  pytest -q

Covers the two physics engines (FEM and the differentiable solver, both
boundary conditions), the metrics, and an end-to-end inverse-recovery check.
"""
import numpy as np
import torch

from beam_simulation import BeamModel, load_library
from datagen import build_dataset, DamageSegment, damage_on_grid
from pinn_model import AegisPINN, AegisConfig, moment_field, deflection_field
from evaluate import all_metrics


def test_fem_matches_analytical():
    """FEM deflection of a SS beam under a sinusoidal load matches closed form."""
    beam = BeamModel(L=1.0, n_elem=200, EI0=1.0, bc="ss")
    EI = beam.EI_from_damage(np.zeros(beam.n_elem))
    for n in (1, 2, 3):
        q0 = (n * np.pi) ** 4
        q = lambda x, n=n, q0=q0: q0 * np.sin(n * np.pi * x)
        u = beam.solve_static(EI, q)
        xs = np.linspace(0, 1, 201)
        w_fem = beam.deflection_at(xs, u)
        w_exact = (q0 / (n * np.pi) ** 4) * np.sin(n * np.pi * xs)
        rel = np.max(np.abs(w_fem - w_exact)) / np.max(np.abs(w_exact))
        assert rel < 1e-3, f"mode {n}: rel err {rel:.2e}"


def test_solver_matches_fem_both_bcs():
    """Differentiable solver reproduces the FEM deflection for SS and cantilever."""
    L, N = 1.0, 400
    x = np.linspace(0, L, N)
    xrow = torch.tensor(x.reshape(1, -1), dtype=torch.float64)
    h = L / (N - 1)
    segs = [DamageSegment(0.6, 0.04, 0.4), DamageSegment(0.3, 0.04, 0.35)]
    EI_true = 1.0 - damage_on_grid(x, segs, L)
    EI_t = torch.tensor(EI_true.reshape(1, -1), dtype=torch.float64)
    loads = load_library(L, 1.0)
    for bc in ("ss", "cantilever"):
        beam = BeamModel(L=L, n_elem=400, EI0=1.0, bc=bc)
        EI_elem = beam.EI_from_damage(1.0 - np.interp(beam.elem_centers(), x, EI_true))
        for q in loads:
            qv = torch.tensor(q(x).reshape(1, -1), dtype=torch.float64)
            m = moment_field(qv, xrow, L, h, bc)
            w_solver = deflection_field(m / EI_t, xrow, L, h, bc).numpy().ravel()
            w_fem = beam.deflection_at(x, beam.solve_static(EI_elem, q))
            rel = np.max(np.abs(w_solver - w_fem)) / (np.max(np.abs(w_fem)) + 1e-12)
            assert rel < 5e-3, f"bc={bc}: rel err {rel:.2e}"


def test_metrics_perfect_on_identity():
    x = np.linspace(0, 1, 400)
    d = damage_on_grid(x, [DamageSegment(0.6, 0.04, 0.4)], 1.0)
    m = all_metrics(x, d, d)
    assert m["localization_error_pct"] == 0.0
    assert m["severity_error_pct"] < 1e-9
    assert m["detection_iou"] > 0.99
    assert m["false_positive_rate"] == 0.0


def test_inverse_recovery_end_to_end():
    """A short training run localizes a single damage to within a few % of span."""
    beam = BeamModel(L=1.0, n_elem=200, EI0=1.0, bc="ss")
    data = build_dataset(beam, [DamageSegment(0.6, 0.04, 0.4)],
                         n_sensors=7, noise_frac=0.02, seed=1)
    cfg = AegisConfig(adam_iters=1200, lbfgs_iters=30, seed=0)
    pinn = AegisPINN(cfg, data["loads"])
    pinn.set_data(data["sensor_x"], data["measurements"], data["w_scale"])
    pinn.train(verbose=False)
    d_pred = pinn.predict_damage(data["x_dense"])
    m = all_metrics(data["x_dense"], d_pred, data["d_true_dense"])
    assert m["localization_error_pct"] < 8.0, m
    assert d_pred.max() > 0.15, "damage not detected"
    # healthy regions stay near zero (few false positives)
    assert m["false_positive_rate"] < 0.25, m


def test_stiffness_bounds():
    """Recovered EI stays in the physical range (0, EI0]."""
    beam = BeamModel(bc="ss")
    data = build_dataset(beam, [DamageSegment(0.5, 0.04, 0.3)], 7, 0.02, seed=1)
    cfg = AegisConfig(adam_iters=300, lbfgs_iters=0)
    pinn = AegisPINN(cfg, data["loads"])
    pinn.set_data(data["sensor_x"], data["measurements"], data["w_scale"])
    pinn.train(verbose=False)
    EI = pinn.predict_EI(data["x_dense"])
    assert EI.min() > 0.0 and EI.max() <= 1.0 + 1e-6
