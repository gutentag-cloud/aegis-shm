"""Verify the differentiable beam solver (moment -> curvature -> deflection)
reproduces the independent FEM deflection for the TRUE stiffness field, for
both simply-supported and cantilever boundary conditions."""
import numpy as np, torch
from beam_simulation import BeamModel, load_library
from datagen import damage_on_grid, DamageSegment
from pinn_model import moment_field, deflection_field

L, EI0, N = 1.0, 1.0, 400
segs = [DamageSegment(0.6, 0.04, 0.40), DamageSegment(0.30, 0.04, 0.35)]
x = np.linspace(0, L, N)
xrow = torch.tensor(x.reshape(1, -1), dtype=torch.float64)
h = L / (N - 1)
EI_true = EI0 * (1.0 - damage_on_grid(x, segs, L))
EI_t = torch.tensor(EI_true.reshape(1, -1), dtype=torch.float64)
loads = load_library(L, 1.0)

overall = 0.0
for bc in ("ss", "cantilever"):
    beam = BeamModel(L=L, n_elem=400, EI0=EI0, bc=bc)
    d_elem = 1.0 - np.interp(beam.elem_centers(), x, EI_true) / EI0
    EI_elem = beam.EI_from_damage(d_elem)
    max_err = 0.0
    for k, q in enumerate(loads):
        qv = torch.tensor(q(x).reshape(1, -1), dtype=torch.float64)
        m = moment_field(qv, xrow, L, h, bc)
        w_solver = deflection_field(m / EI_t, xrow, L, h, bc).numpy().ravel()
        u = beam.solve_static(EI_elem, q)
        w_fem = beam.deflection_at(x, u)
        scale = np.max(np.abs(w_fem)) + 1e-12
        max_err = max(max_err, np.max(np.abs(w_solver - w_fem)) / scale)
    overall = max(overall, max_err)
    print(f"  bc={bc:11s}  max relative deflection error vs FEM: {max_err:.2e}")

assert overall < 5e-3, "differentiable solver does not match FEM!"
print("PASS - differentiable physics solver is consistent with the FEM (both BCs).")
