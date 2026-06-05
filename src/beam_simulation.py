"""
beam_simulation.py
==================
High-fidelity Euler-Bernoulli beam finite-element model used as the physical
"ground truth" for the Aegis structural-health-monitoring study.

The module provides:
  * 2-node cubic-Hermite beam elements (transverse displacement + rotation DOFs)
  * Assembly of the global stiffness matrix K(EI) for an arbitrary, spatially
    varying flexural rigidity field EI(x) (damage = local stiffness loss)
  * Consistent nodal load vectors for *smooth distributed* loads q(x)
    (computed by Gauss-Legendre quadrature of the shape functions)
  * Static solves  K u = f  for simply-supported / cantilever beams
  * Smooth Hermite interpolation of the deflection w(x) anywhere in the span
  * A consistent-mass matrix and a modal solver (used for auxiliary analysis)
  * Sparse, noisy "sensor" sampling utilities

All quantities are non-dimensionalised: span L = 1, healthy rigidity EI0 = 1.
The correctness of the element/assembly code is verified in ``__main__``
against the exact analytical deflection of a simply-supported beam under a
sinusoidal load, w(x) = q0/(EI (n*pi/L)^4) * sin(n*pi*x/L).

Author: Aegis project
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Sequence


# --------------------------------------------------------------------------- #
#  Element-level quantities                                                    #
# --------------------------------------------------------------------------- #
def element_stiffness(EI: float, le: float) -> np.ndarray:
    """Cubic-Hermite Euler-Bernoulli beam element stiffness matrix (4x4).

    DOF order: [w_i, theta_i, w_j, theta_j].
    """
    c = EI / le ** 3
    return c * np.array([
        [ 12.0,      6.0 * le,   -12.0,      6.0 * le],
        [ 6.0 * le,  4.0 * le**2, -6.0 * le,  2.0 * le**2],
        [-12.0,     -6.0 * le,    12.0,     -6.0 * le],
        [ 6.0 * le,  2.0 * le**2, -6.0 * le,  4.0 * le**2],
    ])


def element_mass(rhoA: float, le: float) -> np.ndarray:
    """Consistent mass matrix (4x4) for a uniform beam element."""
    c = rhoA * le / 420.0
    return c * np.array([
        [156.0,      22.0 * le,    54.0,     -13.0 * le],
        [22.0 * le,   4.0 * le**2, 13.0 * le, -3.0 * le**2],
        [54.0,       13.0 * le,   156.0,     -22.0 * le],
        [-13.0 * le, -3.0 * le**2,-22.0 * le,  4.0 * le**2],
    ])


def hermite_shape(xi: np.ndarray, le: float):
    """Hermite cubic shape functions on the natural coordinate xi in [0, 1].

    Returns an array of shape (..., 4) giving [N1, N2, N3, N4] where
    N1, N3 interpolate the nodal displacements and N2, N4 the nodal rotations.
    """
    xi = np.asarray(xi, dtype=float)
    N1 = 1.0 - 3.0 * xi**2 + 2.0 * xi**3
    N2 = le * (xi - 2.0 * xi**2 + xi**3)
    N3 = 3.0 * xi**2 - 2.0 * xi**3
    N4 = le * (-xi**2 + xi**3)
    return np.stack([N1, N2, N3, N4], axis=-1)


# 4-point Gauss-Legendre rule on [0, 1] (exact for polynomials up to degree 7)
_GL_X = 0.5 * (np.array([-0.861136311594053, -0.339981043584856,
                          0.339981043584856,  0.861136311594053]) + 1.0)
_GL_W = 0.5 * np.array([0.347854845137454, 0.652145154862546,
                        0.652145154862546, 0.347854845137454])


# --------------------------------------------------------------------------- #
#  Beam model                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class BeamModel:
    """Euler-Bernoulli beam discretised with `n_elem` Hermite elements."""

    L: float = 1.0
    n_elem: int = 200
    EI0: float = 1.0
    rhoA: float = 1.0
    bc: str = "ss"  # "ss" simply-supported, "cantilever" clamped at x=0

    nodes: np.ndarray = field(init=False)
    le: float = field(init=False)
    n_node: int = field(init=False)
    n_dof: int = field(init=False)

    def __post_init__(self):
        self.n_node = self.n_elem + 1
        self.n_dof = 2 * self.n_node
        self.nodes = np.linspace(0.0, self.L, self.n_node)
        self.le = self.L / self.n_elem

    # ---- element rigidity from a damage field ---------------------------- #
    def EI_from_damage(self, damage_elem: np.ndarray) -> np.ndarray:
        """EI_e = EI0 * (1 - d_e), with d_e the per-element damage severity."""
        damage_elem = np.clip(np.asarray(damage_elem, float), 0.0, 0.999)
        return self.EI0 * (1.0 - damage_elem)

    def elem_centers(self) -> np.ndarray:
        return 0.5 * (self.nodes[:-1] + self.nodes[1:])

    # ---- global assembly ------------------------------------------------- #
    def assemble_K(self, EI_elem: np.ndarray) -> np.ndarray:
        K = np.zeros((self.n_dof, self.n_dof))
        for e in range(self.n_elem):
            ke = element_stiffness(EI_elem[e], self.le)
            d = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
            K[np.ix_(d, d)] += ke
        return K

    def assemble_M(self) -> np.ndarray:
        M = np.zeros((self.n_dof, self.n_dof))
        for e in range(self.n_elem):
            me = element_mass(self.rhoA, self.le)
            d = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
            M[np.ix_(d, d)] += me
        return M

    def consistent_load(self, q_func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """Equivalent nodal force vector for a smooth distributed load q(x)."""
        f = np.zeros(self.n_dof)
        for e in range(self.n_elem):
            x0 = self.nodes[e]
            xq = x0 + _GL_X * self.le               # physical Gauss points
            Nq = hermite_shape(_GL_X, self.le)      # (4pts, 4)
            qv = np.asarray(q_func(xq), float)      # (4pts,)
            fe = self.le * (Nq.T @ (_GL_W * qv))    # (4,)
            d = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
            f[d] += fe
        return f

    # ---- boundary conditions --------------------------------------------- #
    def fixed_dofs(self) -> list[int]:
        if self.bc == "ss":                 # w = 0 at both ends
            return [0, self.n_dof - 2]
        elif self.bc == "cantilever":       # w = theta = 0 at x = 0
            return [0, 1]
        raise ValueError(f"unknown bc {self.bc!r}")

    # ---- static solve ---------------------------------------------------- #
    def solve_static(self, EI_elem: np.ndarray,
                     q_func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        K = self.assemble_K(EI_elem)
        f = self.consistent_load(q_func)
        fixed = self.fixed_dofs()
        free = np.setdiff1d(np.arange(self.n_dof), fixed)
        u = np.zeros(self.n_dof)
        u[free] = np.linalg.solve(K[np.ix_(free, free)], f[free])
        return u

    # ---- post-processing -------------------------------------------------- #
    def deflection_at(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Hermite-interpolate the transverse deflection w at points x."""
        x = np.atleast_1d(np.asarray(x, float))
        w = np.zeros_like(x)
        # locate element index for each x
        idx = np.clip((x / self.le).astype(int), 0, self.n_elem - 1)
        for e in np.unique(idx):
            m = idx == e
            xi = (x[m] - self.nodes[e]) / self.le
            N = hermite_shape(xi, self.le)             # (npts, 4)
            d = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
            w[m] = N @ u[d]
        return w

    def nodal_w(self, u: np.ndarray) -> np.ndarray:
        return u[0::2]

    # ---- modal analysis (auxiliary) -------------------------------------- #
    def modal(self, EI_elem: np.ndarray, n_modes: int = 5):
        from scipy.linalg import eigh
        K = self.assemble_K(EI_elem)
        M = self.assemble_M()
        fixed = self.fixed_dofs()
        free = np.setdiff1d(np.arange(self.n_dof), fixed)
        w2, V = eigh(K[np.ix_(free, free)], M[np.ix_(free, free)])
        w2 = np.clip(w2, 0, None)
        omega = np.sqrt(w2[:n_modes])
        modes = np.zeros((self.n_dof, n_modes))
        modes[free, :] = V[:, :n_modes]
        return omega, modes  # omega in rad/(time); freq = omega/2pi


# --------------------------------------------------------------------------- #
#  Load library + sensor sampling                                             #
# --------------------------------------------------------------------------- #
def load_library(L: float = 1.0, amplitude: float = 1.0) -> list[Callable]:
    """A bank of smooth, statically-admissible distributed load shapes.

    Sinusoidal patch loads of increasing wavenumber probe the structure with
    distinct curvature distributions, which is what makes the inverse
    stiffness problem well-posed across the whole span.
    """
    shapes = []
    for n in (1, 2, 3, 4):
        a = amplitude * (n * np.pi / L) ** 4   # normalise so peak w ~ O(1)
        shapes.append(lambda x, n=n, a=a: a * np.sin(n * np.pi * x / L))
    # two asymmetric Gaussian patch loads to add off-symmetry information
    for x0 in (0.33 * L, 0.66 * L):
        s = 0.08 * L
        a = amplitude * 120.0
        shapes.append(lambda x, x0=x0, s=s, a=a:
                      a * np.exp(-0.5 * ((x - x0) / s) ** 2))
    return shapes


def sample_sensors(beam: BeamModel, u: np.ndarray, sensor_x: np.ndarray,
                   noise_frac: float = 0.0, rng: np.random.Generator | None = None):
    """Return (noisy) deflection measurements at the sensor abscissae."""
    w = beam.deflection_at(sensor_x, u)
    if noise_frac > 0:
        rng = rng or np.random.default_rng()
        scale = noise_frac * np.max(np.abs(w)) if np.max(np.abs(w)) > 0 else noise_frac
        w = w + rng.normal(0.0, scale, size=w.shape)
    return w


# --------------------------------------------------------------------------- #
#  Self-test: FEM vs. exact analytical solution                               #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=== Aegis beam FEM verification ===")
    L, EI0 = 1.0, 1.0
    beam = BeamModel(L=L, n_elem=200, EI0=EI0, bc="ss")
    EI_elem = beam.EI_from_damage(np.zeros(beam.n_elem))

    max_rel_err = 0.0
    for n in (1, 2, 3):
        q0 = (n * np.pi / L) ** 4               # so that exact peak deflection = 1
        q = lambda x, n=n, q0=q0: q0 * np.sin(n * np.pi * x / L)
        u = beam.solve_static(EI_elem, q)
        xs = np.linspace(0, L, 501)
        w_fem = beam.deflection_at(xs, u)
        w_exact = (q0 / (EI0 * (n * np.pi / L) ** 4)) * np.sin(n * np.pi * xs / L)
        rel = np.max(np.abs(w_fem - w_exact)) / np.max(np.abs(w_exact))
        max_rel_err = max(max_rel_err, rel)
        print(f"  mode n={n}:  peak w_fem={np.max(w_fem):+.6f}  "
              f"exact={np.max(w_exact):+.6f}  max-rel-err={rel:.2e}")

    # modal frequencies of a uniform SS beam: omega_n = (n*pi/L)^2 sqrt(EI/rhoA)
    omega, _ = beam.modal(EI_elem, n_modes=4)
    print("  modal check (omega_n / (n*pi)^2, should be ~1):")
    for i, om in enumerate(omega, start=1):
        print(f"    n={i}:  ratio={om / (i * np.pi) ** 2:.4f}")

    assert max_rel_err < 1e-3, "FEM deflection does not match analytical solution!"
    print(f"PASS  max relative deflection error = {max_rel_err:.2e}")
