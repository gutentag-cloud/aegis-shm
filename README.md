# Aegis — Physics-Informed Neural Networks for Sparse-Sensor Structural Damage Localization

> Find hidden damage in a beam or bridge — its **location and severity** — from
> only a handful of sparse, noisy sensors, by embedding the Euler–Bernoulli beam
> equation directly into a neural network.

Aegis is a complete, reproducible research artifact (paper + code + interactive
website) prepared for the International Science and Engineering Fair (ISEF),
**Engineering — Physics**.

---

## Why it matters

Aging beams and bridges accumulate *localized stiffness loss* (cracks, corrosion,
delamination) long before visible failure. Dense sensor arrays can detect it but
are expensive; classical curvature/modal methods amplify measurement noise when
sensors are sparse. Aegis turns the noise-amplifying numerical-differentiation
problem into a **regularized, physics-constrained fitting problem**, so a few
sensors suffice.

## Key results (from `results/metrics.json`)

- **3.8 ± 0.7 % of span** mean localization error (over 4 independent noise
  realizations) across the full 13-scenario suite — single *and* multi-damage.
- **Detection AUC = 1.00** for telling healthy from damaged beams (20 % and 40 %
  damage), from the same sparse sensors.
- Stays at **3.6–3.9 %** with as few as **4 sensors** and up to **8 % noise**.
- **≈12× lower** localization error than a classical curvature baseline, and every
  reconstruction carries a **calibrated ±1σ uncertainty band**.
- Differentiable solver verified against FEM for **both** simply-supported and
  cantilever beams (relative error < 10⁻³).
- Fully reproducible from **exact finite-element physics** — no proprietary data.

Reproduce with `bash run_all.sh` (or follow the steps below).

## How it works (one paragraph)

A neural network represents the unknown stiffness field `EI(x)`. The beam's
response is computed by an **exact, differentiable** Euler–Bernoulli solver
(using the statically determinate bending moment of a load-tested simply
supported beam). Because the whole map `EI(x) → deflection` is differentiable,
the stiffness field is fit to the sparse measurements by gradient descent under
sparsity + total-variation priors (damage is localized and piecewise). This
*hard* physics constraint makes the method immune to the "zero-stiffness
collapse" that destroys naive soft-residual inverse PINNs. Several load cases with
different curvature patterns make the inverse problem well posed across the whole
span.

## Repository layout

```
aegis-shm/
├── src/
│   ├── beam_simulation.py   # Euler–Bernoulli FEM (verified vs analytical soln)
│   ├── pinn_model.py        # Aegis model + differentiable beam solver
│   ├── datagen.py           # damage scenarios + sparse noisy measurements
│   ├── baselines.py         # classical curvature-method baseline
│   ├── evaluate.py          # metrics (localization, severity, IoU, FPR)
│   ├── run_experiments.py   # full experiment suite → results JSON
│   ├── make_figures.py      # publication figures + LaTeX table/macros
│   └── check_solver.py      # solver-vs-FEM consistency test
├── paper/
│   ├── aegis.tex            # journal-style paper (compile with tectonic)
│   ├── aegis.pdf            # compiled paper
│   └── figures/             # figures used by the paper
├── website/                 # interactive demo (open website/index.html)
├── results/                 # results_full.json, metrics.json, figures
├── models/                  # (optional) saved checkpoints
└── requirements.txt
```

## Reproduce everything

```bash
pip install -r requirements.txt

# 1. verify the physics engines
python src/beam_simulation.py      # FEM vs analytical (machine precision)
python src/check_solver.py         # differentiable solver vs FEM

# 2. run the full experiment suite (~5–7 min on a laptop CPU)
python src/run_experiments.py      # → results/*.json, website/data/results.js

# 3. regenerate all figures + the paper's table/macros
python src/make_figures.py

# 4. rebuild the paper
cd paper && tectonic aegis.tex     # → aegis.pdf

# 5. open the website
open website/index.html
```

A quick smoke run is available with `python src/run_experiments.py --quick`.

## Testing & CI

```bash
pip install -r requirements-dev.txt
pytest -q                 # 5 tests: FEM vs analytical, solver vs FEM (both BCs),
                          # metrics, stiffness bounds, end-to-end inverse recovery
```

Every push runs the physics self-tests and the full suite in GitHub Actions
(`.github/workflows/ci.yml`). After pushing, replace `OWNER/REPO` to enable the
badge: `![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)`.

## The website

`website/index.html` is a dependency-free, offline-capable site with six tabs:
**Overview**, **How it works** (a live in-browser beam simulator), **Live demo**
(a scenario explorer driven by the real saved model outputs), **Results**,
**Paper**, and **About**.

## Honest scope & limitations

Validated in simulation on a 1-D Euler–Bernoulli beam under quasi-static load
testing with Gaussian measurement noise. Real structures add shear effects,
support flexibility, temperature variation, model-form error, and dynamic
excitation. The paper's Discussion lays out the path to those extensions. Damage
*severity* is less identifiable than *location* under very sparse sensing (a
width/depth trade-off discussed in the paper); Aegis is most reliable as a
localization-and-screening tool.

## License

Released for educational and research use.
